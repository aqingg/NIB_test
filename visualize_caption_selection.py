"""
方案三：Caption对比与选择可视化 (Caption Selection)
===================================================

功能说明：
    展示Flickr8k数据集中每张图像的5个候选Caption如何通过余弦相似度
    被选择最匹配的一个，并生成详细的可视化分析报告。

使用方法：
    python visualize_caption_selection.py [选项]

常用选项：
    --num_samples N       生成N张可视化图像（默认：5）
    --output_dir PATH     输出目录（默认：outputs/caption_selection）
    --num_steps N         NIB优化步数（默认：10）
    --target_layer N      目标层索引（默认：9）
    --data_root PATH      数据集根目录（默认：datasets）
    --ann_path PATH       标注文件路径（默认：datasets/en_val.json）
    --clip_path PATH      CLIP模型路径（默认：环境变量或models/clip-vit-base-patch32）

示例：
    python visualize_caption_selection.py --num_samples 5

输出文件：
    outputs/caption_selection/
    ├── caption_selection_0001.png
    ├── caption_selection_0002.png
    └── ...
"""

import argparse
import os
import warnings

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from datasets import Flickr8kDataset, collate_fn_flickr8k
from salicncy import nib
from scripts.clip_local import load_clip_local
from pytorch_grad_cam.utils.image import show_cam_on_image

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def normalize(x):
    x_min = x.min()
    x_max = x.max()
    if x_max == x_min:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def extract_text_features(model, input_ids):
    outputs = model.get_text_features(input_ids=input_ids)
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "text_embeds"):
        return outputs.text_embeds
    if hasattr(outputs, "pooler_output"):
        pooled = outputs.pooler_output
        if hasattr(model, "text_projection"):
            return model.text_projection(pooled)
        return pooled
    raise TypeError(f"Unsupported text feature output type: {type(outputs)}")


def extract_image_features(model, pixel_values):
    outputs = model.get_image_features(pixel_values=pixel_values)
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "image_embeds"):
        return outputs.image_embeds
    if hasattr(outputs, "pooler_output"):
        return outputs.pooler_output
    raise TypeError(f"Unsupported image feature output type: {type(outputs)}")


def get_caption_similarities(model, tokenizer, im_feature, captions):
    similarities = []
    for caption in captions:
        tid = torch.tensor([tokenizer.encode(caption, add_special_tokens=True)]).to(im_feature.device)
        tf = extract_text_features(model, tid)
        sim = torch.nn.functional.cosine_similarity(im_feature, tf).item()
        similarities.append(sim)
    return similarities


def create_professional_colormap():
    colors = ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF8000', '#FF0000']
    return LinearSegmentedColormap.from_list('professional', colors, N=256)


def visualize_caption_pro(img, captions, similarities, best_idx,
                          v_saliency, t_saliency, token_words, title=None):
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor('white')

    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1.2],
                          hspace=0.25, wspace=0.2,
                          left=0.05, right=0.95, top=0.92, bottom=0.15)

    cmap = create_professional_colormap()

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img)
    ax_img.set_title('(a) Input Image', fontsize=12, fontweight='bold', pad=8)
    ax_img.axis('off')
    for spine in ax_img.spines.values():
        spine.set_linewidth(1.5)

    sorted_indices = np.argsort(similarities)[::-1]
    sorted_sims = [similarities[i] for i in sorted_indices]
    sorted_caps = [captions[i] for i in sorted_indices]

    ax_bar = fig.add_subplot(gs[0, 1])
    ax_bar.set_title('(b) Caption Similarity Ranking', fontsize=12, fontweight='bold', pad=8)

    y_pos = np.arange(len(sorted_sims))
    colors = plt.cm.RdYlGn_r((np.array(sorted_sims) - min(sorted_sims)) / (max(sorted_sims) - min(sorted_sims) + 1e-8))
    bars = ax_bar.barh(y_pos, sorted_sims, color=colors, edgecolor='black', height=0.6)

    for i, (bar, sim, cap) in enumerate(zip(bars, sorted_sims, sorted_caps)):
        cap_short = cap[:25] + '...' if len(cap) > 25 else cap
        ax_bar.text(-0.01, bar.get_y() + bar.get_height() / 2,
                    f'#{sorted_indices[i] + 1}: {cap_short}',
                    fontsize=10, va='center', ha='right', fontweight='bold')
        ax_bar.text(sim + 0.005, bar.get_y() + bar.get_height() / 2,
                    f'{sim:.4f}', fontsize=9, va='center', color='gray')

    ax_bar.set_xlim(min(sorted_sims) - 0.02, max(sorted_sims) + 0.05)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel('Cosine Similarity', fontsize=10)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)

    ax_vmap = fig.add_subplot(gs[1, 0])
    img_resized = np.array(img.resize((224, 224))).astype(np.float32) / 255.0
    
    if v_saliency.ndim == 3:
        v_saliency_2d = v_saliency[:, :, 0]
    else:
        v_saliency_2d = v_saliency
    
    overlay = show_cam_on_image(img_resized, v_saliency_2d, use_rgb=True)
    ax_vmap.imshow(overlay)
    ax_vmap.set_title('(c) Image Saliency (V-Map)', fontsize=12, fontweight='bold', pad=8)
    ax_vmap.axis('off')
    for spine in ax_vmap.spines.values():
        spine.set_linewidth(1.5)

    ax_detail = fig.add_subplot(gs[1, 1])
    ax_detail.set_title('(d) Selected Caption & Token Importance', fontsize=12, fontweight='bold', pad=8)
    ax_detail.axis('off')

    selected_caption = captions[best_idx]
    ax_detail.text(0.02, 0.9, f'Selected Caption #{best_idx + 1}:', fontsize=11, fontweight='bold', va='top')
    ax_detail.text(0.02, 0.75, f'"{selected_caption}"', fontsize=12, style='italic', va='top', wrap=True)
    ax_detail.text(0.02, 0.65, f'Similarity: {similarities[best_idx]:.4f}', fontsize=11, color='green', fontweight='bold')

    tmap = np.array(t_saliency[1:-1])
    words = [str(w).split('<')[0].strip() for w in token_words[1:-1]]
    valid_len = min(len(words), len(tmap))
    words = words[:valid_len]
    tmap = tmap[:valid_len]

    if len(words) > 0:
        ax_detail.text(0.02, 0.45, 'Token Importance (T-Map):', fontsize=10, fontweight='bold', va='top')
        
        norm_tmap = (tmap - tmap.min()) / (tmap.max() - tmap.min() + 1e-8)
        x_pos = 0.02
        y_pos = 0.3
        max_x = 0.98

        for i, (word, score) in enumerate(zip(words, tmap)):
            color = cmap(norm_tmap[i])
            ax_detail.text(x_pos, y_pos, word, fontsize=11, fontweight='bold',
                          va='center', ha='left',
                          bbox=dict(boxstyle='round,pad=0.2', facecolor=color,
                                   edgecolor='navy', linewidth=0.5, alpha=0.9))
            x_pos += len(word) * 0.02 + 0.04

            if x_pos > max_x and i < len(words) - 1:
                x_pos = 0.02
                y_pos -= 0.12

    max_sim = max(similarities)
    min_sim = min(similarities)
    sim_range = max_sim - min_sim
    avg_sim = np.mean(similarities)

    stats_text = (f'Selection Statistics | Range: {sim_range:.4f} | '
                  f'Average: {avg_sim:.4f} | Best: Caption #{best_idx + 1} ({max_sim:.4f})')
    
    ax_stats = fig.add_axes([0.05, 0.08, 0.9, 0.05])
    ax_stats.axis('off')
    ax_stats.text(0.5, 0.5, stats_text, fontsize=10, ha='center', va='center',
                  fontweight='bold', color='darkblue',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='orange'))

    fig.suptitle('NIB Caption Selection Analysis', fontsize=14, fontweight='bold', y=0.98)

    if title:
        plt.savefig(title, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
    else:
        plt.show()


def main():
    argparser = argparse.ArgumentParser(description="Caption selection visualization using NIB")
    argparser.add_argument("--num_samples", type=int, default=5)
    argparser.add_argument("--output_dir", type=str, default="outputs/caption_selection")
    argparser.add_argument("--num_steps", type=int, default=10)
    argparser.add_argument("--target_layer", type=int, default=9)
    argparser.add_argument("--data_root", type=str, default="datasets")
    argparser.add_argument("--ann_path", type=str, default="datasets/en_val.json")
    argparser.add_argument("--clip_path", type=str, default=None)
    args = argparser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    clip_path = args.clip_path if args.clip_path else os.environ.get("CLIP_PATH", r"D:\NIB-main\models\clip-vit-base-patch32")
    print(f"Loading CLIP model from: {clip_path}")
    model, processor, tokenizer = load_clip_local(clip_path, device)

    print("Loading Flickr8k dataset...")
    dataset = Flickr8kDataset(args.data_root, args.ann_path, image_preprocessor=processor)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn_flickr8k)

    model.eval()

    print(f"\nGenerating {args.num_samples} caption selection visualizations...")
    count = 0
    for imgs, texts, batch_xs in tqdm(dataloader, total=args.num_samples):
        if count >= args.num_samples:
            break

        img = imgs[0]
        captions = texts[0]
        batch_xs = batch_xs.to(device)

        similarities = get_caption_similarities(model, tokenizer, extract_image_features(model, batch_xs), captions)
        best_idx = np.argmax(similarities)

        print(f"\nSample {count + 1}:")
        print(f"  Best Caption #{best_idx + 1}: {captions[best_idx]}")
        print(f"  Similarity: {similarities[best_idx]:.4f}")

        tid = torch.tensor([tokenizer.encode(captions[best_idx], add_special_tokens=True)]).to(device)
        v_saliency, t_saliency = nib(model, [tid], batch_xs, args.num_steps, args.target_layer)

        v_saliency = normalize(np.squeeze(v_saliency[0]))
        if v_saliency.ndim == 3:
            v_saliency = v_saliency.transpose(1, 2, 0)

        t_saliency = normalize(np.squeeze(t_saliency[0]))

        tokens = tokenizer.encode(captions[best_idx], add_special_tokens=True)
        token_words = [tokenizer.decode([t]).strip() for t in tokens]

        output_path = os.path.join(args.output_dir, f"caption_selection_{count + 1:04d}.png")
        visualize_caption_pro(img, captions, similarities, best_idx,
                              v_saliency, t_saliency, token_words, title=output_path)
        print(f"  Saved: {output_path}")

        count += 1

    print(f"\nDone. Saved {count} visualizations to '{args.output_dir}'.")


if __name__ == "__main__":
    main()