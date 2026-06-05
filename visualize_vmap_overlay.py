"""
方案一：图像热力图可视化 (V-Map Overlay)
============================================

功能说明：
    将NIB方法生成的图像显著性图(V-Map)叠加到原图上，直观展示图像中与文本描述最相关的区域。

使用方法：
    python visualize_vmap_overlay.py [选项]

常用选项：
    --num_samples N       生成N张可视化图像（默认：5）
    --output_dir PATH     输出目录（默认：outputs/vmap_overlay）
    --num_steps N         NIB优化步数（默认：10）
    --target_layer N      目标层索引（默认：9）
    --data_root PATH      数据集根目录（默认：datasets）
    --ann_path PATH       标注文件路径（默认：datasets/en_val.json）
    --clip_path PATH      CLIP模型路径（默认：环境变量或models/clip-vit-base-patch32）
    --alpha F             热力图透明度，0-1之间（默认：0.5）

示例：
    python visualize_vmap_overlay.py --num_samples 5
    python visualize_vmap_overlay.py --num_samples 5 --alpha 0.6

输出文件：
    outputs/vmap_overlay/
    ├── vmap_overlay_0001.png
    ├── vmap_overlay_0002.png
    └── ...

依赖：
    - CLIP模型、Flickr8k数据集、salicncy.nib、pytorch_grad_cam
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
from pytorch_grad_cam.utils.image import show_cam_on_image

from datasets import Flickr8kDataset, collate_fn_flickr8k
from salicncy import nib
from scripts.clip_local import load_clip_local

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


def create_professional_colormap():
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF8000', '#FF0000', '#8B0000']
    return LinearSegmentedColormap.from_list('professional', colors, N=256)


def visualize_vmap_professional(img, v_saliency, caption, t_saliency, token_words, title=None):
    fig = plt.figure(figsize=(12, 6))
    fig.patch.set_facecolor('white')

    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1],
                          hspace=0.3, wspace=0.25,
                          left=0.08, right=0.92, top=0.90, bottom=0.15)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img)
    ax1.set_title('(a) Original Image', fontsize=12, fontweight='bold', pad=8)
    ax1.axis('off')
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)

    ax2 = fig.add_subplot(gs[0, 1])
    img_resized = img.resize((224, 224))
    img_np = np.array(img_resized).astype(np.float32) / 255.0

    if v_saliency.ndim == 3:
        v_saliency_2d = v_saliency[:, :, 0]
    else:
        v_saliency_2d = v_saliency

    overlay = show_cam_on_image(img_np, v_saliency_2d, use_rgb=True)
    ax2.imshow(overlay)
    ax2.set_title('(b) NIB Saliency Heatmap', fontsize=12, fontweight='bold', pad=8)
    ax2.axis('off')
    for spine in ax2.spines.values():
        spine.set_linewidth(1.5)

    cmap = create_professional_colormap()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cax = fig.add_axes([0.93, 0.45, 0.015, 0.35])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('Importance', fontsize=10)

    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')

    tmap = np.array(t_saliency[1:-1])
    words = [str(w).split('<')[0].strip() for w in token_words[1:-1]]
    valid_len = min(len(words), len(tmap))
    words = words[:valid_len]
    tmap = tmap[:valid_len]

    if len(words) > 0:
        norm_tmap = (tmap - tmap.min()) / (tmap.max() - tmap.min() + 1e-8)
        x_pos = 0.05
        max_x = 0.95

        ax3.text(0.02, 0.85, 'Caption:', fontsize=11, fontweight='bold', va='top')

        for i, (word, score) in enumerate(zip(words, tmap)):
            color = cmap(norm_tmap[i])
            ax3.text(x_pos, 0.5, word, fontsize=12, fontweight='bold',
                    va='center', ha='left',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor=color,
                             edgecolor='navy', linewidth=0.5, alpha=0.9))
            x_pos += len(word) * 0.025 + 0.05

            if x_pos > max_x and i < len(words) - 1:
                x_pos = 0.05
                ax3.text(0.02, 0.3, 'Continuation:', fontsize=9, fontweight='bold', va='top')
                break

    max_val = float(np.max(v_saliency_2d))
    mean_val = float(np.mean(v_saliency_2d))
    std_val = float(np.std(v_saliency_2d))
    coverage = float(np.sum(v_saliency_2d > 0.5) / v_saliency_2d.size) * 100

    stats_text = (f'Saliency Statistics: '
                  f'Max={max_val:.3f}  '
                  f'Mean={mean_val:.3f}  '
                  f'Std={std_val:.3f}  '
                  f'Coverage={coverage:.1f}%')
    ax3.text(0.5, 0.05, stats_text, fontsize=10, ha='center', va='bottom',
            color='darkblue', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='orange'))

    fig.suptitle('NIB Saliency Visualization', fontsize=14, fontweight='bold', y=0.98)

    if title:
        plt.savefig(title, bbox_inches='tight', dpi=150, facecolor='white', edgecolor='none')
        plt.close()
    else:
        plt.show()


def main():
    argparser = argparse.ArgumentParser(description="Generate professional NIB saliency visualizations")
    argparser.add_argument("--num_samples", type=int, default=5)
    argparser.add_argument("--output_dir", type=str, default="outputs/vmap_overlay")
    argparser.add_argument("--num_steps", type=int, default=10)
    argparser.add_argument("--target_layer", type=int, default=9)
    argparser.add_argument("--data_root", type=str, default="datasets")
    argparser.add_argument("--ann_path", type=str, default="datasets/en_val.json")
    argparser.add_argument("--clip_path", type=str, default=None)
    argparser.add_argument("--alpha", type=float, default=0.5)
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

    print(f"\nGenerating {args.num_samples} visualizations...")
    count = 0
    for imgs, texts, batch_xs in tqdm(dataloader, total=args.num_samples):
        if count >= args.num_samples:
            break

        img = imgs[0]
        captions = texts[0]
        batch_xs = batch_xs.to(device)

        best_caption = captions[0]
        best_sim = -1.0

        with torch.no_grad():
            im_feature = extract_image_features(model, batch_xs)
            for caption in captions:
                tid = torch.tensor([tokenizer.encode(caption, add_special_tokens=True)]).to(device)
                tf = extract_text_features(model, tid)
                sim = torch.nn.functional.cosine_similarity(im_feature, tf).item()
                if sim > best_sim:
                    best_sim = sim
                    best_caption = caption

        print(f"\nSample {count + 1}:")
        print(f"  Caption: {best_caption}")
        print(f"  Similarity: {best_sim:.4f}")

        tid = torch.tensor([tokenizer.encode(best_caption, add_special_tokens=True)]).to(device)
        v_saliency, t_saliency = nib(model, [tid], batch_xs, args.num_steps, args.target_layer)

        v_saliency = normalize(np.squeeze(v_saliency[0]))
        if v_saliency.ndim == 3:
            v_saliency = v_saliency.transpose(1, 2, 0)
        if v_saliency.ndim == 2:
            v_saliency = np.stack([v_saliency] * 3, axis=-1)

        t_saliency = normalize(np.squeeze(t_saliency[0]))

        tokens = tokenizer.encode(best_caption, add_special_tokens=True)
        token_words = [tokenizer.decode([t]).strip() for t in tokens]

        output_path = os.path.join(args.output_dir, f"vmap_overlay_{count + 1:04d}.png")
        visualize_vmap_professional(img, v_saliency, best_caption, t_saliency, token_words, title=output_path)
        print(f"  Saved: {output_path}")

        count += 1

    print(f"\nDone. Saved {count} visualizations to '{args.output_dir}'.")


if __name__ == "__main__":
    main()