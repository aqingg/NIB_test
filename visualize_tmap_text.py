"""
方案二：词级消融实验 (Token-level Ablation Study)
=============================================

功能说明：
    通过词级消融实验验证NIB识别的文本Token重要性是否真正影响CLIP的决策。
    将NIB识别的高分词删除/替换，观察CLIP图文相似度变化，证明NIB的可解释性。

实验设计：
    1. 使用NIB生成文本Token重要性分数(T-Map)
    2. 按重要性排序Token
    3. 依次删除最不重要的Token（用[MASK]替换）
    4. 每次删除后重新计算CLIP相似度
    5. 如果NIB有效，删除重要词时相似度会显著下降

使用方法：
    python visualize_tmap_text.py [选项]

常用选项：
    --num_samples N       生成N张可视化图像（默认：5）
    --output_dir PATH     输出目录（默认：outputs/tmap_ablation）
    --num_steps N         NIB优化步数（默认：10）
    --target_layer N      目标层索引（默认：9）

示例：
    python visualize_tmap_text.py --num_samples 3

输出文件：
    outputs/tmap_ablation/
    ├── tmap_ablation_0001.png
    ├── tmap_ablation_0002.png
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
    colors = ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF8000', '#FF0000']
    return LinearSegmentedColormap.from_list('professional', colors, N=256)


def visualize_token_ablation(img, caption, token_words, t_saliency, 
                            similarities, ratios, title=None):
    tmap = np.array(t_saliency[1:-1])
    words = [str(w).split('<')[0].strip() for w in token_words[1:-1]]
    valid_len = min(len(words), len(tmap))
    words = words[:valid_len]
    tmap = tmap[:valid_len]

    sorted_indices = np.argsort(tmap)[::-1]
    
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor('white')

    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], width_ratios=[1, 1.2],
                          hspace=0.25, wspace=0.2,
                          left=0.05, right=0.95, top=0.92, bottom=0.15)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img)
    ax_img.set_title('(a) Input Image', fontsize=12, fontweight='bold', pad=8)
    ax_img.axis('off')
    for spine in ax_img.spines.values():
        spine.set_linewidth(1.5)

    cmap = create_professional_colormap()
    norm_tmap = (tmap - tmap.min()) / (tmap.max() - tmap.min() + 1e-8)

    ax_tokens = fig.add_subplot(gs[0, 1])
    ax_tokens.set_title('(b) Token Importance (T-Map)', fontsize=12, fontweight='bold', pad=8)
    ax_tokens.axis('off')

    # 准备按重要性排序的单词和分数
    sorted_words = [words[idx] for idx in sorted_indices]
    sorted_scores = [tmap[idx] for idx in sorted_indices]
    sorted_colors = [cmap(norm_tmap[idx]) for idx in sorted_indices]

    # 使用合适的布局，避免重叠
    max_chars_per_line = 30
    line_height = 0.35
    start_y = 0.75
    start_x = 0.02

    current_line = []
    current_line_chars = 0
    current_y = start_y

    for word, score, color in zip(sorted_words, sorted_scores, sorted_colors):
        word_length = len(word)
        if current_line_chars + word_length + 2 > max_chars_per_line and current_line:
            # 绘制当前行
            line_x = start_x
            for w, s, c in current_line:
                ax_tokens.text(line_x, current_y, w, fontsize=11, fontweight='bold',
                              va='center', ha='left',
                              bbox=dict(boxstyle='round,pad=0.2', facecolor=c,
                                       edgecolor='navy', linewidth=0.5, alpha=0.9))
                ax_tokens.text(line_x, current_y - 0.12, f'{s:.3f}', fontsize=9,
                              va='center', ha='left', color='gray')
                line_x += len(w) * 0.03 + 0.03
            
            current_line = []
            current_line_chars = 0
            current_y -= line_height
            
            if current_y < 0.1:
                break
        
        current_line.append((word, score, color))
        current_line_chars += word_length + 2  # +2 for space
    
    # 绘制最后一行
    if current_line:
        line_x = start_x
        for w, s, c in current_line:
            ax_tokens.text(line_x, current_y, w, fontsize=11, fontweight='bold',
                          va='center', ha='left',
                          bbox=dict(boxstyle='round,pad=0.2', facecolor=c,
                                   edgecolor='navy', linewidth=0.5, alpha=0.9))
            ax_tokens.text(line_x, current_y - 0.12, f'{s:.3f}', fontsize=9,
                          va='center', ha='left', color='gray')
            line_x += len(w) * 0.03 + 0.03

    ax_curve = fig.add_subplot(gs[1, 0])
    ax_curve.plot(ratios, similarities, 'o-', linewidth=3, markersize=8,
                  color='#2171b5', markerfacecolor='#6baed6')

    critical_point = np.argmax(np.diff(similarities))
    ax_curve.axvline(ratios[critical_point], color='red', linestyle='--',
                     label=f'Critical: {ratios[critical_point]:.2f}')

    ax_curve.set_xlabel('Tokens Kept (%)', fontsize=11)
    ax_curve.set_ylabel('CLIP Similarity', fontsize=11)
    ax_curve.set_title('(c) Ablation Curve', fontsize=12, fontweight='bold', pad=8)
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend()
    ax_curve.set_xlim(-0.02, 1.02)

    ax_bar = fig.add_subplot(gs[1, 1])
    bar_colors = plt.cm.RdYlGn_r(np.array(similarities) / max(similarities))
    bars = ax_bar.bar(np.arange(len(similarities)), similarities, 
                      color=bar_colors, edgecolor='black')
    
    ax_bar.set_xticks(np.arange(len(similarities)))
    ax_bar.set_xticklabels([f'{int(r*100)}%' for r in ratios], rotation=45, fontsize=9)
    ax_bar.set_xlabel('Tokens Kept', fontsize=11)
    ax_bar.set_ylabel('Similarity', fontsize=11)
    ax_bar.set_title('(d) Ablation Bar Chart', fontsize=12, fontweight='bold', pad=8)
    ax_bar.grid(True, alpha=0.3, axis='y')

    original_sim = similarities[-1]
    max_drop = original_sim - min(similarities)
    drop_ratio = (max_drop / original_sim) * 100

    stats_text = (f'Token Ablation Statistics | Original={original_sim:.4f} | '
                  f'Max Drop={max_drop:.4f} ({drop_ratio:.1f}%) | '
                  f'Critical Point={ratios[critical_point]:.2f}')
    
    stats_ax = fig.add_axes([0.05, 0.08, 0.9, 0.05])
    stats_ax.axis('off')
    stats_ax.text(0.5, 0.5, stats_text, fontsize=10, ha='center', va='center',
                  fontweight='bold', color='darkblue',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='orange'))

    fig.suptitle(f'Token-Level Ablation Study\nOriginal Caption: "{caption}"',
                fontsize=14, fontweight='bold', y=0.98)

    if title:
        plt.savefig(title, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
    else:
        plt.show()


def main():
    argparser = argparse.ArgumentParser(description="Token-level ablation study for NIB")
    argparser.add_argument("--num_samples", type=int, default=3)
    argparser.add_argument("--output_dir", type=str, default="outputs/tmap_ablation")
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

    print(f"\nGenerating {args.num_samples} token ablation visualizations...")
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
            im_feature_norm = im_feature / im_feature.norm(dim=-1, keepdim=True)
            for caption in captions:
                tid = torch.tensor([tokenizer.encode(caption, add_special_tokens=True)]).to(device)
                tf = extract_text_features(model, tid)
                tf_norm = tf / tf.norm(dim=-1, keepdim=True)
                sim = torch.nn.functional.cosine_similarity(im_feature_norm, tf_norm).item()
                if sim > best_sim:
                    best_sim = sim
                    best_caption = caption

        print(f"\nSample {count + 1}:")
        print(f"  Caption: {best_caption}")
        print(f"  Original Similarity: {best_sim:.4f}")

        tid = torch.tensor([tokenizer.encode(best_caption, add_special_tokens=True)]).to(device)
        _, t_saliency = nib(model, [tid], batch_xs, args.num_steps, args.target_layer)

        t_saliency = normalize(np.squeeze(t_saliency[0]))
        tokens = tokenizer.encode(best_caption, add_special_tokens=True)
        token_words = [tokenizer.decode([t]).strip() for t in tokens]

        tmap = np.array(t_saliency[1:-1])
        words = [str(w).split('<')[0].strip() for w in token_words[1:-1]]
        valid_len = min(len(words), len(tmap))
        words = words[:valid_len]
        tmap = tmap[:valid_len]

        sorted_indices = np.argsort(tmap)

        num_tokens = len(words)
        num_ablation = min(num_tokens, 8)
        ratios = np.linspace(0.0, 1.0, num_ablation + 1)[::-1]
        similarities = []

        for ratio in ratios:
            num_to_keep = int(num_tokens * ratio)
            kept_indices = sorted_indices[-num_to_keep:] if num_to_keep > 0 else []
            
            new_words = []
            for i in range(num_tokens):
                if i in kept_indices:
                    new_words.append(words[i])
                else:
                    new_words.append('[MASK]')
            
            masked_caption = ' '.join(new_words)
            
            masked_tid = torch.tensor([tokenizer.encode(masked_caption, add_special_tokens=True)]).to(device)
            masked_tf = extract_text_features(model, masked_tid)
            # 归一化特征以确保余弦相似度在[-1, 1]范围内
            im_feature_norm = im_feature / im_feature.norm(dim=-1, keepdim=True)
            masked_tf_norm = masked_tf / masked_tf.norm(dim=-1, keepdim=True)
            sim = torch.nn.functional.cosine_similarity(im_feature_norm, masked_tf_norm).item()
            similarities.append(sim)

        output_path = os.path.join(args.output_dir, f"tmap_ablation_{count + 1:04d}.png")
        visualize_token_ablation(img, best_caption, token_words, t_saliency,
                                similarities, ratios, title=output_path)

        max_drop = best_sim - min(similarities)
        drop_ratio = (max_drop / best_sim) * 100
        print(f"  Max Similarity Drop: {max_drop:.4f} ({drop_ratio:.1f}%)")
        print(f"  Saved: {output_path}")

        count += 1

    print(f"\nDone. Saved {count} token ablation visualizations to '{args.output_dir}'.")


if __name__ == "__main__":
    main()