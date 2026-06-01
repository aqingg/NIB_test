"""
方案二：文本Token重要性可视化 (T-Map Text)
============================================

功能说明：
    将NIB方法生成的文本Token重要性分数(T-Map)以可视化方式展示，
    通过颜色编码显示每个词对图像-文本匹配的重要程度。

使用方法：
    python visualize_tmap_text.py [选项]

常用选项：
    --num_samples N       生成N张可视化图像（默认：5）
    --output_dir PATH     输出目录（默认：outputs/tmap_text）
    --num_steps N         NIB优化步数（默认：10）
    --target_layer N      目标层索引（默认：9）
    --data_root PATH      数据集根目录（默认：datasets）
    --ann_path PATH       标注文件路径（默认：datasets/en_val.json）
    --clip_path PATH      CLIP模型路径（默认：环境变量或models/clip-vit-base-patch32）
    --viz_type TYPE       可视化类型：combined/heatmap/both（默认：combined）
    --fontsize N          文本字体大小（默认：14）

可视化类型说明：
    combined  - 上下布局：上方为重要性排名条形图，下方为Caption着色展示（默认）
    heatmap  - 纯横向条形图：每个Token及其重要性分数
    both     - 同时生成两种类型

示例：
    # 基本用法：生成5张文本可视化（combined模式）
    python visualize_tmap_text.py --num_samples 5

    # 生成两种可视化类型
    python visualize_tmap_text.py --num_samples 5 --viz_type both

    # 调整字体大小
    python visualize_tmap_text.py --num_samples 5 --fontsize 18

    # 修改NIB参数
    python visualize_tmap_text.py --num_samples 5 --num_steps 20 --target_layer 9

输出文件：
    outputs/tmap_text/
    ├── tmap_text_0001_combined.png    # Combined模式（综合可视化）
    ├── tmap_text_0001_heatmap.png     # Heatmap模式（纯排名图）
    └── ...

可视化说明：
    - 重要性分数越高，颜色越深（蓝色渐变）
    - 最重要的词会加粗并增大字体
    - 每个词下方标注具体的重要性分数
    - 颜色条图例显示分数与颜色的对应关系

依赖：
    - CLIP模型（openai/clip-vit-base-patch32或本地路径）
    - Flickr8k数据集
    - salicncy.nib (NIB显著性方法)
    - matplotlib (可视化绘图)
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
import cv2

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


def create_importance_colormap():
    colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', 
              '#4292c6', '#2171b5', '#08519c', '#08306b']
    return LinearSegmentedColormap.from_list('importance', colors)


def visualize_text_importance(tmap, token_words, title=None, figsize=(14, 6), fontsize=14):
    tmap = np.array(tmap[1:-1])
    token_words = [str(x).split('<')[0].strip() for x in token_words[1:-1]]
    
    valid_len = min(len(token_words), len(tmap))
    token_words = token_words[:valid_len]
    tmap = tmap[:valid_len]

    if len(token_words) == 0:
        return

    sorted_indices = np.argsort(tmap)[::-1]
    top_idx = sorted_indices[0]
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1, 2]})
    fig.patch.set_facecolor('white')
    
    cmap = create_importance_colormap()
    norm_scores = (tmap - tmap.min()) / (tmap.max() - tmap.min() + 1e-8)
    
    ax2 = axes[0]
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('Token Importance Distribution', fontsize=fontsize, fontweight='bold', pad=10)
    
    bar_height = 0.15
    max_bars = 10
    show_indices = sorted_indices[:max_bars] if len(sorted_indices) > max_bars else sorted_indices
    
    for i, idx in enumerate(show_indices):
        y_pos = 0.8 - i * (0.7 / max_bars)
        score = tmap[idx]
        color = cmap(norm_scores[idx])
        width = norm_scores[idx]
        
        rect = mpatches.FancyBboxPatch((0, y_pos - bar_height/2), width * 0.95, bar_height,
                                        boxstyle="round,pad=0.01", facecolor=color, 
                                        edgecolor='darkblue', linewidth=0.5)
        ax2.add_patch(rect)
        
        ax2.text(0.02, y_pos, f'{token_words[idx]}: {score:.4f}', 
                fontsize=fontsize-2, va='center', ha='left', fontweight='bold')
        ax2.text(width * 0.95 + 0.02, y_pos, f'{score:.3f}', 
                fontsize=fontsize-3, va='center', ha='left', color='gray')
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    ax1 = axes[1]
    ax1.set_title('Caption with Token Importance', fontsize=fontsize, fontweight='bold', pad=10)
    
    x_pos = 0.02
    y_pos = 0.5
    max_x = 0.95
    line_height = 0.22
    current_y = y_pos
    
    for i, (word, score) in enumerate(zip(token_words, tmap)):
        color = cmap(norm_scores[i])
        
        if i == top_idx:
            fontweight = 'bold'
            fontsize_display = fontsize + 2
        else:
            fontweight = 'normal'
            fontsize_display = fontsize
        
        ax1.text(x_pos, current_y, word, fontsize=fontsize_display, 
                fontweight=fontweight, va='center', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, 
                         edgecolor='navy', linewidth=0.8, alpha=0.85),
                transform=ax1.transAxes)
        
        x_pos += len(word) * 0.025 + 0.03
        
        if x_pos > max_x and i < len(token_words) - 1:
            x_pos = 0.02
            current_y -= line_height
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.1, 1)
    ax1.axis('off')
    
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.35])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=tmap.min(), vmax=tmap.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('Importance Score', fontsize=fontsize-2)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    if title:
        plt.savefig(title, bbox_inches='tight', dpi=150, facecolor='white', edgecolor='none')
        plt.close()
    else:
        plt.show()


def visualize_text_heatmap_vertical(token_words, scores, title=None, figsize=(10, 8)):
    token_words = [str(x).split('<')[0].strip() for x in token_words[1:-1]]
    scores = np.array(scores[1:-1])
    
    valid_len = min(len(token_words), len(scores))
    token_words = token_words[:valid_len]
    scores = scores[:valid_len]
    
    if len(token_words) == 0:
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    
    n_tokens = len(token_words)
    
    cmap = plt.cm.RdYlGn_r
    norm = plt.Normalize(vmin=scores.min(), vmax=scores.max())
    
    for i, (word, score) in enumerate(zip(token_words, scores)):
        color = cmap(norm(score))
        
        ax.barh(i, score, color=color, edgecolor='black', linewidth=0.5, height=0.7)
        
        ax.text(-0.02, i, word, fontsize=12, va='center', ha='right', fontweight='bold')
        ax.text(score + 0.01, i, f'{score:.3f}', fontsize=10, va='center', ha='left', color='gray')
    
    ax.set_yticks(range(n_tokens))
    ax.set_yticklabels([])
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('Text Token Importance Analysis', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(-0.1, scores.max() * 1.3)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Importance', fontsize=11)
    
    plt.tight_layout()
    
    if title:
        plt.savefig(title, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
    else:
        plt.show()


def main():
    argparser = argparse.ArgumentParser(description="Generate enhanced text token saliency visualizations using NIB")
    argparser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize")
    argparser.add_argument("--output_dir", type=str, default="outputs/tmap_text", help="Output directory")
    argparser.add_argument("--num_steps", type=int, default=10, help="Number of NIB optimization steps")
    argparser.add_argument("--target_layer", type=int, default=9, help="Target layer for NIB")
    argparser.add_argument("--data_root", type=str, default="datasets", help="Dataset root directory")
    argparser.add_argument("--ann_path", type=str, default="datasets/en_val.json", help="Annotation file path")
    argparser.add_argument("--clip_path", type=str, default=None, help="Path to local CLIP model")
    argparser.add_argument("--viz_type", type=str, default="combined", choices=["combined", "heatmap", "both"],
                          help="Visualization type: combined (box+bar), heatmap (bar only), or both")
    argparser.add_argument("--fontsize", type=int, default=14, help="Font size for text tokens")
    args = argparser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.clip_path:
        clip_path = args.clip_path
    else:
        clip_path = os.environ.get("CLIP_PATH", r"D:\NIB-main\models\clip-vit-base-patch32")

    print(f"Loading CLIP model from: {clip_path}")
    model, processor, tokenizer = load_clip_local(clip_path, device)

    print("Loading Flickr8k dataset...")
    dataset = Flickr8kDataset(args.data_root, args.ann_path, image_preprocessor=processor)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn_flickr8k,
    )

    model.eval()

    print(f"\nGenerating {args.num_samples} text saliency visualizations...")
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
        print(f"  Best caption: {best_caption}")
        print(f"  Similarity: {best_sim:.4f}")

        tid = torch.tensor([tokenizer.encode(best_caption, add_special_tokens=True)]).to(device)
        _, t_saliency = nib(model, [tid], batch_xs, args.num_steps, args.target_layer)

        t_saliency = normalize(np.squeeze(t_saliency[0]))

        tokens = tokenizer.encode(best_caption, add_special_tokens=True)
        token_words = [tokenizer.decode([t]).strip() for t in tokens]

        base_name = f"tmap_text_{count + 1:04d}"

        if args.viz_type in ["combined", "both"]:
            output_path = os.path.join(args.output_dir, f"{base_name}_combined.png")
            visualize_text_importance(
                t_saliency,
                token_words,
                title=output_path,
                fontsize=args.fontsize
            )
            print(f"  Saved combined: {output_path}")

        if args.viz_type in ["heatmap", "both"]:
            output_path = os.path.join(args.output_dir, f"{base_name}_heatmap.png")
            visualize_text_heatmap_vertical(
                token_words,
                t_saliency,
                title=output_path
            )
            print(f"  Saved heatmap: {output_path}")

        if args.viz_type == "both":
            print(f"  (Both visualizations saved)")

        count += 1

    print(f"\nDone. Saved {count} visualizations to '{args.output_dir}'.")


if __name__ == "__main__":
    main()