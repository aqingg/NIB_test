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
    --viz_type TYPE       可视化类型：detailed/summary/both（默认：detailed）

可视化类型说明：
    detailed  - 详细模式：原图 + VMap热力图 + 5个Caption对比 + TMap（默认）
    summary   - 摘要模式：原图 + 相似度排名条形图 + Top3详情
    both      - 同时生成两种类型

示例：
    # 基本用法：生成5张详细可视化
    python visualize_caption_selection.py --num_samples 5

    # 生成两种可视化类型
    python visualize_caption_selection.py --num_samples 5 --viz_type both

    # 仅生成摘要模式
    python visualize_caption_selection.py --num_samples 5 --viz_type summary

    # 修改NIB参数
    python visualize_caption_selection.py --num_samples 5 --num_steps 20 --target_layer 9

输出文件：
    outputs/caption_selection/
    ├── caption_selection_0001_detailed.png   # 详细模式
    ├── caption_selection_0001_summary.png      # 摘要模式
    └── ...

Detailed模式内容：
    ┌────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
    │        │Caption 1│Caption 2│Caption 3│Caption 4│Caption 5│
    │  Image │ Sim:0.02│ Sim:0.03│ Sim:0.04│ Sim:0.03│ Sim:0.01│
    │  (VMap)│         │         │ ✓ SELECTED         │         │
    ├────────┴─────────┴─────────┴─────────┴─────────┴─────────┤
    │  T-Map: [A] [child] [climbs] [to] [the] [top]...        │
    └─────────────────────────────────────────────────────────┘

Summary模式内容：
    ┌────────┬─────────────────────────────────────────────────┐
    │        │     Caption Similarity Ranking (条形图)            │
    │  Image │  #1 ████████████████ 0.0435  Caption #3         │
    │        │  #2 █████████████    0.0328  Caption #4         │
    ├────────┼─────────────┬─────────────┬─────────────┤
    │ Top 1  │  Top 2      │  Top 3     │  Summary    │
    └────────┴─────────────┴─────────────┴─────────────┘

选择依据：
    - 使用CLIP图像特征与文本特征的余弦相似度
    - 相似度最高的Caption被选中进行NIB分析
    - 每个Caption显示与选中Caption的相似度差异(Δ)

依赖：
    - CLIP模型（openai/clip-vit-base-patch32或本地路径）
    - Flickr8k数据集
    - salicncy.nib (NIB显著性方法)
    - pytorch_grad_cam (热力图生成)
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


def visualize_caption_selection(img, captions, similarities, best_idx,
                                v_saliency, t_saliency, token_words,
                                title=None, figsize=(18, 10)):
    n_captions = len(captions)

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor('white')

    gs = fig.add_gridspec(2, n_captions + 1, height_ratios=[1, 1.2],
                         width_ratios=[1.5] + [1] * n_captions,
                         hspace=0.3, wspace=0.3)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img)
    ax_img.axis('off')
    ax_img.set_title('Input Image', fontsize=12, fontweight='bold')

    ax_vmap = fig.add_subplot(gs[1, 0])
    vmap_resized = np.array(img.resize((224, 224))) / 255.0
    vmap_display = v_saliency if v_saliency.shape[:2] == (224, 224) else np.array(
        Image.fromarray((v_saliency * 255).astype(np.uint8)).resize((224, 224))) / 255.0
    if vmap_display.ndim == 2:
        vmap_display = np.stack([vmap_display] * 3, axis=-1)
    ax_vmap.imshow(show_cam_on_image(vmap_resized, v_saliency, use_rgb=True))
    ax_vmap.axis('off')
    ax_vmap.set_title('Image Saliency (V-Map)', fontsize=12, fontweight='bold')

    cmap = plt.cm.RdYlGn_r
    norm = plt.Normalize(vmin=min(similarities), vmax=max(similarities))

    for i, (caption, sim) in enumerate(zip(captions, similarities)):
        ax = fig.add_subplot(gs[0, i + 1])

        is_best = (i == best_idx)
        is_worst = (i == np.argmin(similarities))

        bg_color = 'lightgreen' if is_best else ('lightcoral' if is_worst else 'white')

        ax.text(0.5, 0.7, f'Caption {i + 1}', fontsize=10, ha='center', va='center',
               fontweight='bold' if is_best else 'normal',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=bg_color, edgecolor='gray'))

        caption_short = caption[:40] + '...' if len(caption) > 40 else caption
        ax.text(0.5, 0.45, f'"{caption_short}"', fontsize=8, ha='center', va='center',
               wrap=True, style='italic')

        color = cmap(norm(sim))
        rect = mpatches.FancyBboxPatch((0.1, 0.25), 0.8, 0.12,
                                        boxstyle="round,pad=0.02",
                                        facecolor=color, edgecolor='black',
                                        transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(0.5, 0.31, f'Sim: {sim:.4f}', fontsize=9, ha='center', va='center',
               fontweight='bold' if is_best else 'normal')

        if is_best:
            ax.text(0.5, 0.08, '✓ SELECTED', fontsize=10, ha='center', va='center',
                   color='green', fontweight='bold')
        else:
            diff = similarities[best_idx] - sim
            ax.text(0.5, 0.08, f'Δ: {diff:.4f}', fontsize=8, ha='center', va='center',
                   color='gray')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    ax_tmap = fig.add_subplot(gs[1, 1:])
    ax_tmap.set_title('Selected Caption Token Importance (T-Map)', fontsize=12, fontweight='bold', pad=10)

    tmap = np.array(t_saliency[1:-1])
    valid_len = min(len(token_words) - 2, len(tmap))
    tmap = tmap[:valid_len]
    words = [str(w).split('<')[0].strip() for w in token_words[1:1+valid_len]]

    if len(words) > 0:
        norm_tmap = (tmap - tmap.min()) / (tmap.max() - tmap.min() + 1e-8)
        cmap_tmap = plt.cm.Blues

        x_pos = 0.02
        y_pos = 0.5
        max_x = 0.95
        line_height = 0.15

        for i, (word, score) in enumerate(zip(words, tmap)):
            color = cmap_tmap(0.3 + 0.7 * norm_tmap[i])

            ax_tmap.text(x_pos, y_pos, word, fontsize=11, fontweight='bold',
                        va='center', ha='left',
                        bbox=dict(boxstyle='round,pad=0.25', facecolor=color,
                                 edgecolor='navy', linewidth=0.5, alpha=0.85))

            x_pos += len(word) * 0.018 + 0.04

            if x_pos > max_x and i < len(words) - 1:
                x_pos = 0.02
                y_pos -= line_height

    ax_tmap.set_xlim(0, 1)
    ax_tmap.set_ylim(-0.2, 1)
    ax_tmap.axis('off')

    selected_caption = captions[best_idx]
    fig.suptitle(f'Caption Selection Analysis\nBest Match: "{selected_caption}" (Sim: {similarities[best_idx]:.4f})',
                fontsize=14, fontweight='bold', y=0.98)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.55, 0.015, 0.35])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('Similarity Score', fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    if title:
        plt.savefig(title, bbox_inches='tight', dpi=150, facecolor='white', edgecolor='none')
        plt.close()
    else:
        plt.show()


def visualize_multi_caption_comparison(img, captions, similarities, title=None, figsize=(16, 12)):
    n_captions = len(captions)
    sorted_indices = np.argsort(similarities)[::-1]

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor('white')

    gs = fig.add_gridspec(3, 3, height_ratios=[1.5, 1, 0.8],
                         width_ratios=[1, 1, 1], hspace=0.35, wspace=0.25)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img)
    ax_img.axis('off')
    ax_img.set_title('Input Image', fontsize=12, fontweight='bold')

    ax_bar = fig.add_subplot(gs[0, 1:])
    ax_bar.set_title('All Captions Ranked by Similarity', fontsize=12, fontweight='bold')

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, n_captions))
    sorted_sims = [similarities[i] for i in sorted_indices]

    bars = ax_bar.barh(range(n_captions), sorted_sims, color=colors, edgecolor='black', height=0.6)
    ax_bar.set_yticks(range(n_captions))
    ax_bar.set_yticklabels([f'Caption #{i+1}' for i in range(n_captions)], fontsize=10)
    ax_bar.set_xlabel('Cosine Similarity', fontsize=11)
    ax_bar.invert_yaxis()

    for i, (bar, sim) in enumerate(zip(bars, sorted_sims)):
        ax_bar.text(sim + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{sim:.4f}', va='center', fontsize=9)

    ax_bar.set_xlim(min(sorted_sims) - 0.05, max(sorted_sims) + 0.1)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)

    for i, idx in enumerate(sorted_indices[:3]):
        ax = fig.add_subplot(gs[1, i])
        is_best = (i == 0)

        ax.text(0.5, 0.9, f'Rank #{i+1}', fontsize=11, ha='center', va='top', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen' if is_best else 'lightgray',
                        edgecolor='black'))
        ax.text(0.5, 0.6, f'"{captions[idx]}"', fontsize=9, ha='center', va='center',
               wrap=True, style='italic',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))

        color = plt.cm.RdYlGn_r((similarities[idx] - min(similarities)) / (max(similarities) - min(similarities) + 1e-8))
        ax.text(0.5, 0.25, f'Similarity: {similarities[idx]:.4f}', fontsize=10, ha='center', va='center',
               fontweight='bold' if is_best else 'normal',
               color='darkgreen' if is_best else 'black')

        if is_best:
            ax.text(0.5, 0.05, '✓ BEST MATCH', fontsize=10, ha='center', va='center',
                   color='green', fontweight='bold')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    ax_summary = fig.add_subplot(gs[2, :])
    ax_summary.axis('off')

    summary_text = f"Selection Summary: "
    summary_text += f"Best: Caption #{sorted_indices[0]+1} (Sim={similarities[sorted_indices[0]]:.4f}) | "
    summary_text += f"Worst: Caption #{sorted_indices[-1]+1} (Sim={similarities[sorted_indices[-1]]:.4f}) | "
    summary_text += f"Range: {max(similarities)-min(similarities):.4f}"

    ax_summary.text(0.5, 0.6, summary_text, fontsize=11, ha='center', va='center',
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='orange'))

    ax_summary.text(0.5, 0.2,
                   f"Selected for NIB Analysis: Caption #{sorted_indices[0]+1}",
                   fontsize=12, ha='center', va='center', fontweight='bold', color='green')

    plt.tight_layout()

    if title:
        plt.savefig(title, bbox_inches='tight', dpi=150, facecolor='white', edgecolor='none')
        plt.close()
    else:
        plt.show()


def main():
    argparser = argparse.ArgumentParser(description="Caption selection and comparison visualization using NIB")
    argparser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize")
    argparser.add_argument("--output_dir", type=str, default="outputs/caption_selection", help="Output directory")
    argparser.add_argument("--num_steps", type=int, default=10, help="Number of NIB optimization steps")
    argparser.add_argument("--target_layer", type=int, default=9, help="Target layer for NIB")
    argparser.add_argument("--data_root", type=str, default="datasets", help="Dataset root directory")
    argparser.add_argument("--ann_path", type=str, default="datasets/en_val.json", help="Annotation file path")
    argparser.add_argument("--clip_path", type=str, default=None, help="Path to local CLIP model")
    argparser.add_argument("--viz_type", type=str, default="detailed",
                          choices=["detailed", "summary", "both"],
                          help="detailed: V-Map + all captions; summary: ranked bar chart; both: both")
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

    print(f"\nGenerating {args.num_samples} caption selection visualizations...")
    count = 0
    for imgs, texts, batch_xs in tqdm(dataloader, total=args.num_samples):
        if count >= args.num_samples:
            break

        img = imgs[0]
        captions = texts[0]
        batch_xs = batch_xs.to(device)

        print(f"\n{'='*60}")
        print(f"Sample {count + 1}: Processing {len(captions)} captions...")

        similarities = get_caption_similarities(model, tokenizer, extract_image_features(model, batch_xs), captions)

        print(f"\n  Caption Similarities:")
        for i, (cap, sim) in enumerate(zip(captions, similarities)):
            print(f"    [{i+1}] {sim:.4f} - {cap[:60]}...")

        best_idx = np.argmax(similarities)
        print(f"\n  → Selected Caption #{best_idx + 1}: {captions[best_idx]}")

        tid = torch.tensor([tokenizer.encode(captions[best_idx], add_special_tokens=True)]).to(device)
        v_saliency, t_saliency = nib(model, [tid], batch_xs, args.num_steps, args.target_layer)

        v_saliency = normalize(np.squeeze(v_saliency[0]))
        if v_saliency.ndim == 3:
            v_saliency = v_saliency.transpose(1, 2, 0)
        if v_saliency.ndim == 2:
            v_saliency = np.stack([v_saliency] * 3, axis=-1)

        t_saliency = normalize(np.squeeze(t_saliency[0]))

        tokens = tokenizer.encode(captions[best_idx], add_special_tokens=True)
        token_words = [tokenizer.decode([t]).strip() for t in tokens]

        base_name = f"caption_selection_{count + 1:04d}"

        if args.viz_type in ["detailed", "both"]:
            output_path = os.path.join(args.output_dir, f"{base_name}_detailed.png")
            visualize_caption_selection(
                img, captions, similarities, best_idx,
                v_saliency, t_saliency, token_words,
                title=output_path
            )
            print(f"  ✓ Saved detailed: {output_path}")

        if args.viz_type in ["summary", "both"]:
            output_path = os.path.join(args.output_dir, f"{base_name}_summary.png")
            visualize_multi_caption_comparison(
                img, captions, similarities,
                title=output_path
            )
            print(f"  ✓ Saved summary: {output_path}")

        count += 1

    print(f"\n{'='*60}")
    print(f"Done. Saved {count} visualizations to '{args.output_dir}'.")


if __name__ == "__main__":
    main()