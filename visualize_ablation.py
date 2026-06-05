"""
方案四：扰动验证实验 (Ablation Study)
========================================

功能说明：
    通过逐步删除图像中NIB识别的重要/不重要区域，验证NIB热力图真正解释了CLIP的决策依据。

实验设计：
    1. 使用NIB生成图像显著性图
    2. 将图像划分为多个patch
    3. 按显著性分数排序patch
    4. 从最不重要的patch开始逐步删除（设置为灰色）
    5. 每次删除后计算CLIP图文相似度
    6. 绘制相似度随删除比例变化的曲线

使用方法：
    python visualize_ablation.py [选项]

常用选项：
    --num_samples N       生成N个样本（默认：3）
    --output_dir PATH     输出目录（默认：outputs/ablation_study）
    --num_steps N         NIB优化步数（默认：10）
    --target_layer N      目标层索引（默认：9）
    --patch_size N        patch大小（默认：16）
    --num_ablation N      删除阶段数（默认：10）

示例：
    python visualize_ablation.py --num_samples 3 --patch_size 16
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


def get_patch_importance(saliency_map, patch_size=16):
    """计算每个patch的显著性分数"""
    h, w = saliency_map.shape[:2]
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    
    patch_scores = []
    patch_coords = []
    
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            y_start = i * patch_size
            y_end = y_start + patch_size
            x_start = j * patch_size
            x_end = x_start + patch_size
            
            patch_saliency = saliency_map[y_start:y_end, x_start:x_end]
            score = np.mean(patch_saliency)
            
            patch_scores.append(score)
            patch_coords.append((i, j, y_start, y_end, x_start, x_end))
    
    return patch_scores, patch_coords


def apply_ablation(image, patch_coords, sorted_indices, keep_ratio):
    """根据保留比例删除patch"""
    image_copy = np.array(image.copy())
    num_patches = len(patch_coords)
    num_to_remove = int(num_patches * (1 - keep_ratio))
    
    for idx in sorted_indices[:num_to_remove]:
        i, j, y_start, y_end, x_start, x_end = patch_coords[idx]
        image_copy[y_start:y_end, x_start:x_end] = [128, 128, 128]
    
    return Image.fromarray(image_copy)


def visualize_ablation_study(img, caption, processor, model, tokenizer, 
                             v_saliency, patch_size=16, num_ablation=10, title=None):
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('white')
    
    gs = fig.add_gridspec(3, 4, height_ratios=[1.2, 1, 1.5], 
                          width_ratios=[1, 1, 1, 1],
                          hspace=0.3, wspace=0.25)
    
    ax_original = fig.add_subplot(gs[0, 0])
    ax_original.imshow(img)
    ax_original.set_title('(a) Original Image', fontsize=11, fontweight='bold', pad=8)
    ax_original.axis('off')
    
    ax_saliency = fig.add_subplot(gs[0, 1])
    ax_saliency.imshow(v_saliency, cmap='viridis')
    ax_saliency.set_title('(b) NIB Saliency Map', fontsize=11, fontweight='bold', pad=8)
    ax_saliency.axis('off')
    
    ax_overlap = fig.add_subplot(gs[0, 2])
    img_resized = np.array(img.resize(v_saliency.shape[:2][::-1])) / 255.0
    overlay = img_resized * 0.6 + np.stack([v_saliency] * 3, axis=-1) * 0.4
    ax_overlap.imshow(overlay)
    ax_overlap.set_title('(c) Saliency Overlay', fontsize=11, fontweight='bold', pad=8)
    ax_overlap.axis('off')
    
    ax_patches = fig.add_subplot(gs[0, 3])
    h, w = v_saliency.shape[:2]
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    patch_grid = np.zeros((h, w))
    
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            y_start = i * patch_size
            x_start = j * patch_size
            patch_grid[y_start:y_start+patch_size, x_start:x_start+patch_size] = (i + j) % 2
    
    ax_patches.imshow(patch_grid, cmap='gray', alpha=0.5)
    ax_patches.set_title(f'(d) Patch Grid ({num_patches_h}x{num_patches_w})', 
                        fontsize=11, fontweight='bold', pad=8)
    ax_patches.axis('off')
    
    patch_scores, patch_coords = get_patch_importance(v_saliency, patch_size)
    sorted_indices = np.argsort(patch_scores)
    
    ratios = np.linspace(0.0, 1.0, num_ablation + 1)[::-1]
    similarities = []
    
    tid = torch.tensor([tokenizer.encode(caption, add_special_tokens=True)]).to(next(model.parameters()).device)
    
    for ratio in ratios:
        ablated_img = apply_ablation(img, patch_coords, sorted_indices, ratio)
        inputs = processor(images=ablated_img, return_tensors="pt").to(next(model.parameters()).device)
        im_feature = extract_image_features(model, inputs.pixel_values)
        text_feature = extract_text_features(model, tid)
        sim = torch.nn.functional.cosine_similarity(im_feature, text_feature).item()
        similarities.append(sim)
    
    original_sim = similarities[-1]
    
    selected_ratios = [0.0, 0.25, 0.5, 0.75]
    selected_indices = [np.argmin(np.abs(ratios - r)) for r in selected_ratios]
    
    for i, (ratio, idx) in enumerate(zip(selected_ratios, selected_indices)):
        ax = fig.add_subplot(gs[1, i])
        ablated_img = apply_ablation(img, patch_coords, sorted_indices, ratio)
        ax.imshow(ablated_img)
        ax.set_title(f'(e{i+1}) {int(ratio*100)}% kept', fontsize=10, fontweight='bold', pad=5)
        ax.set_xlabel(f'Sim: {similarities[idx]:.4f}', fontsize=9)
        ax.axis('off')
    
    ax_curve = fig.add_subplot(gs[2, :2])
    ax_curve.plot(ratios, similarities, 'o-', linewidth=3, markersize=8, 
                  color='#2171b5', markerfacecolor='#6baed6')
    
    critical_point = np.argmax(np.diff(similarities))
    ax_curve.axvline(ratios[critical_point], color='red', linestyle='--', 
                     label=f'Critical Point: {ratios[critical_point]:.2f}')
    
    ax_curve.set_xlabel('Image Area Kept (%)', fontsize=12)
    ax_curve.set_ylabel('CLIP Similarity', fontsize=12)
    ax_curve.set_title('(f) Similarity vs. Ablation', fontsize=12, fontweight='bold', pad=10)
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend()
    ax_curve.set_xlim(-0.02, 1.02)
    
    ax_bar = fig.add_subplot(gs[2, 2:])
    bar_width = 0.8 / len(ratios)
    bar_colors = plt.cm.RdYlGn_r(np.array(similarities) / max(similarities))
    ax_bar.bar(np.arange(len(ratios)) * bar_width, similarities, width=bar_width, 
               color=bar_colors, edgecolor='black')
    ax_bar.set_xticks(np.arange(len(ratios)) * bar_width + bar_width/2)
    ax_bar.set_xticklabels([f'{int(r*100)}%' for r in ratios], rotation=45, fontsize=9)
    ax_bar.set_xlabel('Image Area Kept', fontsize=11)
    ax_bar.set_ylabel('Similarity', fontsize=11)
    ax_bar.set_title('(g) Ablation Bar Chart', fontsize=12, fontweight='bold', pad=10)
    ax_bar.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f'NIB Ablation Study\nCaption: "{caption}"', 
                fontsize=14, fontweight='bold', y=0.98)
    
    stats_ax = fig.add_axes([0.05, 0.05, 0.9, 0.04])
    stats_ax.axis('off')
    
    max_drop = original_sim - min(similarities)
    drop_ratio = (max_drop / original_sim) * 100
    
    stats_text = (f'Statistics: Original={original_sim:.4f} | Min={min(similarities):.4f} | '
                  f'Max Drop={max_drop:.4f} ({drop_ratio:.1f}%) | '
                  f'Critical Ratio={ratios[critical_point]:.2f}')
    stats_ax.text(0.5, 0.5, stats_text, fontsize=10, ha='center', va='center',
                  fontweight='bold', color='darkblue',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='orange'))
    
    if title:
        plt.savefig(title, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
    else:
        plt.show()
    
    return similarities, ratios


def main():
    argparser = argparse.ArgumentParser(description="NIB Ablation Study - Perturbation Validation")
    argparser.add_argument("--num_samples", type=int, default=3)
    argparser.add_argument("--output_dir", type=str, default="outputs/ablation_study")
    argparser.add_argument("--num_steps", type=int, default=10)
    argparser.add_argument("--target_layer", type=int, default=9)
    argparser.add_argument("--patch_size", type=int, default=16)
    argparser.add_argument("--num_ablation", type=int, default=10)
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

    print(f"\nGenerating {args.num_samples} ablation study visualizations...")
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
        print(f"  Original Similarity: {best_sim:.4f}")

        tid = torch.tensor([tokenizer.encode(best_caption, add_special_tokens=True)]).to(device)
        v_saliency, _ = nib(model, [tid], batch_xs, args.num_steps, args.target_layer)
        
        v_saliency = normalize(np.squeeze(v_saliency[0]))
        if v_saliency.ndim == 3:
            v_saliency = v_saliency.transpose(1, 2, 0)
        if v_saliency.ndim == 2:
            v_saliency = np.stack([v_saliency] * 3, axis=-1)
        v_saliency = v_saliency[:, :, 0]

        output_path = os.path.join(args.output_dir, f"ablation_study_{count + 1:04d}.png")
        similarities, ratios = visualize_ablation_study(
            img, best_caption, processor, model, tokenizer,
            v_saliency, args.patch_size, args.num_ablation, title=output_path
        )

        max_drop = best_sim - min(similarities)
        drop_ratio = (max_drop / best_sim) * 100
        print(f"  Max Similarity Drop: {max_drop:.4f} ({drop_ratio:.1f}%)")
        print(f"  Saved: {output_path}")

        count += 1

    print(f"\nDone. Saved {count} ablation studies to '{args.output_dir}'.")


if __name__ == "__main__":
    main()