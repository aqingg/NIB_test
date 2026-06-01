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
    --save_separate       同时保存原图和热力图（可选）
    --alpha F             热力图透明度，0-1之间（默认：0.5）

示例：
    # 基本用法：生成5张热力图叠加图
    python visualize_vmap_overlay.py --num_samples 5

    # 保存原图、热力图和叠加图
    python visualize_vmap_overlay.py --num_samples 5 --save_separate

    # 调整热力图透明度（更明显的热力效果）
    python visualize_vmap_overlay.py --num_samples 5 --alpha 0.7

    # 修改NIB参数
    python visualize_vmap_overlay.py --num_samples 5 --num_steps 20 --target_layer 9

输出文件：
    outputs/vmap_overlay/
    ├── vmap_overlay_0001.png          # 叠加图（热力图+原图）
    ├── vmap_overlay_0002.png
    ├── heatmap/                       # 纯热力图（可选）
    │   ├── vmap_overlay_0001_heatmap.png
    │   └── ...
    └── original/                      # 原图（可选）
        ├── vmap_overlay_0001_original.png
        └── ...

依赖：
    - CLIP模型（openai/clip-vit-base-patch32或本地路径）
    - Flickr8k数据集
    - salicncy.nib (NIB显著性方法)
    - pytorch_grad_cam (热力图生成)
"""

import argparse
import os
import warnings

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
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


def main():
    argparser = argparse.ArgumentParser(description="Generate image heatmap visualizations using NIB")
    argparser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize")
    argparser.add_argument("--output_dir", type=str, default="outputs/vmap_overlay", help="Output directory")
    argparser.add_argument("--num_steps", type=int, default=10, help="Number of NIB optimization steps")
    argparser.add_argument("--target_layer", type=int, default=9, help="Target layer for NIB")
    argparser.add_argument("--data_root", type=str, default="datasets", help="Dataset root directory")
    argparser.add_argument("--ann_path", type=str, default="datasets/en_val.json", help="Annotation file path")
    argparser.add_argument("--clip_path", type=str, default=None, help="Path to local CLIP model")
    argparser.add_argument("--save_separate", action="store_true", help="Save original and heatmap separately")
    argparser.add_argument("--alpha", type=float, default=0.5, help="Heatmap transparency")
    args = argparser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_separate:
        os.makedirs(os.path.join(args.output_dir, "original"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "heatmap"), exist_ok=True)

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
        print(f"  Best caption: {best_caption}")
        print(f"  Similarity: {best_sim:.4f}")

        tid = torch.tensor([tokenizer.encode(best_caption, add_special_tokens=True)]).to(device)
        v_saliency, _ = nib(model, [tid], batch_xs, args.num_steps, args.target_layer)

        v_saliency = normalize(np.squeeze(v_saliency[0]))
        if v_saliency.ndim == 3:
            v_saliency = v_saliency.transpose(1, 2, 0)

        heatmap_h, heatmap_w = v_saliency.shape[:2]
        img_resized = img.resize((heatmap_w, heatmap_h), Image.BILINEAR)
        img_np = np.array(img_resized).astype(np.float32) / 255.0

        overlay = show_cam_on_image(img_np, v_saliency, use_rgb=True, image_weight=1 - args.alpha)
        
        base_name = f"vmap_overlay_{count + 1:04d}"
        
        overlay_path = os.path.join(args.output_dir, f"{base_name}.png")
        Image.fromarray(overlay).save(overlay_path)
        print(f"  Saved overlay: {overlay_path}")

        if args.save_separate:
            original_path = os.path.join(args.output_dir, "original", f"{base_name}_original.png")
            img_resized.save(original_path)
            
            heatmap_only = (v_saliency * 255).astype(np.uint8)
            heatmap_path = os.path.join(args.output_dir, "heatmap", f"{base_name}_heatmap.png")
            Image.fromarray(heatmap_only).save(heatmap_path)
            print(f"  Saved original: {original_path}")
            print(f"  Saved heatmap: {heatmap_path}")

        count += 1

    print(f"\nDone. Saved {count} overlay images to '{args.output_dir}'.")


if __name__ == "__main__":
    main()