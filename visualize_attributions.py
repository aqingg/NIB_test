from datasets import Flickr8kDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
import torch
import os
import warnings
from salicncy import nib
from scripts.clip_local import load_clip_local
from scripts.plot import visualize_vandt_heatmap
from PIL import Image
import cv2
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def extract_text_features(model, input_ids):
    outputs = model.get_text_features(input_ids=input_ids)
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "pooler_output"):
        return outputs.pooler_output
    return outputs


def extract_image_features(model, pixel_values):
    outputs = model.get_image_features(pixel_values=pixel_values)
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "pooler_output"):
        return outputs.pooler_output
    return outputs


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize")
    argparser.add_argument("--output_dir", type=str, default="results", help="Output directory for visualizations")
    argparser.add_argument("--num_steps", type=int, default=10)
    argparser.add_argument("--target_layer", type=int, default=9)
    args = argparser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading CLIP model...")
    clip_path = os.environ.get("CLIP_PATH", r"D:\NIB-main\models\clip-vit-base-patch32")
    model, processor, tokenizer = load_clip_local(clip_path, device)

    print("Loading Flickr8k dataset...")
    dataset = Flickr8kDataset("datasets", "datasets/en_val.json", image_preprocessor=processor)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: (x[0][0], x[0][1], x[0][2]))

    print(f"Generating visualizations for {args.num_samples} samples...")
    model.eval()

    count = 0
    for img, captions, batch_xs in tqdm(dataloader):
        if count >= args.num_samples:
            break

        best_caption = captions[0]
        best_sim = -1

        with torch.no_grad():
            for caption in captions:
                tid = torch.tensor([tokenizer.encode(caption, add_special_tokens=True)]).to(device)
                tf = extract_text_features(model, tid)
                im_feature = extract_image_features(model, batch_xs.to(device))
                sim = torch.nn.functional.cosine_similarity(im_feature, tf).item()
                if sim > best_sim:
                    best_sim = sim
                    best_caption = caption

        tid = torch.tensor([tokenizer.encode(best_caption, add_special_tokens=True)]).to(device)
        batch_xs = batch_xs.to(device)

        print(f"\nSample {count + 1}:")
        print(f"  Caption: {best_caption}")

        v_saliency, t_saliency = nib(model, [tid], batch_xs, args.num_steps, args.target_layer)

        v_saliency = normalize(v_saliency[0])
        t_saliency = normalize(t_saliency[0])

        v_saliency = np.squeeze(v_saliency)
        if v_saliency.ndim == 3:
            v_saliency = v_saliency.transpose(1, 2, 0)

        heatmap_h, heatmap_w = v_saliency.shape[:2]

        img_resized = img.resize((heatmap_w, heatmap_h), Image.BILINEAR)
        img_np = np.array(img_resized).astype(np.float32) / 255.0

        tokens = tokenizer.encode(best_caption, add_special_tokens=True)
        token_words = [tokenizer.decode([t]).strip() for t in tokens]

        output_path = os.path.join(args.output_dir, f"visualization_{count + 1}.png")
        visualize_vandt_heatmap(
            tmap=t_saliency,
            vmap=v_saliency,
            text_words=token_words,
            image=img_np,
            title=output_path
        )
        print(f"  Saved to: {output_path}")

        count += 1

    print(f"\nDone! Visualizations saved to '{args.output_dir}' directory.")
