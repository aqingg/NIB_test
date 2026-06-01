from pathlib import Path

from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast

REQUIRED_FILES = [
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]
WEIGHT_CANDIDATES = ["pytorch_model.bin", "model.safetensors"]


def validate_clip_dir(clip_path):
    clip_dir = Path(clip_path)
    if not clip_dir.exists():
        raise FileNotFoundError(
            f"Local CLIP path does not exist: {clip_dir}"
        )

    missing = [f for f in REQUIRED_FILES if not (clip_dir / f).exists()]
    weights_ok = any((clip_dir / f).exists() for f in WEIGHT_CANDIDATES)

    if missing or not weights_ok:
        raise FileNotFoundError(
            "Local CLIP files missing. "
            f"Path: {clip_dir} | Missing: {missing} | "
            f"Expected weight file: {WEIGHT_CANDIDATES}"
        )

    return clip_dir


def load_clip_local(clip_path, device):
    clip_dir = validate_clip_dir(clip_path)

    model = CLIPModel.from_pretrained(str(clip_dir), local_files_only=True).to(device)
    processor = CLIPProcessor.from_pretrained(str(clip_dir), local_files_only=True)
    tokenizer = CLIPTokenizerFast.from_pretrained(str(clip_dir), local_files_only=True)

    return model, processor, tokenizer
