import argparse
import os
from typing import Tuple, List

import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

# ------------------- Utility functions ------------------- #

def load_image(path: str) -> Image.Image:
    """Return RGB image loaded from *path*."""
    return Image.open(path).convert("RGB")


def build_pipeline(model: str, device: str) -> StableDiffusionImg2ImgPipeline:
    """Load SD img2img pipeline on the given device."""
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model,
        torch_dtype=torch.float16,
        custom_pipeline="./sd_img2img_hallucination/pipeline_stable_diffusion_img2img_hallucination.py",
    )
    return pipe.to(device)


@torch.no_grad()
def generate(pipe, image: Image.Image, prompt: str, strength: float, guidance: float) -> Tuple[Image.Image, float]:
    """Run the model once and return (generated_image, Hal_x0)."""
    out = pipe(prompt=prompt, image=image, strength=strength, guidance_scale=guidance)

    # Older custom pipeline returns (images, Hal_x0, traj); new diffusers returns a struct with images.
    if hasattr(out, "images"):
        return out.images[0], -1.0  # Hal_x0 unavailable

    syn, hal, traj = out[0].images[0], out[1], out[2]
    
    # 생성 시의 trajectory 를 Tensor Form 으로 저장
    os.makedirs("traj_sdedit", exist_ok=True)
    torch.save(torch.stack(traj), f"traj_sdedit/traj_{hal:.4f}.pth")
    return syn, hal


def random_strength(base: float, span: float) -> float:
    """Uniform random strength around *base* within ±span."""
    return base + span * (2 * torch.rand(1).item() - 1)


def random_guidance(base: float, span: float) -> float:
    """Uniform random guidance scale around *base* within ±span."""
    return base + span * (2 * torch.rand(1).item() - 1)


def save_ranked(images: List[Tuple[float, Image.Image]], stem: str, folder_low: str, folder_high: str, n: int) -> None:
    """Save the *n* lowest and highest Hal_x0 images into separate folders."""
    os.makedirs(folder_low, exist_ok=True)
    os.makedirs(folder_high, exist_ok=True)

    images.sort(key=lambda x: x[0])
    for i in range(min(n, len(images))):
        low_fn  = os.path.join(folder_low,  f"out_{stem}_{i}_{images[i][0]:.4f}.png")
        high_fn = os.path.join(folder_high, f"out_{stem}_{i}_{images[-1-i][0]:.4f}.png")
        images[i][1].save(low_fn)
        images[-1-i][1].save(high_fn)


# ------------------- Main ------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Stable Diffusion Img2Img generation (refactored)")
    p.add_argument("--img_path", required=True, help="Path to input image.")
    p.add_argument("--prompt",   default="Identical Realistic", help="Text prompt guiding generation.")
    p.add_argument("--strength_base", type=float, default=0.55, help="Centre of random strength range.")
    p.add_argument("--strength_span", type=float, default=0.05, help="Half‑width of random strength range.")
    p.add_argument("--guidance_base", type=float, default=7.0, help="Centre of random guidance‑scale range.")
    p.add_argument("--guidance_span", type=float, default=4.0, help="Half‑width of random guidance‑scale range.")
    p.add_argument("--iterations", type=int, default=10, help="Maximum random search iterations.")
    p.add_argument("--num_samples", type=int, default=100, help="Number of ranked images to export to each folder.")
    p.add_argument("--model_name", default="stabilityai/stable-diffusion-2-1-base", help="HuggingFace model ID.")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Device type.")
    p.add_argument("--gpu", default="0", help="GPU ID if CUDA selected.")
    p.add_argument("--out_folder", default="generated_low", help="Folder for lowest Hal_x0 images.")
    p.add_argument("--out_folder_high", default="generated_high", help="Folder for highest Hal_x0 images.")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cpu" if args.device == "cpu" or not torch.cuda.is_available() else f"cuda:{args.gpu}"

    pipe   = build_pipeline(args.model_name, device)
    ref    = load_image(args.img_path)

    imgs_with_scores: List[Tuple[float, Image.Image]] = []
    seen_scores = set()

    for _ in range(args.iterations):
        st = random_strength(args.strength_base, args.strength_span)
        gs = random_guidance(args.guidance_base, args.guidance_span)
        img, score = generate(pipe, ref, args.prompt, st, gs)
        if score in seen_scores:
            continue
        imgs_with_scores.append((score, img))
        seen_scores.add(score)

    stem = os.path.splitext(os.path.basename(args.img_path))[0]
    save_ranked(imgs_with_scores, stem, args.out_folder, args.out_folder_high, args.num_samples)


if __name__ == "__main__":
    main()
