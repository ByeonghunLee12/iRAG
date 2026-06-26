"""iRAG inference (DDIM-based).

Reference-aware super-resolution via the iRAG ref branch:
  z_lr    = VAE.encode(sr_bicubic)
  z_inter = VAE.encode(I_inter)            # I_inter = TTSR(lq, ref [, lr])
  F_cond  = concat( eps_phi(z_lr), eps_phi(z_inter) )   (Eq. 4)
  DDIM samples x_T -> x_0 conditioned on F_cond via SFT.

The TTSR restoration module (carried in the checkpoint) produces the
intermediate image from the bicubic LQ, the reference, and the native LR.

Usage:
  python inference.py \\
    --config configs/irag.yaml \\
    --ckpt   path/to/iRAG.ckpt \\
    --vqgan_ckpt path/to/vqgan_ckpt.ckpt \\
    --init-img path/to/test/sr_bicubic \\
    --ref-img  path/to/test/ref \\
    --lr-img   path/to/test/lr \\
    --outdir   outputs/irag \\
    --ddim_steps 200 --dec_w 0.5 --colorfix_type adain
"""

import argparse, math, os, time
from contextlib import nullcontext

import PIL
import numpy as np
import torch
import torchvision
from PIL import Image
from einops import rearrange, repeat
from omegaconf import OmegaConf
from torch import autocast
from tqdm import trange

from pytorch_lightning import seed_everything

# TF32 matmul: large speed/memory win on Ampere+ GPUs;
# no-op on older GPUs.
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    torch.backends.cuda.matmul.allow_tf32 = True

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from scripts.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization


# ----------------------------------------------------------------------
# Helpers (mirror the DDIM SR helpers)
# ----------------------------------------------------------------------
def space_timesteps(num_timesteps, section_counts):
	if isinstance(section_counts, str):
		if section_counts.startswith("ddim"):
			desired_count = int(section_counts[len("ddim"):])
			for i in range(1, num_timesteps):
				if len(range(0, num_timesteps, i)) == desired_count:
					return set(range(0, num_timesteps, i))
			raise ValueError(f"cannot create exactly {num_timesteps} steps with an integer stride")
		section_counts = [int(x) for x in section_counts.split(",")]
	size_per = num_timesteps // len(section_counts)
	extra = num_timesteps % len(section_counts)
	start_idx = 0
	all_steps = []
	for i, section_count in enumerate(section_counts):
		size = size_per + (1 if i < extra else 0)
		if size < section_count:
			raise ValueError(f"cannot divide section of {size} steps into {section_count}")
		frac_stride = 1 if section_count <= 1 else (size - 1) / (section_count - 1)
		cur_idx = 0.0
		taken_steps = []
		for _ in range(section_count):
			taken_steps.append(start_idx + round(cur_idx))
			cur_idx += frac_stride
		all_steps += taken_steps
		start_idx += size
	return set(all_steps)


def load_model_from_config(config, ckpt, verbose=False):
	print(f"Loading model from {ckpt}")
	pl_sd = torch.load(ckpt, map_location="cpu")
	if "global_step" in pl_sd:
		print(f"Global Step: {pl_sd['global_step']}")
	sd = pl_sd["state_dict"]
	model = instantiate_from_config(config.model)
	m, u = model.load_state_dict(sd, strict=False)
	print('>>>>>>>>>>>>>>>>>>>load results>>>>>>>>>>>>>>>>>>>>>>>')
	if len(m) > 0 and verbose:
		print("missing keys:", m)
	if len(u) > 0 and verbose:
		print("unexpected keys:", u)
	model.cuda()
	model.eval()
	return model


def load_img(path, size=None):
	"""Load image to [-1, 1] CHW tensor with batch dim. If `size` is given,
	resize to that side length (square assumed by caller)."""
	image = Image.open(path).convert("RGB")
	w, h = image.size
	# Match base script: snap to multiples of 32.
	w, h = map(lambda x: x - x % 32, (w, h))
	image = image.resize((w, h), resample=PIL.Image.LANCZOS)
	image = np.array(image).astype(np.float32) / 255.0
	image = image[None].transpose(0, 3, 1, 2)
	return 2. * torch.from_numpy(image) - 1.


def _ensure_dir(path, what):
	if path is None or not os.path.isdir(path):
		raise FileNotFoundError(f"--{what} expected a directory, got: {path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
	parser = argparse.ArgumentParser()

	# I/O
	parser.add_argument("--init-img", type=str, required=True,
		help="dir of LR/sr_bicubic input images (one per file)")
	parser.add_argument("--ref-img", type=str, default=None,
		help="[reference] dir of HR reference images, same filenames as --init-img")
	parser.add_argument("--lr-img", type=str, default=None,
		help="[optional] dir of native LR images (HR/sf). "
		     "If omitted, TTSR synthesizes lr via bicubic_down(lq).")
	parser.add_argument("--inter-img", type=str, default=None,
		help="[precomp mode] dir of precomputed I_inter images, same filenames as --init-img")
	parser.add_argument("--outdir", type=str, default="outputs/irag")

	# Mode (default: read from config)
	parser.add_argument("--mode", type=str, choices=["online", "precomp", "auto"], default="auto",
		help="iRAG inference mode. 'auto' reads config.model.params.use_restoration.")

	# Model / config / ckpts (mirror base script)
	parser.add_argument("--config", type=str, required=True)
	parser.add_argument("--ckpt", type=str, required=True)
	parser.add_argument("--vqgan_ckpt", type=str, required=True)

	# Sampling
	parser.add_argument("--n_samples", type=int, default=1,
		help="batch size; iRAG online mode is memory-heavy, keep small")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--precision", type=str, choices=["full", "autocast"], default="autocast")
	parser.add_argument("--input_size", type=int, default=512)
	parser.add_argument("--dec_w", type=float, default=0.5,
		help="VQGAN/diffusion fusion weight in the CFW decoder")
	parser.add_argument("--colorfix_type", type=str, default="adain",
		choices=["adain", "wavelet", "nofix"])
	parser.add_argument("--ddim_steps", type=int, default=200)
	parser.add_argument("--ddim_eta", type=float, default=0.0)

	# CFG / negative prompt
	parser.add_argument("--scale", type=float, default=7.0,
		help="classifier-free guidance scale (only used when --use_negative_prompt is set)")
	parser.add_argument("--use_negative_prompt", action="store_true")
	parser.add_argument("--use_posi_prompt", action="store_true")

	# Init noise / start point
	parser.add_argument("--no_q_sample", action="store_true",
		help="start from pure Gaussian noise instead of q_sample(z_lr, t=999)")

	opt = parser.parse_args()
	seed_everything(opt.seed)

	# ------------------------------------------------------------------
	# Config + mode detection
	# ------------------------------------------------------------------
	config = OmegaConf.load(opt.config)
	restoration_in_config = bool(getattr(config.model.params, 'use_restoration', False))
	if opt.mode == "auto":
		mode = "online" if restoration_in_config else "precomp"
	else:
		mode = opt.mode
	print(f"[iRAG] mode: {mode} (config.use_restoration={restoration_in_config})")

	# Mode-specific input validation.
	_ensure_dir(opt.init_img, "init-img")
	if mode == "online":
		_ensure_dir(opt.ref_img, "ref-img")
		if opt.lr_img is not None:
			_ensure_dir(opt.lr_img, "lr-img")
	else:
		_ensure_dir(opt.inter_img, "inter-img")

	if opt.colorfix_type == 'adain':
		print('Use adain color correction')
	elif opt.colorfix_type == 'wavelet':
		print('Use wavelet color correction')
	else:
		print('No color correction')

	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	# ------------------------------------------------------------------
	# Models
	# ------------------------------------------------------------------
	model = load_model_from_config(config, opt.ckpt).to(device)

	# Sanity-check: encoder must have ref branch enabled for iRAG inference.
	if not getattr(model.structcond_stage_model, 'use_ref_branch', False):
		raise RuntimeError(
			"structcond_stage_model.use_ref_branch is False. The provided "
			"config/ckpt is not iRAG-compatible. Use an iRAG yaml such as "
			"configs/irag.yaml"
		)
	# Online mode requires the restoration module to be in the ckpt.
	if mode == "online" and getattr(model, 'restoration_module', None) is None:
		raise RuntimeError(
			"online mode requested but model.restoration_module is None. "
			"Either train with use_restoration=true or run in precomp mode."
		)

	vqgan_config = OmegaConf.load("configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml")
	vq_model = load_model_from_config(vqgan_config, opt.vqgan_ckpt).to(device)
	vq_model.decoder.fusion_w = opt.dec_w

	sampler = DDIMSampler(model)

	# Schedule (mirror base script).
	model.register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000,
		linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3)
	model.num_timesteps = 1000
	model = model.to(device)
	ddim_timesteps = sorted(set(space_timesteps(1000, [opt.ddim_steps])))
	sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

	# ------------------------------------------------------------------
	# I/O setup
	# ------------------------------------------------------------------
	os.makedirs(opt.outdir, exist_ok=True)
	sample_path = os.path.join(opt.outdir, "samples")
	os.makedirs(sample_path, exist_ok=True)

	transform = torchvision.transforms.Compose([
		torchvision.transforms.Resize(opt.input_size),
		torchvision.transforms.CenterCrop(opt.input_size),
	])

	img_list_all = sorted(os.listdir(opt.init_img))
	done = set(os.listdir(sample_path))
	img_list = [f for f in img_list_all if f not in done]
	batch_size = opt.n_samples
	niters = math.ceil(len(img_list) / batch_size)
	chunks = [img_list[i * batch_size:(i + 1) * batch_size] for i in range(niters)]

	precision_scope = autocast if opt.precision == "autocast" else nullcontext

	# ------------------------------------------------------------------
	# Sampling loop
	# ------------------------------------------------------------------
	with torch.no_grad(), precision_scope("cuda"), model.ema_scope():
		tic = time.time()
		for n in trange(niters, desc="iRAG sampling"):
			cur_files = chunks[n]

			# --- Load LR (sr_bicubic) batch ---
			lq_list = []
			for fn in cur_files:
				img = load_img(os.path.join(opt.init_img, fn)).to(device)
				img = transform(img)
				lq_list.append(img)
			lq_image = torch.cat(lq_list, dim=0)  # [B, 3, H, W] in [-1, 1]

			# --- Build I_inter ---
			if mode == "online":
				ref_list = []
				for fn in cur_files:
					ref = load_img(os.path.join(opt.ref_img, fn)).to(device)
					ref = transform(ref)
					ref_list.append(ref)
				ref_image = torch.cat(ref_list, dim=0)

				lr_native = None
				if opt.lr_img is not None:
					lr_list = []
					for fn in cur_files:
						p = os.path.join(opt.lr_img, fn)
						# Native LR is HR/sf; do NOT apply the HR transform.
						# load_img returns [-1, 1] tensor at the file's resolution
						# (snapped to mult of 32); accept as-is.
						lr_list.append(load_img(p).to(device))
					lr_native = torch.cat(lr_list, dim=0)

				i_inter = model.restoration_module(lq_image, ref_image, lr_pixel=lr_native)
				i_inter = torch.clamp(i_inter, -1.0, 1.0)
			else:
				inter_list = []
				for fn in cur_files:
					it = load_img(os.path.join(opt.inter_img, fn)).to(device)
					it = transform(it)
					inter_list.append(it)
				i_inter = torch.cat(inter_list, dim=0)

			# --- VAE encode ---
			# For lq we use the CFW VQGAN path (gives the latent + features
			# enc_fea_lq, the latter feeds the CFW decoder for color fidelity).
			# For I_inter we only need the latent, so use the diffusion VAE.
			init_latent_generator, enc_fea_lq = vq_model.encode(lq_image)
			z_lr = model.get_first_stage_encoding(init_latent_generator)
			z_inter = model.get_first_stage_encoding(model.encode_first_stage(i_inter))

			# --- Text cond (empty by default; CFG via negative prompt) ---
			if opt.use_posi_prompt:
				text_init = ['(masterpiece:2), (best quality:2), (realistic:2), (very clear:2)'] * lq_image.size(0)
			else:
				text_init = [''] * lq_image.size(0)
			semantic_c = model.cond_stage_model(text_init)
			nega_semantic_c = None
			if opt.use_negative_prompt:
				negative_text_init = ['3d, cartoon, anime, sketches, (worst quality:2), (low quality:2)'] * lq_image.size(0)
				nega_semantic_c = model.cond_stage_model(negative_text_init)

			# --- Initial latent x_T ---
			if opt.no_q_sample:
				x_T = torch.randn_like(z_lr)
			else:
				noise = torch.randn_like(z_lr)
				t = repeat(torch.tensor([999]), '1 -> b', b=lq_image.size(0)).to(device).long()
				x_T = model.q_sample(x_start=z_lr, t=t, noise=noise)

			# --- DDIM sampling with iRAG ref branch ---
			samples, _ = sampler.ddim_sampling_sr_t(
				cond=semantic_c,
				struct_cond=z_lr,
				struct_cond_ref=z_inter,           # <<< the iRAG ref branch
				shape=z_lr.shape,
				unconditional_conditioning=nega_semantic_c,
				unconditional_guidance_scale=opt.scale if opt.use_negative_prompt else None,
				timesteps=np.array(ddim_timesteps),
				x_T=x_T,
			)

			# --- Decode (CFW) + color correction ---
			x_samples = vq_model.decode(samples * 1. / model.scale_factor, enc_fea_lq)
			if opt.colorfix_type == 'adain':
				x_samples = adaptive_instance_normalization(x_samples, lq_image)
			elif opt.colorfix_type == 'wavelet':
				x_samples = wavelet_reconstruction(x_samples, lq_image)
			x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

			# --- Save ---
			for i in range(x_samples.size(0)):
				arr = (255. * rearrange(x_samples[i].cpu().numpy(), 'c h w -> h w c')).astype(np.uint8)
				Image.fromarray(arr).save(os.path.join(sample_path, cur_files[i]))

		toc = time.time()
		print(f"[iRAG] processed {len(img_list)} images in {toc - tic:.1f}s")

	print(f"Outputs at: {opt.outdir}")


if __name__ == "__main__":
	main()
