import random
from typing import Callable, Optional

import torch
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

class SDEngine:
	def __init__(self, beta_start: float = 0.00085, beta_end: float = 0.012):
		# Device
		self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu"
		# Tokenizer and text encoder
		self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16)
		self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to(self.device)
		# VAE
		self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", torch_dtype=torch.float16).to(self.device)
		# UNet
		self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", torch_dtype=torch.float16).to(self.device)
		# Attention slicing - improves performance on macOS
		if torch.has_mps:
			slice_size = self.unet.config.attention_head_dim // 2
			self.unet.set_attention_slice(slice_size)
		# Scheduler
		self.scheduler = LMSDiscreteScheduler(beta_start=beta_start, beta_end=beta_end, beta_schedule="scaled_linear", num_train_timesteps=1000)

	# We only generate one image at a time for the time being while waiting on following issue to be fixed:
	# https://github.com/huggingface/diffusers/issues/941 & https://github.com/pytorch/pytorch/issues/84039
	def generate(self, prompt: str, width: int = 512, height: int = 512, guidance: float = 7.5,
		seed: int = None, steps: int = 50, callback: Optional[Callable[[int, Image], None]] = None,
		frame_cap: int = None) -> (Image, int):
		conditioned = self.get_embeddings(prompt)
		unconditioned = self.get_embeddings([""], conditioned.shape[1])
		embeddings = torch.cat([unconditioned, conditioned])
		# Generate seed, if necessary
		if seed is None:
			img_seed = random.randrange(2 ** 32 - 1)
		else:
			img_seed = seed
		torch.manual_seed(img_seed)
		latents = torch.randn((1, self.unet.in_channels, height // 8, width // 8))
		self.scheduler.set_timesteps(steps)
		latents = latents.to(self.device).half() * self.scheduler.init_noise_sigma
		# Loop through time steps
		for i, ts in enumerate(tqdm(self.scheduler.timesteps)):
			inp = self.scheduler.scale_model_input(torch.cat([latents] * 2), ts)
			with torch.no_grad():
				tf = ts
				if torch.has_mps:
					tf = ts.type(torch.float32)
				u, t = self.unet(inp, tf, encoder_hidden_states=embeddings).sample.chunk(2)
			pred = u + guidance * (t - u)
			latents = self.scheduler.step(pred, ts, latents).prev_sample
			# Do we have a callback?
			if callback is not None and frame_cap is not None:
				ndx = i + 1
				if ndx % frame_cap == 0:
					img = self.get_image(latents)
					callback(ndx, img)
		with torch.no_grad():
			image = self.get_image(latents)
		return image, img_seed

	def get_embeddings(self, prompt: str, maxlen: int = None) -> torch.FloatTensor:
		if maxlen is None:
			maxlen = self.tokenizer.model_max_length
		text_input = self.tokenizer(prompt, padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt")
		token_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0].half()
		return token_embeddings

	def get_image(self, latents: torch.FloatTensor) -> Image:
		# Scale latents array
		latents = 1 / 0.18215 * latents
		# Decode latents to get image data - it's a [1, 3, 512, 512] tensor here
		with torch.no_grad():
			# Get only the first item since it will be a single image
			data = self.vae.decode(latents).sample[0]
		# The image data is values between -1 and 1, convert to values between 0 and 1
		data = (data / 2 + 0.5).clamp(0, 1)
		# Change order of data to have a [512, 512, 3] tensor
		data = data.detach().cpu().permute(1, 2, 0).numpy()
		# Change data to values between 0 and 255
		data = (data * 255).round().astype("uint8")
		# Get image from data
		image = Image.fromarray(data)
		return image