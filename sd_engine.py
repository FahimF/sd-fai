import random
import torch
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

class SDEngine:
	def __init__(self, beta_start=0.00085, beta_end=0.012):
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

	def generate(self, prompts, width=512, height=512, guidance=7.5, seed=None, steps=50):
		images = []
		seeds = []
		# Loop through the prompts one by one for the time being while waiting on following issue to be fixed:
		# https://github.com/huggingface/diffusers/issues/941 & https://github.com/pytorch/pytorch/issues/84039
		for prompt in prompts:
			text = self.get_embeddings(prompt)
			uncond = self.get_embeddings([""], text.shape[1])
			emb = torch.cat([uncond, text])
			if seed is None:
				img_seed = random.randrange(2 ** 32 - 1)
			else:
				img_seed = seed
			print(f'Image seed is: {img_seed}')
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
					u, t = self.unet(inp, tf, encoder_hidden_states=emb).sample.chunk(2)
				pred = u + guidance * (t - u)
				latents = self.scheduler.step(pred, ts, latents).prev_sample
			with torch.no_grad():
				image = self.get_image(self.vae.decode(1 / 0.18215 * latents).sample)
				images.append(image)
				seeds.append(img_seed)
		return images, seeds

	def get_embeddings(self, prompt, maxlen=None):
		if maxlen is None:
			maxlen = self.tokenizer.model_max_length
		inp = self.tokenizer(prompt, padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt")
		return self.text_encoder(inp.input_ids.to(self.device))[0].half()

	def get_image(self, sample):
		image = (sample / 2 + 0.5).clamp(0, 1)[0].detach().cpu().permute(1, 2, 0).numpy()
		return Image.fromarray((image * 255).round().astype("uint8"))