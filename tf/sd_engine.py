import os
import random
from typing import Callable, Optional

import numpy as np
import tensorflow as tf
from PIL import Image
from keras_cv.models.generative.stable_diffusion.clip_tokenizer import SimpleTokenizer
from keras_cv.models.generative.stable_diffusion.decoder import Decoder
from keras_cv.models.generative.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.models.generative.stable_diffusion.text_encoder import TextEncoder
from tensorflow import keras
from tqdm.auto import tqdm

from tf.simple_scheduler import SimpleScheduler

class SDEngine:
	def __init__(self, width: int = 512, height: int = 512, jit_compile: bool = False):
		self.MAX_PROMPT_LENGTH = 77
		# Reduce Tensorflow logs
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
		# UNet requires multiples of 2**7 = 128 for the images
		width = round(width / 128) * 128
		height = round(height / 128) * 128
		self.width = width
		self.height = height
		# Tokenizer and text encoder
		self.tokenizer = SimpleTokenizer()
		self.text_encoder = TextEncoder(self.MAX_PROMPT_LENGTH)
		# Diffusion model
		self.model = DiffusionModel(height, width, self.MAX_PROMPT_LENGTH)
		# Decoder
		self.decoder = Decoder(height, width)
		# JIT compiling
		self.jit_compile = jit_compile
		if jit_compile:
			self.text_encoder.compile(jit_compile = True)
			self.model.compile(jit_compile = True)
			self.decoder.compile(jit_compile = True)
		# Load weights
		text_encoder_path = keras.utils.get_file(
			origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_encoder.h5",
			file_hash="4789e63e07c0e54d6a34a29b45ce81ece27060c499a709d556c7755b42bb0dc4",
		)
		diffusion_model_path = keras.utils.get_file(
			origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_diffusion_model.h5",
			file_hash="8799ff9763de13d7f30a683d653018e114ed24a6a819667da4f5ee10f9e805fe",
		)
		decoder_path = keras.utils.get_file(
			origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_decoder.h5",
			file_hash="ad350a65cc8bc4a80c8103367e039a3329b4231c2469a1093869a345f55b1962",
		)
		self.text_encoder.load_weights(text_encoder_path)
		self.model.load_weights(diffusion_model_path)
		self.decoder.load_weights(decoder_path)

	def generate(self, prompt: str, width: int = 512, height: int = 512, guidance: float = 7.5,
		seed: int = None, steps: int = 50, callback: Optional[Callable[[int, Image], None]] = None,
		frame_cap: int = None) -> (Image, int):
		batch_size = 1
		# Scheduler
		self.scheduler = SimpleScheduler(num_steps=steps)
		# Embeddings
		conditioned = self.get_embeddings(prompt)
		unconditioned = self.get_embeddings('')
		# Generate seed, if necessary
		if seed is None:
			img_seed = random.randrange(2 ** 32 - 1)
		else:
			img_seed = seed
		# The latents tensor will be [1, 64, 64 4] by default
		latents = tf.random.normal((batch_size, self.height // 8, self.width // 8, 4), seed=seed)
		# Loop through time steps
		lst = list(enumerate(self.scheduler.timesteps))[::-1]
		for i, ts in tqdm(lst):
			t_emb = self.scheduler.get_timestep_embedding(ts, batch_size, latents)
			uncond = self.model.predict_on_batch([latents, t_emb, unconditioned])
			cond = self.model.predict_on_batch([latents, t_emb, conditioned])
			latents = self.scheduler.get_latents(uncond, cond, i, guidance)
			# Do we have a callback?
			if callback is not None and frame_cap is not None:
				ndx = i + 1
				if ndx % frame_cap == 0:
					img = self.get_image(latents)
					callback(ndx, img)
		# Decoding stage
		image = self.get_image(latents)
		return image, img_seed

	def get_embeddings(self, prompt: str, batch_size: int = 1) -> tf.Tensor:
		inputs = self.tokenizer.encode(prompt)
		if len(inputs) > self.MAX_PROMPT_LENGTH:
			raise ValueError(f"Prompt is too long (should be <= {self.MAX_PROMPT_LENGTH} tokens)")
		phrase = inputs + [49407] * (self.MAX_PROMPT_LENGTH - len(inputs))
		phrase = tf.convert_to_tensor([phrase], dtype=tf.int32)
		pos_ids = tf.convert_to_tensor([list(range(self.MAX_PROMPT_LENGTH))], dtype=tf.int32)
		embeddings = self.text_encoder.predict_on_batch([phrase, pos_ids])
		embeddings = tf.squeeze(embeddings)
		if embeddings.shape.rank == 2:
			embeddings = tf.repeat(tf.expand_dims(embeddings, axis=0), batch_size, axis=0)
		return embeddings

	def get_image(self, latents: tf.Tensor) -> Image:
		decoded = self.decoder.predict_on_batch(latents)[0]
		decoded = ((decoded + 1) / 2) * 255
		data = np.clip(decoded, 0, 255).astype("uint8")
		# Get image from data
		image = Image.fromarray(data)
		return image