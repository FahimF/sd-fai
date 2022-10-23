import math

import tensorflow as tf
from keras_cv.models.generative.stable_diffusion.constants import _ALPHAS_CUMPROD

class SimpleScheduler:
	def __init__(self, num_steps: int):
		self.num_steps = num_steps
		self.timesteps = tf.range(1, 1000, 1000 // num_steps)
		self.alphas = [_ALPHAS_CUMPROD[t] for t in self.timesteps]
		self.alphas_prev = [1.0] + self.alphas[:-1]

	def get_timestep_embedding(self, timestep: int, batch_size: int, latents: tf.Tensor, dim:int = 320, max_period: int = 10000) -> tf.Tensor:
		# Save the previous latents
		self.latent_prev = latents
		# Calculate embedding
		half = dim // 2
		freqs = tf.math.exp(-math.log(max_period) * tf.range(0, half, dtype=tf.float32) / half)
		args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
		embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
		embedding = tf.reshape(embedding, [1, -1])
		return tf.repeat(embedding, batch_size, axis=0)

	def get_latents(self, uncond: tf.Tensor, cond: tf.Tensor, index: int, guidance: float):
		noise = uncond + guidance * (cond - uncond)
		a_t, a_prev = self.alphas[index], self.alphas_prev[index]
		pred_x0 = (self.latent_prev - math.sqrt(1 - a_t) * noise) / math.sqrt(a_t)
		res = noise * math.sqrt(1.0 - a_prev) + math.sqrt(a_prev) * pred_x0
		return res
