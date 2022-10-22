import os
import logging
from datetime import datetime
from os.path import exists

from PIL import Image, PngImagePlugin

class SDSupport:
	def __init__(self, args):
		self.args = args
		self.outdir = 'output'
		# Disable warning logging to remove flooding the console
		logging.disable(logging.WARNING)
		if args.outdir is not None:
			self.outdir = args.outdir
		# Check if output directory is there
		if not exists(self.outdir):
			os.mkdir(self.outdir)

	def save__to_png(self, image: Image, seed: int, open: bool = False):
		str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
		name = f'{str}_{seed}.png'
		path = os.path.join(self.outdir, name)
		info = PngImagePlugin.PngInfo()
		meta = f'{self.get_info_string()} -S{seed}'
		info.add_text('Author', meta)
		info.add_text('Title', self.args.prompt)
		info.add_text('Seed', f'{seed}')
		image.save(path, 'PNG', pnginfo=info)
		if open:
			os.system(f'open {path}')
		return path

	def get_info_string(self):
		switches = list()
		switches.append(f'{self.args.prompt}')
		switches.append(f'-s{self.args.steps}')
		switches.append(f'-W{self.args.width}')
		switches.append(f'-H{self.args.height}')
		switches.append(f'-g{self.args.guidance}')
		if self.args.init_image is not None:
			switches.append(f'-I{self.args.init_image}')
		if self.args.strength and self.args.init_image is not None:
			switches.append(f'-f{self.args.strength}')
		return ' '.join(switches)
