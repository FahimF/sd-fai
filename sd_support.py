import logging
import os
from datetime import datetime
from os.path import exists

from PIL import Image, PngImagePlugin

class SDSupport:
	def __init__(self, args):
		self.args = args
		self.outdir = 'output'
		self.framedir = 'frames'
		# Disable warning logging to remove flooding the console
		logging.disable(logging.WARNING)
		if args.outdir is not None:
			self.outdir = args.outdir
		# Check if output directory is there
		if not exists(self.outdir):
			os.mkdir(self.outdir)
		# Check if frames directory is there
		if not exists(self.framedir):
			os.mkdir(self.framedir)

	def save_to_png(self, image: Image, seed: int = None, prefix: str = None, index: int = None, open: bool = False):
		is_frame = False
		if prefix is not None and index is not None:
			is_frame = True
			suffix = f'{index}'.zfill(4)
			name = f"{prefix}_{suffix}.png"
			path = os.path.join(self.framedir, name)
		else:
			str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
			name = f'{str}_{seed}.png'
			path = os.path.join(self.outdir, name)
		if not is_frame:
			info = PngImagePlugin.PngInfo()
			meta = f'{self.get_info_string()} -S{seed}'
			info.add_text('Author', meta)
			info.add_text('Title', ' | '.join(self.args.prompts))
			info.add_text('Seed', f'{seed}')
			image.save(path, 'PNG', pnginfo=info)
			print(f'Image seed: {seed} - path: {path}')
		else:
			image.save(path, 'PNG')
		if open == True:
			os.system(f'open {path}')
		return path

	def get_info_string(self):
		switches = list()
		switches.append(f'{" | ".join(self.args.prompts)}')
		switches.append(f'-s {self.args.steps}')
		switches.append(f'-W {self.args.width}')
		switches.append(f'-H {self.args.height}')
		switches.append(f'-g {self.args.guidance}')
		switches.append(f'-v {self.args.version}')
		switches.append(f'-d {self.args.scheduler}')
		if self.args.init_image is not None:
			switches.append(f'-I {self.args.init_image}')
		if self.args.strength and self.args.init_image is not None:
			switches.append(f'-f {self.args.strength}')
		return ' '.join(switches)
