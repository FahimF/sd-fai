import argparse

from sd_engine import SDEngine
from sd_support import SDSupport

"""
# Stable Diffusion implementation for Deep Learning for Coders course
"""

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--copies', type=int, default=1, help='Number of images to generate')
parser.add_argument('-f', '--strength', default=0.75, type=float, help='How strongly the input image affects the final generated image')
parser.add_argument('-g', '--guidance', type=float, default=7.5, help='How closely you want the final image to be guided by the prompt')
parser.add_argument('-H', '--height', type=int, default=512, help='Image height, multiple of 64')
parser.add_argument('-I', '--init_image', type=str, help='Path to input image for image guidance')
parser.add_argument('-o', '--outdir', type=str, help='The output folder for generated images')
parser.add_argument('-p', '--prompt', type=str, default='Ankh morpork city streets, trending on artstation')
parser.add_argument('-S', '--seed', type=int, help='Image seed; a +ve integer, or use -1 for the previous seed, -2 for the one before that, etc')
parser.add_argument('-s', '--steps', type=int, default=50, help='Number of inference steps')
parser.add_argument('-W', '--width', type=int, default=512, help='Image width, multiple of 64')
args = parser.parse_args()

# Set up Support class
helper = SDSupport(args)
sd = SDEngine()
images, seeds = sd.generate([args.prompt], width=args.width, height=args.height, guidance=args.guidance, seed=args.seed, steps=args.steps)
for ndx, image in enumerate(images):
	seed = seeds[ndx]
	helper.save__to_png(image, seed, open=True)





