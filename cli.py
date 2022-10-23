import argparse
from PIL import Image
from sd_support import SDSupport

"""
# Stable Diffusion implementation for Deep Learning for Coders course
"""

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--copies', type=int, default=1, help='Number of images to generate')
parser.add_argument('-e', '--engine', type=str, default='torch', help='The SD engine to use: PyTorch or TensorFlow. Values: torch or tf')
parser.add_argument('-F', '--frame_cap', type=int, help='Save every nth frame')
parser.add_argument('-f', '--strength', type=float, default=0.75, help='How strongly the input image affects the final generated image')
parser.add_argument('-g', '--guidance', type=float, default=7.5, help='How closely you want the final image to be guided by the prompt')
parser.add_argument('-H', '--height', type=int, default=512, help='Image height, should be a multiple of 8')
parser.add_argument('-I', '--init_image', type=str, help='Path to input image for image guidance')
parser.add_argument('-o', '--outdir', type=str, help='The output folder for generated images')
parser.add_argument('-p', '--prompts', type=str, nargs='+', help='One or more prompts enclosed in quotes')
parser.add_argument('-S', '--seed', type=int, help='Image seed - the same image will be generated for a specific seed')
parser.add_argument('-s', '--steps', type=int, default=50, help='Number of inference steps')
parser.add_argument('-W', '--width', type=int, default=512, help='Image width, should be a multiple of 8')
args = parser.parse_args()
# Argument validations
if args.width % 8 != 0:
	print(f'The provided image width of {args.width} is not divisible by 8.')
	exit(1)
if args.height % 8 != 0:
	print(f'The provided image height of {args.height} is not divisible by 8.')
	exit(1)

# Support class
helper = SDSupport(args)
# SD engiine
if args.engine == 'torch':
	from sd_engine import SDEngine
	sd = SDEngine()
else:
	from sd_engine_tf import SDEngineTF
	sd = SDEngineTF(width=args.width, height=args.height)

# Frame callback
def frame_callback(index: int, image: Image):
	prefix = 'frame' if args.engine == 'torch' else 'tf-frame'
	helper.save_to_png(image, prefix=prefix, index=index)

# Prompt set up
prompts = args.prompts
if args.copies > 1:
	if len(prompts) > 1:
		tmp = []
		for p in prompts:
			items = [p] * args.copies
			tmp.extend(items)
		prompts = tmp
	else:
		prompts = [prompts[0]] * args.copies

for prompt in prompts:
	image, seed = sd.generate(prompt, width=args.width, height=args.height, guidance=args.guidance, seed=args.seed,
		steps=args.steps, callback=frame_callback, frame_cap=args.frame_cap)
	helper.save_to_png(image, seed, open=True)





