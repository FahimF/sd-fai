# Stable Diffusion CLI

A Stable Diffuson script which allows you to generate images from the command-line. You can pass multiple prompts to the engine, or have multiple images generated for the same prompt. You can also save individual frames from the image generation cycle.

## Arguments

```bash
usage: cli.py [-h] [-c COPIES] [-F FRAME_CAP] [-f STRENGTH] [-g GUIDANCE] [-H HEIGHT] [-I INIT_IMAGE] [-o OUTDIR] [-p PROMPTS [PROMPTS ...]] [-S SEED] [-s STEPS] [-W WIDTH]

optional arguments:
  -h, --help            show this help message and exit
  -c COPIES, --copies COPIES
                        Number of images to generate
  -F FRAME_CAP, --frame_cap FRAME_CAP
                        Save every nth frame
  -f STRENGTH, --strength STRENGTH
                        How strongly the input image affects the final generated image
  -g GUIDANCE, --guidance GUIDANCE
                        How closely you want the final image to be guided by the prompt
  -H HEIGHT, --height HEIGHT
                        Image height, should be a multiple of 8
  -I INIT_IMAGE, --init_image INIT_IMAGE
                        Path to input image for image guidance
  -o OUTDIR, --outdir OUTDIR
                        The output folder for generated images
  -p PROMPTS [PROMPTS ...], --prompts PROMPTS [PROMPTS ...]
                        One or more prompts enclosed in quotes
  -S SEED, --seed SEED  Image seed - the same image will be generated for a specific seed
  -s STEPS, --steps STEPS
                        Number of inference steps
  -W WIDTH, --width WIDTH
                        Image width, should be a multiple of 8
```

