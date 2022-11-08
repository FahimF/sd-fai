# Stable Diffusion CLI

A Stable Diffuson script which allows you to generate images from the command-line. You can pass multiple prompts to the engine, or have multiple images generated for the same prompt. You can also save individual frames from the image generation cycle.

The script supports two separate Stable Diffusion engines — one using PyTorch and one using Keras/Tensorflow. You can pick the engine you prefer by using the `-e` parameter (see the arguments list below for more info).

You do not need both PyTorch and Tensorflow installed if all you want to do is use one engine only. The script will work without complaining about the absence of the other library.

## Arguments

```bash
usage: cli.py [-h] [-c COPIES] [-d SCHEDULER] [-e ENGINE] [-F FRAME_CAP] [-f STRENGTH] [-g GUIDANCE] [-H HEIGHT] [-I INIT_IMAGE] [-o OUTDIR] [-p PROMPTS [PROMPTS ...]] [-S SEED]
              [-s STEPS] [-v VERSION] [-W WIDTH]

optional arguments:
  -h, --help            show this help message and exit
  -c COPIES, --copies COPIES
                        Number of images to generate
  -d SCHEDULER, --scheduler SCHEDULER
                        Type of scheduler to use. Options: lmsd, ddim, dpm (DPMSolver), dpmp (DPMSolver++), euler, euler_a (Ancestral). Default: lmsd
  -e ENGINE, --engine ENGINE
                        The SD engine to use: PyTorch or TensorFlow. Values: torch or tf
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
  -v VERSION, --version VERSION
                        Stable Diffusion model version to use. Valid - 1.4, 1.5, defaults to 1.5
  -W WIDTH, --width WIDTH
                        Image width, should be a multiple of 8
```

## Usage Instructions

Basic usage requires passing at least a prompt (or prompts) via the CLI:

```bash
python cli.py -p "a mouse riding a lion"
```

If you want to enter multiple prompts, then they should all follow the `-p` argument and be separated by spaces like this:

```bash
python cli.py -p "a mouse riding a lion" "a tiger riding an elephant"
```

The engine/framework used by default is PyTorch. If you want to switch to using the Tensorflow-based engine, use the `-e` argment:

```bash
python cli.py -p "a mouse riding a lion" -e tf
```

If you want to use a specific scheduler instead of the default scheduler you need to use the `-d` parameter to specify the scheduler and the `-s` parameter to set a different number of steps:

```
python cli.py -p "a mouse riding a lion" -d euler_a -s 40
```

