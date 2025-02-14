import torch
import gc
import argparse
import numpy as np
from pathlib import Path
from contextlib import nullcontext
from accelerate import Accelerator
from accelerate.utils import (
	ProjectConfiguration
)
from accelerate.logging import get_logger
from diffusers import (
	AutoencoderKL,
	DDPMScheduler,
	LCMScheduler,
	StableDiffusionXLPipeline,
	UNet2DConditionModel,
	SchedulerMixin
)
from transformers import (
	AutoTokenizer
)
import common
import typing
import functools

logger = get_logger(__name__)

import re
import itertools
def decompose_string(input_string):
	# Find all substrings within curly braces
	patterns = re.findall(r'\{([^}]*)\}', input_string)
	
	# Split the found substrings by commas to create lists of options
	options = [pattern.split('|') for pattern in patterns]
	
	# Use itertools.product to generate all combinations
	combinations = itertools.product(*options)
	
	# Replace the curly-braced parts in the original string with each combination
	result_strings: typing.List[str] = []
	for combination in combinations:
		temp_string = input_string
		for match, replacement in zip(patterns, combination):
			temp_string = temp_string.replace(f'{{{match}}}', replacement, 1)
		result_strings.append(temp_string)
	
	return result_strings

def load_model(args: argparse.ArgumentParser, scheduler: SchedulerMixin = None):
	text_encoder_cls_one = common.import_model_class_from_model_name_or_path(
		args.pretrained_teacher_model, revision=None
	)
	text_encoder_cls_two = common.import_model_class_from_model_name_or_path(
		args.pretrained_teacher_model, revision=None, subfolder="text_encoder_2"
	)
	text_encoder_one = text_encoder_cls_one.from_pretrained(
		args.pretrained_teacher_model, subfolder="text_encoder", revision=None
	)
	text_encoder_two = text_encoder_cls_two.from_pretrained(
		args.pretrained_teacher_model, subfolder="text_encoder_2", revision=None
	)
	tokenizer_one = AutoTokenizer.from_pretrained(
		args.pretrained_teacher_model, subfolder="tokenizer", revision=None, use_fast=False
	)
	tokenizer_two = AutoTokenizer.from_pretrained(
		args.pretrained_teacher_model, subfolder="tokenizer_2", revision=None, use_fast=False
	)
	if scheduler is None:
		scheduler = DDPMScheduler.from_pretrained(
			args.pretrained_teacher_model, subfolder="scheduler", revision=None
		)
	unet = UNet2DConditionModel.from_pretrained(
		args.pretrained_teacher_model, subfolder="unet", revision=None
	)
	vae_path = (
		args.pretrained_teacher_model
		if args.pretrained_vae_model_name_or_path is None
		else args.pretrained_vae_model_name_or_path
	)
	vae = AutoencoderKL.from_pretrained(
		vae_path, subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
	)

	pipeline = StableDiffusionXLPipeline(
		vae=vae,
		text_encoder=text_encoder_one,
		text_encoder_2=text_encoder_two,
		tokenizer=tokenizer_one,
		tokenizer_2=tokenizer_two,
		unet=unet,
		scheduler=scheduler,
		add_watermarker=False,
	)
	
	return pipeline

def teacher_validation(args: argparse.ArgumentParser):
	pass

def student_validation(args: argparse.ArgumentParser, accelerator: Accelerator, prompts: typing.List[str]):
	pipeline = load_model(args)
	pipeline.scheduler = LCMScheduler.from_config(
		pipeline.scheduler.config
	)
	pipeline.load_lora_weights(args.pretrained_student_lora, args.adapter_name)
	pipeline.fuse_lora()
	pipeline.enable_xformers_memory_efficient_attention()
	pipeline.to(accelerator.device)

	weight_dtype = torch.float32
	if accelerator.mixed_precision == "fp16":
		weight_dtype = torch.float16
	elif accelerator.mixed_precision == "bf16":
		weight_dtype = torch.bfloat16

	if torch.backends.mps.is_available():
		autocast_ctx = nullcontext()
	else:
		autocast_ctx = torch.autocast(accelerator.device.type, dtype=weight_dtype)

	cache_conditions = []
	pipeline.tokenizer



	generators = [
		torch.Generator(device='cpu').manual_seed(args.seed + i)
		for i in range(args.batch)
	]
	
	image_logs = []
	for prompt in prompts:
		with torch.no_grad():
			with autocast_ctx:
				images = []
				for batch in range(len(generators)):
					image = pipeline(
						prompt=prompt,
						num_inference_steps=4,
						num_images_per_prompt=1,
						generator=generators[batch],
						guidance_scale=0.0,
						height=args.resolution,
						width=args.resolution,
					).images[0]
					images.append(image)
				image_logs.append({
					"validation_prompt": prompt, 
					"images": images,
				})

	for tracker in accelerator.trackers:
		if tracker.name == "tensorboard":
			for log in image_logs:
				images = log["images"]
				validation_prompt = log["validation_prompt"]

				formatted_images = []
				for image in images:
					formatted_images.append(np.asarray(image))
				formatted_images = np.stack(formatted_images)

				tracker.writer.add_images(
					validation_prompt, 
					formatted_images, 
					model="student",
					dataformats="NHWC"
				)
		else:
			raise Exception("unsupported tracker: " + tracker.name)

	del pipeline
	gc.collect()
	torch.cuda.empty_cache()

def parse_args(args=None):
	parser = argparse.ArgumentParser(description="Evaluate lcm model.")
	parser.add_argument(
		"--pretrained_teacher_model",
		type=str,
		default=None,
		required=True,
		help="Path to pretrained LDM teacher model or model identifier from huggingface.co/models.",
	)
	parser.add_argument(
		"--pretrained_student_lora",
		type=str,
		required=True,
		help="The output directory where the model predictions and checkpoints will be written.",
	)
	parser.add_argument(
		"--pretrained_vae_model_name_or_path",
		type=str,
		default=None,
		help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
	)
	parser.add_argument(
		"--prompt_matrix",
		type=str,
		required=True,
		help="string represent of a prompt matrix"
	)
	parser.add_argument(
		"--adapter_name",
		type=str,
		required=True,
		help="adapter name of this lora"
	)
	parser.add_argument(
		"--batch",
		type=int,
		default=4,
		help="generate n images for one prompt"
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="the seed to generate images"
	)
	parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
	parser.add_argument(
		"--output",
		type=str,
		default="./logs",
		help="write tensorboard logs to output dir"
	)
	parser.add_argument(
		"--resolution",
		type=int,
		default=1024,
		help="image resolution"
	)
	parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
	return parser.parse_args(args)

def main(args: argparse.ArgumentParser):
	torch.set_grad_enabled(False)

	accelerator_project_config = ProjectConfiguration(
		project_dir=args.output, 
		logging_dir=Path(args.output, "runs")
	)
	accelerator = Accelerator(
		mixed_precision=args.mixed_precision,
		log_with=args.report_to,
		project_config=accelerator_project_config,
	)

	accelerator.init_trackers(
		"validation",
		config={
			"seed": args.seed,
			"prompt_matrix": args.prompt_matrix,
			"adapter_name": args.adapter_name
		}
	)

	prompts = decompose_string(args.prompt_matrix)
	if accelerator.is_main_process:
		print("Will generate the following prompts:")
		for prompt in prompts:
			print("- ", prompt)	

	teacher_validation(args)
	student_validation(args, accelerator, prompts)

	accelerator.end_training()

if __name__ == "__main__":
	args = parse_args()
	main(args)