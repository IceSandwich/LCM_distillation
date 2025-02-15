import argparse
import logging
import os
from contextlib import nullcontext
from pathlib import Path
from typing import override

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import datasets
import common

import diffusers
from diffusers import (
	AutoencoderKL,
	DDPMScheduler,
	UNet2DConditionModel,
)
from transformers import (
	AutoTokenizer,
	CLIPTextModel
)
import typing
from diffusers.utils.import_utils import is_xformers_available

logger = get_logger(__name__)


LOG_DIRNAME = "logs"
MODEL_DIRNAME = "model"

class ThreadControl:
	def IsMainProcess(self) -> bool:
		raise Exception("this is a virtual method.")
	def WaitForEveryOne(self):
		raise Exception("this is a virtual method.")
	def GetAccelerator(self) -> Accelerator:
		raise Exception("this is a virtual method.")

class ProjectConf:
	def __init__(self, args):
		self.prj_dir = args.prj_dir
		self.log_dir = os.path.join(self.prj_dir, "logs")
		self.model_dir = os.path.join(self.prj_dir, "model")
		self.cache_dir = os.path.join(self.prj_dir, "cache")

	def makedirs(self):
		os.makedirs(self.log_dir, exist_ok=True)
		os.makedirs(self.model_dir, exist_ok=True)
		os.makedirs(self.cache_dir, exist_ok=True)

		print("================= Project ===================")
		print("Project dir: ", self.prj_dir)
		print("Log dir: ", self.log_dir)
		print("Model dir: ", self.model_dir)
		print("---------------------------------------------")
	
	@classmethod
	def ApplyToParser(parser: argparse.ArgumentParser):
		parser.add_argument(
			"--prj_dir",
			type=str,
			default="untitled_project",
			help="project dir to contain every outputs."
		)

class TrainConf:
	def __init__(self, args):
		self.epoch:int = args.epoch
		self.repeat:int = args.repeat
		self.mixed_precision:str = args.mixed_precision
		self.gradient_accumulation_steps:int = args.gradient_accumulation_steps
		self.enable_xformers_memory_efficient_attention: bool = args.enable_xformers_memory_efficient_attention
		self.gradient_checkpointing: bool = args.gradient_checkpointing

		self.use_8bit_adam: bool = args.use_8bit_adam

	@classmethod
	def ApplyToParser(parser: argparse.ArgumentParser):
		parser.add_argument(
			"--epoch",
			type=int,
			default=15,
			help="epoch to train"
		)
		parser.add_argument(
			"--repeat",
			type=int,
			default=20,
		)
		parser.add_argument(
			"--mixed_precision",
			type=str,
			default=None,
			choices=["no", "fp16", "bf16"],
			help=(
				"Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
				" 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
				" flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
			),
		)
		parser.add_argument(
			"--gradient_accumulation_steps",
			type=int,
			default=1,
			help="Number of updates steps to accumulate before performing a backward/update pass.",
		)
		parser.add_argument(
			"--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
		)
		parser.add_argument(
			"--gradient_checkpointing",
			action="store_true",
			help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
		)

		# ----Optimizer (Adam)----
		parser.add_argument(
			"--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
		)
		parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
		parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
		parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
		parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
		parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

class LoraConf:
	def __init__(self, args):
		self.lora_rank:int = args.lora_rank
		self.lora_alpha:int = args.lora_alpha
		self.lora_dropout: float = args.lora_dropout
	
	@classmethod
	def ApplyToParser(parser: argparse.ArgumentParser):
		parser.add_argument(
			"--lora_rank",
			type=int,
			default=16,
			help="The rank of the LoRA projection matrix.",
		)
		parser.add_argument(
			"--lora_alpha",
			type=int,
			default=8,
			help=(
				"The value of the LoRA alpha parameter, which controls the scaling factor in front of the LoRA weight"
				" update delta_W. No scaling will be performed if this value is equal to `lora_rank`."
			),
		)
		parser.add_argument(
			"--lora_dropout",
			type=float,
			default=0.0,
			help="The dropout probability for the dropout layer added before applying the LoRA to each layer input.",
		)

class ModelConf:
	def __init__(self, args):
		self.seed: int = args.seed
		self.model: str = args.model
		self.vae: typing.Union[str, None] = args.vae
		self.model_type: str = args.model_type
	
	@classmethod
	def ApplyToParser(parser: argparse.ArgumentParser):
		parser.add_argument(
			"--seed",
			type=int,
			default=42,
			help="A seed for reproducible training."
		)
		parser.add_argument(
			"--model",
			type=str,
			required=True,
			help="Path to pretrained LDM teacher model or model identifier from huggingface.co/models.",
		)
		parser.add_argument(
			"--vae",
			type=str,
			default=None,
			help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
		)
		parser.add_argument(
			"--model_type",
			type=str,
			choices=["sd", "sdxl"],
			required=True,
			help="model type"
		)

class DatasetConf:
	def __init__(self, args):
		self.data_dir: str = args.data_dir

	@classmethod
	def ApplyToParser(parser: argparse.ArgumentParser):
		parser.add_argument(
			"--data_dir",
			type=str,
			help=(
				"A folder containing the training data. Folder contents must follow the structure described in"
				" https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
				" must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
			),
		)

class Dataset:
	def Cache(self):
		raise Exception("this is a virtual method")

class BaseModel:
	def LoadComponents(self):
		raise Exception("this is a virtual method")
	def Freeze(self):
		raise Exception("this is a virtual method")
	def PatchLoraModules(self):
		raise Exception("this is a virtual method")
	def UploadToDevice(self):
		raise Exception("this is a virtual method")
	def CreateOptimizer(self):
		raise Exception("this is a virtual method")
	def GetDatasetAdapter(self) -> Dataset:
		raise Exception("this is a virtual method")

class StableDiffusionDataset(Dataset):
	def __init__(self, args, thread_control: ThreadControl):
		self.thread_control = thread_control
	def Preparse(self):
		pass

class StableDiffusionXLModel(BaseModel):
	def __init__(self, args):
		pass

class StableDiffusionModel(BaseModel):
	def __init__(self, args, thread_control: ThreadControl):
		self.args = args
		self.prj: ProjectConf = ProjectConf(args)
		self.conf = ModelConf(args)
		self.lora_conf = LoraConf(args)
		self.thread_control = thread_control
		self.train_conf = TrainConf(args)

		self.weight_dtype = torch.float32
		if self.thread_control.GetAccelerator().mixed_precision == "fp16":
			self.weight_dtype = torch.float16
		elif self.thread_control.GetAccelerator().mixed_precision == "bf16":
			self.weight_dtype = torch.bfloat16

		self.dataset = StableDiffusionDataset(args, thread_control)

	def LoadComponents(self):
		use_safetensors = self.conf.model.endswith(".safetensors")

		# Create the noise scheduler and the desired noise schedule.
		self.noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(
			self.conf.model, subfolder="scheduler", revision=args.teacher_revision, use_safetensors=use_safetensors
		)

		# DDPMScheduler calculates the alpha and sigma noise schedules (based on the alpha bars) for us
		self.alpha_schedule = torch.sqrt(self.noise_scheduler.alphas_cumprod)
		self.sigma_schedule = torch.sqrt(1 - self.noise_scheduler.alphas_cumprod)
		
		# Initialize the DDIM ODE solver for distillation.
		self.solver = common.DDIMSolver(
			self.noise_scheduler.alphas_cumprod.numpy(),
			timesteps=self.noise_scheduler.config.num_train_timesteps,
			ddim_timesteps=args.num_ddim_timesteps,
		)

		# Load tokenizers from SD 1.X/2.X checkpoint.
		self.tokenizer = AutoTokenizer.from_pretrained(
			self.conf.model, subfolder="tokenizer", revision=None, use_fast=False, use_safetensors=use_safetensors
		)

		# Load text encoders from SD 1.X/2.X checkpoint.
		self.text_encoder = CLIPTextModel.from_pretrained(
			self.conf.model, subfolder="text_encoder", revision=None, use_safetensors=use_safetensors
		)
		
		# Load teacher U-Net from SD 1.X/2.X checkpoint
		self.unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
			self.conf.model, subfolder="unet", revision=None, use_safetensors=use_safetensors
		)

		vae_path = (self.conf.model if self.conf.vae is None else self.conf.vae)
		# Load VAE from SD 1.X/2.X checkpoint
		self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(
			vae_path, subfolder="vae", revision=None, use_safetensors=use_safetensors
		)

	def Freeze(self):
		self.text_encoder.requires_grad_(False)
		self.unet.requires_grad_(False)
		self.vae.requires_grad_(False)

	def PatchLoraModules(self):
		lora_target_modules = [
			"to_q",
			"to_k",
			"to_v",
			"to_out.0",
			"proj_in",
			"proj_out",
			"ff.net.0.proj",
			"ff.net.2",
			"conv1",
			"conv2",
			"conv_shortcut",
			"downsamplers.0.conv",
			"upsamplers.0.conv",
			"time_emb_proj",
		]

		lora_config = LoraConfig(
			r=self.lora_conf.lora_rank,
			target_modules=lora_target_modules,
			lora_alpha=self.lora_conf.lora_alpha,
			lora_dropout=self.lora_conf.lora_dropout,
		)
		self.unet = get_peft_model(self.unet, lora_config)

	def UploadToDevice(self):
		device = self.thread_control.GetAccelerator().device
		self.text_encoder.to(device, dtype=self.weight_dtype)
		self.unet.to(device, dtype=self.weight_dtype)
		if self.conf.vae is None:
			self.vae.to(device)
		else:
			self.vae.to(device, dtype=self.weight_dtype)

		# Also move the alpha and sigma noise schedules to accelerator.device.
		self.alpha_schedule = self.alpha_schedule.to(device)
		self.sigma_schedule = self.sigma_schedule.to(device)

		# Move the ODE solver to accelerator.device.
		self.solver = self.solver.to(device)
	
	def CreateOptimizer(self):
		# Enable optimizations
		if self.train_conf.enable_xformers_memory_efficient_attention:
			if is_xformers_available():
				import xformers

				xformers_version = version.parse(xformers.__version__)
				if xformers_version == version.parse("0.0.16"):
					logger.warning("xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details.")

				self.unet.enable_xformers_memory_efficient_attention()
		else:
			raise ValueError("xformers is not available. Make sure it is installed correctly")

		if self.train_conf.gradient_checkpointing:
			self.unet.enable_gradient_checkpointing()

		if self.train_conf.use_8bit_adam:
			try:
				import bitsandbytes as bnb
			except ImportError:
				raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")

			optimizer_class = bnb.optim.AdamW8bit
		else:
			optimizer_class = torch.optim.AdamW

		# Optimizer creation
		self.optimizer = optimizer_class(
			self.unet.parameters(),
			lr=args.learning_rate,
			betas=(args.adam_beta1, args.adam_beta2),
			weight_decay=args.adam_weight_decay,
			eps=args.adam_epsilon,
		)

	def GetDatasetAdapter(self) -> Dataset:
		return self.dataset

class Model:
	def __init__(self, args, thread_control: ThreadControl):
		set_seed(args.seed)

		self.args = args
		
		if self.args.model_type == "sd":
			self.model = StableDiffusionModel(self.args, thread_control)
		elif self.args.model_type == "sdxl":
			self.model = StableDiffusionXLModel(self.args, thread_control)
		else:
			raise Exception("unhandle model_type:" + self.args.model_type)

	def GetModel(self):
		return self.model
	
	def Prepare(self):
		self.model.LoadComponents()
		self.model.Freeze()
		self.model.PatchLoraModules()
		self.model.UploadToDevice()
		self.model.CreateOptimizer()

	def GetDatasetAdapter(self):
		return self.model.GetDatasetAdapter()

class Train(ThreadControl):
	def __init__(self, args):
		prj = ProjectConf(args)

		accelerator_project_config = ProjectConfiguration(
			project_dir=prj.GetModelDir(), 
			logging_dir=prj.GetLogDir()
		)

		self.accelerator = Accelerator(
			gradient_accumulation_steps=args.gradient_accumulation_steps,
			mixed_precision=args.mixed_precision,
			log_with="tensorboard",
			project_config=accelerator_project_config,
			split_batches=True,  # It's important to set this to True when using webdataset to get the right number of steps for lr scheduling. If set to False, the number of steps will be devide by the number of processes assuming batches are multiplied by the number of processes
		)
		
			# Make one log on every process with the configuration for debugging.
		logging.basicConfig(
			format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
			datefmt="%m/%d/%Y %H:%M:%S",
			level=logging.INFO,
		)
		logger.info(self.accelerator.state, main_process_only=False)
		if self.accelerator.is_local_main_process:
			transformers.utils.logging.set_verbosity_warning()
			diffusers.utils.logging.set_verbosity_info()
		else:
			transformers.utils.logging.set_verbosity_error()
			diffusers.utils.logging.set_verbosity_error()
	
	@override
	def GetAccelerator(self):
		return self.accelerator

	@override
	def IsMainProcess(self) -> bool:
		return self.accelerator.is_main_process
	
	@override
	def WaitForEveryOne(self):
		self.accelerator.wait_for_everyone()

	def LogImage(self, infos: dict):
		for tracker in self.accelerator.trackers:
			if tracker.name == "tensorboard":
				tracker.writer.add_images(dataformats="NHWC", **infos)
			else:
				logger.warning(f"image logging not implemented for {tracker.name}")

	def End(self):
		self.accelerator.end_training()

class Dataset:
	def __init__(self, args, thread_control: ThreadControl):
		self.args = args
		self.thread_control = thread_control
		self.conf = DatasetConf(args)

	def Prepare(self):
		self.conf.data_dir
		pass

	def cache_latents(self):
		pass
	



class SavedCheckpoint:
	@classmethod
	def ApplyToParser(parser: argparse.ArgumentParser):
		parser.add_argument(
			"--checkpoints_total_limit",
			type=int,
			default=None,
			help=("Max number of checkpoints to store."),
		)
		parser.add_argument(
			"--resume_from_checkpoint",
			type=str,
			default=None,
			help=(
				"Whether training should be resumed from a previous checkpoint. Use a path saved by"
				' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
			),
		)
		

def parse_args(args):
	parser = argparse.ArgumentParser(description="Train lcm lora model.")
	ProjectConf.ApplyToParser(parser)
	ModelConf.ApplyToParser(parser)
	TrainConf.ApplyToParser(parser)
	DatasetConf.ApplyToParser(parser)
	SavedCheckpoint.ApplyToParser(parser)
	return parser.parse_args(args)

def main(args: argparse.ArgumentParser):
	ProjectConf(args).makedirs()

	# init accelerate
	train = Train(args)
	model = Model(args, train)

	model.Prepare()
	
	


	train.End()

if __name__ == "__main__":
	args = parse_args()
	main(args)