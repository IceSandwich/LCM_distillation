import gc
import torch
from transformers import CLIPTokenizer, CLIPTextModel
from torch.utils.data import Dataset, DataLoader
from torch.nn import MSELoss
from tqdm import tqdm
import typing
import random
import os

import torch
import torch.nn as nn
import thop
import numpy as np
import lightning as L

from transformers.models.clip.modeling_clip import CLIPTextEmbeddings, CLIPTextConfig

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
	def __init__(self, target_dim: int, **kwargs) -> None:
		super().__init__(**kwargs)
		
		if target_dim != kwargs["d_model"]:
			self.adapter = nn.Linear(kwargs["d_model"], target_dim)
			self.linear2 = nn.Linear(kwargs["dim_feedforward"], target_dim)
			if not self.norm_first:
				self.norm2 = nn.LayerNorm(target_dim)
		else:
			self.adapter = None

	def forward(self, src: torch.Tensor, src_mask: typing.Optional[torch.Tensor] = None, src_key_padding_mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
		x = src
		if self.norm_first:
			x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
			if self.adapter is not None:
				x = self.adapter(x) + self._ff_block(self.norm2(x))
			else:
				x = x + self._ff_block(self.norm2(x))
		else:
			x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
			if self.adapter is not None:
				x = self.norm2(self.adapter(x) + self._ff_block(x))
			else:
				x = self.norm2(x + self._ff_block(x))

		return x

class CustomCLIPEncoder(nn.Module):
	def __init__(self, layer_dims: typing.List[int], num_heads: typing.List[int], intermediate_multiplier=1, dropout=0.1):
		super().__init__()
		self.layers = nn.ModuleList()

		for i, dim in enumerate(layer_dims):
			layer = CustomTransformerEncoderLayer(
				target_dim=layer_dims[i+1] if i < len(layer_dims)-1 else dim,
				d_model=dim,
				nhead=num_heads[i],
				dim_feedforward=dim * intermediate_multiplier,
				dropout=dropout,
				activation="gelu",
				batch_first=True
			)
			self.layers.append(layer)

	def forward(self, hidden_states: torch.Tensor):
		for layer in self.layers:
			hidden_states = layer(hidden_states)
		return hidden_states

class CustomCLIPTextTransformer(nn.Module):
	def __init__(self):
		super().__init__()

		# === 配置维度 ===
		self.vocab_size = 49408
		self.max_position_embeddings = 77

		# Teacher:    MACs:  110.031G Params:  714.737M

		# 32 layers
		# ======================= Config 1 =========================
		# self.layer_dims = [2048] * 6 + [1280] * 20 + [2048] * 6
		# self.num_heads = [16] * 6 + [10] * 20 + [16] * 6  # 注意必须整除 hidden_size
		# ======================= Config 2 =========================  MACs:  25.997G Params:  168.739M Lost: 0.68
		# self.layer_dims = [2048] * 6 + [1704, 1416, 1184, 984, 816] + [768] * 10 + [816, 984, 1184, 1416, 1704] + [2048] * 6 
		# self.num_heads = [16] * 6 + [8] * 20 + [16] * 6 
		# ======================= Config 3 ========================= MACs:  20.673G Params:  134.189M Lost: 0.69
		# self.layer_dims = [2048] * 6 + [1704, 1416, 1184, 984, 816, 768, 640] + [512] * 10 + [640, 768, 816, 984, 1184, 1416, 1704] + [2048] * 2
		# self.num_heads = [16] * 6 + [8] * 24 + [16] * 2
		# ======================= Config 4 ========================= MACs:  19.749G Params:  128.178M Loss: 0.72
		# self.layer_dims = [2048] * 6 + [1704, 1416, 1184, 984, 816, 768] + [768] * 16 + [984, 1416] + [2048] * 2
		# self.num_heads = [16] * 6 + [8] * 24 + [16] * 2
		# ======================= Config 7 ========================= MACs:  18.132G Params:  117.689M Loss: 0.73
		# self.layer_dims = [2048] * 6 + [1704, 1416, 1184, 984, 816, 768] + [512] * 16 + [984, 1416] + [2048] * 2
		# self.num_heads = [16] * 6 + [8] * 24 + [16] * 2

		# 24 layers
		# ====================== Config 5 ========================== MACs:  14.264G Params:  92.576M Loss: 0.70, 0.68
		# self.layer_dims = [2048] * 3 + [1416, 984] + [768] * 14 + [984, 1416] + [2048] * 3
		# self.num_heads = [16] * 3 + [8] * 18 + [16] * 3
		# ====================== Config 6 ========================== MACs:  13.152G Params:  85.360M Loss: 0.72
		# self.layer_dims = [2048] * 3 + [1416, 984] + [768] * 15 + [984, 1416] + [2048] * 2
		# self.num_heads = [16] * 3 + [8] * 19 + [16] * 2

		# 16 layers
		# ====================== Config 8 ========================== MACs:  11.691G Params:  75.886 Loss: 0.69
		# self.layer_dims = [2048] * 3 + [1416, 984] + [768] * 7 + [984, 1416] + [2048] * 2
		# self.num_heads = [16] * 3 + [8] * 11 + [16] * 2

		# 12 layers
		# ====================== Config 8 ========================== MACs:  10.961G Params:  71.149M Loss: 0.69 / 0.419
		# self.layer_dims = [2048] * 3 + [1416, 984] + [768] * 3 + [984, 1416] + [2048] * 2
		# self.num_heads = [16] * 3 + [8] * 7 + [16] * 2
		# ====================== Config 9 ========================== MACs:  7.953G Params:  51.622M Loss: 0.87
		# self.layer_dims = [2048] * 3 + [1416, 984] + [768] * 5 + [984, 1416]
		# self.num_heads = [16] * 3 + [8] * 9
		# ====================== Config 10 ========================= MACs:  8.628G Params:  56.001M Loss: 0.73 / 0.429
		# self.layer_dims = [2048] * 3 + [768] * 7 + [2048] * 2
		# self.num_heads = [16] * 3 + [8] * 7 + [16] * 2
		# ====================== Config 11 ========================= MACs:  8.710G Params:  56.538M Loss: 0.70, 0.71 / 0.428
		# self.layer_dims = [2048] * 2 + [1200] +  [768] * 6 + [1200] + [2048] * 2
		# self.num_heads = [16] * 2 + [8] * 8 + [16] * 2

		# 10 layers
		# ===================== Config 12 ========================= MACs:  8.870G Params:  57.575M Loss: 0.70, 0.707 / 0.419, 0.425
		self.layer_dims = [2048] * 2 + [1200] * 2 +  [768] * 2 + [1200] * 2 + [2048] * 2
		self.num_heads = [16] * 2 + [8] * 6 + [16] * 2



		self.embeddings = CLIPTextEmbeddings(CLIPTextConfig(
			vocab_size=self.vocab_size,
			hidden_size=2048,
			max_position_embeddings=self.max_position_embeddings,
		))

		self.encoder = CustomCLIPEncoder(self.layer_dims, self.num_heads)

		# 输出 projection 层（将不同维度映射到统一输出维度）
		self.final_proj = nn.Linear(self.layer_dims[-1], 2048)

	def forward(self, input_ids: torch.LongTensor):
		# B, T = input_ids.shape
		# position_ids = torch.arange(0, T, device=input_ids.device).unsqueeze(0).expand(B, T)

		# x = self.token_embedding(input_ids) + self.position_embedding(position_ids)  # → [B, T, 2048]
		x = self.embeddings(input_ids)

		x = self.encoder(x)  # → Mixed dim layers

		x = self.final_proj(x)   # → [B, T, 2048]
		return x
	
class CustomCLIPTextModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.text_model = CustomCLIPTextTransformer()
	
	def InitWeights(self):
		def init_weights(module: nn.Module):
			if isinstance(module, nn.Linear):
				# nn.init.normal_(module.weight, std=0.02)
				nn.init.xavier_uniform_(module.weight)
				if module.bias is not None:
					nn.init.zeros_(module.bias)
			elif isinstance(module, nn.Embedding):
				nn.init.normal_(module.weight, std=0.02)
			elif isinstance(module, nn.LayerNorm):
				nn.init.ones_(module.weight)
				nn.init.zeros_(module.bias)
		self.text_model.apply(init_weights)

	def forward(self, input_ids: torch.LongTensor):
		return self.text_model(input_ids)

class TextEncoderXL(nn.Module):
	def __init__(self, pretrained_model_name_or_path: str, **kwargs):
		super().__init__()
		self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer", **kwargs)
		self.tokenizer2: CLIPTokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_2", **kwargs)
		self.text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder", **kwargs)
		self.text_encoder2: CLIPTextModel = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder_2", **kwargs)

	def RequireGrad(self, value: bool):
		for module in [self.tokenizer, self.tokenizer2, self.text_encoder, self.text_encoder2]:
			for param in module.parameters():
				param.requires_grad = value

	def Encode(self, text: typing.List[str], **kwargs):
		return self.tokenizer.encode(text, **kwargs)
	
	def Tokenize(self, text: str):
		return self.tokenizer.tokenize(text)
	
	def Encodxe(self, text: typing.List[str], device: typing.Optional[str] = None):
		inputs_1 = self.tokenizer(
			text,
			padding="max_length",
			truncation=True,
			max_length=77,
			return_tensors="pt"
		)
		inputs_2 = self.tokenizer2(
			text,
			padding="max_length",
			truncation=True,
			max_length=77,
			return_tensors="pt"
		)
		if device is not None:
			inputs_1 = inputs_1.to(device=device)
			inputs_2  = inputs_2.to(device=device)
		return (inputs_1, inputs_2)
	
	def forward(self, input_1: torch.LongTensor, input_2: torch.LongTensor):
		embed_1 = self.text_encoder(input_1).last_hidden_state
		embed_2 = self.text_encoder2(input_2).last_hidden_state
		embed = torch.cat([embed_1, embed_2], dim=-1)
		return embed
	
	def Forward(self, text: typing.List[str], device: typing.Optional[str] = None):
		inputs = self.Tokenize(text, device=device)
		embed = self.Encode(inputs[0], inputs[1])
		return embed
	
# ========== Step 1: Load teacher (SDXL) ==========
# pipe_sdxl = StableDiffusionXLPipeline.from_pretrained(
# 	"stabilityai/stable-diffusion-xl-base-1.0",
# 	torch_dtype=torch.float16
# ).to("cuda")
# teacher_tokenizer = pipe_sdxl.tokenizer
# teacher_encoder = pipe_sdxl.text_encoder  # 只使用主 encoder

# teacher.RequireGrad(False)

# ========== Step 2: Load student (SD1.5) ==========
# pipe_sd15 = StableDiffusionPipeline.from_pretrained(
# 	"runwayml/stable-diffusion-v1-5",
# 	torch_dtype=torch.float16
# ).to("cuda")
# student_tokenizer = pipe_sd15.tokenizer
# student_encoder = pipe_sd15.text_encoder
# print("Creating student model...")
# student = CustomCLIPTextModel()
# student.InitWeights()
# student.text_model.embeddings = teacher.text_encoder2.text_model.embeddings
# student = student.to("cuda")

# =========== Test ================
# input_ids = torch.randint(0, 49408, (2, 77)).to("cuda")
# print("Input: ", input_ids.shape)
# teacher.eval()
# MACs, params = thop.profile(teacher, inputs=(input_ids, input_ids))
# MACs, params = thop.clever_format([MACs, params], '%.3f')
# output = teacher(input_ids, input_ids)
# print("Teacher: ", output.shape, "MACs: ", MACs, "Params: ", params)
# MACs, params = thop.profile(student, inputs=(input_ids,))
# MACs, params = thop.clever_format([MACs, params], '%.3f')
# output = student(input_ids)
# print("Student: ", output.shape, "MACs: ", MACs, "Params: ", params)
# del input_ids, output # → [2, 77, 2048]


# 解冻 student encoder 参数
# for p in student_encoder.parameters():
# 	p.requires_grad = True

# ========== Step 3: Prompt Dataset ==========
class PromptDataset(Dataset):
	def __init__(self, prompts: typing.List[str]):
		self.prompts = prompts

	def __len__(self):
		return len(self.prompts)

	def __getitem__(self, idx):
		return self.prompts[idx]
	
	def Shuffle(self):
		for i in range(len(self.prompts)):
			self.prompts[i] = ', '.join(random.shuffle([x.strip() for x in self.prompts[i].split(',')]))

# 示例 prompt，可换成自定义文本
# prompts = [
# 	"a futuristic city at sunset",
# 	"a cat wearing glasses reading a book",
# 	"an astronaut walking on Mars",
# 	"a medieval knight riding a dragon",
# 	"huangertao, penis, high kick, dynamic pose, outdoor, headshot, low angle",
# 	"Procy, bathroom, spread legs, raised arms, masterpiece, detail, 8k",
# 	"a cat wearing glasses reading a book, under the tree, moonlight, night",
# 	"a medieval knight riding a dragon, indoor, sunlight, inn, castle",
# 	# ...
# ]

class CachedInstanceDataset(Dataset):
	def __init__(self, prompts: typing.List[str]):
		self.prompts = prompts
		self.cached: typing.List[typing.Tuple[torch.LongTensor, torch.Tensor]] = []

	@torch.no_grad()
	def Cache(self, model: TextEncoderXL, device: str = "cpu"):
		self.cached.clear()
		for i in tqdm(range(0, len(self.prompts))):
			tokens: torch.LongTensor = model.Encode(self.prompts[i], padding="max_length", truncation=True, return_tensors="pt").to(device)
			# Get embeddings
			embeds = model(tokens, tokens)
			if embeds.dtype == torch.float16:
				embeds = embeds.to(torch.float32).to(device)
			for token, embed in zip(tokens, embeds):
				self.cached.append((token, embed))

		assert len(self.cached) == len(self.prompts), f"{len(self.cached)} doesn't match the len of prompts: {len(self.prompts)}"

	def __len__(self):
		return len(self.prompts)
	
	def __getitem__(self, idx):
		return self.cached[idx]
	
class CachedDataset(Dataset):
	def __init__(self, dataset_path: str, split: float = 0.95, is_training_set = True):
		self.cache_folder = os.path.join(dataset_path, "cache")
		if not os.path.exists(self.cache_folder):
			print(f"Generating cache in {self.cache_folder}")
			os.makedirs(self.cache_folder, exist_ok=True)
			
			print("Loading teacher model...")
			self.model = TextEncoderXL(
				"D:\MyGithub\LCM_distillation\model",
				torch_dtype = torch.float16
			).to("cuda")
			self.model.eval()

			input_ids = torch.randint(0, 49408, (2, 77)).to("cuda")
			print("Input: ", input_ids.shape)
			MACs, params = thop.profile(self.model, inputs=(input_ids, input_ids))
			MACs, params = thop.clever_format([MACs, params], '%.3f')

			output = self.model(input_ids, input_ids)
			print("Teacher: ", output.shape, "MACs: ", MACs, "Params: ", params)

			with torch.no_grad():
				counter = 0
				for filename in os.listdir(dataset_path):
					if not filename.endswith('.txt'): continue
					with open(os.path.join(dataset_path, filename), 'r', encoding='utf-8') as f:
						prompts = [x.strip() for x in f.readlines() if x.strip() != ""]
					print(f"Accept dataset: {filename} has {len(prompts)} lines.")
					for prompt in tqdm(prompts):
						tokens: torch.LongTensor = self.model.Encode(prompt, padding="max_length", truncation=True, return_tensors="pt").to("cuda")
						embed = self.model(tokens, tokens)
						if embed.dtype == torch.float16:
							embed = embed.to(torch.float32)
						np.savez(os.path.join(self.cache_folder, f"{counter}.npz"), tokens=tokens.detach().cpu().numpy(), embed=embed.detach().cpu().numpy(), prompt=prompt)
						counter += 1
			del self.model
	
		print(f"Using cache dir: {self.cache_folder}")
		self.datasets = os.listdir(self.cache_folder)
		training_len = int(len(self.datasets) * split)
		if is_training_set:
			self.datasets = self.datasets[:training_len]
			print(f"Training set: {len(self.datasets)} items")
		else:
			self.datasets = self.datasets[training_len:]
			print(f"Validation set: {len(self.datasets)} items")

	def __len__(self):
		return len(self.datasets)	
	
	def __getitem__(self, index: int):
		bundle = np.load(os.path.join(self.cache_folder, self.datasets[index]))
		tokens: np.ndarray = bundle["tokens"]
		embed: np.ndarray = bundle["embed"]
		return tokens.squeeze(), embed.squeeze()

print("Prepare dataset...")
train_ds = DataLoader(CachedDataset("datasets"), batch_size=32, shuffle=True)
valid_ds = DataLoader(CachedDataset("datasets", is_training_set=False), batch_size=32, shuffle=False)
# ========== Step 4: Optimizer & Loss ==========
# print("Preparse optimizer...")
# optimizer = torch.optim.AdamW(student.parameters(), lr=4e-5)
# loss_fn = MSELoss()


# ========== Clean Cache =============
gc.collect()
torch.cuda.empty_cache()


class LitTextEncoder(L.LightningModule):
	def __init__(self, repeat = 1):
		super().__init__()
		print("Creating student model...")
		self.model = CustomCLIPTextModel()
		self.model.InitWeights()

		self.loss_fn = MSELoss()
		self.repeat = repeat

	def common_step(self, batch: typing.Tuple[torch.LongTensor, torch.Tensor], batch_idx: int):
		tokens, embeds = batch
		hat = self.model(tokens)
		loss = self.loss_fn(hat, embeds)
		return loss

	def training_step(self, batch: typing.Tuple[torch.LongTensor, torch.Tensor], batch_idx: int):
		loss = self.common_step(batch, batch_idx)
		for i in range(self.repeat - 1):
			loss += self.common_step(batch, batch_idx)
		loss = loss / self.repeat
		self.log("train_loss", loss)
		return loss
	
	def test_step(self, batch: typing.Tuple[torch.LongTensor, torch.Tensor], batch_idx: int):
		loss = self.common_step(batch, batch_idx)
		self.log("test_loss", loss)
	
	def validation_step(self, batch: typing.Tuple[torch.LongTensor, torch.Tensor], batch_idx: int):
		loss = self.common_step(batch, batch_idx)
		self.log("val_loss", loss)
	
	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 20, 2, eta_min=1e-5)
		return [optimizer], [scheduler]
	
model = LitTextEncoder()

trainer = L.Trainer(max_epochs=100, default_root_dir="trains")
trainer.fit(model=model, train_dataloaders=train_ds, val_dataloaders=valid_ds)
		

# ========== Step 5: Training Loop ==========
# for epoch in range(20):
# 	print(f"Epoch {epoch+1}")
# 	# for tokens, targets in tqdm(dataloader):
# 	for batch_prompts in tqdm(dataloader):
# 		batch_prompts: typing.List[str]
# 		with torch.no_grad():
# 			tokens: typing.List[torch.LongTensor] = []
# 			for prompt in batch_prompts:
# 				p = [x.strip() for x in prompt.split(',')]
# 				random.shuffle(p)
# 				p = ', '.join(p)
# 				tk = teacher.Encode(prompts, padding="max_length", truncation=True, return_tensors="pt")
# 				tokens.append(tk)
# 			tokens: torch.LongTensor = torch.concat(tokens, dim=0)
# 			tokens = tokens.to("cuda")
# 			targets = teacher(tokens, tokens)
# 			if targets.dtype == torch.float16:
# 				targets = targets.to(torch.float32)
# 			targets = targets.to("cuda")
# 		# tokens: torch.LongTensor = teacher.Encode(batch_prompts, padding="max_length", truncation=True, return_tensors="pt").to("cuda")
		
# 		student_embeds = student(tokens)

# 		# Get embeddings
# 		# with torch.no_grad():
# 			# teacher_embeds = teacher_encoder(**inputs_teacher).last_hidden_state  # [B, 77, 1280 or 2048]

# 		# student_embeds = student_encoder(**inputs_student).last_hidden_state  # [B, 77, 768]

# 		# Align dimensions if needed (project teacher to student dim)
# 		# if teacher_embeds.shape[-1] != student_embeds.shape[-1]:
# 		# 	proj = torch.nn.Linear(teacher_embeds.shape[-1], student_embeds.shape[-1]).to("cuda")
# 		# 	teacher_embeds = proj(teacher_embeds)
		

# 		# Compute distillation loss
# 		loss = loss_fn(student_embeds, targets)
# 		print()
# 		print("Loss: ", loss.item())

# 		# Optimize
# 		loss.backward()
# 		optimizer.step()
# 		optimizer.zero_grad()

# # ========== Step 6: Save finetuned student ==========
# # student_encoder.save_pretrained("./distilled_sd15_text_encoder")
# # student_tokenizer.save_pretrained("./distilled_sd15_tokenizer")
# print("Done. Loss: ", loss.item())