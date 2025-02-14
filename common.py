from transformers import AutoTokenizer, PretrainedConfig
import numpy as np
import torch
import random

# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoders, tokenizers, is_train=True):
    prompt_embeds_list = []

    captions = []
    for caption in prompt_batch:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

# Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
def compute_embeddings(prompt_batch, original_sizes, crop_coords, text_encoders, tokenizers, resolution, device, weight_dtype, is_train=True):
	def compute_time_ids(original_size, crops_coords_top_left):
		target_size = (resolution, resolution)
		add_time_ids = list(original_size + crops_coords_top_left + target_size)
		add_time_ids = torch.tensor([add_time_ids])
		add_time_ids = add_time_ids.to(device, dtype=weight_dtype)
		return add_time_ids

	prompt_embeds, pooled_prompt_embeds = encode_prompt(prompt_batch, text_encoders, tokenizers, is_train)
	add_text_embeds = pooled_prompt_embeds

	add_time_ids = torch.cat([compute_time_ids(s, c) for s, c in zip(original_sizes, crop_coords)])

	prompt_embeds = prompt_embeds.to(device)
	add_text_embeds = add_text_embeds.to(device)
	unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

	return {"prompt_embeds": prompt_embeds, **unet_added_cond_kwargs}

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")