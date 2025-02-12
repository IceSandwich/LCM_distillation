# Safetensors转Diffusers

```bash
python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path dfdog/model/DragonfruitDogLoraMerge06_v1.safetensors --from_safetensors --dump_path dfdog/diffusers
```



# 读取生成图的metadata

https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/15323



# 训练

```bash
accelerate launch train_lcm_distill_lora_sdxl.py --pretrained_teacher_model=./model --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix --output_dir="YiffyMixv51XL-lora-lcm-sdxl" --mixed_precision="fp16" --train_data_dir="dataset" --resolution=1024 --train_batch_size=1 --gradient_accumulation_steps=1 --gradient_checkpointing --use_8bit_adam --lora_rank=64 --learning_rate=1e-4 --report_to="tensorboard" --lr_scheduler="constant" --lr_warmup_steps=0 --max_train_steps=3000 --checkpointing_steps=500 --validation_steps=50 --seed="0"
```
