Train without flash-attention:
```````shell
# Step 1 - SFT
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/ap-normistral-7b/sft/config_qlora.yaml --load_in_4bit=true --use_flash_attention_2=false

# Step 2 - DPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_dpo.py recipes/ap-normistral-7b/dpo/config_qlora.yaml --use_flash_attention_2=false
```````