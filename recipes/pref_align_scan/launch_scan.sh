#!/bin/bash
configs=("norllm-ai-normistral" "ap-normistral")
loss_types=("sigmoid" "kto_pair" "ipo" "hinge")
betas=("0.7" "0.4" "0.1" "0.8" "0.5" "0.2")

cd ~/alignment-handbook

for config in "${configs[@]}"; do
    for loss_type in "${loss_types[@]}"; do
        for beta in "${betas[@]}"; do
            model_revision="${loss_type}-${beta}"

            ACCELERATE_LOG_LEVEL=info setsid nohup accelerate launch \
            --config_file recipes/accelerate_configs/multi_gpu.yaml \
            --num_processes=1 \
            scripts/run_dpo.py recipes/pref_align_scan/dpo/config_${config}.yaml \
            --output_dir=data_lr5e-6-1e/$config-7b-${loss_type}-beta-${beta} \
            --hub_model_revision=${model_revision} \
            --loss_type=${loss_type} --use_flash_attention_2=false --beta=${beta}
        done
    done
done