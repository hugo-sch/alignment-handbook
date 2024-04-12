#!/bin/bash
# Define an array containing the base configs we wish to fine tune
configs=("aftonposten")
# Define an array of loss types
loss_types=("sigmoid" "kto_pair" "ipo")
# Define an array of beta values
betas=("0.01" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")

cd ~/alignment-handbook

# Outer loop for loss types
for config in "${configs[@]}"; do
    for loss_type in "${loss_types[@]}"; do
        # Inner loop for beta values
        for beta in "${betas[@]}"; do
            model_revision="${loss_type}-${beta}"

            # Launch the job
            ACCELERATE_LOG_LEVEL=info setsid nohup accelerate launch \
            --config_file recipes/accelerate_configs/multi_gpu.yaml \
            --num_processes=1 \
            scripts/run_dpo.py recipes/pref_align_scan/dpo/config_aftonposten.yaml \
            --output_dir=data/$config-6b-align-scan-${loss_type}-beta-${beta} \
            --hub_model_revision=${model_revision} \
            --loss_type=${loss_type} --use_flash_attention_2=false --beta=${beta}
        done
    done
done