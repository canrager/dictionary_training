#!/bin/bash

# Define arrays for different configurations
ARCH=("vanilla" "vanilla" "vanilla" "topk" "topk" "topk")
LAYERS=(5 12 19 5 12 19)
WIDTH=14
NUM_TOKENS=300_000_000
DEVICES=("cuda:0" "cuda:1" "cuda:2" "cuda:3" "cuda:4" "cuda:5")
SAVE_DIR="trained_saes"

# Loop through the configurations
for i in {0..5}; do
    nohup python3 training_sweep.py \
        --save_dir ${SAVE_DIR} \
        --architecture ${ARCH[$i]} \
        --layers ${LAYERS[$i]} \
        --width_exponents ${WIDTH} \
        --num_tokens ${NUM_TOKENS}\
        --device ${DEVICES[$i]} \
        > "./logs/${ARCH[$i]}_l${LAYERS[$i]}_w${WIDTH}_${DEVICES[$i]//:/_}.out" 2>&1 &

        # --no_wandb_logging \
        # --dry_run \

    echo "Started job ${i}/5: ${ARCH[$i]} with ${LAYERS[$i]} layers"
    
    # Optional: add a small delay between job submissions
    sleep 2
done

echo "All jobs submitted!"