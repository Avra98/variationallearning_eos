#!/bin/bash

# Define arrays of learning rates, datasets, and activation functions
learning_rates=(0.5  0.8)
#datasets=("cifar10-1k" "cifar10-2k" "cifar10-5k" "cifar10-10k" "cifar10-20k" "mnist-whole" "mnist-20k" "mnist-5k" "mnist-1k")
datasets=("mnist-20k")
activations=("fc-relu-depth4" "fc-relu-depth6")
#activations=("fc-relu-depth4")
# Define specific GPU IDs
gpu_ids=(0 1)  # Adjust this array based on your available GPU IDs
len_gpu_ids=${#gpu_ids[@]}
gpu_counter=0

# Set up a dictory for log files
log_dir="./training_logs1"
mkdir -p "$log_dir"  # Ensure the directory exists

# Loop over every combination of dataset, learning rate, and activation function
for dataset in "${datasets[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for activation in "${activations[@]}"; do
            # Get the current GPU ID from the list
            current_gpu=${gpu_ids[$((gpu_counter % len_gpu_ids))]}
            log_file="$log_dir/${dataset}_${activation}_lr${lr}_gpu${current_gpu}.log"
            echo "Training on $dataset with learning rate $lr and activation $activation on GPU $current_gpu"

            # Run the training script in background and assign it to the current GPU
            python src/gd.py $dataset $activation mse $lr 700 \
                --acc_goal 0.9999 --loss_goal 0.05 --neigs 2 --eig_freq 5 \
                --iterate_freq 5 --device_id $current_gpu > "$log_file" 2>&1 &

            # Increment GPU counter
            ((gpu_counter++))

            # If ten jobs are launched, wait for them to complete before launching more
            if [ $((gpu_counter % 2)) -eq 0 ]; then
                wait
            fi
        done
    done
done

wait
echo "All training sessions completed. Logs are saved in $log_dir"


# python src/gd.py cifar10-20k fc-relu-depth4 mse 1.5 5000  --acc_goal 0.9999 --loss_goal 0.05 --neigs 2 --eig_freq 100  --iterate_freq 5 --device_id 0