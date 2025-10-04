#!/bin/bash

# Define an array of dataset names
#datasets=("mnist-20k" "mnist-5k" "mnist-1k")  #"cifar10-100"  "cifar10-200")
datasets=( "cifar10-20k")
# Loop through the array and run the command with each dataset in the background
for dataset in "${datasets[@]}"
do
  python src/gd.py "$dataset" fc-tanh mse 0.01 10000 --physical_batch_size 1024 --acc_goal 1.01 --neigs 2 --eig_freq 1000 --device_id 0 &
done

# Wait for all background jobs to finish
wait
