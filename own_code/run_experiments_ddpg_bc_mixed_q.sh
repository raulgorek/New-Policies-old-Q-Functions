#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <task>"
  exit 1
fi

task=$1

seeds=(0 1 2 3)
alphas=(1.0)
betas=(0.25 0.5 0.75)

for seed in ${seeds[@]}; do
    for alpha in ${alphas[@]}; do
        for beta in ${betas[@]}; do
            python ddpg-bc-mixed-q.py --load-path log --task $task --seed $seed --alpha $alpha --beta $beta --epoch 500 > /dev/null 2>&1 &
        done
    done
done

wait

echo "All done"
