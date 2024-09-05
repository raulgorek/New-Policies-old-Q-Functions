#!/bin/bash

tasks=(halfcheetah-medium-v2 walker2d-medium-v2 hopper-medium-v2)
seeds=(0 1 2 3)
alphas=(1)

for seed in ${seeds[@]}; do
    for alpha in ${alphas[@]}; do
        for task in ${tasks[@]}; do
          python ddpg-bc-iql.py --load-path log --task $task --seed $seed --alpha $alpha --epoch 500 > /dev/null 2>&1 &
        done
    done
done

wait

echo "All done"
