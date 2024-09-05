#!/bin/bash

tasks=(halfcheetah-medium-v2 hopper-medium-v2 walker2d-medium-v2)
seeds=(0 1 2 3)
alphas=(1.0)
betas=(1.0)

for seed in ${seeds[@]}; do
    for alpha in ${alphas[@]}; do
        for beta in ${betas[@]}; do
            for task in ${tasks[@]}; do
                python ddpg-bc-mixed-q.py --load-path log --task $task --seed $seed --alpha $alpha --beta $beta --epoch 500 > /dev/null 2>&1 &
            done
        done
    done
done

wait

echo "All done"
