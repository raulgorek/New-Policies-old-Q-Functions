#!/bin/bash

# if [ -z "$1" ]; then
#   echo "Usage: $0 <task>"
#   exit 1
# fi

tasks=(halfcheetah-medium-replay-v2 hopper-medium-replay-v2 walker2d-medium-replay-v2)

seeds=(0 1 2 3)
alphas=(1 30)

for task in ${tasks[@]}; do
    for seed in ${seeds[@]}; do
        for alpha in ${alphas[@]}; do
            python code/ddpg-bc-cql.py --load-path log --task $task --seed $seed --alpha $alpha --epoch 500 > /dev/null 2>&1 &
        done
    done
    wait
    echo "Task $task done"
done

echo "All done"
