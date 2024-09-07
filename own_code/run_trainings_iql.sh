#!/bin/bash

seeds=(0 1 2 3)
tasks=(kitchen-mixed-v0)

for seed in ${seeds[@]}; do
    for task in ${tasks[@]}; do
        python reused_code/run_iql.py --task kitchen-mixed-v0 --seed $seed --epoch 100000 --batch-size 1024 --hidden-dims 512 512 512 --device cpu > /dev/null 2>&1 &
    done
done

wait
echo "All done"