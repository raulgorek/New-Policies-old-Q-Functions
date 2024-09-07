#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <task>"
  exit 1
fi

task=$1

seeds=(0 1 2 3)
alphas=(-8.75)

for seed in ${seeds[@]}; do
    for alpha in ${alphas[@]}; do
        log_file="console_out/task_${task}_seed_${seed}_alpha_${alpha}.log"
        python own_code/ddpg-bc-cql.py --load-path log --task $task --seed $seed --alpha $alpha --epoch 500 > /dev/null 2>&1 &
    done
done

wait

echo "All done"
