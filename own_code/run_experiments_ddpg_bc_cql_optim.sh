#!/bin/bash

if [ -z "$2" ]; then
  echo "Usage: $0 <task> <study_name>"
  exit 1
fi

task=$1

seeds=(0 1 2 3)
alphas=(0 1)
study_name=$2

for seed in ${seeds[@]}; do
    for alpha in ${alphas[@]}; do
        python own_code/ddpg-bc-cql_hyper_optim.py --study-name $study_name --load-path log --task $task --seed $seed --epoch 200 > /dev/null 2>&1 &
    done
done

wait

echo "All done"
