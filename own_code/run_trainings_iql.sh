
seeds=(0 1 2 3)
tasks=(kitchen)

for seed in ${seeds[@]}; do
    for task in ${tasks[@]}; do
        python reused_code/train_iql.py --task $task --seed $seed --epoch 100000 --device mps > /dev/null 2>&1 &
    done
done

wait
echo "All done"