#!/bin/bash

date >> results.txt
echo "" >> results.txt

for i in {1..3}; do
    for num_batches_classification in 0 500 1000; do
        for joint_loss in 0 1; do
            REMARK=round_${i}-${num_batches_classification}-${joint_loss}
            echo $REMARK >> results.txt
            rm -rf checkpoint/
            NUM_BATCHES_CLASSIFICATION=${num_batches_classification} JOINT_LOSS=${joint_loss} \
            python3 src/train.py
            python3 src/evaluate.py >> results.txt
            echo "" >> results.txt
        done
    done
done

