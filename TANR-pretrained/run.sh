#!/bin/bash

date >> results.txt
echo "" >> results.txt

for i in {1..2}; do
    for classification_initiate in 0 1; do
        for joint_loss in 0 1; do
            REMARK=round_${i}-${classification_initiate}-${joint_loss}
            echo $REMARK >> results.txt
            rm -rf checkpoint/
            CLASSIFICATION_INITIATE=${classification_initiate} JOINT_LOSS=${joint_loss} \
            python3 src/train.py
            python3 src/evaluate.py >> results.txt
            echo "" >> results.txt
        done
    done
done

