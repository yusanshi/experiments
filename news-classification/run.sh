#!/bin/bash

date >> results.txt
echo "" >> results.txt

for i in {1..2}; do
    for use_url_text in attention; do
        for text_method_one in lstm; do
            for text_method_two in average; do
                export REMARK=round_${i}-${use_url_text}-${text_method_one}-${text_method_two}
                export USE_URL_TEXT=$use_url_text
                export TEXT_METHOD_ONE=$text_method_one
                export TEXT_METHOD_TWO=$text_method_two
                echo $REMARK >> results.txt
                rm -rf checkpoint/*
                python3 src/train.py
                python3 src/evaluate.py >> results.txt
                echo "" >> results.txt
            done
        done
    done
done

