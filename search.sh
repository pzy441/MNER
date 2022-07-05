#!/usr/bin/env bash

seeds=(32)
temps=(0.14)
temp_lambs=(0.10 0.20 0.30 0.40 0.50)
lambs=(0.40 0.45 0.50 0.60 0.70)
for i in 'twitter2017' # 'twitter2015' 'twitter2017'
do
    for seed in ${seeds[*]}
    do
      for temp in ${temps[*]}
      do
        for temp_lamb in ${temp_lambs[*]}
        do
          for lamb in ${lambs[*]}
          do
            for k in 'TempBert'
            do
              echo '__main__.py'
              echo ${i}
              echo ${k}
              PYTHONIOENCODING=utf-8
              CUDA_VISIBLE_DEVICES=0
              python __main__.py \
              --data_dir=data/${i} \
              --bert_model=bert-base-cased \
              --task_name=${i} \
              --output_dir=out/${i}/ \
              --max_seq_length=128 \
              --do_train \
              --train_batch_size=32 \
              --mm_model ${k} \
              --layer_num1=1 \
              --layer_num2=1 \
              --num_train_epochs=24 \
              --seed $seed \
              --temp $temp \
              --temp_lamb $temp_lamb \
              --lamb $lamb
            done
          done
        done
      done
    done
done