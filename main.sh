#!/usr/bin/env bash
for i in 'twitter2017' # 'twitter2015' 'twitter2017'
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
      --output_dir=out/${i}_${k}_output/ \
      --max_seq_length=128 \
      --do_train \
      --do_eval \
      --train_batch_size=32 \
      --mm_model ${k} \
      --layer_num1=1 \
      --layer_num2=1 \
      --num_train_epochs=24
    done
done
