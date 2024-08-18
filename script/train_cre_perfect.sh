#!/bin/bash
#export CUDA_VISIBLE_DEVICES=1
num=1

# Prepro CDR
python src/prepro.py \
    --dataset CDR \
    --no_dev 1 \
    --perfect 1

for ((i=1;i<=num;i++))
do
  python src/train_cre.py \
    --model_name_or_path /data/pretrained/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --num_epoch 50 \
    --train_batch_size 8 \
    --test_batch_size 16 \
    --learning_rate 3e-5 \
    --warmup_ratio 0.04 \
    --save_path ./model/CDR-cre-perfect.pt \
    --notes CDR-cre-perfect \
    --dataset CDR \
    --no_dev 1
  sleep 10
done

# Prepro GDA
python src/prepro.py \
    --dataset GDA \
    --no_dev 0 \
    --perfect 1
for ((i=1;i<=num;i++))
do
  python src/train_cre.py \
      --model_name_or_path /data/pretrained/BiomedNLP-PubMedBERT-base-uncased-abstract \
      --num_epoch 15 \
      --train_batch_size 8 \
      --test_batch_size 16 \
      --learning_rate 3e-5 \
      --warmup_ratio 0.04 \
      --save_path ./model/GDA-cre-perfect.pt \
      --notes GDA-cre-perfect \
      --dataset GDA \
      --evaluation_steps 1000 \
      --dropout 0.2 \
      --no_dev 0
done

# Prepro BioRED
python src/prepro.py \
    --dataset BioRED \
    --no_dev 1 \
    --perfect 1
for ((i=1;i<=num;i++))
do
  python src/train_cre.py \
      --model_name_or_path /data/pretrained/BiomedNLP-PubMedBERT-base-uncased-abstract \
      --train_batch_size 2 \
      --test_batch_size 4 \
      --num_epoch 50 \
      --learning_rate 3e-5 \
      --warmup_ratio 0.04 \
      --save_path ./model/BioRED-cre-perfect.pt \
      --notes BioRED-cre-perfect \
      --dataset BioRED \
      --dropout 0.2 \
      --no_dev 1
done
