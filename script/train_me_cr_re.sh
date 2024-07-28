#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# CDR dataset
#for ((i=1;i<=10;i++))
#do
#  # Entity Recognition
#  python src/train_me.py \
#      --model_name_or_path /data/pretrained/BiomedNLP-PubMedBERT-base-uncased-abstract \
#      --num_epoch 50 \
#      --save_path ./model/CDR-me_cr_re-me.pt \
#      --notes CDR-me_cr_re-me \
#      --dataset CDR \
#      --no_dev 1
#  sleep 5
#  # Prepro
#  python src/prepro.py \
#      --dataset CDR \
#      --no_dev 1
#  # Coreference Resolution & Relation Extraction
#  for ((j=1;j<=10;j++))
#  do
#    python src/train_cre.py \
#      --model_name_or_path /data/pretrained/BiomedNLP-PubMedBERT-base-uncased-abstract \
#      --num_epoch 50 \
#      --save_path ./model/CDR-me_cr_re-cre.pt \
#      --notes CDR-me_cr_re-cre \
#      --dataset CDR \
#      --no_dev 1
#    sleep 5
#  done
#done

# GDA dataset
# Entity Recognition
#for ((i=1;i<=5;i++))
#do
#  python src/train_me.py \
#      --model_name_or_path /data/pretrained/BiomedNLP-PubMedBERT-base-uncased-abstract \
#      --num_epoch 50 \
#      --save_path ./model/GDA-me_cr_re-me.pt \
#      --notes GDA-me_cr_re-me \
#      --dataset GDA \
#      --evaluation_steps 1000 \
#      --no_dev 0
#  sleep 5
  # Prepro
  python src/prepro.py \
      --dataset GDA \
      --no_dev 0
  # Coreference Resolution & Relation Extraction
  for ((j=1;j<=3;j++))
  do
    python src/train_cre.py \
      --model_name_or_path /data/pretrained/BiomedNLP-PubMedBERT-base-uncased-abstract \
      --num_epoch 15 \
      --save_path ./model/GDA-me_cr_re-cre.pt \
      --notes GDA-me_cr_re-cre \
      --dataset GDA \
      --evaluation_steps 1000 \
      --no_dev 0
    sleep 5
  done
#done

# BioRED dataset
# Entity Recognition
#for ((i=1;i<=3;i++))
#do
#  python src/train_me.py \
#      --model_name_or_path /data/pretrained/BiomedNLP-PubMedBERT-base-uncased-abstract \
#      --num_epoch 50 \
#      --save_path ./model/BioRED-me_cr_re-me.pt \
#      --notes BioRED-me_cr_re-me \
#      --dataset BioRED \
#      --no_dev 1
#  sleep 5
#  # Prepro
#  python src/prepro.py \
#      --dataset BioRED \
#      --no_dev 1
#  # Coreference Resolution & Relation Extraction
#  for ((j=1;j<=10;j++))
#  do
#    python src/train_cre.py \
#        --model_name_or_path /data/pretrained/BiomedNLP-PubMedBERT-base-uncased-abstract \
#        --train_batch_size 2 \
#        --test_batch_size 4 \
#        --num_epoch 50 \
#        --save_path ./model/BioRED-me_cr_re-cre.pt \
#        --notes BioRED-me_cr_re-cre \
#        --dataset BioRED \
#        --no_dev 1
#    sleep 5
#  done
#done
