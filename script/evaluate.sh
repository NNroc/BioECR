#!/bin/bash
#export CUDA_VISIBLE_DEVICES=1

# entity extraction
python src/train_me.py \
    --model_name_or_path /data/pretrained/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --load_path ./model/CDR-me.pt \
    --notes CDR-me_cr_re-me \
    --dataset CDR \
    --no_dev 1
sleep 1
python src/train_me.py \
    --model_name_or_path /data/pretrained/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --load_path ./model/GDA-me.pt \
    --notes GDA-me_cr_re-me \
    --dataset GDA \
    --no_dev 0
sleep 1
python src/train_me.py \
    --model_name_or_path /data/pretrained/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --load_path ./model/BioRED-me.pt \
    --notes BioRED-me_cr_re-me \
    --dataset BioRED \
    --no_dev 1
sleep 1

# end-to-end
python src/prepro.py \
    --dataset CDR \
    --no_dev 1
python src/prepro.py \
    --dataset GDA \
    --no_dev 0
python src/prepro.py \
    --dataset BioRED \
    --no_dev 1

python src/train_cre.py \
    --model_name_or_path /data/pretrained/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --load_path ./model/CDR-cre.pt \
    --notes CDR-me_cr_re-cre \
    --dataset CDR \
    --no_dev 1
sleep 1
python src/train_cre.py \
    --model_name_or_path /data/pretrained/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --load_path ./model/GDA-cre.pt \
    --notes GDA-me_cr_re-cre \
    --dataset GDA \
    --no_dev 0
sleep 1
python src/train_cre.py \
    --model_name_or_path /data/pretrained/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --train_batch_size 2 \
    --test_batch_size 4 \
    --load_path ./model/BioRED-cre.pt \
    --notes BioRED-me_cr_re-cre \
    --dataset BioRED \
    --no_dev 1
sleep 1


# perfect
python src/prepro.py \
    --dataset CDR \
    --no_dev 1 \
    --perfect 1
python src/prepro.py \
    --dataset GDA \
    --no_dev 0 \
    --perfect 1
python src/prepro.py \
    --dataset BioRED \
    --no_dev 1 \
    --perfect 1
python src/train_cre.py \
    --model_name_or_path /data/pretrained/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --load_path ./model/CDR-cre-p.pt \
    --num_epoch 50 \
    --train_batch_size 8 \
    --test_batch_size 16 \
    --save_path ./model/CDR-cre-perfect.pt \
    --notes CDR-cre-perfect \
    --dataset CDR \
    --no_dev 1
sleep 1
python src/train_cre.py \
    --model_name_or_path /data/pretrained/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --train_batch_size 8 \
    --test_batch_size 16 \
    --load_path ./model/GDA-cre-p.pt \
    --notes GDA-cre-perfect \
    --dataset GDA \
    --evaluation_steps 1000 \
    --no_dev 0
sleep 1
python src/train_cre.py \
    --model_name_or_path /data/pretrained/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --train_batch_size 2 \
    --test_batch_size 4 \
    --load_path ./model/BioRED-cre-p.pt \
    --notes BioRED-cre-perfect \
    --dataset BioRED \
    --no_dev 1
sleep 1
