python src/train_cre.py \
    --model_name_or_path /data/pretrained/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --num_epoch 50 \
    --learning_rate 3e-5 \
    --warmup_ratio 0.04 \
    --save_path ./model/BioRED-PubMedBERT-base-tag.pt \
    --notes BioRED-PubMedBERT-base-cre \
    --dataset BioRED \
    --no_dev 1