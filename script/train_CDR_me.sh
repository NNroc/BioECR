python src/train_me.py \
    --train_file train.json \
    --dev_file dev.json \
    --test_file test.json \
    --model_name_or_path /data/pretrained/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --num_epoch 30 \
    --learning_rate 5e-5 \
    --notes CDR-PubMedBERT-base-me  \
    --dataset CDR \
    --no_dev 1
