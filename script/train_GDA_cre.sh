python src/train_cre.py \
    --model_name_or_path /data/pretrained/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --num_epoch 10 \
    --learning_rate 3e-5 \
    --warmup_ratio 0.04 \
    --save_path ./model/GDA-PubMedBERT-base-cre.pt \
    --notes GDA-PubMedBERT-base \
    --dataset GDA \
    --evaluation_steps 400 \
    --dropout 0.2 \
    --no_dev 0
