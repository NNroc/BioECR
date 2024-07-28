python src/train_cre.py \
  --model_name_or_path /data/pretrained/BiomedNLP-PubMedBERT-base-uncased-abstract \
  --num_epoch 30 \
  --learning_rate 3e-5 \
  --warmup_ratio 0.04 \
  --save_path ./model/CDR-cre-no_dev.pt \
  --notes CDR-PubMedBERT-base-cre \
  --dataset CDR \
  --no_dev 1 \
  --alpha 0.5
sleep 10