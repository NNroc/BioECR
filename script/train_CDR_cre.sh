python src/train_cre.py \
  --model_name_or_path /data/pretrained/BiomedNLP-PubMedBERT-base-uncased-abstract \
  --num_epoch 30 \
  --save_path ./model/CDR-cre-no_dev.pt \
  --notes CDR-PubMedBERT-base-cre \
  --dataset CDR \
  --no_dev 1
sleep 10