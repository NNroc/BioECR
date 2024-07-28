for d in "train" "dev" "test";
do
    python3 ./data_preprocess/process.py \
                      -i ./data/CDR/${d}.pubtator \
                      -o ./data/CDR/pre/${d}.json \
                      --data CDR
done