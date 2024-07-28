for d in "train" "dev" "test";
do
    python3 ./data_preprocess/process.py \
                      -i ./data/BioRED/${d}.pubtator \
                      -o ./data/BioRED/pre/${d}.json \
                      --data BioRED
done