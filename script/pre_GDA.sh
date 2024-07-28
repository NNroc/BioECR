python3 ./data_preprocess/gda2pubtator.py \
              --input_folder ./data/GDA/training_data/ \
              --output_file ./data/GDA/training.pubtator

python3 ./data_preprocess/gda2pubtator.py \
              --input_folder ./data/GDA/testing_data/ \
              --output_file ./data/GDA/test.pubtator

python3 ./data_preprocess/split_gda.py

for d in "train" "dev" "test";
do
    python3 ./data_preprocess/process.py \
                      -i ./data/GDA/${d}.pubtator \
                      -o ./data/GDA/pre/${d}.json \
                      --data GDA
done
