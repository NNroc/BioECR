# BioECR
Code for [](https://1)

## Dataset
The [CDR](https://academic.oup.com/database/article/doi/10.1093/database/baw068/2630414) dataset can be downloaded following the instructions at [here](https://biocreative.bioinformatics.udel.edu/media/store/files/2016/CDR_Data.zip). 
The [GDA](https://link.springer.com/chapter/10.1007/978-3-030-17083-7_17) dataset can be downloaded following the instructions at [here](https://bitbucket.org/alexwuhkucs/gda-extraction/get/fd4a7409365e.zip). 
The [BioRED](https://academic.oup.com/bib/article/23/5/bbac282/6645993) dataset can be downloaded following the instructions at [here](https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/).
The [PubMedBERT](https://doi.org/10.1145/3458754) pre-trained language model can be downloaded following the instructions at [here](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract/tree/main).
The fine-tuned model (CDR-me.pt, CDR-cre.pt, CDR-cre-p.pt ...) can be downloaded following the instructions at [here](https://huggingface.co/NNroc/BioECR/tree/main)
Several samples have been added to the data folder (`./data/CDR/`, `./data/GDA/`, and `./data/BioRED/`)

The expected structure of files is:
```
biodoc
 |-- dataset
 |    |-- BioRED
 |    |    |-- train.pubtator
 |    |    |-- dev.pubtator
 |    |    |-- test.pubtator
 |    |-- CDR
 |    |    |-- train.pubtator
 |    |    |-- dev.pubtator
 |    |    |-- test.pubtator
 |    |-- GDA
 |    |    |-- train.pubtator
 |    |    |-- dev.pubtator
 |    |    |-- test.pubtator
 |    |    |-- train_gda_docs
 |    |    |-- training.pubtator
 |-- model
 |    |-- CDR-me.pt
 |    |-- GDA-me.pt
 |    |-- BioRED-me.pt
 |    |-- CDR-cre.pt
 |    |-- GDA-cre.pt
 |    |-- BioRED-cre.pt
 |    |-- CDR-cre-p.pt
 |    |-- GDA-cre-p.pt
 |    |-- BioRED-cre-p.pt
```
It is worth noting that BioECR needs to generate intermediate data (`{}.json`) from preprocess results and (`{}-gc.json`) from entity extraction prediction results.

## Environment
Due to the unavailability of the original address for geniass-1.00.tar.gz, we have provided geniass-1.00.tar.gz here/ data_preprocess/commom/
```
conda create -n BioECR python=3.8
conda activate BioECR
pip install -r requirements.txt
cd ./data_preprocess/common
tar xvzf geniass-1.00.tar.gz
cd geniass
make
cd ../../../
chmod +x ./data_preprocess/common/
```

## Preprocessing Data
```
bash ./script/pre_CDR.sh
bash ./script/pre_GDA.sh
bash ./script/pre_BioRED.sh
```

## Training & Evaluation
You can train BioECR model in the following steps:

1. Run entity extraction with `src/train_me.py`.
2. Preprocess span prediction with `src/prepro.py`. Specify the dev and test predictions with `--dev_file` and `--test_file` arguments respectively.
3. Run coreference resolution & relation extraction with `src/train_cre.py`.

For example, training CDR model.
```
bash script/train_CDR_me.sh
bash script/train_CDR_cre.sh
```

training CR and RE in perfect entity extraction.
```
bash script/train_cre_perfect.sh
```

End to end training (including three datasets).
```
bash script/train_me_cr_re.sh
```

Only evaluate model (including three datasets).
```
bash script/evaluate.sh
```
The evaluation results are provided in the log. To evaluate RE result on test data, you should first save the model using `--save_path` argument before training. The model correponds to the best dev results will be saved. After that, You can evaluate the saved model by setting the `--load_path` argument, and the program will generate a test file `result.json`.
