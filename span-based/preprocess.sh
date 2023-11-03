#!bin/bash
# Download Bio-LM from GitHub
cd data
wget https://dl.fbaipublicfiles.com/biolm/RoBERTa-large-PM-M3-Voc-hf.tar.gz
tar -zxvf RoBERTa-large-PM-M3-Voc-hf.tar.gz

# Add token to tokenizer
cd ../src
python3 add_tokens_to_tokenizer.py --bert_model_path ../data/RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf

# Preprocess data
cd loader
python3 loader.py
