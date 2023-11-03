# N2C2-2022 Span-based model
The implementation of Span-based model for n2c2-2022

## Requirements
```
pip3 install -r requirements.txt  # TODO
```

## Usage
### Download Bio-LM and Preprocess
```
sh preprocess.sh
```

### Train
```
cd src
python3 main.py \
    --config config/config.json \
    --train_data ../data/train.json \
```
