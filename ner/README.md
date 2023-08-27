# 1. Environment
Dockerfile for NER model: `docker/Dockerfile`.

# 2. Prepare Data
- Bio-LM (Lewis et al., 2020)
    - Run the following script.
        ```bash
        bash scripts/prepare_transformer.sh
        ```
        - This script downloads the official bio-lm checkpoint and prepares the modified vocabulary that includes the additional tokens (\\n, \\r, \\t) as the special tokens.
        - The modified PLM will be put in `REPOSITORY_ROOT_DIR/data/transformers/RoBERTa-large-PM-M3-Voc-NewLines`
        - The official github repository: https://github.com/facebookresearch/bio-lm

-  n2c2-2022 track 1 dataset
    - Put the n2c2-2022 track 1 dataset under REPOSITORY_ROOT_DIR/data/
        - The training script assumes there are REPOSITORY_ROOT_DIR/data/trainingdata_v3/train/ and REPOSITORY_ROOT_DIR/data/trainingdata_v3/dev/ directories.
        - example directory structure:
            ```
            REPOSITORY_ROOT_DIR/
            ├── data/
            │   └── trainingdata_v3/
            │       ├── train/
            │       │   ├── 101-01.ann
            │       │   ├── 101-01.txt
            │       │   └── ...
            │       └── dev/
            │           ├── 100-01.ann
            │           ├── 100-01.txt
            │           └── ...
            └── ner/
                ├── configs
                ├── scripts
                └── ...
            ```

# 3. Run Training and Validation
```bash
cd src/
# NOTE: current working directory should be REPOSITORY_ROOT_DIR/ner/src
MODEL_SAVE_DIR="../../data/ner-models/n2c2-2022-track1"
BASE_CONFIG_FILE_PATH="../configs/run-config_n2c2-2022.json"

# `--config_file ${BASE_CONFIG_FILE_PATH}`: indicate the base configuration. when a command line argument is set that is also written in the configuration, the argument's value will be used.
# `--save_dir ${MODEL_SAVE_DIR}`: checkpoints and log files will be stored here.
python3 main_ner.py --mode train --config_file ${BASE_CONFIG_FILE_PATH} --save_dir ${MODEL_SAVE_DIR}

cat ${MODEL_SAVE_DIR}/log.txt # log file
cat ${MODEL_SAVE_DIR}/best-scores.json # best epochs and corresponding f-scores.

VALIDATION_DATA_DIR="../../data/trainingdata_v3/dev" # if you have the test set, change the path here.
VALIDATION_OUTPUT_DIR="../../data/ner-output"

# `--config_file ${MODEL_SAVE_DIR}/run-config.json`: reuse the configuration that is used at training.
# `init ${MODEL_SAVE_DIR}/model-ema_best.pt`: initialize the parameters with the best checkpoint of the parameter-averaging model by Exponential Moving Average.
# `--test_corpus_dir ${VALIDATION_DATA_DIR}`: indicate the test set directory.
# `--eval_output_dir ${VALIDATION_OUTPUT_DIR}` predictions will be stored here.
python3 main_ner.py --mode eval-test --config_file ${MODEL_SAVE_DIR}/run-config.json --init ${MODEL_SAVE_DIR}/model-ema_best.pt --test_corpus_dir ${VALIDATION_DATA_DIR} --eval_output_dir ${VALIDATION_OUTPUT_DIR}

ls ${VALIDATION_OUTPUT_DIR} # see outputs.
```

# NOTE
- Trained model used in the workshop and the paper: https://github.com/tti-coin/n2c2-2022_track1/releases/tag/ner

- To use another pretrained weight, use `--bert_name /path-or-name/of/bert` option when running `main_ner.py`.
    - example: `python3 main_ner.py --bert_name bert-base-uncased --mode train --config_file ${BASE_CONFIG_FILE_PATH} --save_dir ${MODEL_SAVE_DIR}`

- Any BRAT-format files can be processed by the script. Run `main_ner.py` with options `--train_corpus_dir /path/to/train_dir` and `--dev_corpus_dir /path/to/dev_dir` to indicate the dataset directories.

