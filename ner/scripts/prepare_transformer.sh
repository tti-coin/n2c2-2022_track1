#!/bin/bash
REPOSITORY_ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE:-$0}" )" && cd .. && cd .. && pwd )"
DATA_DIR=${REPOSITORY_ROOT_DIR}/data

BIO_LM_CHECKPOINT_URL="https://dl.fbaipublicfiles.com/biolm/RoBERTa-large-PM-M3-Voc-hf.tar.gz"

echo "repository root dir: ${REPOSITORY_ROOT_DIR}"
echo "data dir           : ${DATA_DIR}"
if [ ! -d "${DATA_DIR}" ]; then
    echo "data dir does not exist. check the directory structure."
    exit -1
fi

echo "download bio-lm from ${BIO_LM_CHECKPOINT_URL} to temp dir"
tmp_dir=`mktemp -d`
(cd ${tmp_dir} && curl -L -O ${BIO_LM_CHECKPOINT_URL})

echo "extract"
(cd ${tmp_dir} && tar zxf RoBERTa-large-PM-M3-Voc-hf.tar.gz)

echo "copy to data dir"
mkdir -p ${DATA_DIR}/transformers
rsync -av ${tmp_dir}/RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf ${DATA_DIR}/transformers

echo "remove tempdir"
rm -r ${tmp_dir}

echo "add [\\r, \\n, \\t] as special tokens to bio-lm's vocabulary."
echo "NOTE: expects transformers==4.17.0"
python3 -c """
import transformers, json
BASE_CHECKPOINT_DIR = '${DATA_DIR}/transformers/RoBERTa-large-PM-M3-Voc-hf'
NL_SAVE_DIR = '${DATA_DIR}/transformers/RoBERTa-large-PM-M3-Voc-NewLines'

tokenizer = transformers.AutoTokenizer.from_pretrained(BASE_CHECKPOINT_DIR)
tokenizer.model_max_length = 512
tokenizer.add_special_tokens({'additional_special_tokens':['\n','\r','\t']})
tokenizer.save_pretrained(NL_SAVE_DIR)

with open(NL_SAVE_DIR+'/tokenizer_config.json') as f:
    tokenizer_config = json.load(f)
if 'model_max_length' not in tokenizer_config:
    tokenizer_config['model_max_length'] = 512
    with open(NL_SAVE_DIR+'/tokenizer_config.json', 'w') as f:
        json.dump(tokenizer_config, f)

config = transformers.AutoConfig.from_pretrained(BASE_CHECKPOINT_DIR)
model = transformers.AutoModel.from_pretrained(BASE_CHECKPOINT_DIR, config=config)
model.resize_token_embeddings(len(tokenizer))
model.save_pretrained(NL_SAVE_DIR)
"""

echo "done"
