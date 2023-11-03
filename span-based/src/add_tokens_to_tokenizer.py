import argparse
from pathlib import Path

import transformers
from transformers import AutoConfig, AutoModel


def get_args():
    """Get arguments from command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model_path", type=str, help="Path to BERT model")
    return parser.parse_args()


def save_tokenizer_and_model(args: argparse.Namespace) -> None:
    init_path = args.bert_model_path / "extended"
    config = AutoConfig.from_pretrained(args.bert_model_path)
    bert_tokenizer = transformers.AutoTokenizer.from_pretrained(args.bert_model_path, config=config)
    bert_model = AutoModel.from_pretrained(args.bert_model_path, config=config)
    test_sentence = "A\nB\tC\tD<MEDICATION>E</MEDICATION>"
    assert bert_tokenizer.tokenize(test_sentence) != ["A", "\n", "B", "\t", "C", "\t", "D", "<MEDICATION>", "E", "</MEDICATION>"]

    # Add tokens (\n, \t, \r, <drug>)
    additional_tokens = ["\n", "\t", "\r", "<MEDICATION>", "</MEDICATION>"]
    bert_tokenizer.add_special_tokens({"additional_special_tokens": additional_tokens})
    bert_model.resize_token_embeddings(len(bert_tokenizer))
    assert bert_tokenizer.tokenize(test_sentence) == ["A", "\n", "B", "\t", "C", "\t", "D", "<MEDICATION>", "E", "</MEDICATION>"]

    # Save model
    bert_tokenizer.save_pretrained(init_path)
    bert_model.save_pretrained(init_path)


if __name__ == "__main__":
    args = get_args()
    args.bert_model_path = Path(args.bert_model_path)
    save_tokenizer_and_model(args=args)
