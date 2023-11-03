import argparse
import os
import sys
from pathlib import Path

import pytest
import transformers

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.loader import (
    Attribute,
    Entity,
    create_entities,
    create_label_dict,
    load_raw_data,
    preprocess,
    split_windows,
)


@pytest.mark.parametrize(
    "test_fname, test_passage_idx, expected_label_dict",
    [
        (
            "199-03",
            1,
            {
                "478|479": [{"Event": [1, 0, 0], "Action": [0, 0, 0, 0, 0, 0, 0], "Negation": [0, 0], "Temporality": [0, 0, 0, 0], "Certainty": [0, 0, 0, 0], "Actor": [0, 0, 0]}],
                "986|986": [{"Event": [0, 1, 0], "Action": [0, 0, 0, 0, 1, 0, 0], "Negation": [0, 1], "Temporality": [1, 0, 0, 0], "Certainty": [1, 0, 0, 0], "Actor": [1, 0, 0]}],
                "1030|1030": [{"Event": [0, 1, 0], "Action": [0, 0, 0, 0, 1, 0, 0], "Negation": [0, 1], "Temporality": [1, 0, 0, 0], "Certainty": [1, 0, 0, 0], "Actor": [1, 0, 0]}],
                "1040|1041": [{"Event": [0, 1, 0], "Action": [0, 0, 0, 0, 1, 0, 0], "Negation": [0, 1], "Temporality": [1, 0, 0, 0], "Certainty": [1, 0, 0, 0], "Actor": [1, 0, 0]}],
                "1269|1269": [{"Event": [1, 0, 0], "Action": [0, 0, 0, 0, 0, 0, 0], "Negation": [0, 0], "Temporality": [0, 0, 0, 0], "Certainty": [0, 0, 0, 0], "Actor": [0, 0, 0]}],
                "1374|1375": [{"Event": [0, 1, 0], "Action": [0, 0, 0, 0, 1, 0, 0], "Negation": [0, 1], "Temporality": [1, 0, 0, 0], "Certainty": [1, 0, 0, 0], "Actor": [1, 0, 0]}],
                "1391|1391": [{"Event": [1, 0, 0], "Action": [0, 0, 0, 0, 0, 0, 0], "Negation": [0, 0], "Temporality": [0, 0, 0, 0], "Certainty": [0, 0, 0, 0], "Actor": [0, 0, 0]}],
                "1488|1488": [{"Event": [1, 0, 0], "Action": [0, 0, 0, 0, 0, 0, 0], "Negation": [0, 0], "Temporality": [0, 0, 0, 0], "Certainty": [0, 0, 0, 0], "Actor": [0, 0, 0]}],
            },
        ),
        (  # multi label
            "194-01",
            8,
            {
                "22|23": [
                    {"Event": [0, 1, 0], "Action": [1, 0, 0, 0, 0, 0, 0], "Negation": [0, 1], "Temporality": [1, 0, 0, 0], "Certainty": [1, 0, 0, 0], "Actor": [1, 0, 0]},
                    {"Event": [0, 1, 0], "Action": [0, 1, 0, 0, 0, 0, 0], "Negation": [0, 1], "Temporality": [1, 0, 0, 0], "Certainty": [1, 0, 0, 0], "Actor": [1, 0, 0]},
                ],
            },
        ),
    ],
)
def test_create_label_dict(test_fname, test_passage_idx, expected_label_dict):
    parser = argparse.ArgumentParser()
    args = parser.parse_args(args=[])
    args.cached_corpus_path = Path("../data/preprocessed_data/corpus.json")
    args.cached_dataset_path = Path("../data/preprocessed_data/dataset.json")
    args.force_preprocess = False
    corpus, _ = preprocess(args=args)
    for fname, annotated_passages in corpus.train_fname2annotated_passages.items():
        if fname != test_fname:
            continue
        for passage_idx, annotated_passage in enumerate(annotated_passages):
            if passage_idx != test_passage_idx:
                continue
            label_dict = create_label_dict(entities=annotated_passage.entities)
            print(label_dict)
            assert label_dict == expected_label_dict


@pytest.mark.parametrize(
    "test_token_ids, test_entities, expected_window_token_ids_list, expected_window_entities_list",
    [
        (
            [100, 101, 102, 103, 104, 105, 106],
            [
                Entity(ann_id="E1", text="test1", s_tok_idx=1, e_tok_idx=1, attr=Attribute(event="NoDisposition", action=None, negation=None, temporality=None, certainty=None, actor=None)),
                Entity(ann_id="E2", text="test2", s_tok_idx=3, e_tok_idx=3, attr=Attribute(event="NoDisposition", action=None, negation=None, temporality=None, certainty=None, actor=None)),
                Entity(ann_id="E3", text="test3", s_tok_idx=5, e_tok_idx=5, attr=Attribute(event="NoDisposition", action=None, negation=None, temporality=None, certainty=None, actor=None)),
            ],
            [[0, 100, 101, 102, 103, 2], [0, 102, 103, 104, 105, 2], [0, 104, 105, 106, 2]],
            [
                [
                    Entity(ann_id="E1", text="test1", s_tok_idx=2, e_tok_idx=2, attr=Attribute(event="NoDisposition", action=None, negation=None, temporality=None, certainty=None, actor=None)),
                    Entity(ann_id="E2", text="test2", s_tok_idx=4, e_tok_idx=4, attr=Attribute(event="NoDisposition", action=None, negation=None, temporality=None, certainty=None, actor=None)),
                ],
                [
                    Entity(ann_id="E3", text="test3", s_tok_idx=4, e_tok_idx=4, attr=Attribute(event="NoDisposition", action=None, negation=None, temporality=None, certainty=None, actor=None)),
                ],
                [],
            ],
        )
    ],
)
def test_split_windows(test_token_ids, test_entities, expected_window_token_ids_list, expected_window_entities_list):
    bert_tokenizer = transformers.AutoTokenizer.from_pretrained(Path("../data/RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf"))
    window_token_ids_list, window_entities_list = split_windows(bert_tokenizer=bert_tokenizer, token_ids=test_token_ids, entities=test_entities, max_sequence_length=6, window_size=2)
    for idx, (window_token_ids, window_entities) in enumerate(zip(window_token_ids_list, window_entities_list)):
        print(window_token_ids)
        print(window_entities)
        assert window_token_ids == expected_window_token_ids_list[idx]
        assert window_entities == expected_window_entities_list[idx]


def test_create_instances():  # TODO
    assert True


@pytest.mark.parametrize(
    "test_fname, test_passage_idx, expected_target_entities",
    [
        (  # 101-01
            "101-01",
            1,
            [],
        ),
        (  # 101-01
            "101-01",
            3,
            [
                Entity(ann_id="E2", text="Lipitor", s_tok_idx=135, e_tok_idx=136, attr=Attribute(event="NoDisposition", action=None, negation=None, temporality=None, certainty=None, actor=None)),
                Entity(ann_id="E3", text="Synthroid", s_tok_idx=522, e_tok_idx=524, attr=Attribute(event="NoDisposition", action=None, negation=None, temporality=None, certainty=None, actor=None)),
            ],
        ),
        (  # 109-04
            "109-04",
            1,
            [
                Entity(
                    ann_id="E11",
                    text="UF heparin",
                    s_tok_idx=175,
                    e_tok_idx=176,
                    attr=Attribute(event="Disposition", action="Start", negation="NotNegated", temporality="Past", certainty="Certain", actor="Physician"),
                ),
                Entity(ann_id="E2", text="coumadin", s_tok_idx=208, e_tok_idx=210, attr=Attribute(event="NoDisposition", action=None, negation=None, temporality=None, certainty=None, actor=None)),
                Entity(ann_id="E3", text="Advair", s_tok_idx=224, e_tok_idx=225, attr=Attribute(event="NoDisposition", action=None, negation=None, temporality=None, certainty=None, actor=None)),
                Entity(ann_id="E4", text="Lipitor", s_tok_idx=227, e_tok_idx=228, attr=Attribute(event="NoDisposition", action=None, negation=None, temporality=None, certainty=None, actor=None)),
                Entity(ann_id="E5", text="HCTZ", s_tok_idx=230, e_tok_idx=231, attr=Attribute(event="NoDisposition", action=None, negation=None, temporality=None, certainty=None, actor=None)),
                Entity(ann_id="E6", text="heparin", s_tok_idx=233, e_tok_idx=233, attr=Attribute(event="NoDisposition", action=None, negation=None, temporality=None, certainty=None, actor=None)),
                Entity(
                    ann_id="E12",
                    text="lytics",
                    s_tok_idx=557,
                    e_tok_idx=558,
                    attr=Attribute(event="Disposition", action="Start", negation="Negated", temporality="Present", certainty="Certain", actor="Physician"),
                ),
                Entity(
                    ann_id="E10",
                    text="low molecular weight heparin",
                    s_tok_idx=580,
                    e_tok_idx=583,
                    attr=Attribute(event="Disposition", action="Start", negation="NotNegated", temporality="Present", certainty="Certain", actor="Physician"),
                ),
                Entity(
                    ann_id="E8",
                    text="spiriva",
                    s_tok_idx=595,
                    e_tok_idx=597,
                    attr=Attribute(event="Disposition", action="Start", negation="NotNegated", temporality="Present", certainty="Certain", actor="Physician"),
                ),
                Entity(
                    ann_id="E9",
                    text="albuterol",
                    s_tok_idx=599,
                    e_tok_idx=600,
                    attr=Attribute(event="Disposition", action="Start", negation="NotNegated", temporality="Present", certainty="Certain", actor="Physician"),
                ),
            ],
        ),
    ],
)
def test_create_entities(test_fname, test_passage_idx, expected_target_entities):
    fname2passage_texts, fname2annotation = load_raw_data(Path("../data/original_data/train"))
    # Set up test data
    test_passage_text = fname2passage_texts[test_fname][test_passage_idx]
    test_annotation = fname2annotation[test_fname]
    test_passage_offset = sum([len(passage) for passage in fname2passage_texts[test_fname][:test_passage_idx]])
    test_bert_tokenizer = transformers.AutoTokenizer.from_pretrained(Path("../data/RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf"))
    # Run test
    target_entities, _ = create_entities(test_bert_tokenizer, test_passage_text, test_passage_offset, test_annotation)
    print(target_entities)
    assert target_entities == expected_target_entities
