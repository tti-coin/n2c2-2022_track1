# %%
import argparse
import math
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Optional

import pydantic
import transformers


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model_path", default="../data/RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf", type=Path)
    parser.add_argument("--train_data_path", default="../data/original_data/train", type=Path)
    parser.add_argument("--eval_data_path", default="../data/original_data/dev", type=Path)
    parser.add_argument("--cached_corpus_path", default="../data/preprocessed_data/corpus.json", type=Path)
    parser.add_argument("--cached_dataset_path", default="../data/preprocessed_data/dataset.json", type=Path)
    parser.add_argument("--force_preprocess", action="store_true")
    return parser.parse_args()


class Attribute(pydantic.BaseModel):
    event: str
    action: Optional[str] = None
    negation: Optional[str] = None
    temporality: Optional[str] = None
    certainty: Optional[str] = None
    actor: Optional[str] = None


class Entity(pydantic.BaseModel):
    ann_id: str
    text: str
    s_tok_idx: int
    e_tok_idx: int
    attr: Attribute


class AnnotatedPassage(pydantic.BaseModel):
    passage_offset: int
    passage_text: str
    token_ids: list[int]
    entities: list[Entity]


class Corpus(pydantic.BaseModel):
    bert_model_path: Path
    train_fname2annotated_passages: dict[str, list[AnnotatedPassage]]
    eval_fname2annotated_passages: dict[str, list[AnnotatedPassage]]
    pad_token_id: int


class Instance(pydantic.BaseModel):
    instance_id: str
    entity_key: str
    token_ids: list[int]
    entity_range: tuple[int, int]
    label: list[dict[str, list[int]]]
    text: str


class SpanDataset(pydantic.BaseModel):
    train_instances: list[Instance]
    eval_instances: list[Instance]
    bert_model_path: Path
    pad_token_id: int

    @classmethod
    def create_span_dataset(cls, corpus: Corpus) -> "SpanDataset":
        bert_tokenizer = transformers.AutoTokenizer.from_pretrained(args.bert_model_path)
        train_instances = create_instances(fname2annotated_passages=corpus.train_fname2annotated_passages, bert_tokenizer=bert_tokenizer)
        eval_instances = create_instances(fname2annotated_passages=corpus.eval_fname2annotated_passages, bert_tokenizer=bert_tokenizer)
        pad_token_id = corpus.pad_token_id
        return cls(train_instances=train_instances, eval_instances=eval_instances, bert_model_path=corpus.bert_model_path, pad_token_id=pad_token_id)


def create_label_dict(entities: list[Entity]) -> dict[str, list[dict[str, list[int]]]]:
    EVENT_MAPPING = {"NoDisposition": 0, "Disposition": 1, "Undetermined": 2}
    ACTION_MAPPING = {"Start": 0, "Stop": 1, "Increase": 2, "Decrease": 3, "UniqueDose": 4, "OtherChange": 5, "Unknown": 6}
    NEGATION_MAPPING = {"Negated": 0, "NotNegated": 1}
    TEMPORALITY_MAPPING = {"Past": 0, "Present": 1, "Future": 2, "Unknown": 3}
    CERTAINTY_MAPPING = {"Certain": 0, "Hypothetical": 1, "Conditional": 2, "Unknown": 3}
    ACTOR_MAPPING = {"Physician": 0, "Patient": 1, "Unknown": 2}

    label_dict = defaultdict(list)
    for entity in entities:  # multi label
        e_key = str(entity.s_tok_idx) + "|" + str(entity.e_tok_idx)
        label = {
            "Event": [0] * 3,
            "Action": [0] * 7,
            "Negation": [0] * 2,
            "Temporality": [0] * 4,
            "Certainty": [0] * 4,
            "Actor": [0] * 3,
        }
        label["Event"][EVENT_MAPPING[entity.attr.event]] = 1
        if entity.attr.action is not None:
            label["Action"][ACTION_MAPPING[entity.attr.action]] = 1
        if entity.attr.negation is not None:
            label["Negation"][NEGATION_MAPPING[entity.attr.negation]] = 1
        if entity.attr.temporality is not None:
            label["Temporality"][TEMPORALITY_MAPPING[entity.attr.temporality]] = 1
        if entity.attr.certainty is not None:
            label["Certainty"][CERTAINTY_MAPPING[entity.attr.certainty]] = 1
        if entity.attr.actor is not None:
            label["Actor"][ACTOR_MAPPING[entity.attr.actor]] = 1

        label_dict[e_key].append(label)
    return label_dict


def split_windows(
    bert_tokenizer: transformers.PreTrainedTokenizer, token_ids: list[int], entities: list[Entity], max_sequence_length: int = 510, window_size: int = 128
) -> tuple[list[list[int]], list[list[Entity]]]:
    def get_window_ranges(sequence_length: int) -> list[tuple[int, int]]:
        if sequence_length <= (max_sequence_length - 2):
            num_dup = 1
        else:
            outsize = sequence_length - (max_sequence_length - 2)
            num_dup = 1 + math.ceil(outsize / window_size)
        ranges = list()
        for d in range(num_dup):
            start = window_size * d
            ranges.append((start, min(sequence_length, start + max_sequence_length - 2)))
        return ranges

    seq_len = len(token_ids)
    ranges = get_window_ranges(sequence_length=seq_len)
    window_token_ids_list, window_entities_list = [], []
    seen_entities = set()
    for start, end in ranges:
        valid_entities = list()
        for entity in entities:
            e_key = str(entity.s_tok_idx) + "|" + str(entity.e_tok_idx)
            if (start <= entity.s_tok_idx) and (entity.e_tok_idx <= end) and (e_key not in seen_entities):
                valid_entities.append(entity)
                seen_entities.add(e_key)

        window_token_ids = bert_tokenizer.build_inputs_with_special_tokens(token_ids[start:end])
        window_entities = list()
        for entity in valid_entities:
            new_entity = deepcopy(entity)
            new_entity.s_tok_idx = entity.s_tok_idx - start + 1
            new_entity.e_tok_idx = entity.e_tok_idx - start + 1
            window_entities.append(new_entity)
        window_token_ids_list.append(window_token_ids)
        window_entities_list.append(window_entities)
    return window_token_ids_list, window_entities_list


def create_instances(fname2annotated_passages: dict[str, list[AnnotatedPassage]], bert_tokenizer: transformers.PreTrainedTokenizer) -> list[Instance]:
    """
    Create instances from the given annotated passages.
    """

    instances = []
    s_medication_tok_idx = bert_tokenizer.encode("<MEDICATION>")[0]
    e_medication_tok_idx = bert_tokenizer.encode("</MEDICATION>")[0]
    for fname, annotated_passages in fname2annotated_passages.items():
        for passage_idx, annotated_passage in enumerate(annotated_passages):
            entities = sorted(annotated_passage.entities, key=lambda x: x.s_tok_idx)
            label_dict = create_label_dict(entities=entities)
            window_token_ids_list, window_entities_list = split_windows(bert_tokenizer=bert_tokenizer, token_ids=annotated_passage.token_ids, entities=entities)
            for window_token_ids, window_entities in zip(window_token_ids_list, window_entities_list):
                for entity in window_entities:
                    instance_id = f"{fname}|{passage_idx}|{entity.ann_id}"
                    entity_key = str(entity.s_tok_idx) + "|" + str(entity.e_tok_idx)
                    # Add special tokens (<MEDICATION>, </MEDICATION>)
                    new_window_token_ids = (
                        window_token_ids[: entity.s_tok_idx]
                        + [s_medication_tok_idx]
                        + window_token_ids[entity.s_tok_idx : entity.e_tok_idx]
                        + [e_medication_tok_idx]
                        + window_token_ids[entity.e_tok_idx :]
                    )
                    instance = Instance(
                        instance_id=instance_id,
                        entity_key=entity_key,
                        token_ids=new_window_token_ids,
                        entity_range=(entity.s_tok_idx + 1, entity.e_tok_idx + 1),
                        label=label_dict[entity_key],
                        text=entity.text,
                    )
                    instances.append(instance)
    return instances


def create_entities(bert_tokenizer: transformers.AutoTokenizer, passage_text: str, passage_offset: int, annotation: list[str]) -> tuple[list[Entity], list[int]]:
    """
    Create entities from the given passage text and annotation.
    """

    def parse_annotation(annotation: list[str]) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
        eid2val: dict[str, dict[str, str]] = {}
        tid_eid_mapping: dict[str, str] = {}
        for line in annotation:
            if line.strip().startswith("E"):
                eid = line.strip().split("\t")[0]
                assert eid not in tid_eid_mapping.values()  # eid should be unique
                event_val, tid = line.strip().split("\t")[1].split(":")
                eid2val[eid] = {"event": event_val}
                tid_eid_mapping[tid] = eid
            elif line.strip().startswith("A"):
                attr_type, eid, attr_val = line.strip().split("\t")[1].split()
                assert attr_type not in eid2val[eid]
                eid2val[eid][attr_type.lower()] = attr_val
            else:
                assert line.strip().split("\t")[0][0] == "T"
        return eid2val, tid_eid_mapping

    encoded = bert_tokenizer(passage_text, truncation=False, return_offsets_mapping=True, add_special_tokens=False)
    token_ids, offset_mapping = encoded["input_ids"], encoded["offset_mapping"]
    eid2val, tid_eid_mapping = parse_annotation(annotation)

    target_entities = []
    for line in annotation:
        if not line.strip().startswith("T"):
            continue
        tid, tag_m, tag_text = line.strip().split(sep="\t", maxsplit=2)
        offset = tag_m.replace(";", " ").split()[1:]
        tag_start, tag_end = int(offset[0]), int(offset[-1])
        # Check if this mention is in passage
        if tag_start < passage_offset or passage_offset + len(passage_text) <= tag_end:
            continue

        tag_start, tag_end = tag_start - passage_offset, tag_end - passage_offset

        try:
            assert passage_text[tag_start:tag_end].replace("\t", "").replace("  ", "") == tag_text
        except AssertionError:
            assert passage_text[tag_start:tag_end].replace("\t", "").replace("\n", "").replace(" ", "") == tag_text.replace(" ", "").replace("\n", "")

        starts, ends = zip(*offset_mapping)
        for s_tok_idx, ref_s in enumerate(starts):
            if ref_s == tag_start:
                break
        for e_tok_idx, ref_e in enumerate(ends):
            if ref_e == tag_end:
                break
        eid = tid_eid_mapping[tid]
        attr = Attribute(**eid2val[eid])
        target_entities.append(Entity(ann_id=eid, text=tag_text, s_tok_idx=s_tok_idx, e_tok_idx=e_tok_idx, attr=attr))

    target_entities = sorted(target_entities, key=lambda x: x.s_tok_idx)  # Sort by start token index
    # Check if decoded tokens are same with original tokens
    for target_entity in target_entities:
        try:
            assert bert_tokenizer.decode(token_ids[target_entity.s_tok_idx : target_entity.e_tok_idx + 1]).strip() == target_entity.text
        except AssertionError:
            pass
            # print(f"Decoding error: {bert_tokenizer.decode(token_ids[target_entity.s_tok_idx : target_entity.e_tok_idx + 1]).strip()} != {target_entity.text}")
    return target_entities, token_ids


def load_raw_data(data_path: Path) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    Load raw data from data_path
    """

    def split_document_to_passages(sentences: list[str]) -> list[str]:
        blank_num = 0
        passage_texts = []
        passage_text = ""
        for sentence in sentences:
            if sentence == "\n":
                passage_text += sentence
                blank_num += 1
                continue

            if blank_num >= 2:
                passage_texts.append(passage_text)
                passage_text = sentence
            else:
                passage_text += sentence
            blank_num = 0

        passage_texts.append(passage_text)
        return passage_texts

    fname2passage_texts, fname2annotation = defaultdict(list), defaultdict(list)
    for file_path in data_path.glob("*"):
        if not str(file_path).endswith(".txt") and not str(file_path).endswith(".ann"):
            continue
        fname = str(file_path).rsplit("/", 1)[-1].split(".")[0]
        with open(file_path, encoding="utf-8") as fn:
            if str(file_path).endswith(".txt"):
                fname2passage_texts[fname] = split_document_to_passages(fn.readlines())
            elif str(file_path).endswith(".ann"):
                fname2annotation[fname] = [line.strip() for line in fn.readlines()]
    assert len(fname2passage_texts) == len(fname2annotation)
    return fname2passage_texts, fname2annotation


def create_annotated_passages(bert_tokenizer: transformers.PreTrainedTokenizer, data_path: Path, is_train: bool) -> dict[str, list[AnnotatedPassage]]:
    """
    Create a dictionary of annotated passages from the given data path.
    """
    fname2passage_texts, fname2annotation = load_raw_data(data_path=data_path)
    fname2annotated_passages = {}
    for fname, passage_texts in fname2passage_texts.items():
        annotation = fname2annotation[fname] if is_train else []
        passage_offset = 0
        annotated_passages = []
        for passage_text in passage_texts:
            entities, token_ids = create_entities(bert_tokenizer=bert_tokenizer, passage_text=passage_text, passage_offset=passage_offset, annotation=annotation)
            annotated_passage = AnnotatedPassage(passage_offset=passage_offset, passage_text=passage_text, token_ids=token_ids, entities=entities)
            annotated_passages.append(annotated_passage)
            passage_offset += len(passage_text)
        fname2annotated_passages[fname] = annotated_passages
    return fname2annotated_passages


def create_corpus(args: argparse.Namespace) -> Corpus:
    """
    Create a corpus from the given arguments.
    """
    bert_tokenizer = transformers.AutoTokenizer.from_pretrained(args.bert_model_path)
    train_fname2annotated_passages = create_annotated_passages(bert_tokenizer=bert_tokenizer, data_path=args.train_data_path, is_train=True)
    eval_fname2annotated_passages = create_annotated_passages(bert_tokenizer=bert_tokenizer, data_path=args.eval_data_path, is_train=False)
    return Corpus(
        bert_model_path=args.bert_model_path,
        train_fname2annotated_passages=train_fname2annotated_passages,
        eval_fname2annotated_passages=eval_fname2annotated_passages,
        pad_token_id=bert_tokenizer.pad_token_id,
    )


def preprocess(args: argparse.Namespace) -> tuple[Corpus, SpanDataset]:
    """
    Create a corpus and a span dataset.
    """
    # Create corpus
    if args.cached_corpus_path is None or not args.cached_corpus_path.exists() or args.force_preprocess:
        corpus = create_corpus(args=args)
        if args.cached_corpus_path is not None:
            args.cached_corpus_path.parent.mkdir(parents=True, exist_ok=True)
            with open(args.cached_corpus_path, "w") as f:
                print(corpus.model_dump_json(), file=f)
    else:
        with open(args.cached_corpus_path) as f:
            corpus = Corpus.model_validate_json(f.read())

    # Create span dataset
    if args.cached_dataset_path is None or not args.cached_dataset_path.exists() or args.force_preprocess:
        span_dataset = SpanDataset.create_span_dataset(corpus=corpus)
        if args.cached_dataset_path is not None:
            args.cached_dataset_path.parent.mkdir(parents=True, exist_ok=True)
            with open(args.cached_dataset_path, "w") as f:
                print(span_dataset.model_dump_json(), file=f)
    else:
        with open(args.cached_dataset_path) as f:
            span_dataset = SpanDataset.model_validate_json(f.read())
    return corpus, span_dataset


if __name__ == "__main__":
    args = get_args()
    span_dataset = preprocess(args=args)
# %%
