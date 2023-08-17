import pathlib
import lzma

import pydantic

import brat_loader as BL
import ner as NER

def brat_annotation_to_instances(doc:BL.Document, tokenizer:NER.TokenizerInterface, max_length:int, stride:int, doc_name:str="(not-given)") -> list[NER.NERInstance]:
    # list up all spans
    # - discontinuous spans will be split into distinct spans
    ner_spans:list[NER.NERSpan] = list()
    for ner_annotation in doc.text_bounds:
        for s, span in enumerate(ner_annotation.spans):
            ner_spans.append(NER.NERSpan(start=span.start, end=span.end, id=f'{ner_annotation.id}_{s}'))

    # remove overlapping spans
    has_span = [False for _ in range(len(doc.text))]
    pos_to_span_id = [None for _ in range(len(doc.text))]
    unique_ner_spans = list()
    for span in ner_spans:
        for pos in range(span.start, span.end):
            if has_span[pos]:
                print(f'span {span.id} in document {doc_name} is ignored because there is another overlapping span {pos_to_span_id[pos]}')
                break
        else:
            for pos in range(span.start, span.end):
                has_span[pos] = True
                pos_to_span_id[pos] = span.id
            unique_ner_spans.append(span)
    ner_spans = unique_ner_spans

    # tokenize and split into subsequences
    doc_instances = NER.NERInstance.build(
        text=doc.text, spans=ner_spans, id=doc_name,
        tokenizer=tokenizer, add_special_tokens=True,
        truncation=NER.NERTruncationScheme.SPLIT, max_length=max_length, stride=stride, fit_to_token=True, add_split_idx_to_id=True,
    )
    return doc_instances


class Instances(pydantic.BaseModel):
    instances: list[NER.NERInstance]

def load_dir(dir_path:pathlib.Path | str, cache_file_path:pathlib.Path | str, tokenizer:NER.TokenizerInterface, max_length:int, stride:int) -> list[NER.NERInstance]:
    if type(dir_path) is str:
        dir_path = pathlib.Path(dir_path)
    if type(cache_file_path) is str:
        cache_file_path = pathlib.Path(cache_file_path)

    if cache_file_path.exists():
        with lzma.open(cache_file_path, "rt") as f:
            instances = Instances.model_validate_json(f.read()).instances
        return instances

    doc_name_to_documents = BL.load_dir(dir_path)
    instances: list[NER.NERInstance] = list()
    for doc_name in sorted(doc_name_to_documents.keys()):
        doc = doc_name_to_documents[doc_name]
        doc_instances = brat_annotation_to_instances(doc=doc, tokenizer=tokenizer, max_length=max_length, stride=stride, doc_name=doc_name)
        instances.extend(doc_instances)

    # save
    cache_file_path.parent.mkdir(exist_ok=True)
    with lzma.open(cache_file_path, "wt") as f:
        print(Instances(instances=instances).model_dump_json(), file=f)

    return instances
