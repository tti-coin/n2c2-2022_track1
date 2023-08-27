# %%
import logging as _logging
_logger = _logging.getLogger(__name__)
_logger.setLevel(_logging.INFO)
_ch = _logging.StreamHandler()
_ch.setLevel(_logging.INFO)
_formatter = _logging.Formatter('%(name)s - %(levelname)s:%(message)s')
_ch.setFormatter(_formatter)
_logger.addHandler(_ch)

# %%
import collections
import json
import pathlib
import lzma

from contextlib import ExitStack as NullContext

from typing import Optional, Callable
import dataclasses

import pydantic
import numpy as np
import torch
import transformers

from utils.argparse import DataclassArgumentParser
from utils.dataloader import SelectiveDataset
from utils.averagemanager import AverageManager
from utils.chunk import to_chunk
from utils.progress_bar import closing_tqdm

import brat_loader as BRAT
import ner as NER
import modeling as M

# %%
@dataclasses.dataclass
class RunConfig:
    mode:str = dataclasses.field(default="", metadata={"choices":["train", "eval-dev", "eval-test"], "required":True})

    config_file:Optional[str] = dataclasses.field(default=None, metadata={"help":"set path/to/run-config.json to reuse the config."})
    init_model:Optional[str] = None

    save_dir:Optional[str] = None
    eval_output_dir:Optional[str] = None
    force_save:bool = dataclasses.field(default=False, metadata={"help":"when set, ignore existence of save_dir or eval_output_dir. This will overwrite existing logs in the directories."})

    train_corpus_dir:str = "../../data/trainingdata_v3/train"
    dev_corpus_dir:str = "../../data/trainingdata_v3/dev"
    test_corpus_dir:Optional[str] = None
    cache_dir:str = "../../data/ner-preprocessed"
    use_dataset_cache:bool = True

    bert_name:str = "../../data/transformers/RoBERTa-large-PM-M3-Voc-NewLines"
    max_length:int = 512
    stride:int = 128
    boundary_exclusion_size:int = 32
    do_rstrip_text:bool = False
    learning_rate:float = 2e-5
    weight_decay:float = 1e-5
    dropout_output:float = 0.2
    clip_grad:bool = True
    clipping_grad_value:float = 1.0
    use_average:bool = True
    batch_size:int = 8
    max_epoch:int = 50

    ner_label_scheme:str = dataclasses.field(default="span_only", metadata={"choices":["span_only", "single_label"]})
    labels:str = dataclasses.field(default='["Entity"]', metadata={"help":"give labels list as json string. e.g. '[\"Entity\"]' or '[\"Disposition\", \"NoDisposition\", \"Undetermined\"]'"})

    process_batch_size:int = 4
    device:str = "cuda:0"

    def get_labels(self) -> list[str]:
        return json.loads(self.labels)

    def __post_init__(self):
        if self.ner_label_scheme == "span_only":
            assert len(self.get_labels()) == 1, f'number of labels must be 1 when ner_label_scheme is "span_only", but {len(self.get_labels())=}. ({self.get_labels()=})'
        elif self.ner_label_scheme == "single_label":
            pass
        else:
            raise ValueError(self.ner_label_scheme)


# %%
def brat_annotation_to_instances(doc:BRAT.Document, tokenizer:NER.TokenizerInterface, max_length:int, stride:int, rstrip_text:bool, doc_name:str="(not-given)") -> tuple[list[NER.NERInstance], NER.NERInstance]:
    text = doc.text
    if rstrip_text:
        text = text.rstrip()

    # list up all spans
    # - discontinuous spans will be split into distinct spans
    ner_spans:list[NER.NERSpan] = list()
    for test_bound in doc.text_bounds:
        for s, span in enumerate(test_bound.spans):
            ner_spans.append(NER.NERSpan(start=span.start, end=span.end, label=test_bound.type, id=f'{test_bound.id}_{s}'))

    # remove overlapping spans
    has_span = [False for _ in range(len(text))]
    pos_to_span_id = [None for _ in range(len(text))]
    unique_ner_spans = list()
    for span in ner_spans:
        for pos in range(span.start, span.end):
            if has_span[pos]:
                _logger.warning(f'span {span.id} in document {doc_name} is ignored because there is another overlapping span {pos_to_span_id[pos]}')
                break
        else:
            for pos in range(span.start, span.end):
                has_span[pos] = True
                pos_to_span_id[pos] = span.id
            unique_ner_spans.append(span)
    ner_spans = unique_ner_spans

    # tokenize and split into subsequences
    doc_instances, full_doc_instance = NER.NERInstance.build(
        text=text, spans=ner_spans, id=doc_name,
        tokenizer=tokenizer, add_special_tokens=True,
        truncation=NER.NERTruncationScheme.SPLIT, max_length=max_length, stride=stride, fit_token_span=NER.NERSpanFittingScheme.MAXIMIZE, add_split_idx_to_id=False, return_non_truncated=True,
    )
    return doc_instances, full_doc_instance

def load_dir(dir_path:pathlib.Path | str, tokenizer:NER.TokenizerInterface, max_length:int, stride:int, rstrip_text:bool, use_cache:bool, cache_dir_path:pathlib.Path | str) -> list[NER.NERInstance]:
    class Cache(pydantic.BaseModel):
        instances: list[NER.NERInstance]
        doc_name_to_full_doc_instance: dict[str, NER.NERInstance]

    if type(dir_path) is str:
        dir_path = pathlib.Path(dir_path)
    if type(cache_dir_path) is str:
        cache_dir_path = pathlib.Path(cache_dir_path)

    cache_file_path = cache_dir_path / f'cache_{str(dir_path).replace("/","+")}_{tokenizer.name_or_path.replace("/","+")}_L{max_length}_S{stride}_R{int(rstrip_text)}.json.xz'

    if use_cache and cache_file_path.exists():
        with lzma.open(cache_file_path, "rt") as f:
            cache = Cache.model_validate_json(f.read())
        return cache.instances, cache.doc_name_to_full_doc_instance

    doc_name_to_documents = BRAT.load_dir(dir_path)
    instances: list[NER.NERInstance] = list()
    doc_name_to_full_doc_instance: dict[str, NER.NERInstance] = dict()
    for doc_name in sorted(doc_name_to_documents.keys()):
        doc = doc_name_to_documents[doc_name]
        doc_instances, full_doc_instance = brat_annotation_to_instances(doc=doc, tokenizer=tokenizer, max_length=max_length, stride=stride, doc_name=doc_name, rstrip_text=rstrip_text)
        instances.extend(doc_instances)
        doc_name_to_full_doc_instance[doc_name] = full_doc_instance

    # save
    if use_cache:
        cache_file_path.parent.mkdir(exist_ok=True)
        with lzma.open(cache_file_path, "wt") as f:
            print(Cache(instances=instances, doc_name_to_full_doc_instance=doc_name_to_full_doc_instance).model_dump_json(), file=f)

    return instances, doc_name_to_full_doc_instance

def build_dataset(instances:list[NER.NERInstance], label_scheme:str|NER.NERLabelScheme, label_to_id:Callable[[str,],int]|dict[str,int], device:str) -> SelectiveDataset:
    fields = [
        {"name":"id", "mapping":lambda x:x.id},
        {"name":"input_ids", "mapping":lambda x:x.token_ids, "dtype":torch.long, "device":device, "padding":True, "padding_value":0, "padding_mask":True},
        {"name":"label", "mapping":lambda x:x.get_sequence_label(tagging_scheme=NER.NERTaggingScheme.BILOU, label_scheme=label_scheme, label_to_id=label_to_id), "dtype":torch.long, "device":device, "padding":True, "padding_value":0},
        {"name":"token_len", "mapping":lambda x:len(x.token_ids)},
        {"name":"instance", "mapping":lambda x:x},
    ]
    return SelectiveDataset(instances, fields)

# %%
_ce_criterion = torch.nn.CrossEntropyLoss()
def criterion(logits, labels, attention_mask):
    # logits: [B,seq_len,5]
    # label: [B,seq_len,5]
    inds = torch.where(attention_mask)
    num_logits = len(inds)
    loss = _ce_criterion(logits[inds], labels[inds])
    return loss, num_logits

class Meter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.doc_id_to_pred_spans = collections.defaultdict(set)
        self.doc_id_to_gold_spans = collections.defaultdict(set)
    def update(self, gold_spans:list[NER.NERSpan], pred_spans:list[NER.NERSpan], doc_id):
        self.doc_id_to_gold_spans[doc_id] |= {(span.start, span.end, span.label) for span in gold_spans}
        self.doc_id_to_pred_spans[doc_id] |= {(span.start, span.end, span.label) for span in pred_spans}
    def calc(self, eps:float=1e-8):
        num_correct = 0
        num_preds = 0
        num_golds = 0
        for doc_id in self.doc_id_to_gold_spans.keys():
            golds = self.doc_id_to_gold_spans[doc_id]
            preds = self.doc_id_to_pred_spans[doc_id]
            num_preds += len(preds)
            num_golds += len(golds)
            num_correct += len(golds & preds)
        precision = num_correct / max(1, num_preds)
        recall = num_correct / max(1, num_golds)
        f_score = 2*precision*recall / max(eps, precision+recall)
        return {
            "precision":precision, "recall":recall, "f-score":f_score,
            "PRF":f'{precision:.4f}/{recall:.4f}/{f_score:.4f}',
        }
    def total_gold_span_num(self) -> int:
        return sum(len(spans) for spans in self.doc_id_to_gold_spans.values())

    def output_predictions_as_ann_to_dir(self, dir_path:str|pathlib.Path, id_to_label:dict[int,str], doc_id_to_text_func:Optional[Callable[[str], str]]=None):
        if type(dir_path) is str:
            dir_path = pathlib.Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        for doc_id, spans in self.doc_id_to_pred_spans.items():
            with open(dir_path / f'{doc_id}.ann', "w") as f:
                for s, span in enumerate(sorted(spans), start=1):
                    id_ = f'T{s}'
                    entity_label = id_to_label[span[2]]
                    if doc_id_to_text_func is not None:
                        text_bound = BRAT.TextBound.from_span(BRAT.Span(start=span[0], end=span[1]), document_text=doc_id_to_text_func(doc_id), id=id_, type=entity_label)
                    else:
                        entity_len = span[1] - span[0]
                        dummy_reference_text = "X" * entity_len
                        text_bound = BRAT.TextBound(id=id_, type=entity_label, span=BRAT.Span(start=span[0], end=span[1]), reference_text=dummy_reference_text)
                    print(text_bound.to_annotation_text(), file=f)


# %%
def run_epoch(
    model:M.Model, dataset:SelectiveDataset, doc_name_to_full_instance:dict[str,NER.NERInstance], label_scheme:str|NER.NERLabelScheme, label_to_id:Callable[[str,],int]|dict[str,int],
    boundary_exclusion_size:int, process_batch_size:int, is_training:bool,
    opts:Optional[list]=None, train_batch_size:Optional[int]=None, clip_grad:Optional[bool]=None, clipping_grad_value:Optional[float]=None,
):
    if is_training:
        assert opts is not None
        assert train_batch_size is not None
        assert clip_grad is not None
        assert clipping_grad_value is not None

        assert train_batch_size % process_batch_size == 0, f'train_batch_size must be a multiple of process_batch_size, but {train_batch_size=} and {process_batch_size=}'
        num_accumulate_steps = train_batch_size // process_batch_size

        model.train()

    else:
        num_accumulate_steps = 1

        model.eval()

    total_loss = 0.0
    meter_by_union = Meter()
    doc_name_to_split_instance_and_logits_pairs = collections.defaultdict(list)
    run_epoch_desc_prefix = "run_epoch"
    with (
        closing_tqdm(dataset.dataloader(batch_size=process_batch_size, shuffle=is_training), desc=run_epoch_desc_prefix) as dataloader_pbar,
        NullContext() if is_training else torch.no_grad()
    ):
        for minibatches in to_chunk(dataloader_pbar, num_accumulate_steps):
            for minibatch in minibatches:
                logits = model(input_ids=minibatch["input_ids"],attention_mask=minibatch["input_ids_mask"])
                loss, num_logits = criterion(logits=logits, labels=minibatch["label"], attention_mask=minibatch["input_ids_mask"])
                total_loss += loss.item() * num_logits

                for instance, step_logits, step_token_len in zip(minibatch["instance"], logits.detach().cpu().numpy(), minibatch["token_len"]):
                    step_logits = step_logits[:step_token_len]
                    step_preds:list[NER.NERSpan] = NER.viterbi_decode(step_logits, tagging_scheme=NER.NERTaggingScheme.BILOU, label_scheme=label_scheme, scalar_logit_for_token_level=True, as_spans=True)
                    step_preds = instance.decode_token_span_to_char_span(step_preds, strip=True, recover_split=True)
                    step_preds = [span for span in step_preds if span.start < span.end]
                    meter_by_union.update(gold_spans=[span.map_label(label_to_id) for span in instance.recover_char_spans_wrt_split_offset(instance.spans)], pred_spans=step_preds, doc_id=instance.id)
                    doc_name_to_split_instance_and_logits_pairs[instance.id].append((instance, step_logits))

                if is_training:
                    (loss/len(minibatches)).backward()

            if is_training:
                if clip_grad:
                    step_grad = torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_grad_value)
                for opt in opts:
                    opt.step()
                model.zero_grad()

            desc = f'{run_epoch_desc_prefix} loss={loss.item():.4e}  P/R/F(union)={meter_by_union.calc()["PRF"]}'
            if is_training and clip_grad:
                desc += f' norm(grad)={step_grad:.4f}'
            dataloader_pbar.set_description(desc)

    meter_by_ensemble = Meter()
    ensemble_desc_prefix = "ensemble-over-splits"
    with closing_tqdm(doc_name_to_split_instance_and_logits_pairs.items(), total=len(doc_name_to_split_instance_and_logits_pairs), desc=ensemble_desc_prefix) as pbar:
        for doc_name, split_and_logits_pairs in pbar:
            non_split_instance = doc_name_to_full_instance[doc_name]
            splits, logits = zip(*split_and_logits_pairs)
            ensembled_logits = np.stack(NER.ensemble_split_sequences(splits, logits, aggregate="mean", boundary_exclusion_size=boundary_exclusion_size), axis=0) # [seq_len, 5(=BILOU)]
            ensembled_preds:list[NER.NERSpan] = NER.viterbi_decode(ensembled_logits, tagging_scheme=NER.NERTaggingScheme.BILOU, label_scheme=label_scheme, scalar_logit_for_token_level=True, as_spans=True)
            ensembled_preds:list[NER.NERSpan] = non_split_instance.decode_token_span_to_char_span(ensembled_preds, strip=True, is_token_span_starting_after_special_tokens=True)
            ensembled_preds = [span for span in ensembled_preds if span.start < span.end]
            meter_by_ensemble.update(gold_spans=[span.map_label(label_to_id) for span in non_split_instance.spans], pred_spans=ensembled_preds, doc_id=doc_name)
            pbar.set_description(f'{ensemble_desc_prefix} P/R/F/(ensemble)={meter_by_ensemble.calc()["PRF"]}')
    assert meter_by_ensemble.total_gold_span_num() == meter_by_union.total_gold_span_num()

    return total_loss, meter_by_union, meter_by_ensemble



# %%
run_config = DataclassArgumentParser(RunConfig).parse_args()
if run_config.config_file is not None:
    with open(run_config.config_file) as f:
        default_run_config = RunConfig(**json.load(f))
    run_config = DataclassArgumentParser(RunConfig, default_value=default_run_config).parse_args()
if run_config.mode == "eval-test":
    assert run_config.test_corpus_dir is not None, 'need to set path to test_corpus_dir by "--test_corpus_dir /path/to/test_corpus_dir/"'
# _logger.info(f'{run_config=}')

id_to_label = {i:label for i,label in enumerate(run_config.get_labels())}
num_classes = len(id_to_label)
if run_config.ner_label_scheme == "span_only":
    label_to_id_func = lambda _: 0
elif run_config.ner_label_scheme == "single_label":
    _label_to_id = {v:k for k,v in id_to_label.items()}
    # label_to_id_func = lambda x:_label_to_id.get(x, 0)
    label_to_id_func = _label_to_id.__getitem__
else:
    raise ValueError(run_config.ner_label_scheme)

model_config = M.ModelConfig(bert_name=run_config.bert_name, dropout_output=run_config.dropout_output, output_dim=num_classes*4+1, id_to_label=id_to_label)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_config.bert_name, trim_offsets=False)
model = M.Model(model_config=model_config)
if run_config.init_model is not None:
    model.load_state_dict(torch.load(run_config.init_model, map_location="cpu"))
model.to(run_config.device)

# %%
if run_config.mode == "train":
    train_instances, train_doc_name_to_full_instance = load_dir(run_config.train_corpus_dir, tokenizer=tokenizer, max_length=run_config.max_length, stride=run_config.stride, rstrip_text=run_config.do_rstrip_text, use_cache=run_config.use_dataset_cache, cache_dir_path=pathlib.Path(run_config.cache_dir))
    dev_instances, dev_doc_name_to_full_instance = load_dir(run_config.dev_corpus_dir, tokenizer=tokenizer, max_length=run_config.max_length, stride=run_config.stride, rstrip_text=run_config.do_rstrip_text, use_cache=run_config.use_dataset_cache, cache_dir_path=pathlib.Path(run_config.cache_dir))
    train_dataset = build_dataset(train_instances, label_scheme=run_config.ner_label_scheme, label_to_id=label_to_id_func, device=run_config.device)
    dev_dataset = build_dataset(dev_instances, label_scheme=run_config.ner_label_scheme, label_to_id=label_to_id_func, device=run_config.device)

    opts = list()
    opt = torch.optim.AdamW(model.parameters(), run_config.learning_rate, weight_decay=run_config.weight_decay)
    opts.append(opt)
    if run_config.use_average:
        model_average = AverageManager(model.parameters())
        model_average.to(run_config.device)
        opts.append(model_average)

    # prepare save_dir_path
    save_dir_path = None
    if run_config.save_dir is not None:
        save_dir_path = pathlib.Path(run_config.save_dir)
        if save_dir_path.exists():
            if run_config.force_save:
                _logger.warning(f'{run_config.save_dir=} already exists, but force_save is set to be True. NOTE: This will result overwriting existing logs and checkpoints in the save_dir.')
            else:
                raise ValueError(f'{run_config.save_dir=} already exists. Set other directory, or run with `python3 main.py --force_save` to ignore this error. (NOTE: --force_save will result overwriting existing logs and checkpoints in the save_dir.)')
        else:
            save_dir_path.mkdir(parents=True)

        def tee_print(*args, **kwargs):
            assert "file" not in kwargs
            with open(save_dir_path/"log.txt", "a") as f:
                print(*args, **kwargs, file=f)
            print(*args, **kwargs)

        with open(save_dir_path/"log.txt", "w"):
            pass # make and clear file
        with open(save_dir_path/"run-config.json", "w") as f:
            json.dump(dataclasses.asdict(run_config), f)
        with open(save_dir_path/"model-config.json", "w") as f:
            print(model_config.model_dump_json(), file=f)
    else:
        tee_print = print

    best_scores = dict()
    for epoch in range(1,run_config.max_epoch+1):
        _logger.info(f'{epoch=}')
        # train
        train_loss, train_meter_by_union, train_meter_by_ensemble = run_epoch(
            model=model, dataset=train_dataset, doc_name_to_full_instance=train_doc_name_to_full_instance, label_scheme=run_config.ner_label_scheme, label_to_id=label_to_id_func,
            boundary_exclusion_size=run_config.boundary_exclusion_size, process_batch_size=run_config.process_batch_size, is_training=True,
            opts=opts, train_batch_size=run_config.batch_size, clip_grad=run_config.clip_grad, clipping_grad_value=run_config.clipping_grad_value,
        )
        report = f'epoch={epoch} target=train loss={train_loss:.4e} P/R/F(union)={train_meter_by_union.calc()["PRF"]} P/R/F(ensemble)={train_meter_by_ensemble.calc()["PRF"]}'

        # dev
        dev_loss, dev_meter_by_union, dev_meter_by_ensemble = run_epoch(
            model=model, dataset=dev_dataset, doc_name_to_full_instance=dev_doc_name_to_full_instance, label_scheme=run_config.ner_label_scheme, label_to_id=label_to_id_func,
            boundary_exclusion_size=run_config.boundary_exclusion_size, process_batch_size=run_config.process_batch_size, is_training=False,
        )
        report += f'\nepoch={epoch} target=dev loss={dev_loss:.4e} P/R/F(union)={dev_meter_by_union.calc()["PRF"]} P/R/F(ensemble)={dev_meter_by_ensemble.calc()["PRF"]}'

        if run_config.use_average:
            with model_average.average_context():
                dev_ave_loss, dev_ave_meter_by_union, dev_ave_meter_by_ensemble = run_epoch(
                    model=model, dataset=dev_dataset, doc_name_to_full_instance=dev_doc_name_to_full_instance, label_scheme=run_config.ner_label_scheme, label_to_id=label_to_id_func,
                    boundary_exclusion_size=run_config.boundary_exclusion_size, process_batch_size=run_config.process_batch_size, is_training=False,
                )
            report += f'\nepoch={epoch} target=dev(average) loss={dev_ave_loss:.4e} P/R/F(union)={dev_ave_meter_by_union.calc()["PRF"]} P/R/F(ensemble)={dev_ave_meter_by_ensemble.calc()["PRF"]}'


        # save
        has_saved = False
        dev_f_score = dev_meter_by_ensemble.calc()["f-score"]
        if best_scores.get("dev/f", -1.0) < dev_f_score:
            best_scores["dev/f"] = dev_f_score
            best_scores["dev/epoch"] = epoch
            if save_dir_path is not None:
                torch.save(model.state_dict(), str(save_dir_path / "model_best.pt"))
                dev_meter_by_ensemble.output_predictions_as_ann_to_dir(save_dir_path/"preds_ensemble", id_to_label=id_to_label, doc_id_to_text_func=lambda x:dev_doc_name_to_full_instance[x].text)
                has_saved = True
        if run_config.use_average:
            dev_ave_f_score = dev_ave_meter_by_ensemble.calc()["f-score"]
            if best_scores.get("dev-ave/f", -1.0) < dev_ave_f_score:
                best_scores["dev-ave/f"] = dev_ave_f_score
                best_scores["dev-ave/epoch"] = epoch
                if save_dir_path is not None:
                    with model_average.average_context():
                        torch.save(model.state_dict(), str(save_dir_path / "model-ema_best.pt"))
                    dev_ave_meter_by_ensemble.output_predictions_as_ann_to_dir(save_dir_path/"preds-ema_ensemble", id_to_label=id_to_label, doc_id_to_text_func=lambda x:dev_doc_name_to_full_instance[x].text)
                    has_saved = True
        if has_saved:
            with open(save_dir_path/"best-scores.json", "w") as f:
                json.dump(best_scores, f)

        tee_print(report)

elif run_config.mode in ["eval-dev", "eval-test"]:
    # load data
    if run_config.mode == "eval-dev":
        corpus_dir = run_config.dev_corpus_dir
    elif run_config.mode == "eval-test":
        corpus_dir = run_config.test_corpus_dir
    else:
        raise ValueError(run_config.mode)
    val_instances, val_doc_name_to_full_instance = load_dir(corpus_dir, tokenizer=tokenizer, max_length=run_config.max_length, stride=run_config.stride, rstrip_text=run_config.do_rstrip_text, use_cache=run_config.use_dataset_cache, cache_dir_path=pathlib.Path(run_config.cache_dir))
    val_dataset = build_dataset(val_instances, label_scheme=run_config.ner_label_scheme, label_to_id=label_to_id_func, device=run_config.device)

    # prepare output directory
    if run_config.eval_output_dir is None:
        _logger.info(f'`eval_output_dir` is not set. To save evaluation results, run with `python3 main.py --eval_output_dir /path/to/eval_output_dir`.')
        eval_output_dir_path = None
    else:
        eval_output_dir_path = pathlib.Path(run_config.eval_output_dir)

        if eval_output_dir_path.exists():
            if run_config.force_save:
                _logger.warning(f'eval_output_dir_path={str(eval_output_dir_path)} already exists, but force_save is set to be True. NOTE: This will result overwriting existing logs.')
            else:
                raise ValueError(f'eval_output_dir_path={str(eval_output_dir_path)} already exists. Set other directory, or run with `python3 main.py --force_save` to ignore this error. (NOTE: --force_save will result overwriting existing logs.)')
        else:
            eval_output_dir_path.mkdir(parents=True)

    val_loss, val_meter_by_union, val_meter_by_ensemble = run_epoch(
        model=model, dataset=val_dataset, doc_name_to_full_instance=val_doc_name_to_full_instance, label_scheme=run_config.ner_label_scheme, label_to_id=label_to_id_func,
        boundary_exclusion_size=run_config.boundary_exclusion_size, process_batch_size=run_config.process_batch_size, is_training=False,
    )
    report = f'target={run_config.mode} loss={val_loss:.4e} P/R/F(union)={val_meter_by_union.calc()["PRF"]} P/R/F(ensemble)={val_meter_by_ensemble.calc()["PRF"]}'

    if run_config.eval_output_dir is not None:
        with open(eval_output_dir_path/"run-config.json", "w") as f:
            json.dump(dataclasses.asdict(run_config), f)

        val_meter_by_union.output_predictions_as_ann_to_dir(eval_output_dir_path/"preds_union", id_to_label=id_to_label, doc_id_to_text_func=lambda x:val_doc_name_to_full_instance[x].text)
        val_meter_by_ensemble.output_predictions_as_ann_to_dir(eval_output_dir_path/"preds_ensemble", id_to_label=id_to_label, doc_id_to_text_func=lambda x:val_doc_name_to_full_instance[x].text)

        scores = {
            "union": val_meter_by_union.calc(),
            "ensemble": val_meter_by_ensemble.calc(),
        }
        with open(eval_output_dir_path/"scores.json", "w") as f:
            json.dump(scores, f)

        with open(eval_output_dir_path/"report.txt", "w") as f:
            print(report, file=f)

    print(report)


