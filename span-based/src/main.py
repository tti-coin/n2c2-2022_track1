# %%
import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import optuna
import pydantic
import torch
from tqdm import tqdm

from loader import AnnotatedPassage, Attribute, Corpus, SpanDataset
from model import Model, ModelConfig
from utils.dataset import SelectiveDataset
from utils.utils import (
    cal_fscore,
    custom_criterion,
    fix_seed,
    get_optimizer_grouped_parameters,
)


class Config(pydantic.BaseModel):
    bert_model_path: Path
    cached_dataset_path: Path
    cached_corpus_path: Path
    cached_config_path: Optional[Path] = None
    mode: str = "training"
    init_model_path: Optional[Path] = None
    saved_model_path: Optional[Path] = None
    batch_size: int = 32
    early_stopping: int = 15
    max_epoch: int = 1
    random_seed: int = 42
    other_lr: float = 1e-5
    bert_lr: float = 5e-6
    dropout_rate: float = 0.8
    lr_decay: float = 0.95
    bert_layers: int = 1
    window_token: int = 0

    @classmethod
    def create_config(cls, args: argparse.Namespace) -> "Config":
        if args.cached_config_path is None or not args.cached_config_path.exists():
            config = cls(**vars(args))
        else:
            with open(args.cached_config_path) as f:
                config = cls.model_validate_json(f.read())
        return config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model_path", default="../data/RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf", type=Path)
    parser.add_argument("--cached_dataset_path", default="../data/preprocessed_data/dataset.json", type=Path)
    parser.add_argument("--cached_corpus_path", default="../data/preprocessed_data/corpus.json", type=Path)
    parser.add_argument("--cached_config_path", default="../data/preprocessed_data/config.json", type=Path)
    parser.add_argument("--mode", type=str, default="training", choices=["tuning", "training", "evaluation"])
    parser.add_argument("--init_model_path", default=None)
    parser.add_argument("--saved_model_path", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--early_stopping", type=int, default=15)
    parser.add_argument("--max_epoch", type=int, default=50)
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()


if '--transport="tcp"' in sys.argv:  # vscode jupyter
    parser = argparse.ArgumentParser()
    args = parser.parse_args(args=[])
    args.bert_model_path = Path("../data/RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf")
    args.cached_dataset_path = Path("../data/preprocessed_data/dataset.json")
    args.cached_corpus_path = Path("../data/preprocessed_data/corpus.json")
    args.cached_config_path = None
    args.mode = "training"
    args.init_model_path = None
    args.saved_model_path = None
    args.batch_size = 32
    args.max_epoch = 50
    args.random_seed = 42
else:
    args = get_args()


def create_dataloader(args: argparse.Namespace) -> tuple[SelectiveDataset, SelectiveDataset]:
    # Load span dataset
    with open(args.cached_dataset_path) as f:
        span_dataset = SpanDataset.model_validate_json(f.read())
    assert args.bert_model_path == span_dataset.bert_model_path

    selectors = [
        {"name": "entity_key"},
        {
            "name": "token_ids",
            "dtype": torch.long,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "padding": True,
            "padding_value": span_dataset.pad_token_id,
            "padding_mask": True,
        },
        {"name": "entity_range", "dtype": torch.int},
        {"name": "label"},
        {"name": "text"},
    ]

    train_dataset = SelectiveDataset(span_dataset.train_instances, selectors)
    eval_dataset = SelectiveDataset(span_dataset.eval_instances, selectors)
    train_dataloader = train_dataset.dataloader(args.batch_size, shuffle=True)
    eval_dataloader = eval_dataset.dataloader(args.batch_size, shuffle=False)
    return train_dataloader, eval_dataloader


def create_entity_key2gold_labels(fname2annotated_passages: dict[str, list[AnnotatedPassage]]) -> dict[str, list[Attribute]]:
    entity_key2gold_labels = defaultdict(list)
    for annotated_passages in fname2annotated_passages.values():
        for annotated_passage in annotated_passages:
            for entity in annotated_passage.entities:
                entity_key = str(entity.s_tok_idx) + "|" + str(entity.e_tok_idx)
                entity_key2gold_labels[entity_key].append(
                    Attribute(
                        event=entity.att.event,
                        action=entity.att.action,
                        negation=entity.att.negation,
                        temporality=entity.att.temporality,
                        certainty=entity.att.certainty,
                        actor=entity.att.actor,
                    )
                )
    return entity_key2gold_labels


def create_entity_key2pred_labels(outputs_list: list[torch.Tensor], entity_keys: list[list[str]]) -> dict[str, list[Attribute]]:
    EVENT_LIST = ["NoDisposition", "Disposition", "Undetermined"]
    ACTION_LIST = ["Start", "Stop", "Increase", "Decrease", "UniqueDose", "OtherChange", "Unknown"]
    NEGATION_LIST = ["Negated", "NotNegated"]
    TEMPORALITY_LIST = ["Past", "Present", "Future", "Unknown"]
    CERTAINTY_LIST = ["Certain", "Hypothetical", "Conditional", "Unknown"]
    ACTOR_LIST = ["Physician", "Patient", "Unknown"]

    entity_key2pred_labels = defaultdict(list)
    for outputs, minibatch_entity_keys in zip(outputs_list, entity_keys):
        for entity_key, output in zip(minibatch_entity_keys, outputs):
            event_logit = output[0:3]
            action_logit = output[3:10]
            negation_logit = output[10:12]
            temporality_logit = output[12:16]
            certainty_logit = output[16:20]
            actor_logit = output[20:23]

            event_pred = EVENT_LIST[int(torch.argmax(event_logit))]
            if event_pred == "NoDisposition":
                entity_key2pred_labels[entity_key].append(Attribute(Event="NoDisposition", Action=None, Negation=None, Temporality=None, Certainty=None, Actor=None))
            elif event_pred == "Disposition":
                action_pred = ACTION_LIST[int(torch.argmax(action_logit))]
                negation_pred = NEGATION_LIST[int(torch.argmax(negation_logit))]
                temporality_pred = TEMPORALITY_LIST[int(torch.argmax(temporality_logit))]
                certainty_pred = CERTAINTY_LIST[int(torch.argmax(certainty_logit))]
                actor_pred = ACTOR_LIST[int(torch.argmax(actor_logit))]
                entity_key2pred_labels[entity_key].append(
                    Attribute(
                        event="Disposition",
                        action=action_pred,
                        negation=negation_pred,
                        temporality=temporality_pred,
                        certainty=certainty_pred,
                        actor=actor_pred,
                    )
                )
            else:
                entity_key2pred_labels[entity_key].append(Attribute(Event="Undetermined", Action=None, Negation=None, Temporality=None, Certainty=None, Actor=None))
    return entity_key2pred_labels


def _train(config: Config, corpus: Corpus, train_dataloader: SelectiveDataset, eval_dataloader: SelectiveDataset, skip_train: bool = False):
    model = Model(ModelConfig(bert_model_path=config.bert_model_path, dropout_rate=config.dropout_rate, bert_layers=config.bert_layers, window_token=config.window_token))
    if config.init_model_path is not None:
        model.load_state_dict(torch.load(config.init_model_path))
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    optimizer_parameters = get_optimizer_grouped_parameters(
        model=model,
        other_lr=config.other_lr,
        bert_lr=config.bert_lr,
        lr_decay=config.lr_decay,
    )
    optimizer = torch.optim.AdamW(optimizer_parameters)

    best_eval_fscore = -1.0
    train_entity_key2gold_labels = create_entity_key2gold_labels(fname2annotated_passages=corpus.train_fname2annotated_passages)
    eval_entity_key2gold_labels = create_entity_key2gold_labels(fname2annotated_passages=corpus.eval_fname2annotated_passages)
    for epoch in range(1, config.max_epoch + 1):
        if skip_train:
            print(f"Epoch: {epoch}epoch")
            # Training
            model.train()
            train_loss = 0.0
            outputs_list, entity_keys = [], []
            for mini_batch in tqdm(train_dataloader):
                entity_keys.append(mini_batch["entity_key"])
                token_ids, token_ids_mask = mini_batch["token_ids"], mini_batch["token_ids_mask"]
                s_tok_idxs, e_tok_idxs = mini_batch["entity_range"][:, 0], mini_batch["entity_range"][:, 1]
                batch_labels = mini_batch["label"]
                optimizer.zero_grad()
                outputs = model(token_ids=token_ids, token_ids_mask=token_ids_mask, s_tok_idxs=s_tok_idxs, e_tok_idxs=e_tok_idxs)
                loss = custom_criterion(outputs=outputs, batch_labels=batch_labels)
                outputs_list.append(outputs.to("cpu"))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_entity_key2pred_labels = create_entity_key2pred_labels(outputs_list=outputs_list, entity_keys=entity_keys)
            train_precision, train_recall, train_fscore = cal_fscore(entity_key2gold_labels=train_entity_key2gold_labels, entity_key2pred_labels=train_entity_key2pred_labels)
            print(f"Train:p={train_precision}, r={train_recall}, f={train_fscore}")
        else:
            print("Skip training")

        # Evaluation
        model.eval()
        dev_loss = 0.0
        outputs_list, entity_keys = [], []
        with torch.no_grad():
            for minibatch in eval_dataloader:
                entity_keys.append(minibatch["entity_key"])
                token_ids, token_ids_mask = minibatch["token_ids"], minibatch["token_ids_mask"]
                s_tok_idxs, e_tok_idxs = minibatch["entity_range"][:, 0], minibatch["entity_range"][:, 1]
                batch_labels = minibatch["label"]

                outputs = model(token_ids=token_ids, token_ids_mask=token_ids_mask, s_tok_idxs=s_tok_idxs, e_tok_idxs=e_tok_idxs)
                loss = custom_criterion(outputs=outputs, batch_labels=batch_labels)
                outputs_list.append(outputs.to("cpu"))
                dev_loss += loss.item()

        eval_entity_key2pred_labels = create_entity_key2pred_labels(outputs_list=outputs_list, entity_keys=entity_keys)
        eval_precision, eval_recall, eval_fscore = cal_fscore(entity_key2gold_labels=eval_entity_key2gold_labels, entity_key2pred_labels=eval_entity_key2pred_labels)
        print(f"Eval:p={eval_precision}, r={eval_recall}, f={eval_fscore}")
        if best_eval_fscore < eval_fscore:
            best_eval_fscore = eval_fscore
            early_stopping_round = 0
            if config.saved_model_path is not None and not skip_train:
                torch.save(model.state_dict(), config.saved_model_path)
        else:
            early_stopping_round += 1

        # Early Stopping
        if early_stopping_round >= config.early_stopping:
            print("Early Stopping ...")
            break
        if skip_train:
            break
        yield eval_fscore

    if config.save_output_path is not None:
        # Output brat format
        output_brat()  # TODO


def train(config: Config, corpus: Corpus, train_dataloader: SelectiveDataset, eval_dataloader: SelectiveDataset, skip_train: bool = False) -> list[float]:
    return list(_train(config=config, corpus=corpus, train_dataloader=train_dataloader, eval_dataloader=eval_dataloader, skip_train=skip_train))


def main(args: argparse.Namespace) -> None:
    # Load Corpus
    with open(args.cached_corpus_path) as f:
        corpus = Corpus.model_validate_json(f.read())
    train_dataloader, eval_dataloader = create_dataloader(args=args)

    if args.mode == "tuning":

        def objective(trial):
            # Set up hyper-parameters
            args.other_lr = trial.suggest_loguniform("other_lr", 1e-5, 1e-4)
            args.bert_lr = trial.suggest_loguniform("bert_lr", 5e-7, 5e-5)
            args.dropout_rate = trial.suggest_loguniform("dropout", 0.1, 1.0)
            args.lr_decay = trial.suggest_loguniform("lr_decay", 0.95, 1.0)
            args.bert_layers = trial.suggest_int("bert_layers", 1, 4)
            args.window_token = trial.suggest_int("window_token", 0, 5)
            config = Config.create_config(args=args)

            best_score = -1
            for step, score in enumerate(_train(config=config, corpus=corpus, train_dataloader=train_dataloader, eval_dataloader=eval_dataloader)):
                best_score = max(best_score, score)
                trial.report(score, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return best_score

        pruner = optuna.pruners.SuccessiveHalvingPruner()
        study = optuna.create_study(
            direction="maximize",
            pruner=pruner,
        )
        study.optimize(objective, n_trials=args.optuna_n_trials)
        print(f"best dev accuracy: {study.best_value}")
        print(f"best parameter: {study.best_params}")
    elif args.mode == "training":
        config = Config.create_config(args=args)
        train(config=config, corpus=corpus, train_dataloader=train_dataloader, eval_dataloader=eval_dataloader)
    else:
        config = Config.create_config(args=args)
        train(config=config, corpus=corpus, train_dataloader=train_dataloader, eval_dataloader=eval_dataloader, skip_train=True)


if __name__ == "__main__":
    fix_seed(args.random_seed)
    main(args=args)


# %%
