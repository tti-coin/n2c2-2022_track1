# %%
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from loader import Attribute


def fix_seed(seed):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_optimizer_grouped_parameters(model: torch.nn.Module, other_lr: float, bert_lr: float, lr_decay: float) -> list:
    model_type = "bert"
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "bert" not in n],
            "weight_decay": 0.0,
            "lr": other_lr,
        },
    ]
    layers = [getattr(model, model_type).embeddings] + list(getattr(model, model_type).encoder.layer)
    layers.reverse()
    for layer in layers:
        bert_lr *= lr_decay
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": bert_lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": bert_lr,
            },
        ]
    return optimizer_grouped_parameters


def custom_criterion(outputs: torch.Tensor, batch_labels: list[list[dict[str, list[int]]]]) -> torch.Tensor:
    # Create gold labels
    event_gold_labels, action_gold_labels, negation_gold_labels, temporality_gold_labels, certainty_gold_labels, actor_gold_labels = [], [], [], [], [], []
    masks = []  # If Event is not "Disposition", not calculate attribute loss

    for labels in batch_labels:
        event_gold_label, action_gold_label, negation_gold_label, temporality_gold_label, certainty_gold_label, actor_gold_label = (
            np.zeros(3),
            np.zeros(7),
            np.zeros(2),
            np.zeros(4),
            np.zeros(4),
            np.zeros(3),
        )
        for label in labels:
            event_gold_label += np.array(label["Event"])
            action_gold_label += np.array(label["Action"])
            negation_gold_label += np.array(label["Negation"])
            temporality_gold_label += np.array(label["Temporality"])
            certainty_gold_label += np.array(label["Certainty"])
            actor_gold_label += np.array(label["Actor"])

        # event_label = np.array([0, 2, 0]) -> np.array([0, 1, 0])
        event_gold_label[event_gold_label != 0] = 1
        action_gold_label[action_gold_label != 0] = 1
        negation_gold_label[negation_gold_label != 0] = 1
        temporality_gold_label[temporality_gold_label != 0] = 1
        certainty_gold_label[certainty_gold_label != 0] = 1
        actor_gold_label[actor_gold_label != 0] = 1
        masks += [True] if event_gold_label[1] == 1 else [False]

        event_gold_labels.append(torch.tensor(event_gold_label).unsqueeze(0))
        action_gold_labels.append(torch.tensor(action_gold_label).unsqueeze(0))
        negation_gold_labels.append(torch.tensor(negation_gold_label).unsqueeze(0))
        temporality_gold_labels.append(torch.tensor(temporality_gold_label).unsqueeze(0))
        certainty_gold_labels.append(torch.tensor(certainty_gold_label).unsqueeze(0))
        actor_gold_labels.append(torch.tensor(actor_gold_label).unsqueeze(0))

    t_event_gold_labels = torch.cat(event_gold_labels)
    t_action_gold_labels = torch.cat(action_gold_labels)
    t_negation_gold_labels = torch.cat(negation_gold_labels)
    t_temporality_gold_labels = torch.cat(temporality_gold_labels)
    t_certainty_gold_labels = torch.cat(certainty_gold_labels)
    t_actor_gold_labels = torch.cat(actor_gold_labels)

    # Split outputs
    event_outputs, action_outputs, negation_outputs, temporality_outputs, certainty_outputs, actor_outputs = (
        outputs[:, 0:3],
        outputs[:, 3:10],
        outputs[:, 10:12],
        outputs[:, 12:16],
        outputs[:, 16:20],
        outputs[:, 20:23],
    )

    # Calculate loss
    criterion = nn.BCEWithLogitsLoss()
    losses = criterion(
        event_outputs,
        t_event_gold_labels.float().to(
            "cuda" if torch.cuda.is_available() else "cpu",
        ),
    )
    # Event == Disposition (gold)
    if sum(masks) != 0:
        t_masks = torch.tensor(masks)
        losses += criterion(
            action_outputs[t_masks],
            t_action_gold_labels[t_masks]
            .float()
            .to(
                "cuda" if torch.cuda.is_available() else "cpu",
            ),
        )
        losses += criterion(
            negation_outputs[t_masks],
            t_negation_gold_labels[t_masks]
            .float()
            .to(
                "cuda" if torch.cuda.is_available() else "cpu",
            ),
        )
        losses += criterion(
            temporality_outputs[t_masks],
            t_temporality_gold_labels[t_masks]
            .float()
            .to(
                "cuda" if torch.cuda.is_available() else "cpu",
            ),
        )
        losses += criterion(
            certainty_outputs[t_masks],
            t_certainty_gold_labels[t_masks]
            .float()
            .to(
                "cuda" if torch.cuda.is_available() else "cpu",
            ),
        )
        losses += criterion(
            actor_outputs[t_masks],
            t_actor_gold_labels[t_masks]
            .float()
            .to(
                "cuda" if torch.cuda.is_available() else "cpu",
            ),
        )
    return losses


def cal_fscore(entity_key2gold_labels: dict[str, list[Attribute]], entity_key2pred_labels: dict[str, list[Attribute]]) -> tuple[float, float, float]:
    tp, fn, fp = 0, 0, 0
    for entity_key, gold_labels in entity_key2gold_labels.items():
        pred_labels = entity_key2pred_labels[entity_key]

        for gold_label in gold_labels:
            if gold_label in pred_labels:
                tp += 1
            else:
                fn += 1

        for pred_label in pred_labels:
            if pred_label not in gold_labels:
                fp += 1

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    print(f"tp:{tp}, fn:{fn}, fp:{fp}")
    f_score = (2 * recall * precision) / (recall + precision + 1e-10)
    return precision, recall, f_score


def output_brat(target_json, bert, args):
    n2c2_path = args.n2c2_path
    eval_path = args.eval_path
    for i, (fname, passages) in enumerate(target_json.items()):
        output_list = list()
        a_id = 1
        for pidx, passage in enumerate(passages):
            for entity in passage.entities:
                id = entity.ann_id
                text = entity.text
                event = entity.att.Event
                offset_s = passage.starts[entity.start] + passage.offset
                offset_e = passage.ends[entity.end - 1] + passage.offset

                text = passage.text[offset_s - passage.offset : offset_e - passage.offset]
                output_line = "T" + id[1:] + "\t" + event + " " + str(offset_s) + " " + str(offset_e) + "\t" + text
                output_list.append(output_line)

                output_line = "E" + id[1:] + "\t" + event + ":T" + id[1:] + " "
                output_list.append(output_line)

                if event != "Disposition":
                    continue

                if entity.att.Action is not None:
                    output_line = "A" + str(a_id) + "\t" + "Action" + " " + id + " " + entity.att.Action
                    output_list.append(output_line)
                    a_id += 1

                if entity.att.Negation is not None:
                    output_line = "A" + str(a_id) + "\t" + "Negation" + " " + id + " " + entity.att.Negation
                    output_list.append(output_line)
                    a_id += 1

                if entity.att.Temporality is not None:
                    output_line = "A" + str(a_id) + "\t" + "Temporality" + " " + id + " " + entity.att.Temporality
                    output_list.append(output_line)
                    a_id += 1

                if entity.att.Certainty is not None:
                    output_line = "A" + str(a_id) + "\t" + "Certainty" + " " + id + " " + entity.att.Certainty
                    output_list.append(output_line)
                    a_id += 1

                if entity.att.Actor is not None:
                    output_line = "A" + str(a_id) + "\t" + "Actor" + " " + id + " " + entity.att.Actor
                    output_list.append(output_line)
                    a_id += 1

        pred_dir = n2c2_path / "data/revision_results/ida" / args.eval_name / (bert + ".pred")
        if not pred_dir.exists():
            pred_dir.mkdir()

        with open(pred_dir / (fname + ".ann"), "w") as f:
            for output_line in output_list:
                f.write("%s\n" % output_line)

    print("save {}".format(pred_dir))
