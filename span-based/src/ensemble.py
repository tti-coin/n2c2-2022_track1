# %%
import glob
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n2c2_path", type=str)
    parser.add_argument("--thres", type=int, default=10)
    parser.add_argument("--pred_dirs", type=str)
    return parser.parse_args()


def create_dataset(args: argparse.Namespace):
    pred_dirs = glob.glob(str(args.n2c2_path / "data/revision_results" / args.pred_dirs / "*"))
    dataset = defaultdict(dict)
    for pred_dir in pred_dirs:
        if pred_dir.split("/")[-1] == "ensemble.pred":
            print("skip ensemble.pred")
            continue
        if pred_dir.split(".")[-1] == "txt" or pred_dir.split(".")[-1] == "ann":
            continue
        for file_path in Path(pred_dir).glob("*"):
            fname = str(file_path).split("/")[-1].split(".")[0]
            dataset[pred_dir][fname] = list()
            with open(file_path) as f:
                for line in f.readlines():
                    dataset[pred_dir][fname].append(line)
    return dataset


def count_pred(dataset: Dict[str, Dict[str, List[str]]]):
    count_dict = dict()
    for dir_name, values in dataset.items():
        for fname, annotations in values.items():
            eid_dict = dict()
            e_t_mapper = dict()
            t_e_mapper = dict()
            t_offset = dict()
            for line in annotations:
                if line.strip().startswith("E"):
                    e_id, mapper_m = line.strip().split("\t")
                    e_type, tag_id = mapper_m.split(":")
                    assert e_id not in e_t_mapper
                    assert tag_id not in t_e_mapper
                    e_t_mapper[e_id] = tag_id
                    t_e_mapper[tag_id] = e_id
                    eid_dict[e_id] = {"event": e_type}
                elif line.strip().startswith("A"):
                    attr_id, attr_m = line.strip().split("\t")
                    attr_type, e_id, attr_val = attr_m.split(" ")
                    assert attr_type not in eid_dict[e_id].keys()
                    eid_dict[e_id][attr_type] = attr_val
                else:
                    try:
                        t_id, attr_t, text = line.strip().split("\t")
                    except:
                        print(fname)
                        continue
                    _, start, end = attr_t.split(" ")
                    t_offset[t_id] = (start, end, text)
            for eid, attr in eid_dict.items():
                tid = e_t_mapper[eid]
                offset = t_offset[tid]
                key = fname + "|" + offset[0] + "|" + offset[1] + "|" + offset[2]
                if key not in count_dict.keys():
                    count_dict[key] = defaultdict(int)
                if attr["event"] == "Disposition":
                    attr_str = attr["event"] + "|" + attr["Action"] + "|" + attr["Negation"] + "|" + attr["Temporality"] + "|" + attr["Certainty"] + "|" + attr["Actor"]
                else:
                    attr_str = attr["event"]
                count_dict[key][attr_str] += 1
    return count_dict


def crete_output_dict(args: argparse.Namespace, count_dict: Dict[str, Dict[str, int]]):
    output_dict = defaultdict(list)
    for key, attr_dict in count_dict.items():
        max_cnt = 0
        pred = ""
        flag = True
        attr_dict = sorted(attr_dict.items(), key=lambda x: x[1], reverse=True)
        s = set()
        for attr, cnt in attr_dict:
            if max_cnt < cnt:
                max_cnt = cnt
                pred = attr
            if args.thres < cnt:
                if len(s) and attr.split("|")[0] not in s:
                    continue
                flag = False
                output_dict[key.split("|")[0]].append(key.split("|")[1:] + [attr])
                s.add(attr.split("|")[0])
        if flag:
            output_dict[key.split("|")[0]].append(key.split("|")[1:] + [pred])
    return output_dict


def save_ensemble_result(args: argparse.Namespace, output_dict: Dict[str, List[str]]):
    for fname, labels in output_dict.items():
        a_id = 1
        output_list = list()
        for idx, label in enumerate(labels):
            start, end, text, attr_dict = label
            attr_list = attr_dict.split("|")
            output_line = "T" + str(idx) + "\t" + attr_list[0] + " " + str(start) + " " + str(end) + "\t" + text
            output_list.append(output_line)
            output_line = "E" + str(idx) + "\t" + attr_list[0] + ":T" + str(idx) + " "
            output_list.append(output_line)

            if attr_list[0] != "Disposition":
                continue

            output_line = "A" + str(a_id) + "\t" + "Action" + " " + "E" + str(idx) + " " + attr_list[1]
            output_list.append(output_line)
            a_id += 1

            output_line = "A" + str(a_id) + "\t" + "Negation" + " " + "E" + str(idx) + " " + attr_list[2]
            output_list.append(output_line)
            a_id += 1

            output_line = "A" + str(a_id) + "\t" + "Temporality" + " " + "E" + str(idx) + " " + attr_list[3]
            output_list.append(output_line)
            a_id += 1

            output_line = "A" + str(a_id) + "\t" + "Certainty" + " " + "E" + str(idx) + " " + attr_list[4]
            output_list.append(output_line)
            a_id += 1

            output_line = "A" + str(a_id) + "\t" + "Actor" + " " + "E" + str(idx) + " " + attr_list[5]
            output_list.append(output_line)
            a_id += 1
        pred_dir = args.n2c2_path / "data/revision_results" / args.pred_dirs / "ensemble.pred"
        if not pred_dir.exists():
            pred_dir.mkdir()
        with open(pred_dir / (fname + ".ann"), "w") as f:
            for output_line in output_list:
                f.write("%s\n" % output_line)

    print("save {}".format(pred_dir))


if __name__ == "__main__":
    args = get_args()
    args.n2c2_path = Path(args.n2c2_path)
    dataset = create_dataset(args)
    count_dict = count_pred(dataset)
    output_dict = crete_output_dict(args, count_dict)
    save_ensemble_result(args, output_dict)
