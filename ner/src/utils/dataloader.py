from itertools import zip_longest

from typing import Union

import numpy as np
import torch
import torch.utils.data


def get_padded_shape(seq):
    assert type(seq) in [list, tuple]
    shape = _get_padded_shape_inner(seq)
    return shape
def _get_padded_shape_inner(seq, _current_depth=0):
    if any(type(deep) in [list, tuple] for deep in seq):
        deep_shapes = [_get_padded_shape_inner(deep_seq, _current_depth=_current_depth+1) for deep_seq in seq]
        deep_shape = [max(lens) for lens in zip_longest(*deep_shapes, fillvalue=0)]
        return [len(seq), *deep_shape]
    else:
        return [len(seq)]

def pad(seq, padding_value):
    assert type(seq) is list
    shape = get_padded_shape(seq)
    padded, mask = _pad_inner(seq, padding_value=padding_value, shape=shape)
    return padded, mask
def _pad_inner(seq, padding_value, shape, _current_depth=0):
    assert type(seq) is list
    if _current_depth == len(shape) - 1:
        pad_vec = [padding_value]
        pad_vec = pad_vec * (shape[_current_depth] - len(seq))
        mask = [1] * len(seq) + [0] * len(pad_vec)
        return seq + pad_vec, mask
    else:
        deep_seqs_and_masks = [_pad_inner(deep_seq, padding_value=padding_value, shape=shape, _current_depth=_current_depth+1) for deep_seq in seq]
        if len(deep_seqs_and_masks) > 0:
            deep_seqs, deep_mask = map(list, list(zip(*deep_seqs_and_masks)))
        else:
            deep_seqs, deep_mask = [], []

        pad_vec = [padding_value]
        zero_vec = [0]
        for l in reversed(shape[_current_depth+1:]):
            pad_vec = pad_vec * l
            zero_vec = zero_vec * l
            pad_vec = [pad_vec]
            zero_vec = [zero_vec]
        pad_vec = pad_vec * (shape[_current_depth] - len(seq))
        mask = deep_mask + zero_vec * (shape[_current_depth] - len(seq))
        return deep_seqs + pad_vec, mask

class Selector:
    def __init__(self, name, origin=None, mapping=None, dtype=None, device=None, padding=False, padding_value=0, padding_mask=False):
        assert not ((origin is not None) and (mapping is not None)), "cannot set both origin and mapping"
        self.name = name
        self._origin = origin
        self.mapping = mapping
        self.dtype = dtype
        self.device = device
        self.padding = padding
        self.padding_value = padding_value
        self.padding_mask = padding_mask
    @property
    def origin(self):
        if self._origin is not None:
            return self._origin
        elif self.mapping is not None:
            return None
        else:
            return self.name

    def select(self, instance):
        if self.mapping is not None:
            return self.mapping(instance)
        else:
            return instance[self.origin]

class SelectiveDataset(torch.utils.data.Dataset):
    """
    example use:
    i1 = {"id":"instance1", "foo":32, "bar":[[1,2]]}
    i2 = {"id":"instance2", "foo":50, "bar":[[10],[32],[5]]}
    i3 = {"id":"instance3", "foo":43, "bar":[], "baz":-1}
    instances = [i1,i2,i3,i1,i1,i1,i1]

    device = torch.device("cpu")
    # device = torch.device("cuda:0")
    selectors = [
        {"name":"id"},
        {"name":"foo_original", "mapping":lambda instance:instance["foo"], "dtype":torch.long},
        {"name":"foo_square", "mapping":lambda instance:instance["foo"]**2, "dtype":torch.float},
        {"name":"bar", "dtype":torch.float, "device":device, "padding":True, "padding_value":-42, "padding_mask":True},
    ]
    dataset = SelectiveDataset(instances, selectors, sort_key=lambda x:len(x["bar"]))
    dataloader = dataset.dataloader(batch_size=4, shuffle=True)
    for minibatch in dataloader:
        assert type(minibatch) is dict
        print(minibatch)
    """
    def __init__(self, instances, selectors, sort_key=None, ordered=False, rng_state:Union[int, np.random.RandomState]=12345):
        assert all(type(selector) in [Selector, dict] for selector in selectors)
        selectors = [selector if type(selector) is Selector else Selector(**selector) for selector in selectors]
        assert len(selectors) == len(set(s.name for s in selectors)), "cannot use a same name multiple times."

        self.instances = instances
        self.selectors = list(selectors)
        self.sort_key = sort_key

        if isinstance(rng_state, int):
            self.rng = np.random.RandomState(rng_state)
        elif isinstance(rng_state, np.random.RandomState):
            self.rng = rng_state
        else:
            raise ValueError(rng_state)
        self.ordered = ordered
        self.order = list(range(len(self.instances)))
        self.num_shuffled = 0
        if self.ordered:
            self.shuffle_order()

    def shuffle_order(self):
        order = list(range(len(self.instances)))
        self.ordered = True
        self.rng.shuffle(order)
        self.order = order
        self.num_shuffled += 1
        return self

    def __getitem__(self, idx):
        instance = self.instances[self.order[idx]]
        return {selector.name:selector.select(instance) for selector in self.selectors}

    def __len__(self):
        return len(self.instances)

    def collate_fn(self, instances):
        if self.sort_key is not None:
            instances = sorted(instances, key=self.sort_key, reverse=True)

        outputs = dict()
        for selector in self.selectors:
            key = selector.name
            values = [instance[key] for instance in instances]

            if selector.padding:
                values, masks = pad(values, selector.padding_value)

                if selector.padding_mask:
                    masks = torch.FloatTensor(masks)
                    if selector.device is not None:
                        masks = masks.to(selector.device)
                    outputs[key + "_mask"] = masks

            if selector.dtype is not None:
                values = torch.tensor(values, dtype=selector.dtype)
                if selector.device is not None:
                    values = values.to(selector.device)

            outputs[key] = values

        return outputs

    def dataloader(self, batch_size, shuffle, *args, **kwargs):
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn, *args, **kwargs)

