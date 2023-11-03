from itertools import zip_longest

import torch


def get_padded_shape(seq):
    assert type(seq) in [list, tuple]
    shape = _get_padded_shape(seq)
    return shape


def _get_padded_shape(seq, _current_depth=0):
    if any(type(deep) in [list, tuple] for deep in seq):
        deep_shapes = [_get_padded_shape(deep_seq, _current_depth=_current_depth + 1) for deep_seq in seq]
        deep_shape = [max(lens) for lens in zip_longest(*deep_shapes, fillvalue=0)]
        return [len(seq), *deep_shape]
    else:
        return [len(seq)]


def pad(seq, padding_value):
    assert type(seq) is list
    shape = get_padded_shape(seq)
    padded, mask = _pad(seq, padding_value=padding_value, shape=shape)
    return padded, mask


def _pad(seq, padding_value, shape, _current_depth=0):
    assert type(seq) is list
    if _current_depth == len(shape) - 1:
        pad_vec = [padding_value]
        pad_vec = pad_vec * (shape[_current_depth] - len(seq))
        mask = [1] * len(seq) + [0] * len(pad_vec)
        return seq + pad_vec, mask
    else:
        deep_seqs_and_masks = [
            _pad(
                deep_seq,
                padding_value=padding_value,
                shape=shape,
                _current_depth=_current_depth + 1,
            )
            for deep_seq in seq
        ]
        if len(deep_seqs_and_masks) > 0:
            deep_seqs, deep_mask = map(list, list(zip(*deep_seqs_and_masks)))
        else:
            deep_seqs, deep_mask = [], []

        pad_vec = [padding_value]
        zero_vec = [0]
        for l in reversed(shape[_current_depth + 1 :]):
            pad_vec = pad_vec * l
            zero_vec = zero_vec * l
            pad_vec = [pad_vec]
            zero_vec = [zero_vec]
        pad_vec = pad_vec * (shape[_current_depth] - len(seq))
        mask = deep_mask + zero_vec * (shape[_current_depth] - len(seq))
        return deep_seqs + pad_vec, mask


class Selector:
    def __init__(
        self,
        name,
        origin=None,
        mapping=None,
        dtype=None,
        device=None,
        padding=False,
        padding_value=0,
        padding_mask=False,
    ):
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
    def __init__(self, instances, selectors, sort_key=None):
        self.instances = [instance if isinstance(instance, dict) else instance.dict() for instance in instances]
        assert all(type(selector) in [Selector, dict] for selector in selectors)
        selectors = [selector if isinstance(selector, Selector) else Selector(**selector) for selector in selectors]
        assert len(selectors) == len(set(s.name for s in selectors)), "cannot use a same name multiple times."
        self.selectors = list(selectors)
        self.sort_key = sort_key

    def __getitem__(self, idx):
        instance = self.instances[idx]
        return {selector.name: selector.select(instance) for selector in self.selectors}

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
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            *args,
            **kwargs,
        )
