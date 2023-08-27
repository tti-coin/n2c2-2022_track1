# %%
_TWO_BECAUSE_OF_SPECIAL_TOKEN = 2

import re as _re
import math as _math
import collections as _collections
import enum as _enum
from typing import Optional as _Optional, Any as _Any, Annotated as _Annotated, Generic as _Generic, TypeVar as _TypeVar, Iterable as _Iterable, Callable as _Callable

import pydantic as _pydantic
import numpy as _numpy
import transformers as _transformers

from utils.logging_util import make_logger as _make_logger
_logger, _log_once, change_logging_level = _make_logger(__name__, default_level="WARNING")

# %%

# %%
class TokenizerInterface:
    init_kwargs:dict[str,_Any] = dict()
    def __call__(self, *args, **kwargs) -> dict:
        raise NotImplementedError("TokenizerInterface.__call__")
    def build_inputs_with_special_tokens(self, *args, **kwargs) -> list[int]:
        raise NotImplementedError("TokenizerInterface.build_inputs_with_special_tokens")

# %%
NERLabelType = int | str

class NERSpan(_pydantic.BaseModel):
    start: int
    end: int
    label: NERLabelType = 0
    id: _Optional[str] = None

    @_pydantic.model_validator(mode="after")
    def check_some(self):
        if not (self.start <= self.end):
            raise ValueError(f'self.start must be equal to or less than self.end, but {self.start=} {self.end=} ({self=})')
        return self

    model_config = _pydantic.ConfigDict(
        validate_assignment=True,
        frozen=True,
    )

    def without_id(self):
        return self.model_copy(update={"id":None}, deep=True)

    def map_label(self, label_mapping:dict[NERLabelType,int]|dict[int,NERLabelType]|_Callable[[NERLabelType,],int]|_Callable[[int,],NERLabelType]) -> "NERSpan":
        if isinstance(label_mapping, dict):
            label_mapping = label_mapping.__getitem__
        out = self.model_copy(update={"label":label_mapping(self.label)}, deep=True)
        return out


class NERTaggingScheme(str, _enum.Enum):
    BILOU = "bilou"
    BIO = "bio"
    TOKEN_LEVEL = "token_level"

class NERLabelScheme(str, _enum.Enum):
    SINGLE_LABEL = "single_label"
    MULTI_LABEL = "multi_label"
    SPAN_ONLY = "span_only"

class NERSpanTag(int, _enum.Enum):
    O = 0
    B = 1
    I = 2
    L = 3
    U = 4

class NERTruncationScheme(str, _enum.Enum):
    NONE = "none"
    TRUNCATE = "truncate"
    SPLIT = "split"

class NERSpanFittingScheme(str, _enum.Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class NERSplitInfo(_pydantic.BaseModel):
    num_splits:int
    split_index:int
    char_offset:int
    token_offset:int

    def is_initial_split(self) -> bool:
        return self.split_index == 0
    def is_last_split(self) -> bool:
        return self.num_splits == (self.split_index + 1)

class NERAdditionalInfo(_pydantic.BaseModel):
    split: _Optional[NERSplitInfo] = None
    forward_special_token_size: int = _pydantic.Field(default=0, description="number of tokens that are not included in originall text and placed in front of it (e.g., [BOS] or additional prompt).")
    backward_special_token_size: int = _pydantic.Field(default=0, description="number of tokens that are not included in originall text and placed behind it (e.g., [SEP] or additional prompt).")
    note: _Optional[_Any] = None

def serialize_without_default(mod:_pydantic.BaseModel) -> dict[str, _Any]:
    return mod.model_dump(exclude_defaults=True)


class NERInstance(_pydantic.BaseModel):
    text: str
    spans: list[NERSpan]
    id: _Any = None

    token_ids: _Optional[list[int]] = None
    token_spans: _Optional[list[NERSpan]] = None
    offset_mapping: _Optional[list[tuple[int,int]]] = None
    has_added_special_tokens: bool = False

    info: _Annotated[NERAdditionalInfo, _pydantic.PlainSerializer(serialize_without_default, when_used="unless-none")] = _pydantic.Field(default_factory=NERAdditionalInfo)

    model_config = _pydantic.ConfigDict(
        validate_assignment=True,
    )

    def only_label_(self, label_or_labels:NERLabelType|_Iterable[NERLabelType]) -> "NERInstance":
        if isinstance(label_or_labels, NERLabelType):
            label_or_labels = {label_or_labels}
        elif not isinstance(label_or_labels, set):
            label_or_labels = set(label_or_labels)
        self.spans = [span for span in self.spans if span.label in label_or_labels]
        if self.token_spans is not None:
            self.token_spans = [token_span for token_span in self.token_spans if token_span.label in label_or_labels]
        return self
    def only_label(self, label_or_labels:NERLabelType|_Iterable[NERLabelType], deep:bool=True) -> "NERInstance":
        out = self.model_copy(deep=deep)
        return out.only_label_(label_or_labels)

    def map_labels_(self, label_mapping:dict[NERLabelType,int]|dict[int,NERLabelType]|_Callable[[NERLabelType,],int]|_Callable[[int,],NERLabelType]) -> "NERInstance":
        self.spans = [span.map_label(label_mapping) for span in self.spans]
        if self.token_spans is not None:
            self.token_spans = [token_span.map_label(label_mapping) for token_span in self.token_spans]
        return self
    def map_labels(self, label_mapping:dict[NERLabelType,int]|dict[int,NERLabelType]|_Callable[[NERLabelType,],int]|_Callable[[int,],NERLabelType], deep:bool=True) -> "NERInstance":
        out = self.model_copy(deep=deep)
        return out.map_labels_(label_mapping)

    @classmethod
    def build(cls, text:str, spans:list[NERSpan], id:_Any=None, *, tokenizer:_Optional[TokenizerInterface]=None, add_special_tokens:_Optional[bool]=None, truncation:None|bool|NERTruncationScheme=None, max_length:_Optional[int]=None, stride:_Optional[int]=None, fit_token_span:_Optional[NERSpanFittingScheme]=None, add_split_idx_to_id:_Optional[bool]=None, return_non_truncated:_Optional[bool]=None, ignore_trim_offsets:_Optional[bool]=None, tokenizer_other_kwargs:_Optional[dict]=None):
        spans = [NERSpan.model_validate(span) for span in spans]
        spans = [span if type(span) is NERSpan else NERSpan(*span) for span in spans]
        out = cls(text=text, spans=spans, id=id)
        if tokenizer is not None:
            encode_func_args = dict()
            for key in ["add_special_tokens", "truncation", "max_length", "stride", "fit_token_span", "add_split_idx_to_id", "return_non_truncated", "ignore_trim_offsets", "tokenizer_other_kwargs"]:
                value = eval(key)
                if value is not None:
                    encode_func_args[key] = value
            out = out.encode_(tokenizer, **encode_func_args)
        return out

    def encode_(self, tokenizer:TokenizerInterface, *, add_special_tokens:bool=False, truncation:None|bool|NERTruncationScheme=NERTruncationScheme.NONE, max_length:_Optional[int]=None, stride:_Optional[int]=None, fit_token_span:NERSpanFittingScheme=NERSpanFittingScheme.MAXIMIZE, add_split_idx_to_id:bool=False, return_non_truncated:bool=False, ignore_trim_offsets:bool=False, tokenizer_other_kwargs:_Optional[dict]=None):
        assert not self.has_added_special_tokens

        if tokenizer.init_kwargs.get("trim_offsets", True) :
            _tokenizer_name_for_message = getattr(tokenizer, 'name_or_path', None)
            _tokenizer_name_for_message = f'"{_tokenizer_name_for_message}"' if _tokenizer_name_for_message is not None else "..."
            message = f"Tokenizer's `trim_offsets` parameter should be False, however, is set to be True or not set. This will lead the mismatch at offset_mapping. Consider to initialize tokenizer with `trim_offsets=False` (e.g., `AutoTokenizer.from_pretrained({_tokenizer_name_for_message}, trim_offsets=False)`)."
            if not ignore_trim_offsets:
                message += " To ignore this warning, set `ignore_trim_offsets=True` at calling `NERInstance.build` or `NERInstance.encode_`."
                raise ValueError(message)
            else:
                _log_once(message=message, level="WARNING")

        if isinstance(truncation, bool):
            if truncation:
                truncation = NERTruncationScheme.TRUNCATE
            else:
                truncation = NERTruncationScheme.NONE
        elif truncation is None:
            truncation = NERTruncationScheme.NONE

        if tokenizer_other_kwargs is None:
            tokenizer_other_kwargs = dict()
        else:
            tokenizer_other_kwargs = dict(tokenizer_other_kwargs)

        assert "truncation" not in tokenizer_other_kwargs, '"truncation" option can be only given at the direct argument of encode_ function.'
        for key in ["add_special_tokens", "max_length"]:
            if key in tokenizer_other_kwargs:
                _log_once(message=f'found the argument "{key}" in "tokenizer_other_kwargs". change to giving the argument directly to the fucntion.', level="WARNING")
                exec(f'{key} = tokenizer_other_kwargs["{key}"]')
        tokenizer_other_kwargs["add_special_tokens"] = False
        if max_length is not None:
            tokenizer_other_kwargs["max_length"] = max_length

        if truncation == NERTruncationScheme.NONE:
            pass
        elif truncation == NERTruncationScheme.TRUNCATE:
            if return_non_truncated: raise NotImplementedError("return_non_truncated")
            assert max_length is not None
            tokenizer_other_kwargs["truncation"] = True
            if add_special_tokens:
                max_length = max_length - _TWO_BECAUSE_OF_SPECIAL_TOKEN
                tokenizer_other_kwargs["max_length"] = tokenizer_other_kwargs["max_length"] - _TWO_BECAUSE_OF_SPECIAL_TOKEN
        elif truncation == NERTruncationScheme.SPLIT:
            assert max_length is not None
            assert stride is not None
            if add_special_tokens:
                assert 0 < stride <= (max_length - _TWO_BECAUSE_OF_SPECIAL_TOKEN), (stride, max_length)
            else:
                assert 0 < stride <= max_length, (stride, max_length)
            tokenizer_other_kwargs["truncation"] = False
            tokenizer_other_kwargs["max_length"] = None
        else:
            raise ValueError(truncation)

        enc = tokenizer(self.text, return_offsets_mapping=True, **tokenizer_other_kwargs)
        self.token_ids = enc["input_ids"]
        self.offset_mapping = enc["offset_mapping"]

        start_to_token_id = {start:i for i,[start,end] in enumerate(self.offset_mapping)}
        end_to_token_id = {end:i for i,[start,end] in enumerate(self.offset_mapping)}

        token_spans = list()
        offset_mapping_start_with_sentinel = [start for start,_ in self.offset_mapping] + [self.offset_mapping[-1][1]]
        offset_mapping_end_with_sentinel = [0] + [end for _,end in self.offset_mapping]
        for span in self.spans:
            if fit_token_span == NERSpanFittingScheme.MAXIMIZE:
                for st in range(len(self.token_ids)):
                    if offset_mapping_end_with_sentinel[st] <= span.start < offset_mapping_end_with_sentinel[st+1]:
                        break
                else:
                    continue

                for et_minus_one in range(max(0,st-1), len(self.token_ids)):
                    if offset_mapping_start_with_sentinel[et_minus_one] < span.end <= offset_mapping_start_with_sentinel[et_minus_one+1]:
                        break
                else:
                    # continue
                    # raise ValueError(self)
                    pass

                token_spans.append(NERSpan(start=st,end=et_minus_one+1, label=span.label, id=span.id))

            elif fit_token_span == NERSpanFittingScheme.MINIMIZE:
                st = start_to_token_id.get(span.start, None)
                et_minus_one = end_to_token_id.get(span.end, None)
                if (st is not None) and (et_minus_one is not None):
                    token_spans.append(NERSpan(start=st,end=et_minus_one+1, label=span.label, id=span.id))

            else:
                raise ValueError(fit_token_span)

        self.token_spans = token_spans

        if truncation == NERTruncationScheme.SPLIT:
            if add_special_tokens:
                outs = self._split_with_size(max_length=max_length-_TWO_BECAUSE_OF_SPECIAL_TOKEN, stride=stride, add_split_idx_to_id=add_split_idx_to_id)
                for out in outs:
                    out.with_special_tokens_(tokenizer=tokenizer)
            else:
                outs = self._split_with_size(max_length=max_length, stride=stride, add_split_idx_to_id=add_split_idx_to_id)
            if return_non_truncated:
                return outs, self
            else:
                return outs

        else:
            if add_special_tokens:
                self.with_special_tokens_(tokenizer=tokenizer)
            return self

    def with_special_tokens_(self, tokenizer:TokenizerInterface):
        assert not self.has_added_special_tokens, f'already special tokens are added. id:{self.id}'

        token_len_wo_sp_tokens = len(self.token_ids)
        new_input_ids = tokenizer.build_inputs_with_special_tokens(self.token_ids)
        assert len(new_input_ids) == token_len_wo_sp_tokens + _TWO_BECAUSE_OF_SPECIAL_TOKEN

        new_offset_mapping, new_token_spans = self._padded_mappings_and_token_spans(num_forward_padding=_TWO_BECAUSE_OF_SPECIAL_TOKEN//2, num_backward_padding=_TWO_BECAUSE_OF_SPECIAL_TOKEN//2)

        self.token_ids = new_input_ids
        self.offset_mapping = new_offset_mapping
        self.token_spans = new_token_spans
        self.has_added_special_tokens = True
        self.info.forward_special_token_size = _TWO_BECAUSE_OF_SPECIAL_TOKEN // 2
        self.info.backward_special_token_size = _TWO_BECAUSE_OF_SPECIAL_TOKEN // 2
        return self

    def with_special_tokens(self, tokenizer:TokenizerInterface):
        out = self.model_copy(deep=True)
        return out.with_special_tokens_(tokenizer=tokenizer)

    def with_query_and_special_tokens_(self, tokenizer:TokenizerInterface, encoded_query:list[int], max_length:int):
        assert not self.has_added_special_tokens, f'must be without special tokens. id:{self.id}'

        new_input_ids = tokenizer.build_inputs_with_special_tokens(encoded_query, self.token_ids)
        exceeded_len = max(0, len(new_input_ids) - max_length)
        if exceeded_len > 0:
            self._truncate_back_tokens_(size=exceeded_len)
            new_input_ids = tokenizer.build_inputs_with_special_tokens(encoded_query, self.token_ids)

        assert (self.token_ids[-1] == new_input_ids[-2]) and (self.token_ids[-1] != new_input_ids[-1]) # ... SOME_TOKEN [SEP]
        num_backward_padding = 1
        num_forward_padding = len(new_input_ids) - num_backward_padding - len(self.token_ids)
        new_offset_mapping, new_token_spans = self._padded_mappings_and_token_spans(num_forward_padding=num_forward_padding, num_backward_padding=num_backward_padding)
        assert len(new_offset_mapping) == len(new_input_ids)

        self.token_ids = new_input_ids
        self.offset_mapping = new_offset_mapping
        self.token_spans = new_token_spans
        self.has_added_special_tokens = True
        self.info.forward_special_token_size = (_TWO_BECAUSE_OF_SPECIAL_TOKEN // 2) + len(encoded_query) + 1 # [BOS] prompt [SEP] ...

        return self

    def with_query_and_special_tokens(self, tokenizer:TokenizerInterface, encoded_query:list[int], max_length:int):
        out = self.model_copy(deep=True)
        return out.with_query_and_special_tokens_(tokenizer=tokenizer, encoded_query=encoded_query, max_length=max_length)

    def _split_with_size(self, max_length, stride, add_split_idx_to_id:bool):
        assert not self.has_added_special_tokens, f'must be without special tokens. id:{self.id}'

        num_splits = 1 + _math.ceil(max(0, len(self.token_ids) - max_length) / stride)

        if num_splits == 1:
            outs = [self.model_copy(deep=True)]
            outs[0].info.split = NERSplitInfo(num_splits=num_splits, split_index=0, char_offset=0, token_offset=0)
        else:
            outs = [self.model_copy(deep=True) for _ in range(num_splits)]
            for split_idx, out in enumerate(outs):
                token_start = stride * split_idx
                token_end = token_start + max_length
                out.token_ids = out.token_ids[token_start:token_end]
                out.offset_mapping = out.offset_mapping[token_start:token_end]

                char_start,_ = out.offset_mapping[0]
                _,char_end = out.offset_mapping[-1]
                out.text = out.text[char_start:char_end]

                out.offset_mapping = [(start-char_start, end-char_start) for [start,end] in out.offset_mapping]

                out.spans = [span for span in out.spans if (char_start<=span.start) and (span.end <= char_end)]
                out.spans = [span.model_copy(update={"start":span.start-char_start, "end":span.end-char_start}) for span in out.spans]

                out.token_spans = [token_span for token_span in out.token_spans if (token_start<=token_span.start) and (token_span.end <= token_end)]
                out.token_spans = [token_span.model_copy(update={"start":token_span.start-token_start, "end":token_span.end-token_start}) for token_span in out.token_spans]

                out.info.split = NERSplitInfo(num_splits=num_splits, split_index=split_idx, char_offset=char_start, token_offset=token_start)

        if add_split_idx_to_id:
            outs = [out.model_copy(update={"id":out.id + f"_{split_idx}"}) for split_idx, out in enumerate(outs)]

        return outs

    def _padded_mappings_and_token_spans(self, num_forward_padding:int, num_backward_padding:int):
        _,last_position = self.offset_mapping[-1]
        padded_offset_mapping = [(0,0) for _ in range(num_forward_padding)] + self.offset_mapping + [(last_position,last_position) for _ in range(num_backward_padding)]

        padded_token_spans = list()
        for span in self.token_spans:
            new_s = span.start + num_forward_padding
            new_e = span.end + num_forward_padding
            copied_span = span.model_copy(update={"start":new_s, "end":new_e})
            padded_token_spans.append(copied_span)

        return padded_offset_mapping, padded_token_spans

    def _truncate_back_tokens_(self, size:int):
        assert not self.has_added_special_tokens, 'cannot truncate after special tokens has been added. id:{self.id}'
        assert size >= 0
        self.token_ids = self.token_ids[:-size]
        self.offset_mapping = self.offset_mapping[:-size]

        new_token_len = len(self.token_ids)
        new_token_spans = list()
        for span in self.token_spans:
            # if span.end > new_token_len:
            #     continue
            if span.start >= new_token_len:
                continue
            if span.end > new_token_len:
                span = span.model_copy(update={"end":new_token_len})

            new_token_spans.append(span)

        self.token_spans = new_token_spans
        return self


    @_pydantic.validate_call
    def get_sequence_label(self, tagging_scheme:NERTaggingScheme=NERTaggingScheme.BILOU, label_scheme:NERLabelScheme=NERLabelScheme.SINGLE_LABEL, label_to_id:_Optional[dict[NERLabelType,int]|_Callable[[NERLabelType,],int]]=None, num_class_without_negative:_Optional[int]=None, strict:bool=True) -> list[int] | list[list[int]]:
        """
        output := [label_0, label_1, label_2, ...]

        if label_scheme==SINGLE_LABEL:
            label_t := int
            t: timestep

            if tagging_scheme==BILOU:
                label_t \\in {0:O, 1:B-class_0, 2:I-class_0, 3:L-class_0, 4:U-class_0, 5:B-class_1, 6:I-class_1, ..., 4*n+1:B-class_n, 4*n+2:I-class_n, 4*n+3:L-class_n, 4*n+4:U-class_n, ...}
            if tagging_scheme==BIO:
                label_t \\in {0:O, 1:B-class_0, 2:I-class_0, 3:B-class_1, 4:I-class_1, ..., 2*n+1:B-class_n, 2*n+2:I-class_n, ...}
            if tagging_scheme==TOKEN_LEVEL:
                label_t \\in {0:O, 1:class_0, 2:class_1, ..., n+1:class_n, ...}

        if label_scheme==MULTI_LABEL:
            label_t := [label_t_class_0, label_t_class_1, ..., label_t_class_N]
            N: num_class_without_negative

            label_t_class_k := int
            k: the index of the class

            if tagging_scheme==BILOU:
                label_t_class_k \\in {0:O, 1:B, 2:I, 3:L, 4:U}
            if tagging_scheme==BIO:
                label_t_class_k \\in {0:O, 1:B, 2:I}
            if tagging_scheme==TOKEN_LEVEL:
                label_t_class_k \\in {0:Negative, 1:Positive}

        if label_scheme==SPAN_ONLY:
            label_t := int
            t: timestep

            if tagging_scheme==BILOU:
                label_t \\in {0:O, 1:B, 2:I, 3:L, 4:U}
            if tagging_scheme==BIO:
                label_t \\in {0:O, 1:B, 2:I}
            if tagging_scheme==TOKEN_LEVEL:
                label_t \\in {0:Negative, 1:Positive}


        label_to_id must be specified except for label_scheme==SPAN_ONLY or all labels of token_spans are integers.

        num_class_without_negative must be specified when label_scheme==MULTI_LABEL.
        """
        # NOTE: comment out this check because this check is performed at converting spans to labels.
        # if (label_to_id is None) and (label_scheme != NERLabelScheme.SPAN_ONLY):
        #     assert all([isinstance(span.label, int) for span in self.token_spans]), f'labels must be integers when label_to_id is not specified. {self.token_spans}'

        if label_scheme == NERLabelScheme.MULTI_LABEL:
            assert num_class_without_negative is not None, "num_class_without_negative must be specified under the multi-labelling setting."

        if label_scheme in [NERLabelScheme.SINGLE_LABEL, NERLabelScheme.SPAN_ONLY]:
            out = [int(NERSpanTag.O) for _ in range(len(self.token_ids))]
        elif label_scheme == NERLabelScheme.MULTI_LABEL:
            transposed_out = [[int(NERSpanTag.O) for _ in range(len(self.token_ids))] for _ in range(num_class_without_negative)]
        else:
            raise ValueError(label_scheme)

        for span in self.token_spans:
            if label_scheme == NERLabelScheme.SPAN_ONLY:
                pass
            elif label_to_id is None:
                if not isinstance(span.label, int):
                    raise ValueError(f'labels must be integers when label_to_id is not specified. {span=}')
                label_id = span.label
            else:
                if isinstance(label_to_id, dict):
                    label_to_id = label_to_id.__getitem__
                label_id = label_to_id(span.label)

            if tagging_scheme == NERTaggingScheme.BILOU:
                if label_scheme == NERLabelScheme.SINGLE_LABEL:
                    target_out = out
                    class_offset = label_id * 4 # 4 <- len({B, I, L, U})
                elif label_scheme == NERLabelScheme.SPAN_ONLY:
                    target_out = out
                    class_offset = 0
                elif label_scheme == NERLabelScheme.MULTI_LABEL:
                    target_out = transposed_out[label_id]
                    class_offset = 0
                else:
                    raise ValueError(label_scheme)

                if strict:
                    assert all([t == NERSpanTag.O for t in target_out[span.start:span.end]]), f'there must not be overlapped spans: {self.token_spans}'

                if span.start + 1 == span.end:
                    target_out[span.start] = NERSpanTag.U + class_offset
                else:
                    target_out[span.start] = NERSpanTag.B + class_offset
                    target_out[span.end-1] = NERSpanTag.L + class_offset
                    for i in range(span.start+1, span.end-1):
                        target_out[i] = NERSpanTag.I + class_offset

            elif tagging_scheme == NERTaggingScheme.BIO:
                if label_scheme == NERLabelScheme.SINGLE_LABEL:
                    target_out = out
                    class_offset = label_id * 2 # 2 <- len({B, I})
                elif label_scheme == NERLabelScheme.SPAN_ONLY:
                    target_out = out
                    class_offset = 0
                elif label_scheme == NERLabelScheme.MULTI_LABEL:
                    target_out = transposed_out[label_id]
                    class_offset = 0
                else:
                    raise ValueError(label_scheme)

                if strict:
                    assert all([t == NERSpanTag.O for t in target_out[span.start:span.end]]), f'there must not be overlapped spans: {self.token_spans}'

                target_out[span.start] = NERSpanTag.B + class_offset
                for i in range(span.start+1, span.end):
                    target_out[i] = NERSpanTag.I + class_offset

            elif tagging_scheme == NERTaggingScheme.TOKEN_LEVEL:
                if label_scheme == NERLabelScheme.SINGLE_LABEL:
                    target_out = out
                    class_offset = label_id
                elif label_scheme == NERLabelScheme.SPAN_ONLY:
                    target_out = out
                    class_offset = 0
                elif label_scheme == NERLabelScheme.MULTI_LABEL:
                    target_out = transposed_out[label_id]
                    class_offset = 0
                else:
                    raise ValueError(label_scheme)

                if strict:
                    assert all([t == NERSpanTag.O for t in target_out[span.start:span.end]]), f'there must not be overlapped spans: {self.token_spans}'

                for i in range(span.start, span.end):
                    target_out[i] = 1 + class_offset

            else:
                raise ValueError(tagging_scheme)

        if label_scheme == NERLabelScheme.MULTI_LABEL:
            out = [list(v) for v in zip(*transposed_out)]

        return out

    def decode_token_span_to_char_span(self, span:NERSpan|list[NERSpan], strip:bool=True, recover_split:bool=False, is_token_span_starting_after_special_tokens:bool=False) -> NERSpan | list[NERSpan]:
        if not isinstance(span, NERSpan):
            return [self.decode_token_span_to_char_span(s, strip=strip, recover_split=recover_split, is_token_span_starting_after_special_tokens=is_token_span_starting_after_special_tokens) for s in span]

        skip_offset = 0
        if is_token_span_starting_after_special_tokens:
            skip_offset = skip_offset + self.info.forward_special_token_size

        char_start, _ = self.offset_mapping[skip_offset+span.start]
        if span.start == span.end:
            char_end = char_start
        else:
            _,char_end = self.offset_mapping[skip_offset+span.end-1]
        out = NERSpan(start=char_start, end=char_end, label=span.label, id=span.id)

        if strip:
            out = self.strip_char_spans(out)
        if recover_split:
            out = self.recover_char_spans_wrt_split_offset(out)
        return out

    def strip_char_spans(self, span:NERSpan|list[NERSpan]) -> NERSpan | list[NERSpan]:
        if not isinstance(span, NERSpan):
            return [self.strip_char_spans(s) for s in span]

        forward_blank_span = _re.match("^\\s*", self.text[span.start:span.end]).span()
        forward_blank_size = forward_blank_span[1] - forward_blank_span[0]
        backward_blank_span = _re.search("\\s*$", self.text[span.start+forward_blank_size:span.end]).span()
        backward_blank_size = backward_blank_span[1] - backward_blank_span[0]
        out = span.model_copy(update={"start":span.start+forward_blank_size, "end":span.end-backward_blank_size})
        return out

    def recover_char_spans_wrt_split_offset(self, span:NERSpan|list[NERSpan]) -> NERSpan | list[NERSpan]:
        if not isinstance(span, NERSpan):
            return [self.recover_char_spans_wrt_split_offset(s) for s in span]

        split_offset = self.info.split.char_offset
        out = span.model_copy(update={"start":span.start+split_offset, "end":span.end+split_offset})
        return out




# %%
def convert_sequence_label_to_spans(sequence_label:list[int]|list[list[int]], tagging_scheme:NERTaggingScheme=NERTaggingScheme.BILOU, label_scheme:NERLabelScheme=NERLabelScheme.SINGLE_LABEL, id_to_label:_Optional[dict[int,NERLabelType]|_Callable[[int,],NERLabelType]]=None) -> list[NERSpan]:
    if label_scheme in [NERLabelScheme.SINGLE_LABEL, NERLabelScheme.SPAN_ONLY]:
        if tagging_scheme == NERTaggingScheme.TOKEN_LEVEL:
            outs = [NERSpan(start=t,end=t+1,label=label-1) for t, label in enumerate(sequence_label) if label != 0]

        elif tagging_scheme == NERTaggingScheme.BILOU:
            outs = list()
            start = None
            class_ = None

            for t, label in enumerate(sequence_label):
                if label == 0: # "O"
                    # assert start is None
                    if start is not None:
                        _logger.warning(f'span ends without "L" at timestep {t-1}. treat as termination.: {sequence_label}')
                        outs.append(NERSpan(start=start, end=t, label=class_))
                        start = None
                        class_ = None

                elif (label-1) % 4 == 0: # "B"
                    # assert start is None
                    if start is not None:
                        _logger.warning(f'span ends without "L" at timestep {t-1}. treat as termination.: {sequence_label}')
                        outs.append(NERSpan(start=start, end=t, label=class_))
                        # start = None
                        # class_ = None
                    start = t
                    class_ = (label-1) // 4

                elif (label-1) % 4 == 1: # "I"
                    # assert start is not None
                    if start is None:
                        _logger.warning(f'span starts without "B" at timestep {t}. treat as new beginning.: {sequence_label}')
                        start = t
                        class_ = (label-1) // 4

                    # assert class_ == ((label-1) // 4)
                    if class_ != ((label-1) // 4):
                        _logger.warning(f'span class is incosistent at timestep {t}. treat as new beginning: {sequence_label}')
                        outs.append(NERSpan(start=start, end=t, label=class_))
                        # start = None
                        # class_ = None
                        start = t
                        class_ = (label-1) // 4

                elif (label-1) % 4 == 2: # "L"
                    # assert start is not None
                    if start is None:
                        _logger.warning(f'span starts without "B" at timestep {t}. treat as new beginning.: {sequence_label}')
                        start = t
                        class_ = (label-1) // 4

                    # assert class_ == ((label-1) // 4)
                    if class_ != ((label-1) // 4):
                        _logger.warning(f'span class is incosistent at timestep {t}. treat as new beginning: {sequence_label}')
                        outs.append(NERSpan(start=start, end=t, label=class_))
                        # start = None
                        # class_ = None
                        start = t
                        class_ = (label-1) // 4

                    outs.append(NERSpan(start=start, end=t+1, label=class_))
                    start = None
                    class_ = None

                elif (label-1) % 4 == 3: # "U"
                    # assert start is None
                    if start is not None:
                        _logger.warning(f'span ends without "L" at timestep {t-1}. treat as termination.: {sequence_label}')
                        outs.append(NERSpan(start=start, end=t, label=class_))
                        start = None
                        class_ = None
                    outs.append(NERSpan(start=t, end=t+1, label=(label-1)//4))

                else:
                    raise ValueError(label)

            # check the last span is terminated
            if start is not None:
                _logger.warning(f'span ends without "L" at timestep {t}. treat as termination.: {sequence_label}')
                outs.append(NERSpan(start=start, end=t+1, label=class_))
                start = None
                class_ = None

        elif tagging_scheme == NERTaggingScheme.BIO:
            outs = list()
            start = None
            class_ = None

            for t, label in enumerate(sequence_label):
                if label == 0: # "O"
                    if start is not None:
                        outs.append(NERSpan(start=start, end=t, label=class_))
                        start = None
                        class_ = None

                elif (label-1) % 2 == 0: # "B"
                    if start is not None:
                        outs.append(NERSpan(start=start, end=t, label=class_))
                        # start = None
                        # class_ = None
                    start = t
                    class_ = (label-1) // 2

                elif (label-1) % 2 == 1: # "I"
                    # assert start is not None
                    if start is None:
                        _logger.warning(f'span starts without "B" at timestep {t}. treat as new beginning.: {sequence_label}')
                        start = t
                        class_ = (label-1) // 2

                    # assert class_ == ((label-1) // 2)
                    if class_ != ((label-1) // 2):
                        _logger.warning(f'span class is incosistent at timestep {t}. treat as new beginning: {sequence_label}')
                        outs.append(NERSpan(start=start, end=t, label=class_))
                        # start = None
                        # class_ = None
                        start = t
                        class_ = (label-1) // 2

                else:
                    raise ValueError(label)

            # check the last span is terminated
            if start is not None:
                outs.append(NERSpan(start=start, end=t+1, label=class_))
                start = None
                class_ = None

        else:
            raise ValueError(tagging_scheme)

    elif label_scheme == NERLabelScheme.MULTI_LABEL:
        num_class = len(sequence_label[0])
        outs = list()
        for c in range(num_class):
            sequence_label_class_c = [labels[c] for labels in sequence_label]
            spans_class_c = convert_sequence_label_to_spans(sequence_label=sequence_label_class_c, tagging_scheme=tagging_scheme, label_scheme=NERLabelScheme.SPAN_ONLY)
            outs.extend([NERSpan(start=span.start, end=span.end, label=c) for span in spans_class_c])

    else:
        raise ValueError(label_scheme)

    if id_to_label is not None:
        if label_scheme == NERLabelScheme.SPAN_ONLY:
            _log_once(message=f'id_to_label is ignored when label_scheme==NERLabelScheme.SPAN_ONLY', level="INFO")
        else:
            outs = [span.map_label(id_to_label) for span in outs]
    return outs

@_pydantic.validate_call
def viterbi_decode(logits_sequence:list[float]|list[list[float]]|list[list[list[float]]], tagging_scheme:NERTaggingScheme=NERTaggingScheme.BILOU, label_scheme:NERLabelScheme=NERLabelScheme.SINGLE_LABEL, scalar_logit_for_token_level:bool=False, as_spans:bool=False, id_to_label:_Optional[dict[int,NERLabelType]|_Callable[[int,],NERLabelType]]=None) -> list[int] | list[list[int]] | list[NERSpan]:
    """
    input: logits_sequence
    - 2D or 3D float list. shape==[seq_len, [num_class,] num_label].

    output: sequence_label

    if (scalar_logit_for_token_level == True) and (tagging_scheme == TOKEN_LEVEL) and (label_scheme in [MULTI_LABEL, SPAN_ONLY]),
    the expected shape of logits_sequence is [seq_len, [num_class,]] and each value logits_sequence[i[,j]] will be treated as a logit of the positive probability.
    otherwise, the shape should be [seq_len, [num_class,] 2] (2:(logit_negative, logit_positive)).


    if label_scheme==SINGLE_LABEL:
        logits_sequence := [logits_time-step_0, logits_time-step_1, ...]

        if tagging_scheme==BILOU:
            logits_time-step_t := [logit_O, logit_B-class_0, logit_I-class_0, logit_L-class_0, logit_U-class_0, logit_B-class_1, logit_I-class_1, ...]
            len(logits_time-step_t) == num_class*4 + 1 (4 <= {B,I,L,U})

        if tagging_scheme==BIO
            logits_time-step_t := [logit_O, logit_B-class_0, logit_I-class_0, logit_B-class_1, logit_I-class_1, ...]
            len(logits_time-step_t) == num_class*2 + 1 (2 <= {B,I})

        if tagging_scheme==TOKEN_LEVEL
            logits_time-step_t := [logit_O, logit_class_0, logit_class_1, ...]
            len(logits_time-step_t) == num_class*1 + 1 (1 <= {Positive})


    if label_scheme==MULTI_LABEL:
        logits_sequence := [logits_time-step_0, logits_time-step_1, ...]
        logits_time-step_t := [logits_class_0, logits_class_1, ...]

        if tagging_scheme==BILOU:
            logits_class_c := [logit_O, logit_B, logit_I, logit_L, logit_U]

        if tagging_scheme==BIO
            logits_class_c := [logit_O, logit_B, logit_I]

        if tagging_scheme==TOKEN_LEVEL
            logits_class_c := [logit_Negative, logit_Positive]


    if label_scheme==SPAN_ONLY:
        logits_sequence := [logits_time-step_0, logits_time-step_1, ...]

        if tagging_scheme==BILOU:
            logits_time-step_t := [logit_O, logit_B, logit_I, logit_L, logit_U]

        if tagging_scheme==BIO
            logits_time-step_t := [logit_O, logit_B, logit_I]

        if tagging_scheme==TOKEN_LEVEL
            logits_time-step_t := [logit_Negative, logit_Positive]
    """
    if as_spans:
        sequence_label = viterbi_decode(logits_sequence=logits_sequence, tagging_scheme=tagging_scheme, label_scheme=label_scheme, scalar_logit_for_token_level=scalar_logit_for_token_level, as_spans=False)
        return convert_sequence_label_to_spans(sequence_label=sequence_label, tagging_scheme=tagging_scheme, label_scheme=label_scheme, id_to_label=id_to_label)


    logits_sequence:_numpy.ndarray = _numpy.array(logits_sequence, dtype=_numpy.float32)

    if label_scheme == NERLabelScheme.SINGLE_LABEL:
        assert len(logits_sequence.shape) == 2, logits_sequence.shape

        if tagging_scheme == NERTaggingScheme.BILOU:
            num_class = (logits_sequence.shape[1] - 1) // 4
        elif tagging_scheme == NERTaggingScheme.BIO:
            num_class = (logits_sequence.shape[1] - 1) // 2
        elif tagging_scheme == NERTaggingScheme.TOKEN_LEVEL:
            num_class = logits_sequence.shape[1] - 1
        else:
            raise ValueError(tagging_scheme)
    elif label_scheme == NERLabelScheme.MULTI_LABEL:
        if (tagging_scheme == NERTaggingScheme.TOKEN_LEVEL) and scalar_logit_for_token_level:
            assert len(logits_sequence.shape) == 2, logits_sequence.shape
        else:
            assert len(logits_sequence.shape) == 3, logits_sequence.shape

        decoded = list()
        for class_i in range(logits_sequence.shape[1]):
            logit_sequence_class_i = logits_sequence[:,class_i]
            decoded_class_i = viterbi_decode(logits_sequence=logit_sequence_class_i, tagging_scheme=tagging_scheme, label_scheme=NERLabelScheme.SPAN_ONLY, scalar_logit_for_token_level=scalar_logit_for_token_level)
            decoded.append(decoded_class_i)
        decoded = list(zip(*decoded)) # [num_class, seq_len] -> [seq_len, num_class]
        return decoded
    elif label_scheme == NERLabelScheme.SPAN_ONLY:
        if (tagging_scheme == NERTaggingScheme.TOKEN_LEVEL) and scalar_logit_for_token_level:
            assert len(logits_sequence.shape) == 1, logits_sequence.shape
        else:
            assert len(logits_sequence.shape) == 2, logits_sequence.shape

        num_class = 1
    else:
        raise ValueError(label_scheme)

    if tagging_scheme == NERTaggingScheme.BILOU:
        num_label_wo_O = 4

        transition_paths = [None for _ in range(1+num_label_wo_O*num_class)]
        terminations = [0] + [num_label_wo_O*c + L_or_U for c in range(num_class) for L_or_U in [3,4]]
        transition_paths[0] = "from-termination"
        for c in range(num_class):
            # B
            transition_paths[num_label_wo_O*c+1] = "from-termination"
            # I
            transition_paths[num_label_wo_O*c+2] = [num_label_wo_O*c+1, num_label_wo_O*c+2] # from B or I
            # L
            transition_paths[num_label_wo_O*c+3] = [num_label_wo_O*c+1, num_label_wo_O*c+2] # from B or I
            # U
            transition_paths[num_label_wo_O*c+4] = "from-termination"

    elif tagging_scheme == NERTaggingScheme.BIO:
        num_label_wo_O = 2

        transition_paths = [None for _ in range(1+num_label_wo_O*num_class)]
        terminations = list(range(1+num_label_wo_O*num_class))
        transition_paths[0] = "from-termination"
        for c in range(num_class):
            # B
            transition_paths[num_label_wo_O*c+1] = "from-termination"
            # I
            transition_paths[num_label_wo_O*c+2] = [num_label_wo_O*c+1, num_label_wo_O*c+2] # from B or I

    elif tagging_scheme == NERTaggingScheme.TOKEN_LEVEL:
        if label_scheme == NERLabelScheme.SPAN_ONLY and scalar_logit_for_token_level:
            return (logits_sequence > 0).astype(_numpy.int32).tolist()
        else:
            return logits_sequence.argmax(-1).tolist()

    else:
        raise ValueError(tagging_scheme)

    # init
    past_trajs_history = list()
    cumsum_logits_history = list()
    cumsum_logits_history.append([0.0] + [-_numpy.inf] * (num_label_wo_O*num_class)) # sentinel

    # forward
    for timestep in range(logits_sequence.shape[0]):
        logits = logits_sequence[timestep]

        prev_cumsum_logits = cumsum_logits_history[-1]
        cumsum_logits = [None for _ in range(1+num_label_wo_O*num_class)]
        past_trajs = [None for _ in range(1+num_label_wo_O*num_class)]

        prev_best_terminal_index = max(terminations, key=lambda i:prev_cumsum_logits[i])
        prev_best_terminal_cumsum_logit = prev_cumsum_logits[prev_best_terminal_index]

        for l in range(1+num_label_wo_O*num_class):
            if transition_paths[l] == "from-termination": # O/B/U
                past_trajs[l] = prev_best_terminal_index
                cumsum_logits[l] = logits[l] + prev_best_terminal_cumsum_logit
            else: # I/L
                past_traj = max(transition_paths[l], key=lambda i:prev_cumsum_logits[i])
                past_trajs[l] = past_traj
                cumsum_logits[l] = logits[l] + prev_cumsum_logits[past_traj]

        cumsum_logits_history.append(cumsum_logits)
        past_trajs_history.append(past_trajs)

    # backward
    last_cumsum_logits = cumsum_logits_history[-1]
    traj = max(terminations, key=lambda i:last_cumsum_logits[i])
    reverse_trajectory = list()
    for past_trajs in reversed(past_trajs_history):
        reverse_trajectory.append(traj)
        traj = past_trajs[traj]
    sequence_label = list(reversed(reverse_trajectory))
    return sequence_label

def merge_spans(spans:list[NERSpan]) -> list[NERSpan]:
    if len(spans) == 0:
        return list()
    max_end = max([0] + [span.end for span in spans])
    initial_table = [False for _ in range(max_end)]
    label_class_to_table = _collections.defaultdict(lambda: list(initial_table))
    for span in spans:
        for i in range(span.start, span.end):
            label_class_to_table[span.label][i] = True

    outs = list()
    for label, table in label_class_to_table.items():
        t = 0
        while t < max_end:
            if table[t]:
                start = t
                while (t < max_end) and table[t]:
                    t += 1
                end = t
                outs.append(NERSpan(start=start,end=end,label=label))
                continue
            else:
                t += 1
                continue

    return outs


# %%
class NERAggregateScheme(str, _enum.Enum):
    NONE = "none"
    MEAN = "mean"
    SET = "set"

_ElementT = _TypeVar("_ElementT")
class NERMultiElementSparseSequence(_Generic[_ElementT]):
    def __init__(self) -> None:
        self._body: list[list[_ElementT]] = list()

    def __len__(self) -> int:
        return len(self._body)

    def _expand(self, max_size) -> "NERMultiElementSparseSequence":
        current_size = len(self)
        if current_size < max_size:
            new_values = [list() for _ in range(current_size, max_size)]
            self._body.extend(new_values)
        return self

    def values_at(self, idx:int|slice, allow_expand:bool=True) -> list[_ElementT] | list[list[_ElementT]]:
        if allow_expand:
            if isinstance(idx, int):
                expected_size = idx + 1
            elif isinstance(idx, slice):
                if idx.stop is not None:
                    expected_size = idx.stop
                else:
                    expected_size = 0
            else:
                raise ValueError(f'type(idx) must be either int or slice, but {type(idx)=} (value={idx}).')
            self._expand(expected_size)
        return self._body[idx]

    def __getitem__(self, idx:int|slice) -> list[_ElementT] | list[list[_ElementT]]:
        return self.values_at(idx=idx, allow_expand=True)

    def registor(self, offset:int, values:list[_ElementT]) -> None:
        target_sequence = self[offset:offset+len(values)]
        for i, value in enumerate(values):
            target_sequence[i].append(value)
        return

    def aggregate(self, scheme:NERAggregateScheme, allow_missing:bool=False, missing_value:_Any=None) -> list:
        out = list()
        for i in range(len(self)):
            values:list[_ElementT] = self.values_at(i, allow_expand=False)
            if len(values) == 0:
                if not allow_missing:
                    raise ValueError(f'There is no value at index {i} while allow_missing=False')
                else:
                    aggregated_value = missing_value
            else:
                if scheme == NERAggregateScheme.NONE:
                    aggregated_value = values
                elif scheme == NERAggregateScheme.MEAN:
                    aggregated_value:_ElementT = sum(values) / len(values)
                elif scheme == NERAggregateScheme.SET:
                    aggregated_value = set(values)
                else:
                    raise NotImplementedError(scheme)
            out.append(aggregated_value)
        return out

@_pydantic.validate_call
def ensemble_split_sequences(splits:list[NERInstance], corresponding_token_sequences:list[_Any], aggregate:NERAggregateScheme, boundary_exclusion_size:_pydantic.NonNegativeInt=0, allow_missing:bool=False, missing_value:_Any=None) -> list:
    # NOTE: comment-out since this function now uses _pydantic.validate_call with the type of NonNegativeInt for boundary_exclusion_size.
    #assert boundary_exclusion_size >= 0, f'{boundary_exclusion_size=} must be equal to or greater than 0.'

    aggregator = NERMultiElementSparseSequence()
    for split, sequence in zip(splits, corresponding_token_sequences):
        seq_len = len(sequence)
        assert len(split.token_ids) == seq_len, f'{len(split.token_ids)=} must be equal to len(sequence)={seq_len}'

        token_offset = 0
        if split.info.split is not None:
            token_offset = split.info.split.token_offset

        # NOTE: we cannot solely use `-split.info.backward_special_token_size` for the end index of `sequence` but must prepare it explicitly since `split.info.backward_special_token_size` could be 0.
        sequence_start = split.info.forward_special_token_size
        sequence_end = seq_len - split.info.backward_special_token_size

        # exclude some tokens that are near the splitting boundary from aggregation.
        if (boundary_exclusion_size > 0) and (split.info.split is not None):
            if not split.info.split.is_initial_split():
                sequence_start = sequence_start + boundary_exclusion_size
                token_offset = token_offset + boundary_exclusion_size
            if not split.info.split.is_last_split():
                sequence_end = sequence_end - boundary_exclusion_size
            assert sequence_start < sequence_end, f'{boundary_exclusion_size=} must be small enough to be {sequence_start=} < {sequence_end=} so there exist aggregation targets.'

        aggregator.registor(offset=token_offset, values=sequence[sequence_start:sequence_end])

    return aggregator.aggregate(scheme=aggregate, allow_missing=allow_missing, missing_value=missing_value)


# %%
if __name__ == "__main__":
    tok = _transformers.AutoTokenizer.from_pretrained("bert-base-cased", trim_offsets=False)
    # %%
    _instance_without_sp = NERInstance.build(
        text = "there is very long text in this instance. we need to truncate this instance so that the bert model can take this as the input.",
        spans = [{"start":9,"end":9+14, "label":0, "id":"first-span:class_0:very long text"}, {"start":88,"end":88+10, "label":3, "id":"second-span:class_3:the bert model"}, {"start":116,"end":116+9, "label":2, "id":"third-span:class_2:the input"}],
    )
    _num_class = 4
    _instance_without_sp.encode_(tokenizer=tok, add_special_tokens=False, truncation=False)
    _instance_with_sp = _instance_without_sp.with_special_tokens(tok)
    _bilou_gold_label = _instance_with_sp.get_sequence_label(tagging_scheme=NERTaggingScheme.BILOU, label_scheme=NERLabelScheme.SINGLE_LABEL)
    _multi_label_indep_gold_label = _instance_with_sp.get_sequence_label(tagging_scheme=NERTaggingScheme.TOKEN_LEVEL, label_scheme=NERLabelScheme.MULTI_LABEL, num_class_without_negative=_num_class)
    print("_instance_without_sp:", _instance_with_sp)
    print("_bilou_gold_label:", _bilou_gold_label)
    assert _bilou_gold_label == [0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 14, 15, 0, 0, 0, 0, 9, 11, 0, 0]
    print("_multi_label_indep_gold_label:", _multi_label_indep_gold_label)
    assert _multi_label_indep_gold_label == [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    # %%
    _instance_with_sp_and_string_labels = NERInstance.build(
        text = "there is very long text in this instance. we need to truncate this instance so that the bert model can take this as the input.",
        spans = [{"start":9,"end":9+14, "label":"class_0", "id":"first-span:class_0:very long text"}, {"start":88,"end":88+10, "label":"class_3", "id":"second-span:class_3:the bert model"}, {"start":116,"end":116+9, "label":"class_2", "id":"third-span:class_2:the input"}],
        tokenizer = tok, add_special_tokens=True, truncation=False,
    )
    _label_to_id1 = {"class_0":0, "class_1":1, "class_2":2, "class_3":3}
    _bilou_gold_label_with_label_mapping = _instance_with_sp_and_string_labels.get_sequence_label(tagging_scheme=NERTaggingScheme.BILOU, label_scheme=NERLabelScheme.SINGLE_LABEL, label_to_id=_label_to_id1)
    print("_bilou_gold_label_with_label_mapping:", _bilou_gold_label_with_label_mapping)
    assert _bilou_gold_label_with_label_mapping == _bilou_gold_label

    # %%
    _encoded_query = tok("what model is used ?", add_special_tokens=False)["input_ids"]
    _instance_with_query = _instance_without_sp.only_label(3).with_query_and_special_tokens(tokenizer=tok, encoded_query=_encoded_query, max_length=30)
    _gold_label_for_query = _instance_with_query.get_sequence_label(tagging_scheme=NERTaggingScheme.BILOU, label_scheme=NERLabelScheme.SPAN_ONLY)
    print(_instance_with_query)
    print()
    print("label:", _gold_label_for_query)
    assert _gold_label_for_query == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 0]

    # %%
    _instances_split_by_stride = NERInstance.build(
        text = "there is very long text in this instance. we need to truncate this instance so that the bert model can take this as the input.",
        spans = [{"start":9,"end":9+14, "label":0, "id":"first-span:class_0:very long text"}, {"start":88,"end":88+10, "label":3, "id":"second-span:class_3:the bert model"}, {"start":116,"end":116+9, "label":2, "id":"third-span:class_2:the input"}],
        tokenizer = tok,
        add_special_tokens=True,
        truncation = NERTruncationScheme.SPLIT,
        max_length = 8,
        stride = 2,
    )
    print(type(_instances_split_by_stride), type(_instances_split_by_stride[0]), len(_instances_split_by_stride))
    assert len(_instances_split_by_stride) == 13

    # %%
    # step_preds = BNER.viterbi_decode(step_logit, tagging_scheme=model_config.tagging_scheme, label_scheme=model_config.label_scheme, scalar_logit_for_independent=True, as_spans=True)
    # step_preds = instance.decode_token_span_to_char_span(step_preds, strip=True, recover_split=True)


# %%
