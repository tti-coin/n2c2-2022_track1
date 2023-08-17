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
import pathlib
import enum
import collections
from typing import Annotated, Optional, Any, Callable

import pydantic
import pydantic.functional_validators as PV

# %%
class DiscontinuousSpanScheme(str, enum.Enum):
    SPLIT = "split"
    MERGE_BY_FILL = "merge_by_fill"

class Config(pydantic.BaseModel):
    discontinuous_span: DiscontinuousSpanScheme



# %%
def _at_least_one(values:Any) -> Any:
    if len(values) == 0:
        raise ValueError(f'values must have at least 1 element, but {values=}')
    return values
def _is_initial(initials:str | set[str]) -> Callable[[str],str]:
    if type(initials) is str:
        initials = {initials}
    def validator(id:str) -> str:
        if id[0] not in initials:
            raise ValueError(f'the initial character of id must be one of {initials}, but {id[0]=} ({id=})')
        return id
    return validator

class Span(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    start: int
    end: int

    @pydantic.model_validator(mode="after")
    def check_some(self):
        if not (self.start < self.end):
            raise ValueError(f'start must be less than end, but {self.start=} and {self.end=} ({self=})')
        return self


def _wrap_to_list(elem_type):
    def validator(elem_or_elems:Any) -> list[elem_type]:
        elem_or_elems = pydantic.TypeAdapter(elem_type | list[elem_type]).validate_python(elem_or_elems)
        elems = [elem_or_elems] if isinstance(elem_or_elems, elem_type) else elem_or_elems
        return elems
    return validator

def _spans_in_ascending_order(spans:list[Span]) -> list[Span]:
    for i in range(0, len(spans)-1):
        before = spans[i]
        after = spans[i+1]
        if not (before.end <= after.start):
            raise ValueError(f'discontinuous spans must be in ascending order, but {spans=}')
    return spans
_Span_or_Spans_type = Annotated[list[Span], pydantic.BeforeValidator(_wrap_to_list(Span)), pydantic.AfterValidator(_at_least_one), pydantic.AfterValidator(_spans_in_ascending_order)]
class TextBound(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(populate_by_name=True)

    id: Annotated[str, pydantic.AfterValidator(_is_initial("T"))]
    type: str
    spans: _Span_or_Spans_type = pydantic.Field(alias="span")
    reference_text: str

    @property
    def is_discontinuous(self) -> bool:
        return len(self.spans) > 1
    @property
    def span(self) -> Span:
        if self.is_discontinuous:
            raise ValueError(f'TextBound.span can be called iff the text bound contains exactly one span, but {self=}')
        return self.spans[0]
    @property
    def start(self) -> int:
        return self.spans[0].start
    @property
    def end(self) -> int:
        return self.spans[-1].end

    def to_annotation_text(self) -> str:
        text_bounds = ";".join([f'{span.start} {span.end}' for span in self.spans])
        type_and_text_bounds = f'{self.type} {text_bounds}'
        out = "\t".join([self.id, type_and_text_bounds, self.reference_text])
        return out

    @classmethod
    @pydantic.validate_call
    def from_span(cls, span_or_spans:_Span_or_Spans_type, document_text:str, id:str, type:str):
        reference_text = " ".join([document_text[span.start:span.end] for span in span_or_spans])
        if "\n" in reference_text:
            _logger.info(f'Found a new-line character "\\n" in the reference_text. it is replaced with a blank character.')
            reference_text = reference_text.replace("\n", " ")
        return cls(id=id, type=type, spans=span_or_spans, reference_text=reference_text)

class Argument(pydantic.BaseModel):
    role: str
    arg_id: str


class Event(pydantic.BaseModel):
    id: Annotated[str, pydantic.AfterValidator(_is_initial("E"))]
    type: str
    trigger_id: str
    args: list[Argument]

    def to_annotation_text(self) -> str:
        second_field = " ".join([f'{self.type}:{self.trigger_id}'] + [f'{arg.role}:{arg.arg_id}' for arg in self.args])
        out = "\t".join([self.id, second_field])
        return out


class Relation(pydantic.BaseModel):
    id: Annotated[str, pydantic.AfterValidator(_is_initial("R"))]
    type: str
    args: Annotated[list[Argument], pydantic.AfterValidator(_at_least_one)]

    def to_annotation_text(self) -> str:
        second_field = " ".join([self.type] + [f'{arg.role}:{arg.arg_id}' for arg in self.args])
        out = "\t".join([self.id, second_field])
        return out


class Attribute(pydantic.BaseModel):
    id: Annotated[str, pydantic.AfterValidator(_is_initial(["A", "M"]))]
    name: str
    target_id: str
    value: bool | str

    def to_annotation_text(self) -> str:
        if type(self.value) is bool:
            if self.value:
                second_field = f'{self.name} {self.target_id} true'
            else:
                second_field = f'{self.name} {self.target_id}'
        else:
            second_field = f'{self.name} {self.target_id} {self.value}'
        out = "\t".join([self.id, second_field])
        return out


class Normalization(pydantic.BaseModel):
    id: Annotated[str, pydantic.AfterValidator(_is_initial("N"))]
    target_id: str
    resource_id: str
    entry_id: str
    type: str = "Reference"

    def to_annotation_text(self) -> str:
        raise NotImplementedError("normalization annotation has reference-text field, but currently it is not implemented thus to_annotation_text also cannot be process.")


class Note(pydantic.BaseModel):
    id: Annotated[str, pydantic.AfterValidator(_is_initial("#"))]
    text: str

    def to_annotation_text(self) -> str:
        return "\t".join([self.id, self.text])


class Document(pydantic.BaseModel):
    text: Optional[str] = None
    text_bounds: list[TextBound] = pydantic.Field(default_factory=list)
    relations: list[Relation] = pydantic.Field(default_factory=list)
    events: list[Event] = pydantic.Field(default_factory=list)
    attributes: list[Attribute] = pydantic.Field(default_factory=list)
    normalizations: list[Normalization] = pydantic.Field(default_factory=list)
    notes: list[Note] = pydantic.Field(default_factory=list)

    def get_all_annotations(self) -> list:
        return [annotation for annotations in [self.text_bounds, self.relations, self.events, self.attributes, self.normalizations, self.notes] for annotation in annotations]
    _id_to_annotation: dict = pydantic.PrivateAttr(default_factory=dict)
    def get_annotation_by_id(self, id:str) -> TextBound | Relation | Event | Attribute | Normalization | Note:
        if id not in self._id_to_annotation:
            self._id_to_annotation = {annotation.id:annotation for annotation in self.get_all_annotations()}
        return self._id_to_annotation[id]

    def to_annotation_text(self) -> str:
        return "\n".join([annotation.to_annotation_text() for annotation in self.get_all_annotations()])

# %%
def _parse_text_bound_annotation(line:str) -> TextBound:
    assert line[0] == "T"
    id_, type_and_text_bounds, reference_text = line.rstrip("\r\n").split("\t", maxsplit=3)
    type_, text_bounds = type_and_text_bounds.split(" ", maxsplit=1)
    text_bounds = [list(map(int, start_end.split())) for start_end in text_bounds.split(";")]
    spans = [Span(start=start, end=end) for start,end in text_bounds]
    return TextBound(id=id_, type=type_, spans=spans, reference_text=reference_text)

def _parse_event_annotation(line:str) -> Event:
    assert line[0] == "E"
    id_, trigger_and_args = line.rstrip().split("\t")
    trigger, *args = trigger_and_args.split(" ")
    type_, trigger_id = trigger.split(":")
    args = [arg.split(":") for arg in args]
    return Event(id=id_, type=type_, trigger_id=trigger_id, args=[Argument(role=role, arg_id=arg_id) for role,arg_id in args])

def _parse_relation_annotation(line:str) -> Relation:
    # TODO: implement for Equiv relation
    assert line[0] == "R"
    id_, type_and_args = line.rstrip().split("\t")
    type_, *args = type_and_args.split(" ")
    args = [arg.split(":") for arg in args]
    return Relation(id=id_, type=type_, args=[Argument(role=role, arg_id=arg_id) for role,arg_id in args])

def _parse_attribute_annotation(line) -> Attribute:
    assert line[0] in {"A", "M"}
    id_, attribute_info = line.rstrip().split("\t")
    attribute_info = attribute_info.split(" ")
    if len(attribute_info) == 2:
        attribute_name, target_id = attribute_info
        attribute_value = False
    elif len(attribute_info) == 3:
        attribute_name, target_id, attribute_value = attribute_info
        if attribute_value == "true":
            attribute_value = True
    else:
        raise ValueError(f'attirbute annotation must have either two or three element in the values field, but {attribute_info=}')
    return Attribute(id=id_, name=attribute_name, target_id=target_id, value=attribute_value)

def _parse_normalization_annotation(line:str) -> Normalization:
    assert line[0] == "N"
    id_, norm_info = line.rstrip("\r\n").split("\t")[:2]
    type_, target_id, resource_and_entry = norm_info.split(" ")
    resource_id, entry_id = resource_and_entry.split(":")
    return Normalization(id=id_, target_id=target_id, resource_id=resource_id, entry_id=entry_id, type=type_)

def _parse_note_annotation(line:str) -> Note:
    assert line[0] == "#"
    id_, note_text = line.rstrip("\r\n").split("\t", maxsplit=1)
    return Note(id=id_, text=note_text)


# %%
def parse_ann_file(ann_file) -> Document:
    initial_to_annotations = collections.defaultdict(list)

    for line in ann_file:
        if len(line.strip()) == 0: continue
        line = line.rstrip("\r\n")

        initial = line[0]
        if initial == "M":
            initial = "A"

        parsed_line = {
            "T": _parse_text_bound_annotation,
            "R": _parse_relation_annotation,
            "E": _parse_event_annotation,
            "A": _parse_attribute_annotation,
            "N": _parse_normalization_annotation,
            "#": _parse_note_annotation,
        }[initial](line)

        initial_to_annotations[initial].append(parsed_line)

    initial_to_field_name = {
        "T": "text_bounds",
        "R": "relations",
        "E": "events",
        "A": "attributes",
        "N": "normalizations",
        "#": "notes",
    }

    return Document(**{initial_to_field_name[initial]:field_annotations for initial, field_annotations in initial_to_annotations.items()})

def load_file(text_file, ann_file) -> Document:
    text = text_file.read()
    doc = parse_ann_file(ann_file=ann_file)
    return doc.model_copy(update={"text":text})

def load_dir(dir_path: pathlib.Path | str) -> dict[str,Document]:
    if type(dir_path) is str:
        dir_path = pathlib.Path(dir_path)

    assert dir_path.is_dir(), f'{dir_path=} must be a directory'

    stem_to_document = dict()
    stems = set()
    for text_path in dir_path.glob("./*.txt"):
        stem = text_path.stem
        ann_path = dir_path / f'{stem}.ann'

        if ann_path.exists():
            with open(text_path) as text_file, open(ann_path) as ann_file:
                doc = load_file(text_file=text_file, ann_file=ann_file)
        else:
            _logger.warning(f'corresponding ann file does not exist for {str(text_path)}')
            with open(text_path) as text_file:
                doc = Document(text=text_file.read())
        stem_to_document[stem] = doc
        stems.add(stem)

    for ann_path in dir_path.glob("./*.ann"):
        stem = ann_path.stem
        if stem not in stems:
            _logger.warning(f'corresponding text file does not exist for {str(ann_path)}')
            with open(ann_path) as ann_file:
                doc = parse_ann_file(ann_file=ann_file)
                stem_to_document[stem] = doc

    return stem_to_document


# %%

# %%

