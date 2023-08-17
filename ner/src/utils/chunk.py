from typing import TypeVar, Iterable, Generator, List

ValueType = TypeVar("ValueType")
def to_chunk(iterable: Iterable[ValueType], chunk_size:int) -> Generator[List[ValueType], None, None]:
    assert chunk_size > 0
    chunk = list()
    for val in iterable:
        chunk.append(val)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = list()
    if len(chunk) > 0:
        yield chunk

