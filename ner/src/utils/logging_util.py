import logging as _logging
from typing import Optional as _Optional, Protocol as _Protocol, Hashable as _Hashable

class _ChangeLoggingLevelFunction(_Protocol):
    def __call__(self, level:str|int) -> None: ...

class _LogOnceFunction(_Protocol):
    def __call__(self, message:str, level:str|int, id:_Optional[_Hashable]=None, enable:bool=True) -> None: ...
class _LogOnce:
    def __init__(self, logger) -> None:
        self.logger = logger
        self.keys = set()
    def __call__(self, message:str, level:str|int, id:_Optional[_Hashable]=None, enable:bool=True) -> None:
        if enable:
            if id is None:
                id = message
            if id not in self.keys:
                if isinstance(level, str):
                    level = _logging.getLevelName(level)
                self.logger.log(level, message)
                self.keys.add(id)

def make_logger(name, default_level:str|int="WARNING") -> tuple[_logging.Logger, _LogOnceFunction, _ChangeLoggingLevelFunction]:
    if isinstance(default_level, str):
        default_level = _logging.getLevelName(default_level)
    logger = _logging.getLogger(name)
    logger.setLevel(default_level)
    ch = _logging.StreamHandler()
    ch.setLevel(default_level)
    formatter = _logging.Formatter('%(name)s - %(levelname)s:%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    log_once = _LogOnce(logger=logger)

    def change_logging_level(level:str|int) -> None:
        if isinstance(level, str):
            level = _logging.getLevelName(level)
        logger.setLevel(level)
        ch.setLevel(level)

    return logger, log_once, change_logging_level


