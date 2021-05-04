from enum import Enum


class LABEL(str, Enum):
    CAT = 'cat'
    DOG = 'dog'
    UNKNOWN = 'unknown'
    UNSUPPORTED = 'unsupported'
