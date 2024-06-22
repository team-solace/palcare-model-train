from dataclasses import dataclass
from typing import List, TypedDict


class FormattedFinetuneData(TypedDict):
    system: str
    input: str
    output: str
