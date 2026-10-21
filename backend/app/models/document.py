from dataclasses import dataclass
from typing import Any


@dataclass
class Document:
    text: str
    metadata: dict[str, Any]