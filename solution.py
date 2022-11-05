from pathlib import Path
from typing import List, Iterable, Set, Tuple


class Solution:

    def __init__(self, model_path: Path):
        self._model = None

    def predict(self, texts: List[str]) -> Iterable[Set[Tuple[int, int, str]]]:
        return [set() for _ in texts]

