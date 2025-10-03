import abc
from typing import *

import numpy as np


class AnswerGenerator(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        data: np.ndarray,
        text: str,
    ) -> str: ...
