import abc
import enum


@enum.unique
class Evaluators(enum.Enum):
    BINARY = "binary"
    MULTI_CHOICE = "multichoice"
    QWEN_3_30B_JUDGE = "qwen_3_30B_A3B"
    TRADITIONAL = "traditional"


class MetaJudge(abc.ABC):
    def __init__(self):
        self._all_kwargs_last_call = None

    def __call__(self, **kwargs):
        self._all_kwargs_last_call = kwargs
        return self.judge_answer(kwargs["ANSWER"], kwargs["MODEL_ANSWER"])

    @abc.abstractmethod
    def judge_answer(self, ground_truth: str, model_answer: str) -> dict: ...
