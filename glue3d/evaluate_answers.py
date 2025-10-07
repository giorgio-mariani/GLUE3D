from typing import *
from datasets import load_dataset
import tqdm

from glue3d.data import QATasks
from glue3d.evaluators import Evaluators

DEFAULT_EVALUATORS = {
    QATasks.BINARY: Evaluators.BINARY,
    QATasks.MULTICHOICE: Evaluators.MULTI_CHOICE,
    QATasks.CAPTION: Evaluators.TRADITIONAL,
}


def instantiate_evaluator(qa_evaluator: Evaluators) -> Callable:
    def _binary_eval(**kwargs):
        y_true, y_pred = kwargs["ANSWER"], kwargs["MODEL_ANSWER"]
        assert type(y_true) == type(y_pred) == bool
        return y_true == y_pred

    def _multi_eval(**kwargs):
        y_true, y_pred = kwargs["ANSWER"], kwargs["MODEL_ANSWER"]
        assert type(y_true) == type(y_pred) == str
        assert y_true in ["A", "B", "C", "D"]
        assert y_pred in ["A", "B", "C", "D"]
        return y_true == y_pred

    from glue3d.evaluators.loaders import load_qwen3_30B_A3B_model
    from glue3d.evaluators.qwen3 import Qwen3Judge
    from glue3d.evaluators.misc import TraditionalCaptionMetricEvaluator

    if qa_evaluator == Evaluators.QWEN_3_30B_JUDGE:
        answer_evaluator = Qwen3Judge(*load_qwen3_30B_A3B_model())
    elif qa_evaluator == Evaluators.TRADITIONAL:
        answer_evaluator = TraditionalCaptionMetricEvaluator()
    elif qa_evaluator == Evaluators.BINARY:
        answer_evaluator = _binary_eval
    elif qa_evaluator == Evaluators.MULTI_CHOICE:
        answer_evaluator = _multi_eval
    else:
        assert False

    return answer_evaluator


def evaluate_GLUE3D_answers(
    benchmark_task: str,
    model_answer_data: Union[str, "pd.DataFrame"],
    answer_evaluator: Union[Callable, str, None] = None,
    output_file: Optional[str] = None,
) -> "pd.DataFrame":
    """Evaluates the model answers against the ground truth data using the provided answer evaluator.

    Args:
        benchmark_task str: Ground truth data config name.
        model_answer_data (Union[str,pd.DataFrame]): DataFrame (or csv filename) containing the model answers.
        answer_evaluator (Callable): Function to evaluate the model answers against the ground truth ones.
        output_file str: Output filename, must be a .csv

    Returns:
        pd.DataFrame: DataFrame containing the evaluation results.
    """

    benchmark_task = QATasks(benchmark_task)
    answer_evaluator = Evaluators(answer_evaluator) if isinstance(answer_evaluator, str) else answer_evaluator

    if answer_evaluator is None:
        answer_evaluator = DEFAULT_EVALUATORS[benchmark_task]

    if isinstance(answer_evaluator, Evaluators):
        answer_evaluator = instantiate_evaluator(answer_evaluator)

    # Check if output file exists
    if output_file is not None:
        if output_file.exists():
            raise FileExistsError(
                f"Output file {output_file} already exists. Please remove it or choose a different name."
            )
        elif output_file.parent.exists() is False:
            raise FileNotFoundError(f"Output directory {output_file.parent} does not exist. Please create it first.")
        elif output_file.suffix != ".csv":
            raise ValueError(f"Output file must have a .csv extension, got {output_file.suffix}.")

    import pandas as pd  # Importing here to avoid bugs with llama.cpp and pandas

    oid_k = "OBJECT_ID"
    qid_k = "QUESTION_ID"
    ma_k = "MODEL_ANSWER"
    a_k = "ANSWER"

    # Check if answers is a DataFrame
    if isinstance(model_answer_data, str):
        model_answer_data = pd.read_csv(model_answer_data, index=[oid_k, qid_k])

    if not isinstance(model_answer_data, pd.DataFrame):
        raise ValueError("Answers should be a pandas DataFrame.")

    # Check that index is composed of OBJECT_ID and QUESTION_ID
    if not {oid_k, qid_k} == set(model_answer_data.index.names):
        raise ValueError(f"Answers DataFrame index must be composed of '{oid_k}' and '{qid_k}'.")

    # Check that answers contains MODEL_ANSWER
    if not ma_k in model_answer_data.columns:
        raise ValueError(f"Answers DataFrame must contain the '{ma_k}' column.")

    # Load ground truth data
    ground_truth_data = load_dataset("giorgio-mariani-1/GLUE3D", benchmark_task.value, split="test")
    ground_truth_data = pd.DataFrame.from_records(list(ground_truth_data))
    ground_truth_data = ground_truth_data[[oid_k, qid_k, a_k]].set_index([oid_k, qid_k])

    # Check that all MODEL_ANSWER values are a subset of ANSWER values
    valid_values = set(ground_truth_data[a_k].dropna().unique())
    difference = set(model_answer_data[ma_k].dropna().unique()).difference(valid_values)
    if len(difference) > 0 and benchmark_task != QATasks.CAPTION:
        raise ValueError(f"invalid MODEL_ANSWER values: {difference}. Valid values: {valid_values}")

    data = model_answer_data.join(ground_truth_data, validate="1:1")

    # For every entry, pass input data to the evaluator
    output_records = []
    for (oid, qid), entry in tqdm.tqdm(data.iterrows(), total=len(data)):
        evaluation_output = answer_evaluator(**entry)

        # Normalize evaluation output
        if isinstance(evaluation_output, dict):
            pass
        elif isinstance(evaluation_output, (list, tuple)):
            evaluation_output = {f"VALUE_{i}": v for i, v in enumerate(evaluation_output)}
        else:
            evaluation_output = {"VALUE": evaluation_output}

        # Append the evaluation record to the output
        output_records += [{oid_k: oid, qid_k: qid, **evaluation_output}]

    output_data = pd.DataFrame.from_records(output_records)

    if output_file is not None:
        output_data.to_csv(output_file, index=False)
    return output_data
