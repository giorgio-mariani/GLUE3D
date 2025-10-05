from typing import *
from datasets import load_dataset
import tqdm

from glue3d.evaluators import Evaluators


def evaluate_GLUE3D_answers(
    ground_truth_data: Union[str, "pd.DataFrame"],
    model_answer_data: "pd.DataFrame",
    answer_evaluator: Union[Callable, str],
) -> "pd.DataFrame":
    """Evaluates the model answers against the ground truth data using the provided answer evaluator.
    Args:
        ground_truth_data (Union[str, pd.DataFrame]): Ground truth data config name or pandas DataFrame.
        model_answer_data (pd.DataFrame): DataFrame containing the model answers.
        answer_evaluator (Callable): Function to evaluate the model answers against the ground truth ones.
    Returns:
        pd.DataFrame: DataFrame containing the evaluation results.
    """
    if isinstance(answer_evaluator, str):

        judge = Evaluators(answer_evaluator)

        if judge == Evaluators.QWEN_3_30B_JUDGE:
            from glue3d.evaluators.loaders import load_qwen3_30B_A3B_model
            from glue3d.evaluators.qwen3 import Qwen3Judge

            answer_evaluator = Qwen3Judge(*load_qwen3_30B_A3B_model())

        elif judge == Evaluators.TRADITIONAL:
            from glue3d.evaluators.misc import TraditionalCaptionMetricEvaluator

            answer_evaluator = TraditionalCaptionMetricEvaluator()
        elif judge == Evaluators.BINARY:

            def _binary_eval(**kwargs):
                y_true, y_pred = kwargs["ANSWER"], kwargs["MODEL_ANSWER"]
                assert type(y_true) == type(y_pred) == bool
                return y_true == y_pred

            answer_evaluator = _binary_eval
        elif judge == Evaluators.MULTI_CHOICE:

            def _multi_eval(**kwargs):
                y_true, y_pred = kwargs["ANSWER"], kwargs["MODEL_ANSWER"]
                assert type(y_true) == type(y_pred) == str
                assert y_true in ["A", "B", "C", "D"]
                assert y_pred in ["A", "B", "C", "D"]
                return y_true == y_pred

            answer_evaluator = _multi_eval
        else:
            assert False

    import pandas as pd  # Importing here to avoid bugs with llama.cpp and pandas

    oid_k = "OBJECT_ID"
    qid_k = "QUESTION_ID"
    ma_k = "MODEL_ANSWER"
    a_k = "ANSWER"

    # Check if answers is a DataFrame
    if not isinstance(model_answer_data, pd.DataFrame):
        raise ValueError("Answers should be a pandas DataFrame.")

    # Check that index is composed of OBJECT_ID and QUESTION_ID
    if not {oid_k, qid_k} == set(model_answer_data.index.names):
        raise ValueError("Answers DataFrame index must be composed of 'OBJECT_ID' and 'QUESTION_ID'.")

    # Check that answers contains MODEL_ANSWER
    if not ma_k in model_answer_data.columns:
        raise ValueError(f"Answers DataFrame must contain the '{ma_k}' column.")

    # Load ground truth data
    if isinstance(ground_truth_data, str):
        ground_truth_data = load_dataset("giorgio-mariani-1/GLUE3D", ground_truth_data, split="test")
        ground_truth_data = pd.DataFrame.from_records(list(ground_truth_data))
        ground_truth_data = ground_truth_data[[oid_k, qid_k, a_k]].set_index([oid_k, qid_k])
    else:
        if not isinstance(ground_truth_data, pd.DataFrame):
            raise ValueError("ground_truth_data must be a string or a pandas DataFrame.")

        if not {oid_k, qid_k} == set(ground_truth_data.index.names):
            raise ValueError(f"ground_truth_data DataFrame index must be composed of '{oid_k}' and '{qid_k}'.")

        if not a_k in ground_truth_data.columns:
            raise ValueError(f"ground_truth_data DataFrame must contain the '{a_k}' column.")

    # Check that all MODEL_ANSWER values are a subset of ANSWER values
    valid_values = set(ground_truth_data[a_k].dropna().unique())
    difference = set(model_answer_data[ma_k].dropna().unique()).difference(valid_values)
    if len(difference) > 0:
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

    output_data = pd.DataFrame.from_records(output_records).set_index([oid_k, qid_k])
    return output_data
