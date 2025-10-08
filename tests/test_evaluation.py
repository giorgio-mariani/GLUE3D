import pandas as pd
import numpy as np
from datasets import load_dataset

import pytest

from glue3d.evaluate_answers import evaluate_GLUE3D_answers


def _get_gt_data(task):
    ground_truth_data = load_dataset("giorgio-mariani-1/GLUE3D", task, split="test")
    ground_truth_data = pd.DataFrame.from_records(list(ground_truth_data))
    return ground_truth_data.set_index(["OBJECT_ID", "QUESTION_ID"])


def test_binary_task_evaluation():
    task = "binary_task"

    gt = _get_gt_data(task)
    num_entries = len(gt)

    gt["MODEL_ANSWER"] = gt["ANSWER"]
    del gt["ANSWER"]

    # Assert acuracy 1.0 (gt)
    df = evaluate_GLUE3D_answers(task, gt)
    assert df["VALUE"].sum() == num_entries

    # Assert accuracy around 0.5 (random)
    gt["MODEL_ANSWER"] = False
    df = evaluate_GLUE3D_answers(task, gt)
    assert np.abs(df["VALUE"].mean() - 0.5).item() <= 0.075


def test_multi_task_evaluation():
    task = "multiplechoice_task"

    gt = _get_gt_data(task)
    num_entries = len(gt)

    gt["MODEL_ANSWER"] = gt["ANSWER"]
    del gt["ANSWER"]

    # Assert acuracy 1.0 (gt)
    df = evaluate_GLUE3D_answers(task, gt)
    assert df["VALUE"].sum() == num_entries

    # Assert accuracy around 0.5 (random)
    gt["MODEL_ANSWER"] = "A"
    df = evaluate_GLUE3D_answers(task, gt)
    assert np.abs(df["VALUE"].mean() - 0.25).item() <= 0.05


def test_captioning_evaluation():
    task = "captioning_task"

    gt = _get_gt_data(task)

    gt["MODEL_ANSWER"] = gt["ANSWER"]
    del gt["ANSWER"]

    # Assert acuracy 1.0 (gt)
    df = evaluate_GLUE3D_answers(task, gt)
    assert np.isclose(df["BLEU-1"].mean(), 100.0)

    gt["MODEL_ANSWER"] = "The quick brown fox jumps over the lazy dog."

    df = evaluate_GLUE3D_answers(task, gt)
    assert df["BLEU-1"].mean() <= 2.0


@pytest.mark.test_judge
def test_captioning_judge():
    task = "captioning_task"
    gt = _get_gt_data(task)
    gt["MODEL_ANSWER"] = "The quick brown fox jumps over the lazy dog."
    del gt["ANSWER"]

    df = evaluate_GLUE3D_answers(task, gt, answer_evaluator="qwen_3_30B_A3B")
    assert df["QWEN_SCORE"].mean() <= 2.0, df.columns
