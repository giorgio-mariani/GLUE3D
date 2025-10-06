import os
from pathlib import Path
from typing import *

import pandas as pd
from tqdm import tqdm

from glue3d.data import QATasks, load_GLUE3D_benchmark
from glue3d.models import AnswerGenerator


def process_binary(answer: str):
    if answer is not ["Yes", "No"]:
        raise ValueError(f"Output answer '{answer}' should be either 'Yes' or 'No', found {answer} instead!")

    return answer == "Yes"


def process_multichoice(answer: str):
    choices = ["A", "B", "C", "D"]
    if answer not in choices:
        raise ValueError(f"Output answer '{answer}' should be one of {choices}, found {type(answer)} instead!")

    return answer


def process_caption(answer: str):
    if not isinstance(answer, str):
        raise ValueError(f"Output answer '{answer}' should be a string, found {type(answer)} instead!")
    return answer


def generate_GLUE3D_answers(
    qa_task: str,
    dataset_type: str,
    answer_generator: AnswerGenerator,
    output_file: Optional[Path] = None,
) -> pd.DataFrame:
    output_file = Path(output_file)

    # Task to answer checker
    processors = {
        QATasks.BINARY: process_binary,
        QATasks.MULTICHOICE: process_multichoice,
        QATasks.CAPTION: process_caption,
    }

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

    # Load benchmark data
    cache_dir = os.environ.get("GLUE3D_CACHE_DIR", None)
    if cache_dir is None:
        cache_dir = Path(".cache/glue3d").absolute()
        print(f"Warning: 'GLUE3D_CACHE_DIR' is not set. Using default cache directory ({cache_dir}).")

    dataset = load_GLUE3D_benchmark(dataset_type, qa_task, cache_dir=cache_dir)
    task_processor = processors[QATasks(qa_task)]

    responses = []
    for batch in tqdm(dataset):

        # Get IDS
        object_id = batch["object_id"]  # <- string
        question_id = batch["question_id"]  # <- string

        # Get data
        question_data = batch["data"]  # <- tensor of shape N, C(6)
        question = batch["question"]  # string

        # Compute anser
        answer = answer_generator(question_data, question)  # List of strings
        answer = task_processor(answer)

        # Append results to output file
        responses.append({"OBJECT_ID": object_id, "QUESTION_ID": question_id, "MODEL_ANSWER": answer})

    # Save the results to a CSV file
    responses_df = pd.DataFrame.from_records(responses)
    if output_file is not None:
        responses_df.to_csv(output_file, index=False)
    return responses_df
