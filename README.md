# GLUE3D

Official implementation of **GLUE3D: General Language Understanding Evaluation for 3D Point Clouds**.

---

## Installation

Install from PyPI:

```bash
pip install glue3d
```

---

## Generating Answers

You can generate answers for the benchmark tasks in two main ways:

1. **Using the dataset loader** (`load_GLUE3D_benchmark`) with your own model and code.
2. **Using the built-in AnswerGenerator interface** with `generate_GLUE3D_answers`.


### Option 1: Using `load_GLUE3D_benchmark`

The benchmark data can be loaded directly using:

```python
import pandas as pd
from glue3d.data import load_GLUE3D_benchmark

dataset = load_GLUE3D_benchmark(
    dataset_name="GLUE3D-points-8k",  # or "GLUE3D-images", "GLUE3D-multiview", "GLUE3D-points"
    qa_task="binary_task",             # or "multiplechoice_task", "captioning_task"
    cache_dir=None,                    # Optional; defaults to './cache' or $GLUE3D_CACHE_DIR
)
```

> **Note:** The loader uses a local cache directory. You can customize it via the `GLUE3D_CACHE_DIR` environment variable.

Once loaded, iterate through the dataset to generate answers:

```python
your_model = ...  # Load your 3D-LLM

model_answers = []
for x in dataset:
    oid = x["object_id"]
    qid = x["question_id"]
    q = x["question"]
    pc = x["data"]  # e.g., (8192 x 6) np.ndarray for "GLUE3D-points-8K"

    answer = your_model.answer_question(pc, q)
    model_answers.append({
        "OBJECT_ID": oid,
        "QUESTION_ID": qid,
        "MODEL_ANSWER": answer,
    })

# Save results
pd.DataFrame.from_records(model_answers).to_csv("qa.csv", index=False)
```

> **Answer format note:**
> Ensure your answers follow the expected format for each task (`binary_task`, `multiplechoice_task`, `captioning_task`).
 - For the `binary_task`, the model answer must be a boolean.
 - For the `multiplechoice_task`, the model answer must be one of `A`, `B`, `C`, `D`.
 - For the `captioning_task` the model answer must be a string.

---

### Option 2: Using the `AnswerGenerator` Interface

You can subclass the provided `HFAnswerGenerator` classes for a cleaner integration.
Your custom generator must implement the `GeneratorMixin` interface.

```python
import numpy as np
from typing import override
from glue3d import generate_GLUE3D_answers
from glue3d.models.hf import BinaryHFAnswerGenerator, MultichoiceHFAnswerGenerator, CaptioningHFGenerator

# Example custom AnswerGenerator for the binary task
class YourAnswerGenerator(BinaryHFAnswerGenerator):
    def __init__(self, your_model, tokenizer):
        super().__init__(your_model, tokenizer)

    @override
    def prepare_inputs(self, data: np.ndarray, text: str) -> dict:
        ... # Preprocess data (e.g., tokenize text, move tensors to device, apply chat templates)
        return {
            "input_ids": ...,
            "points": ...,
            "do_sample": ...,
            "stopping_criteria": ...,
        }
```

Then, generate answers across the entire benchmark:

```python
your_model = ...
answer_gen = YourAnswerGenerator(your_model)

qa_answers = generate_GLUE3D_answers(
    qa_task="binary_task",
    dataset_type="GLUE3D-points-8K",
    answer_generator=answer_gen,
)

# `qa_answers` is returned as a pandas DataFrame
```

---

## Evaluation

After generating a `qa.csv` file containing question–answer pairs for the binary task, e.g.:

```csv
OBJECT_ID,QUESTION_ID,VALUE
dc5c798,0fbac6,True
dc5c798,556cc4,False
```

evaluate the results from the command line:

```bash
glue3d evaluate --input-file qa.csv --output-file out.csv --task binary_task
```

---
