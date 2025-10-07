<h1 align="center">
GLUE3D: General Language Understanding Evaluation for 3D Point Clouds
</h1>
<p align="center">
Giorgio Mariani, Alessandro Raganato, Simone Melzi, Gabriella Pasi
</p>

Official implementation of **GLUE3D: General Language Understanding Evaluation for 3D Point Clouds**.

GLUE3D is a Q&A benchmark for evaluation of 3D-LLMs object understanding capabilities. It is built around 128 richly textured surfaces spanning creatures, objects, architecture and transport. Each surface is provided as a 50 k-point RGB point cloud, a 8K-point RGB point cloud, a 512 × 512 RGB rendering, and five RGB-D multiviews. These multiple representations enable point-for-point evaluation across several modalities.

GLUE3D consists of three Q&A task types: *binary question answering*, *multiple-choice question answering*, and *open-ended captioning*. This diverse set of tasks enables a more robust and comprehensive assessment of multimodal understanding in 3D-LLMs.

![](assets/teaser.jpg)

---

## Installation

To evaluate your question-answering model on GLUE3D, we offer a PyPI package that can be easily installed with the command:

```bash
pip install glue3d
```


You can install glue3d from source if you want the latest changes in the library or are interested in contributing. However, the latest version may not be stable. Feel free to open an [issue](https://github.com/giorgio-mariani/GLUE3D/issues) if you encounter an error.

```bash
git clone https://github.com/giorgio-mariani/GLUE3D.git
cd GLUE3D

pip install -e .
```

---

## Answer generation

To evaluate your model, first you need to generate your 3D-LLM's answers for the desired GLUE3D task. You can do so in two main ways:

1. Using the dataset loader (`load_GLUE3D_benchmark`) with your own model and code.
2. Using the built-in AnswerGenerator interface with `generate_GLUE3D_answers`. This option is to be preferred if your model follows `huggingface` causal generation procedure (e.g., `LlavaLlamaForCausalLM`).

<details>
<summary> Option 1: Using load_GLUE3D_benchmark </summary>

### Using `load_GLUE3D_benchmark`

The GLUE3D benchmark data can be (down)loaded using:

```python
import pandas as pd
from glue3d.data import load_GLUE3D_benchmark

dataset = load_GLUE3D_benchmark(
    dataset_name="GLUE3D-points-8k", # or "GLUE3D-images", "GLUE3D-multiview", "GLUE3D-points"
    qa_task="binary_task",           # or "multiplechoice_task", "captioning_task"
    cache_dir=None,                  # Optional; defaults to './cache' or $GLUE3D_CACHE_DIR
)
```



This procedure loads to memory and prepare the necessary GLUE3D data for the specified Q&A task and data-type. It also automatically downloads all necessary data to disk if this is not yet stored. The available tasks are `binary_task`, `multiplechoice_task`, `captioning_task`. Note that the loader uses a local cache directory. You can customize it via the `GLUE3D_CACHE_DIR` environment variable.

Once the GLUE3D data is loaded, you can iterate through the dataset to generate answers for each question in the Q&A task:

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

> [!IMPORTANT]
>  Ensure your answers follow the expected format for each task.
> - For the `binary_task`, the model answer must be a boolean object (either `True` or `False`).
> - For the `multiplechoice_task`, the model answer must be one of `A`, `B`, `C`, `D`.
> - For the `captioning_task` the model answer must be a string.

</details>

<details>
<summary> Option 2: Using the `AnswerGenerator` Interface </summary>
    
### Using the `AnswerGenerator` Interface

If your 3D-LLM inherits the `GeneratorMixin` class (e.g., `LlavaLlamaForCausalLM`), then it is possible to use our `*HFAnswerGenerator` abstract classes to simplify the generation process. The only requirement is to implement the `prepare_inputs` function, which takes in input the point cloud (or image) and the question and returns the keyword inputs for the `GeneratorMixin.generate()` method:

```python
import numpy as np
from typing import override
from glue3d import generate_GLUE3D_answers
from glue3d.models.hf import (
    BinaryHFAnswerGenerator,
    MultichoiceHFAnswerGenerator,
    CaptioningHFGenerator
)

# Example custom AnswerGenerator for the binary task
class YourAnswerGenerator(BinaryHFAnswerGenerator): # <- Swap with MultichoiceHFAnswerGenerator
    def __init__(self, your_model, tokenizer):      #   or CaptioningHFGenerator for other tasks.
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

Once you have your custom implementation, generation can be simply done by calling `generate_GLUE3D_answers` on your target dataset-type and Q&A task:
```python
your_model = ...
answer_gen = YourAnswerGenerator(your_model)

qa_answers = generate_GLUE3D_answers(
    qa_task="binary_task",
    dataset_type="GLUE3D-points-8K",
    answer_generator=answer_gen,
)

# `qa_answers` is returned as a pandas DataFrame
qa_answers.to_csv("qa.csv", index=False)
```

</details>

---

## Q&A evaluation
As result of the [answer generation](#answer-generation) step, you should have a `.CSV` file containing the question-answer  pairs for a given task. The file (let us call it `binary-qa.csv`) should have a structure similar to

```csv
OBJECT_ID, QUESTION_ID, VALUE
dc5c798, 0fbac6, True
dc5c798, 556cc4, False
...
```

It is then possible to evaluate the answers produced by your model using the `glue3d evaluate` CLI command:
```bash
glue3d evaluate --input-file binary-qa.csv --output-file out.csv --task binary_task
```
Or equivalently, using Python
```python
from glu3d.evaluate_answers import evaluate_GLUE3D_answers

out = evaluate_GLUE3D_answers("binary_task", "binary-qa.csv")
out.to_csv("out.csv")
```

For the binary and multiple choice tasks, the output is a dataframe which indicates extact match between the question answer and the model provided one. For the captioning task, results scores for *BLEU*, *METEOR*, *ROUGE-L*, *S-BERT*, and *SimCSE* are provided. All scores are scaled to range between 0-100.

> [!NOTE]
> For the captioning task it is also possible to change the evaluator to use qwen3-30B-A3B as a judge. To do so use the command:
> ```bash
>glue3d evaluate --input-file captions.csv --output-file out.csv --task captioning_task --evaluator qwen_3_30B_A3B
>```

---
