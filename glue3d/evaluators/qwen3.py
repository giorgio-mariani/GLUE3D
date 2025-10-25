import re
from typing import *

from glue3d.evaluators.base import MetaJudge


PROMPT_GLUE3D_CAPTION = """Evaluate a model-generated caption against a human-provided caption (ground truth).
The former is the output from a captioning system; the latter is written by a human annotator.

Consider the two captions and provide a score representing how coherent the two sentences are with each other.
Your response should consist of a single confidence score ranging from 0 to 100.

Below are several examples of question-answer pairs along with their corresponding confidence scores:

question1: How many oranges will there be if 1/3 of them are removed?
answer from model: There will be 6 left.
answer from label: As there are 9 oranges in total, there will be 6 oranges left if 1/3 of them are removed.
confidence score: 100

question2: What is this object?
answer from model: This is a bathtub
answer from label: This is a dirty bathtub.
confidence score: 80

question3: What is this object?
answer from model: This is a bottle of water
answer from label: This is a bottle of oil
confidence score: 50

question4: What does the boy have in his right hand?
answer from model: He is holding a white cup in his right hand.
answer from label: He is holding a sword in his right hand.
confidence score: 0

Next, I will give you the following inputs:
question: What is this object?
answer from model: {model_output}
answer from label: {ground_truth}

Please remember: your output should contain **only** the confidence score, no words or punctuation, just the number.
"""


PROMPT_GLUE3D_OPEN_QA = """Evaluate a model-generated answer against a human-provided answer (ground truth).
The former is the output from a Visual Q&A system; the latter is written by a human annotator.

Consider the two answers and provide a score representing how consistent is the model-generated answer with the human-provided one.
Your response should consist of a single confidence score ranging from 0 to 100.
Focus on the information content provided in model-generated answer. Is it consistent with the ground truth? Is it not?

The score should **not** be influenced by:
 - The syntax of the answer; "There are three sticks." is equivalent to "3 sticks".
 - The presence of additional information in model generated answer should not negatively influence the score.
 - The confidence of the answer should not negatively influence the score: "This appears to be an apple" should be score the same as "This is an apple."
 - Grammar errors should not influence the confidence score.

---

Below are several examples of question-answer pairs along with their corresponding confidence scores:

question: How many oranges will there be if 1/3 of them are removed?
answer from model: There will be 6 left.
answer from label: As there are 9 oranges in total, there will be 6 oranges left if 1/3 of them are removed.
confidence score: 100

question: How many wheels doe the car have?
answer from model: 3
answer from label: The car has three wheels.
confidence score: 100

question: What is the content of the bottle?
answer from model: This is a bottle of water.
answer from label: This is a bottle of oil.
confidence score: 0

question: What does the boy have in his right hand?
answer from model: He is holding a white cup in his right hand.
answer from label: The boy is holding a white glass in his right hand.
confidence score: 50

---

Next, estimate the confidence score for the following answer pair:
question: {question}?
answer from model: {model_output}
answer from label: {ground_truth}

Please remember, your output should contain **only** the confidence score, no words or punctuation, just the number.
"""


class Qwen3Judge(MetaJudge):
    def __init__(self, model, prompt=PROMPT_GLUE3D_CAPTION):
        super().__init__()
        self.model = model
        self.prompt = prompt

    def parse_generation(self, response: str):
        return int(response)

    def judge_answer(self, ground_truth: str, model_output: str) -> Mapping[str, Any]:

        response = self.chat_complete(self.prompt.format(ground_truth=ground_truth, model_output=model_output))
        score = self.parse_generation(response)

        return {"QWEN_SCORE": score}

    def chat_complete(self, prompt_string: str):

        messages = [
            {"role": "system", "content": "You are an LLM Judge.\n/no_think"},
            {"role": "user", "content": prompt_string},
        ]

        output = self.model.create_chat_completion(messages, top_k=1)
        assitant_response = output["choices"][0]["message"]["content"]

        match = re.fullmatch("<think>(?P<thought>.|\n)*</think>(\s)*(?P<response>(.|\n)*)", assitant_response)
        return match.group("response")


class Qwen3JudgeOpenQA(Qwen3Judge):
    def __init__(self, model):
        super().__init__(model, PROMPT_GLUE3D_OPEN_QA)

    def judge_answer(self, ground_truth, model_output):
        prompt = self.prompt.format(
            question=self._all_kwargs_last_call["QUESTION"],
            ground_truth=ground_truth,
            model_output=model_output,
        )
        response = self.chat_complete(prompt)
        score = self.parse_generation(response)

        return {"QWEN_SCORE": score}
