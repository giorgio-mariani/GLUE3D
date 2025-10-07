import abc
from typing import *

from glue3d.models.base import AnswerGenerator


class HFAnswerGenerator(AnswerGenerator):
    def __init__(self, model, tokenizer, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.kwargs = kwargs

    @abc.abstractmethod
    def prepare_data(self, data: Any, text: str) -> Any: ...

    def __call__(self, data: Any, text: str) -> str:
        import torch

        with torch.inference_mode():
            inputs = self.prepare_data(data, text).to(self.model.device)

            # Inference: Generation of the output
            output_ids = self.model.generate(**inputs, **self.kwargs)
            generated_ids = [
                output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            (output,) = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
        return output.strip()


class MultichoiceHFGenerator(HFAnswerGenerator):
    def __init__(self, model, tokenizer, choices: List[str] = ("A", "B", "C", "D"), **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.choices = choices
        self.choice_ids = {}
        for c in choices:
            (c_token,) = self.tokenizer.encode(c, add_special_tokens=False)
            self.choice_ids[c] = c_token

    def __call__(self, data: Any, text: str) -> str:
        import torch

        with torch.inference_mode():
            inputs = self.prepare_data(data, text).to(self.model.device)

            # Inference: Generation of the output
            model_output = self.model(**inputs, return_dict=True, **self.kwargs)

            # Generate outputs
            answer_prob = {}
            logits = model_output.logits[:, -1:].cpu()  # <- [B, N, D]
            probs = torch.softmax(logits, dim=-1)
            for choice, choice_id in self.choice_ids.items():
                answer_prob[choice] = probs[:, :, choice_id].item()

            return max(answer_prob, key=answer_prob.get)


class BinaryHFGenerator(MultichoiceHFGenerator):
    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(model, tokenizer, choices=["Yes", "No"], **kwargs)


class CaptioningHFGenerator(HFAnswerGenerator):
    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(model, tokenizer, max_new_tokens=128, **kwargs)
