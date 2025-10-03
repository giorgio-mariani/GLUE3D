from typing import *

from glue3d.models.base import AnswerGenerator


class TextAnswerGenerator(AnswerGenerator):
    def __init__(self, model, tokenizer, system_msg: Optional[str] = None, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.kwargs = kwargs
        if system_msg is not None:
            self.prefix_msgs = ({"role": "system", "content": system_msg},)
        else:
            self.prefix_msgs = tuple()

    def __call__(self, data: str, text: str) -> str:
        import torch

        with torch.inference_mode():
            # Preparation for inference
            device = self.model.device
            caption, question = data, text
            qa_message = dict(
                role="user",
                content=f"Consider the following: {caption}\n Answer with the most plausible option. {question}",
            )
            inputs = self.tokenizer.apply_chat_template(
                self.prefix_msgs + (qa_message,),
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)

            # Inference: Generation of the output
            output_ids = self.model.generate(**inputs, **self.kwargs)
            generated_ids = [
                output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            outputs = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            (output,) = outputs
            return output.strip()


class MultichoiceTextAnswerGenerator(TextAnswerGenerator):
    def __init__(self, model, tokenizer, choices, postprocess_fn=None, **kwargs):
        super().__init__(model, tokenizer, **kwargs)

        self.post_process_fn = postprocess_fn if postprocess_fn is not None else lambda x: x
        self.choice_ids = {}
        for c in choices:
            (c_token,) = self.tokenizer.encode(c, add_special_tokens=False)
            self.choice_ids[c] = c_token

    def __call__(self, data: str, text: str) -> str:

        import torch

        with torch.inference_mode():
            # Preparation for inference
            device = self.model.device
            caption, question = data, text
            qa_message = dict(
                role="user",
                content=f"Consider the following: {caption}\n Answer with the most plausible option. {question}",
            )
            inputs = self.tokenizer.apply_chat_template(
                self.prefix_msgs + (qa_message,),
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)

            # Inference: Generation of the output
            model_output = self.model(**inputs, return_dict=True, **self.kwargs)

            # Generate outputs
            answer_prob = {}
            logits = model_output.logits[:, -1:]  # <- [B, N, D]
            probs = torch.softmax(logits, dim=-1)
            for choice, choice_id in self.choice_ids.items():
                answer_prob[choice] = probs[:, :, choice_id].item()

            return self.post_process_fn(max(answer_prob, key=answer_prob.get))
