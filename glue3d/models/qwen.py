from typing import *

from glue3d.models.base import AnswerGenerator


class QwenAnswerGenerator(AnswerGenerator):
    def __init__(
        self,
        model,
        processor,
        **kwargs,
    ):
        self.model = model
        self.processor = processor
        self.kwargs = kwargs

    def prepare_data(self, data: Any, text: str) -> Any:
        from qwen_vl_utils import process_vision_info

        if isinstance(data, dict):
            images = data["images"]
        else:
            images = [data]

        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": img} for img in images] + [{"type": "text", "text": text}],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs

    def __call__(
        self,
        data: Any,
        text: str,
    ) -> str:
        import torch

        with torch.inference_mode():
            inputs = self.prepare_data(data, text).to(self.model.device)

            # Inference: Generation of the output
            output_ids = self.model.generate(**inputs, **self.kwargs)
            generated_ids = [
                output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            outputs = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            (output,) = outputs
        return output.strip()


class MultichoiceQwenAnswerGenerator(QwenAnswerGenerator):
    def __init__(self, model, processor, choices, postprocess_fn=None, **kwargs):
        super().__init__(model, processor, **kwargs)
        self.post_process_fn = postprocess_fn if postprocess_fn is not None else lambda x: x

        self.choice_ids = {}
        for c in choices:
            (c_token,) = self.processor.tokenizer.encode(c, add_special_tokens=False)
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

            output = self.post_process_fn(max(answer_prob, key=answer_prob.get))
        return output
