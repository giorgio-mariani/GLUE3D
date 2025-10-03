from typing import *

import numpy as np

from glue3d.models import AnswerGenerator


class PointLLMAnswerGenerator(AnswerGenerator):
    def __init__(
        self,
        model: "PointLLMLlamaForCausalLM",
        tokenizer: "PreTrainedTokenizer",
        **kwargs,
    ):

        # Load pointllm only when instantiating
        from pointllm.conversation import conv_templates, SeparatorStyle
        from pointllm.model import PointLLMLlamaForCausalLM

        assert isinstance(model, PointLLMLlamaForCausalLM)
        # assert isinstance(tokenizer, PreTrainedTokenizer)

        self.model = model
        self.tokenizer = tokenizer
        self.conv_template = conv_templates["vicuna_v1_1"].copy()
        self.kwargs = kwargs

        self.stop_string = (
            self.conv_template.sep if self.conv_template.sep_style != SeparatorStyle.TWO else self.conv_template.sep2
        )
        self.point_backbone_config = self.model.get_model().point_backbone_config

    @property
    def device(self):
        return self.model.device

    @property
    def dtype(self):
        return self.model.dtype

    def prepare_text(self, text: str) -> str:
        point_backbone_config = self.point_backbone_config
        point_token_len = point_backbone_config["point_token_len"]
        point_patch_token = point_backbone_config["default_point_patch_token"]
        point_start_token = point_backbone_config["default_point_start_token"]
        point_end_token = point_backbone_config["default_point_end_token"]
        mm_use_point_start_end = point_backbone_config["mm_use_point_start_end"]

        if mm_use_point_start_end:
            user_text = point_start_token + point_patch_token * point_token_len + point_end_token + "\n" + text
        else:
            user_text = point_patch_token * point_token_len + "\n" + text

        conv = self.conv_template.copy()
        conv.append_message(conv.roles[0], user_text)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def __call__(self, data: np.ndarray, text: str) -> str:
        import torch

        with torch.inference_mode():
            point_clouds = torch.from_numpy(data).to(device=self.device, dtype=self.dtype).unsqueeze(0)
            texts = [text]
            device, dtype = self.model.device, self.model.dtype

            prepared_texts = [self.prepare_text(t) for t in texts]
            input_ids = self.tokenizer(prepared_texts, return_tensors="pt").input_ids.to(device=device)
            batch_size, sequence_len = input_ids.shape

            # Prepare stopping criteria
            from pointllm.model.utils import KeywordsStoppingCriteria

            stopping_criteria = KeywordsStoppingCriteria([self.stop_string], self.tokenizer, input_ids)

            # Generate outputs
            self.model.eval()
            output_ids = self.model.generate(
                input_ids,
                point_clouds=point_clouds.to(device=device, dtype=dtype),
                **self.kwargs,
                stopping_criteria=[stopping_criteria],
            )  # <- (B, L)

            assert (input_ids != output_ids[:, :sequence_len]).sum().item() == 0
            outputs = self.tokenizer.batch_decode(output_ids[:, sequence_len:], skip_special_tokens=True)
            (output,) = outputs
            return output.strip()


class MultiChoicePointLLMAnswerGenerator(PointLLMAnswerGenerator):
    def __init__(self, model, tokenizer, choices: List[str], postprocess_fn: Optional[Callable] = None, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.choice_ids = {}
        for c in choices:
            (c_token,) = self.tokenizer.encode(c, add_special_tokens=False)
            self.choice_ids[c] = c_token
        self.post_process_fn = postprocess_fn if not postprocess_fn is None else lambda x: x

    def __call__(self, data: np.ndarray, text: str) -> str:
        import torch

        with torch.inference_mode():
            point_clouds = torch.from_numpy(data).to(device=self.device, dtype=self.dtype).unsqueeze(0)
            texts = [text]

            device, dtype = self.model.device, self.model.dtype
            prepared_texts = [self.prepare_text(t) for t in texts]

            input_ids = self.tokenizer(prepared_texts, return_tensors="pt").input_ids.to(device=device)
            batch_size, sequence_len = input_ids.shape
            assert batch_size == 1

            # Generate outputs
            self.model.eval()
            answer_prob = {}
            model_output = self.model(
                input_ids,
                return_dict=True,
                point_clouds=point_clouds.to(device=device, dtype=dtype),
                **self.kwargs,
            )

            logits = model_output.logits[:, -1:]  # <- [B, N, D]
            probs = torch.softmax(logits, dim=-1)
            for choice, choice_id in self.choice_ids.items():
                answer_prob[choice] = probs[:, :, choice_id].item()

            return self.post_process_fn(max(answer_prob, key=answer_prob.get))
