from typing import *

import numpy as np

from glue3d.models import AnswerGenerator


class MiniGPT3DAnswerGenerator(AnswerGenerator):
    def __init__(self, model: "MiniGPT_3D", **kwargs):
        super().__init__()

        from minigpt4.models.minigpt_v2 import MiniGPT_3D
        from minigpt4.conversation.conversation import CONV_VISION

        self.model = model
        self.kwargs = kwargs
        self.conv_template = CONV_VISION.copy()
        self.conv_template.system = ""

    def prepare_text(self, text: str) -> str:
        conv = self.conv_template.copy()
        conv.append_message(conv.roles[0], "<PC><PointCloudHere></PC> <s>[INST] {}".format(text))
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    @property
    def device(self):
        return self.model.llama_model.device

    @property
    def dtype(self):
        return self.model.llama_model.dtype

    def __call__(self, data: np.ndarray, text: str) -> str:
        import torch

        with torch.inference_mode():
            point_clouds = torch.from_numpy(data).to(device=self.device, dtype=self.dtype).unsqueeze(0)
            texts = [text]
            self.model.eval()
            point_clouds = point_clouds.to(device=self.device, dtype=self.dtype)

            prepared_texts = [self.prepare_text(t) for t in texts]
            answers = self.model.generate(point_clouds, prepared_texts, **self.kwargs)

            outputs = []
            for answer in answers:
                answer = answer.lower().replace("<unk>", "").strip()
                answer = answer.split("###")[0]  # remove the stop sign '###'
                answer = answer.split("Assistant:")[-1].strip()
                outputs.append(answer)

            (outputs,) = outputs
            return outputs


class MultiChoiceMiniGPT3DAnswerGenerator(MiniGPT3DAnswerGenerator):
    def __init__(self, model, choices: List[str], postprocess_fn: Optional[Callable] = None, **kwargs):
        super().__init__(model, **kwargs)
        self.tokenizer, self.llm = self.model.llama_tokenizer, self.model.llama_model
        self.post_process_fn = postprocess_fn if postprocess_fn is not None else lambda x: x

        self.choice_ids = {}
        for c in choices:
            (c_token,) = self.tokenizer.encode(c, add_special_tokens=False)
            self.choice_ids[c] = c_token

    def _create_input_emebbings(self, texts: List[str], point_clouds: "torch.Tensor"):
        import torch

        minigpt3d = self.model
        point_clouds = point_clouds.to(dtype=self.dtype, device=self.device)

        pc_embeds, _ = minigpt3d.encode_pc(point_clouds)
        pc_lists = [[pc_emb[None]] for pc_emb in pc_embeds]
        batch_embs = [minigpt3d.get_context_emb(text, pc_list) for text, pc_list in zip(texts, pc_lists)]
        batch_size = len(batch_embs)

        # Padding --------
        max_len = max([emb.shape[1] for emb in batch_embs])
        emb_dim = batch_embs[0].shape[2]

        embs = torch.zeros([batch_size, max_len, emb_dim], dtype=self.dtype, device=self.device)
        attn_mask = torch.zeros([batch_size, max_len], dtype=torch.int, device=self.device)

        for i, emb in enumerate(batch_embs):
            emb_len = emb.shape[1]
            embs[i, -emb_len:] = emb[0]
            attn_mask[i, -emb_len:] = 1
        return embs, attn_mask

    def __call__(self, data: np.ndarray, text: str) -> str:
        import torch

        with torch.inference_mode():
            self.model.eval()
            point_clouds = torch.from_numpy(data).to(device=self.device, dtype=self.dtype).unsqueeze(0)
            texts = [text]
            assert len(texts) == 1
            prepared_texts = [self.prepare_text(t) for t in texts]
            input_embs, attn_mask = self._create_input_emebbings(prepared_texts, point_clouds)

            llm = self.model.llama_model
            with self.model.maybe_autocast():
                model_output = llm(
                    inputs_embeds=input_embs.to(dtype=self.dtype, device=self.device),
                    attention_mask=attn_mask.to(dtype=self.dtype, device=self.device),
                    # bos_token_id=50256,
                    # pad_token_id=tokenizer.eos_token_id,
                    return_dict=True,
                    **self.kwargs,
                )

            # Generate outputs
            answer_prob = {}
            logits = model_output.logits[:, -1:]  # <- [B, N, D]
            probs = torch.softmax(logits, dim=-1)
            for choice, choice_id in self.choice_ids.items():
                answer_prob[choice] = probs[:, :, choice_id].item()

            return self.post_process_fn(max(answer_prob, key=answer_prob.get))
