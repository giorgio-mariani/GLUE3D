from typing import *


def load_phi3_5_mini_instruct():
    from transformers import AutoTokenizer
    from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    model = Phi3ForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        # quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype="bfloat16",
        device_map="auto",
    )
    return model, tokenizer


def load_qwen3_30B_A3B_model() -> Tuple[Any]:
    from llama_cpp import Llama

    return (
        Llama(
            model_path="/data/Qwen_Qwen3-30B-A3B-Q4_K_M.gguf",
            n_gpu_layers=-1,
            n_ctx=8000,
            verbose=True,
        ),
    )
