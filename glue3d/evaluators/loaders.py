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
    from huggingface_hub import hf_hub_download

    model_filepath = hf_hub_download(
        repo_id="unsloth/Qwen3-30B-A3B-GGUF",
        filename="Qwen3-30B-A3B-Q4_K_M.gguf",
    )

    return (
        Llama(
            model_path=model_filepath,
            n_gpu_layers=-1,
            n_ctx=8000,
            verbose=True,
        ),
    )
