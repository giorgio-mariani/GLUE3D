from huggingface_hub import snapshot_download

snapshot_download(repo_id="YuanTang96/MiniGPT-3D", local_dir=".", ignore_patterns=["README.md"])
