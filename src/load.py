from huggingface_hub import snapshot_download

# 모델 다운로드 경로 지정
local_dir = "models/KMA-xtts-v2-model-v0.8"
model_name = "jsswon/KMA-xtts-v2-model-v0.8"

# 모델 다운로드
snapshot_download(repo_id=model_name, local_dir=local_dir)