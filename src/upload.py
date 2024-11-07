from huggingface_hub import create_repo
from huggingface_hub import upload_file

SPEAKER_NAME = "KMA"
REPO_NAME = f"{SPEAKER_NAME}-xtts-v2-model-v0.8"
ROOT_MODEL_DIR = "/convei_nas2/intern/jungsoo/c-arm-tts/tts-for-human/run/KMA-v0.8"
# ROOT_MODEL_DIR = "/convei_nas2/intern/jungsoo/c-arm-tts/tts-for-human/TTS/run/training/GPT_XTTS_v2.0_BBANGHYONG_FT-November-02-2024_06+44PM-27450b1"
# ROOT_MODEL_DIR = "/convei_nas2/intern/jungsoo/c-arm-tts/tts-for-human/TTS/run/training/GPT_XTTS_v2.0_BBANGHYONG_FT-November-02-2024_08+19PM-27450b1"
# ROOT_MODEL_DIR = "/convei_nas2/intern/jungsoo/c-arm-tts/tts-for-human/TTS/run/training/GPT_XTTS_v2.0_HJH_FT-November-04-2024_12+35PM-27450b1"
# create_repo(REPO_NAME, private=False) 

file_paths = [
    f"{ROOT_MODEL_DIR}/config.json",
    f"{ROOT_MODEL_DIR}/best_model.pth",
    f"{ROOT_MODEL_DIR}/checkpoint_224604.pth",
    f"{ROOT_MODEL_DIR}/model.pth",
    f"{ROOT_MODEL_DIR}/train_xtts.py",
    f"{ROOT_MODEL_DIR}/trainer_0_log.txt",

] 

for file_path in file_paths:
    path_in_repo = file_path.split("/")[-1]

    upload_file(
        path_or_fileobj=file_path,
        path_in_repo=path_in_repo,
        repo_id=f"jsswon/{REPO_NAME}",
        repo_type="model"
    )
    print(f"파일 {path_in_repo} 업로드")

print("종료")