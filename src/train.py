import os
import librosa
import torch
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager
from trainer import Trainer, TrainerArgs

# 현재 디렉토리 가져오기
current_dir = os.getcwd()
data_path = os.path.join(current_dir, "dataset/KMA/wavs")  # 현재 디렉토리에 기반한 전체 경로

# 데이터셋 분석을 위한 리스트 초기화
sample_rates = []
audio_lengths = []

# 오디오 파일 정보 추출
for file_name in os.listdir(data_path):
    if file_name.endswith(".wav"):
        file_path = os.path.join(data_path, file_name)

        # 오디오 파일 로드
        audio, sample_rate = librosa.load(file_path, sr=None)

        # 샘플레이트 및 오디오 길이 저장
        sample_rates.append(sample_rate)
        audio_lengths.append(len(audio))

# 1. 샘플레이트 설정
sample_rate = max(set(sample_rates), key=sample_rates.count)  # 가장 빈번한 샘플레이트 선택

# 2. 오디오 길이 설정
max_wav_length = max(audio_lengths)  # 가장 긴 오디오 길이
min_wav_length = min(audio_lengths)  # 가장 짧은 오디오 길이

# 3. Conditioning Length 계산
min_conditioning_length = int(min_wav_length * 0.3)  # 최소 길이의 30%로 설정
max_conditioning_length = int(max_wav_length * 0.7)  # 최대 길이의 70%로 설정

# 결과 출력
print("현재 디렉토리:", current_dir)
print("데이터 경로:", data_path)
print("샘플레이트:", sample_rate)
print("최대 오디오 길이 (samples):", max_wav_length)
print("최소 오디오 길이 (samples):", min_wav_length)
print("최소 Conditioning 길이:", min_conditioning_length)
print("최대 Conditioning 길이:", max_conditioning_length)

# 모델 파일 경로 설정
OUT_PATH = os.path.join(current_dir, "run")
CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

# DVAE 및 MEL normalization 파일 경로 설정
DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, "dvae.pth")
MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, "mel_stats.pth")
TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, "vocab.json")
XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, "model.pth")

# 학습 파라미터 설정
model_args = GPTArgs(
    max_conditioning_length=max_conditioning_length,
    min_conditioning_length=min_conditioning_length,
    debug_loading_failures=True,
    max_wav_length=max_wav_length,
    max_text_length=200,
    mel_norm_file=MEL_NORM_FILE,
    dvae_checkpoint=DVAE_CHECKPOINT,
    xtts_checkpoint=XTTS_CHECKPOINT,
    tokenizer_file=TOKENIZER_FILE,
    gpt_num_audio_tokens=1026,
    gpt_start_audio_token=1024,
    gpt_stop_audio_token=1025,
    gpt_use_masking_gt_prompt_approach=True,
    gpt_use_perceiver_resampler=True,
)

# 오디오 설정
audio_config = XttsAudioConfig(sample_rate=sample_rate, dvae_sample_rate=sample_rate, output_sample_rate=24000)

# 학습 설정
config = GPTTrainerConfig(
    run_eval=True,
    epochs=1000,
    output_path=OUT_PATH,
    model_args=model_args,
    run_name="kma_fine_tune_korean",
    project_name="KMA_Korean_TTS",
    run_description="Fine-tuning XTTS-v2 on Korean KMA dataset",
    audio=audio_config,
    batch_size=1,
    batch_group_size=48,
    eval_batch_size=1,
    num_loader_workers=8,
    eval_split_max_size=256,
    print_step=50,
    plot_step=100,
    save_step=5000,
    save_n_checkpoints=1,
    save_checkpoints=True,
    optimizer="AdamW",
    lr=5e-6,
    test_sentences=[
        {
            "text": "안녕하세요, 오늘 기분이 어떠세요?",
            "speaker_wav": os.path.join(data_path, "sample_reference.wav"),
            "language": "ko",
        }
    ],
)

# 데이터셋 설정 및 로드
dataset_config = BaseDatasetConfig(
    formatter="ljspeech", meta_file_train="metadata.csv", language="ko", path=os.path.join(current_dir, "dataset/KMA")
)
train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True, eval_split_size=0.02)

# Trainer 인스턴스 생성 및 학습 시작
trainer = Trainer(
    TrainerArgs(
        restore_path=None,
        skip_train_epoch=False,
        start_with_eval=True,
        grad_accum_steps=252,
    ),
    config,
    output_path=OUT_PATH,
    model=None,  # 모델을 직접 초기화하는 코드 필요 시 추가
    train_samples=train_samples,
    eval_samples=eval_samples,
)

trainer.fit()
