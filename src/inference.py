# import os
# from TTS.api import TTS
# import IPython
#
# # Define paths
# output_path = "C:/Users/ojuic/Documents/tts-for-prettygirl/run/kma_fine_tune_korean-November-02-2024_05+48AM-0000000"
# config_path = os.path.join(output_path, "config.json")  # Config file for the model
# speaker_wav_path = "C:/Users/ojuic/Documents/tts-for-prettygirl/dataset/KMA/wavs/0033_G2A3E1S0C1_KMA_000001.wav"  # 화자 오디오 파일 경로
# output_wav_path = "output_inference.wav"
#
# # Load the TTS model
# try:
#     tts = TTS(model_path=output_path, config_path=config_path)
#     # Run inference with speaker_wav specified
#     tts.tts_to_file(text="안녕하세요! 반가워요, 키랏!", speaker_wav=speaker_wav_path, file_path=output_wav_path, language="ko")
#     print("Inference completed, audio saved at:", output_wav_path)
#     IPython.display.Audio(output_wav_path)
# except FileNotFoundError as e:
#     print(f"File not found: {e}")

import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Define paths based on your environment
# output_path = "/convei_nas2/intern/jungsoo/c-arm-tts/tts-for-human/run/kma_fine_tune_korean-November-02-2024_06+12PM-27450b1"
# output_path = "/convei_nas2/intern/jungsoo/c-arm-tts/tts-for-human/TTS/run/training/GPT_XTTS_v2.0_BBANGHYONG_FT-November-02-2024_06+44PM-27450b1"
# output_path ="/convei_nas2/intern/jungsoo/c-arm-tts/tts-for-human/TTS/run/training/GPT_XTTS_v2.0_BBANGHYONG_FT-November-02-2024_08+19PM-27450b1"
# output_path = "/convei_nas2/intern/jungsoo/c-arm-tts/tts-for-human/TTS/run/training/GPT_XTTS_v2.0_HJH_FT-November-04-2024_12+35PM-27450b1"
output_path = "/convei_nas2/intern/jungsoo/c-arm-tts/tts-for-human/TTS/run/training/GPT_XTTS_v2.0_HJH_Trial2_FT-November-05-2024_12+13AM-27450b1" # trial2
# output_path = "/convei_nas2/intern/jungsoo/c-arm-tts/tts-for-human/TTS/run/training/GPT_XTTS_v2.0_HJH_Trial3_FT-November-05-2024_12+14AM-27450b1" # trial3
# output_path = "/convei_nas2/intern/jungsoo/c-arm-tts/tts-for-human/TTS/run/training/GPT_XTTS_v2.0_HJH_Trial4_High_Pitch_FT-November-05-2024_02+43PM-27450b1" # HJH High pitch
# output_path = "/convei_nas2/intern/jungsoo/c-arm-tts/tts-for-human/TTS/run/training/GPT_XTTS_v2.0_KMA_Trial5_High_Pitch_FT-November-05-2024_04+18PM-27450b1" # KMA High pitch
# output_path = "/convei_nas2/intern/jungsoo/c-arm-tts/tts-for-human/TTS/run/training/GPT_XTTS_v2.0_PMK_Trial6_High_Pitch_FT-November-05-2024_05+48PM-27450b1" # PMK High pitch

config_path = os.path.join(output_path, "config.json")
checkpoint_path = output_path  # assuming model checkpoint is in this directory
output_wav_path = "HJH_trial2_14.wav"
# speaker_wav_path = "/convei_nas2/intern/jungsoo/c-arm-tts/tts-for-human/dataset/KMA/wavs/sample_reference.wav"
speaker_wav_path = "/convei_nas2/intern/jungsoo/c-arm-tts/tts-for-human/dataset/0050_G2A4E7S0C2_HJH/wavs/0050_G2A4E7S0C2_HJH_000167.wav" # HJH
# speaker_wav_path = "/convei_nas2/intern/jungsoo/c-arm-tts/tts-for-human/dataset/0050_G2A4E7S0C2_HJH/wavs/0050_G2A4E7S0C2_HJH_000880.wav" # HJH High_pitch
# speaker_wav_path = "/convei_nas2/intern/jungsoo/c-arm-tts/tts-for-human/dataset/KMA/wavs/0033_G2A3E1S0C1_KMA_000474.wav" # KMA High_pitch
# speaker_wav_path = "/convei_nas2/intern/jungsoo/c-arm-tts/tts-for-human/dataset/0045_G2A3E1S0C1_PMK/wavs/0045_G2A3E1S0C1_PMK_000833.wav" # PMK High_pitch

# Load model configuration
print("Loading model configuration...")
config = XttsConfig()
config.load_json(config_path)

# Initialize and load the model
print("Initializing and loading model...")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=checkpoint_path, use_deepspeed=False, vocab_path='/convei_nas2/intern/jungsoo/c-arm-tts/tts-for-human/run/XTTS_v2.0_original_model_files/vocab.json')
model.cuda()

# Compute speaker latents using reference audio
print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[speaker_wav_path])

# Run inference with Korean text
print("Running inference...")
# output_text = "안녕하세요! 좋은 목소리를 내기 위해 많이 노력했습니다... 잘 부탁드립니다.."
# output_text = "안녕하세요! 좋은 목소리를 내기 위해 많이 노력했어요... 잘 부탁드립니다!"
# output_text = "안녕~! 좋은 목소리를 내기 위해 많이 노력했어~ 잘 부탁해~"
# output_text = "안녕하세요! 선생님! 제 목소리는 어떤가요?"


out = model.inference(
    text=output_text,
    language="ko",
    gpt_cond_latent=gpt_cond_latent,
    speaker_embedding=speaker_embedding,
    temperature=0.4  # Optional parameter for adjusting temperature
)

# Save the output to a WAV file
torchaudio.save(output_wav_path, torch.tensor(out["wav"]).unsqueeze(0), config.audio.output_sample_rate)
print(f"Inference completed, audio saved at: {output_wav_path}")

# export LD_LIBRARY_PATH=""