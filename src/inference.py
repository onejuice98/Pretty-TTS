import os
import torch
import torchaudio
import argparse

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from util.processing import load_json

REFERENCE_SPEAKER = {
    "HJH-xtts-v2-model": "dataset/0050_G2A4E7S0C2_HJH/wavs/0050_G2A4E7S0C2_HJH_000167.wav",
    "HJH-xtts-v2-model-v1.1": "dataset/0050_G2A4E7S0C2_HJH/wavs/0050_G2A4E7S0C2_HJH_000167.wav",
    "HJH-xtts-v2-model-v1.2": "dataset/0050_G2A4E7S0C2_HJH/wavs/0050_G2A4E7S0C2_HJH_000167.wav",
    "HJH-xtts-v2-model-v1.3": "dataset/0050_G2A4E7S0C2_HJH/wavs/0050_G2A4E7S0C2_HJH_000880.wav",
    "KMA-xtts-v2-model-v0.9": "dataset/KMA/wavs/0050_G2A4E7S0C2_HJH_000880.wav",
    "KMA-xtts-v2-model-v1.1": "dataset/KMA/wavs/0033_G2A3E1S0C1_KMA_000474.wav",
    "KMA-xtts-v2-model": "dataset/KMA/wavs/0033_G2A3E1S0C1_KMA_000474.wav",
    "KMA-xtts-v2-model-v0.8": "dataset/KMA/wavs/0033_G2A3E1S0C1_KMA_000474.wav",
    "PMK-xtts-v2-model-v1": "dataset/0045_G2A3E1S0C1_PMK/wavs/0045_G2A3E1S0C1_PMK_000833.wav"
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_source", type=str, required=True, help="Audio(dataset) source name")
    parser.add_argument("--model_name", type=str, required=True, choices=REFERENCE_SPEAKER.keys(), help="Select model")
    parser.add_argument("--all_mode", type=str, required=True, choices=["true", "false"], help="Select mode for all")
    parser.add_argument("--auto_utterance", type=str, required=True, choices=["true", "false"], help="Select mode for auto utterance")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    audio_source, model_name, all_mode, auto_utterance = args.audio_source, args.model_name, args.all_mode, args.auto_utterance

    output_text = "안녕하세요! 선생님! 제 목소리는 어떤가요?"
    output_text_list = load_json("dataset/utterance.json") if auto_utterance == "true" else [{"text": output_text}]

    model_names = REFERENCE_SPEAKER.keys() if all_mode == "true" else [model_name]

    for model_name in model_names:
        print(f"Processing model: {model_name}")
        config_path = f"models/{model_name}/config.json"
        speaker_wav_path = REFERENCE_SPEAKER[model_name]

        # Load model configuration
        print("Loading model configuration...")
        config = XttsConfig()
        config.load_json(config_path)

        # Initialize and load model
        print("Initializing and loading model...")
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_dir=f"models/{model_name}",
            use_deepspeed=False,
            vocab_path=f"run/XTTS_v2.0_original_model_files/vocab.json"
        )
        model.cuda()

        # Compute speaker latents using reference audio
        print("Computing speaker latents...")
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[speaker_wav_path])

        # Ensure output directory exists
        model_output_dir = f"output/{model_name}"
        os.makedirs(model_output_dir, exist_ok=True)

        # Run inference and save each utterance
        for i, utterance in enumerate(output_text_list):
            utterance_text = utterance["text"]

            # Inference
            print(f"Running inference for {model_name} - Utterance {i}...")
            out = model.inference(
                text=utterance_text,
                language="ko",
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=0.4
            )

            # Determine output file path
            if auto_utterance == "true":
                output_wav_path = f"{model_output_dir}/{i}.wav"
            else:
                existing_files = sorted([int(f.split(".")[0]) for f in os.listdir(model_output_dir) if f.endswith(".wav")])
                index = (existing_files[-1] + 1) if existing_files else 0
                output_wav_path = f"{model_output_dir}/{index}.wav"

            # Save the output to a WAV file
            torchaudio.save(output_wav_path, torch.tensor(out["wav"]).unsqueeze(0), config.audio.output_sample_rate)
            print(f"Inference completed for {model_name} - Utterance {i}, audio saved at: {output_wav_path}")
