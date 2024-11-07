import os
import glob
import librosa
import argparse
import pandas as pd
import numpy as np

from util.processing import get_file_list, load_json

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audio_source",
        type=str,
        required=True,
        help="Audio(dataset) source name"
    )
    parser.add_argument(
        "--pitch_threshold",
        type=int,
        required=True,
        help="If audio pitch exceed threshold, it is filtered."
    )
    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
        default="metadata",
        help="Name of output metadata."
    )
    
    return parser.parse_args()

# # 0050_G2A4E7S0C2_HJH, KMA, 0045_G2A3E1S0C1_PMK
# NAME = "0045_G2A3E1S0C1_PMK"
# label_path = "/convei_nas2/intern/jungsoo/c-arm-tts/dataset/TL4"
# mp3_path = "/convei_nas2/intern/jungsoo/c-arm-tts/dataset/TS4"
# # HJH: 1600 / KMA: 1690 / PMK: 1546
# pitch_threshold = 1546
#
# # /convei_nas2/intern/jungsoo/c-arm-tts/tts-for-human/dataset/0045_G2A3E1S0C1_PMK
# # /convei_nas2/intern/jungsoo/c-arm-tts/dataset/TS4/3.발성캐릭터/1.아동/0050_G2A4E7S0C2_HJH
# # 0045_G2A3E1S0C1_PMK

def get_metadata(audio_source, pitch_threshold, output_name):
    voice_label_path_list = sorted([f for f in glob.glob("dataset/TL4" + "/**/**/*") if audio_source in f])
    voice_mp3_path_list = sorted([f for f in glob.glob("dataset/TS4" + "/**/**/*") if audio_source in f])

    label_csv = []

    for voice_label_path in voice_label_path_list:
        voice_label_list = get_file_list(voice_label_path, extension="json")
        for voice_label in voice_label_list:
            voice_data = load_json(f"{voice_label_path}/{voice_label}")

            file_name = voice_data["파일정보"]["FileName"].split(".")[0]
            org_label_text = voice_data["전사정보"]["TransLabelText"]
            trans_label_text = voice_data["전사정보"]["TransLabelText"]
            character_emotion = voice_data["화자정보"]["CharacterEmotion"]

            # 오디오 파일 경로 설정 및 피치 계산
            audio_path = os.path.join(voice_mp3_path_list[0], f"{file_name}.wav")
            if os.path.exists(audio_path):
                y, sr = librosa.load(audio_path, sr=None)
                pitches, _ = librosa.core.piptrack(y=y, sr=sr)
                pitches_nonzero = pitches[pitches > 0]
                
                # 평균 피치가 threshold 이상인 경우에만 추가
                if len(pitches_nonzero) > 0 and np.mean(pitches_nonzero) >= pitch_threshold:
                    label_csv.append([file_name, org_label_text, trans_label_text])

    df = pd.DataFrame(label_csv, columns=["FileName", "TransLabelText", "TransLabelText"])
    csv_path = f"dataset/{audio_source}/{output_name}.csv"
    df.to_csv(csv_path, index=False, sep="|", header=False)

    print("CSV 파일 저장 완료:", csv_path)
    
if __name__ == "__main__":
    args = parse_args()
    audio_source, pitch_threshold, output_name = args.audio_source, args.pitch_threshold, args.output_name
    
    get_metadata(audio_source, pitch_threshold, output_name)

    