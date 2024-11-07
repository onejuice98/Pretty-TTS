import os
import librosa
import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audio_source",
        type=str,
        required=True,
        help="Audio(dataset) source name"
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="Path to the CSV file containing filenames"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    audio_source = args.audio_source
    csv_file = args.csv_file
    data_path = f"dataset/{audio_source}/wavs"

    # 기본 통계 초기화
    sample_rates = []
    audio_lengths = []
    total_files = 0
    total_frames = 0
    total_seconds = 0
    total_size = 0
    max_input_length = 0
    max_output_length = 0

    # CSV 파일 읽기
    df = pd.read_csv(csv_file, delimiter='|', header=None)
    filenames = df[0].tolist()  # 첫 번째 컬럼에서 파일명 가져오기

    # wav 파일 처리
    for file_name in filenames:
        wav_file = file_name + ".wav"  # Assuming filenames don't include ".wav"
        file_path = os.path.join(data_path, wav_file)

        if os.path.exists(file_path):
            # 파일 크기 누적
            total_size += os.path.getsize(file_path)

            # 오디오 파일 로드 (librosa)
            audio, sample_rate = librosa.load(file_path, sr=None)

            # 샘플레이트 및 오디오 길이 저장
            sample_rates.append(sample_rate)
            audio_lengths.append(len(audio))

            # 총 프레임과 총 시간 누적
            frames = len(audio)
            duration = frames / float(sample_rate)

            total_frames += frames
            total_seconds += duration

            # 최대 입력/출력 길이 계산
            max_input_length = max(max_input_length, frames // sample_rate)  # 초 단위
            max_output_length = max(max_output_length, frames)

            # 파일 개수 증가
            total_files += 1

    # 전체 용량을 GB 단위로 변환
    total_size_gb = total_size / (1024 ** 3)

    # 샘플레이트 설정
    sample_rate = max(set(sample_rates), key=sample_rates.count)  # 가장 빈번한 샘플레이트 선택

    # 오디오 길이 설정
    max_wav_length = max(audio_lengths)  # 가장 긴 오디오 길이
    min_wav_length = min(audio_lengths)  # 가장 짧은 오디오 길이

    # Conditioning Length 계산
    min_conditioning_length = int(min_wav_length * 0.3)  # 예시로 최소 길이의 30%로 설정
    max_conditioning_length = int(max_wav_length * 0.7)  # 예시로 최대 길이의 70%로 설정

    # 결과 출력
    print("About KMA Dataset")
    print(f"{total_files} *.wav files (about {round(total_size_gb, 1)}GB)")
    print(f"Wrote {total_files} utterances, {total_frames} frames ({round(total_seconds / 3600, 2)} hours)")
    print(f"Max input length: {max_input_length}")
    print(f"Max output length: {max_output_length}")
    print("샘플레이트:", sample_rate)
    print("최대 오디오 길이 (samples):", max_wav_length)
    print("최소 오디오 길이 (samples):", min_wav_length)
    print("최소 Conditioning 길이:", min_conditioning_length)
    print("최대 Conditioning 길이:", max_conditioning_length)
