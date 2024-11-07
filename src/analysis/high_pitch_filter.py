import os
import librosa
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audio_source",
        type=str,
        required=True,
        help="Audio(dataset) source name"
    )
    
    return parser.parse_args()

def get_mean_pitch(audio_dir, audio_source):
    pitches_all = []

    # 모든 파일의 피치를 계산하고 평균 피치를 구함
    for filename in os.listdir(audio_dir):
        if audio_source not in filename:
            continue
        
        if filename.endswith('.wav'):
            filepath = os.path.join(audio_dir, filename)
            
            # 오디오 파일 로드
            y, sr = librosa.load(filepath, sr=None)
            
            # 피치 추출
            pitches, _ = librosa.core.piptrack(y=y, sr=sr)
            
            # 피치가 0이 아닌 값들만 수집
            pitches_nonzero = pitches[pitches > 0]
            if len(pitches_nonzero) > 0:
                mean_pitch = np.mean(pitches_nonzero)
                pitches_all.append(mean_pitch)

    pitch_threshold = np.mean(pitches_all)
    print(f"Calculated pitch threshold: {pitch_threshold}")
    return pitch_threshold

def analysis_audio(audio_dir, audio_source, pitch_threshold):
    high_pitch_files = []
    for filename in os.listdir(audio_dir):
        if audio_source not in filename:
            continue
        
        if filename.endswith('.wav'):
            filepath = os.path.join(audio_dir, filename)
            
            y, sr = librosa.load(filepath, sr=None)
            
            pitches, _ = librosa.core.piptrack(y=y, sr=sr)
            
            pitches_nonzero = pitches[pitches > 0]
            if len(pitches_nonzero) > 0:
                mean_pitch = np.mean(pitches_nonzero)
                
                if mean_pitch >= pitch_threshold:
                    high_pitch_files.append(filename)

    print("High-pitch files:", high_pitch_files)
    print(f"Total high-pitch samples: {len(high_pitch_files)}")

if __name__ == "__main__":
    args = parse_args()
    audio_source = args.audio_source
    
    audio_dir = f"dataset/{audio_source}/wavs"

    pitch_threshold = get_mean_pitch(audio_dir, audio_source)
    analysis_audio(audio_dir, audio_source, pitch_threshold)
