[![python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)

### Install
```
git clone https://github.com/onejuice98/Pretty-TTS.git
pip install -r requirements.txt
```

### Load Dataset
**Download Link**
[AIHub 감성 및 발화 스타일별 음성합성 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=466)
- TL4.zip | 130.60 MB
- TS4.zip | 34.47 GB
<br>
Download at directory `dataset/TL4`, `dataset/TS4`
<br>
**Processing Dataset & Make Dataset LJSpeech Format**
```
python src/ljspeech_format.py
```

### Train
Use [Coqui/TTS](https://github.com/coqui-ai/TTS)
Execute train code

### Inference
1. Upload Exist Pretty TTS [(HuggingFace)](https://huggingface.co/jsswon/PMK-xtts-v2-model-v1) with `src/uplad.py`
2. Change Utterance in `dataset/utterance.json`
3. Execute script `bash scripts/inference.bash`
