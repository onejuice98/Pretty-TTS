Base Model: [XTTS-v2](https://huggingface.co/coqui/XTTS-v2)
<br>
Best Pretty Girl Model is [here](https://huggingface.co/jsswon/PMK-xtts-v2-model-v1).

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

<br>

At Root Directory & Modify `scripts/filter.bash` for tailored dataset

```
base scripts/filter.bash
```

### Train
Use [Coqui/TTS](https://github.com/coqui-ai/TTS)

**Exeucute train code at TTS directory**

```
CUDA_VISIBLE_DEVICES="0" python TTS/train.py
```
Train Environment: NVIDIA RTX A5000 (24GB)
<br>
Training Duration: 6.275 hours
<br>
Hyperparameter is [here](https://huggingface.co/jsswon/PMK-xtts-v2-model-v1/blob/main/train.py).

### Inference
0. Execute `mkdir models` at Root directory
1. Upload Exist Pretty TTS [(HuggingFace)](https://huggingface.co/jsswon/PMK-xtts-v2-model-v1) with `src/uplad.py`
2. Modify Utterance in `dataset/utterance.json`
3. Execute script `bash scripts/inference.bash`

### Example Best Model Inference
[Sample Audio](output/PMK-xtts-v2-model-v1/1.wav)
