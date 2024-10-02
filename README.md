# XAI606_ngkim
신경망 응용 및 실습 
## I. Project Title

High-Fidelity Expressive Style Transfer for Text-to-Speech Synthesis

## II. Project Introduction

### Objective

The goal of this project is to develop a text-to-speech (TTS) model capable of high-fidelity expressive style transfer.
We aim to improve upon existing TTS systems by enabling more natural and varied prosody, and better style capturing and transfer for a single speaker across different emotions.

### Motivation

Current TTS systems often struggle to produce truly expressive and natural-sounding speech, especially when trying to transfer speaking styles within a speaker's voice. This project is motivated by the need for TTS models that can:

1. Accurately capture and reproduce various speaking styles and emotions of speaker
2. Transfer speaking styles (e.g., emotional states, prosody) within the same speaker's voice
3. Generate speech with improved naturalness and expressiveness

By developing such a system, we hope to enable more engaging and versatile voice applications, improve the naturalness of synthesized speech, and advance the state-of-the-art in expressive speech synthesis.

## III. Dataset Description

We use the Emotional Speech Dataset (ESD) for this project, specifically focusing on 10 English speakers. 
The ESD is a high-quality emotional speech dataset designed for emotional voice conversion and emotional text-to-speech tasks. 

For detailed information about the ESD and to access the raw dataset, please visit:
https://hltsingapore.github.io/ESD/

Note: While the link above provides access to the raw ESD, for this project we will be using a preprocessed and binarized version of the dataset.

Our preprocessed dataset consists of 1750 utterances per speaker, divided equally among 5 emotions (neutral, happy, angry, sad, surprise), with 350 utterances per emotion. The dataset has been split into training, validation, and test sets as follows:

### Training Set
- All 10 speakers from ESD
- For each speaker:
  - 300 utterances per emotion
  - Total of 1500 utterances per speaker (300 x 5 emotions)
- Overall total: 15,000 utterances (1500 x 10 speakers)

### Validation Set
- Same 10 speakers as in the training set
- For each speaker:
  - 20 utterances per emotion
  - Total of 100 utterances per speaker (20 x 5 emotions)
- Overall total: 1,000 utterances (100 x 10 speakers)

### Test Set
- Same 10 speakers as in the training and validation sets
- For each speaker:
  - 30 utterances per emotion
  - Total of 150 utterances per speaker (30 x 5 emotions)
- Overall total: 1,500 utterances (150 x 10 speakers)

The specific utterance ranges for each set are:

- Training: 51-350, 401-700, 751-1050, 1101-1400, 1451-1750
- Validation: 1-20, 351-370, 701-720, 1051-1070, 1401-1420
- Test: 21-50, 371-400, 721-750, 1071-1100, 1421-1450

### Preprocessed Data Format

The dataset has been preprocessed and binarized. The structure of the preprocessed data is as follows:

- `train.data`, `valid.data`, `test.data`: Binarized data files for each set
- `train.idx`, `valid.idx`, `test.idx`: Index files for the binarized data
- `train_lengths.npy`, `valid_lengths.npy`, `test_lengths.npy`: NumPy files containing mel-spectrogram lengths
- `train_f0s_mean_std.npy`, `valid_f0s_mean_std.npy`, `test_f0s_mean_std.npy`: NumPy files with F0 statistics
- `emo_map.json`: JSON file mapping emotion labels to numerical IDs
- `phone_set.json`: JSON file containing the set of phonemes used
- `spk_map.json`: JSON file mapping speaker IDs to numerical IDs

These preprocessed files contain all necessary information for model training, including mel-spectrograms, phoneme sequences, F0 contours, and emotion embeddings.

All preprocessed datasets are available from (https://drive.google.com/drive/folders/19qV2xjEh32XOU8HHj10b17NE1b4GvlF0?usp=sharing)

We encourage participants to focus on developing models that can effectively capture and transfer expressive styles within a speaker's voice, demonstrating high-quality and natural-sounding results across different emotions.

### Dependencies

A suitable [conda](https://conda.io/) environment named `ExpTTS` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate ExpTTS
```

### Training 

```bash
CUDA_VISIBLE_DEVICES=$GPU python tasks/run.py --config modules/GenerSpeech/config/generspeech.yaml  --exp_name GenerSpeech --reset
```

### Inference 

```bash
CUDA_VISIBLE_DEVICES=$GPU python tasks/run.py --config modules/GenerSpeech/config/generspeech.yaml  --exp_name GenerSpeech --infer
```
