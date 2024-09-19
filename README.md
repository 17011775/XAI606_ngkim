# XAI606_ngkim
신경망 응용 및 실습 
I. Project Title
GenerSpeech: Multi-Speaker Expressive Text-to-Speech with Fine-Grained Style Control
II. Project Introduction
Objective
The goal of this project is to develop a text-to-speech (TTS) model capable of generating high-quality, expressive speech for multiple speakers with fine-grained style control. We aim to improve upon existing TTS systems by enabling more natural and varied prosody, better speaker similarity, and the ability to transfer speaking styles across speakers.
Motivation
Current TTS systems often struggle to produce truly expressive and natural-sounding speech, especially when dealing with multiple speakers or trying to transfer speaking styles. This project is motivated by the need for more flexible and capable TTS models that can:

Generate speech with natural prosody and expressiveness
Accurately capture and reproduce speaker identities
Transfer speaking styles (e.g. emotional states, emphasis patterns) between speakers
Allow fine-grained control over speech attributes like pitch, speaking rate, and emphasis

By developing such a system, we hope to enable more natural and engaging voice interfaces, improve accessibility technologies, and advance the state-of-the-art in speech synthesis.
III. Dataset Description
We will use a multi-speaker dataset derived from LibriTTS and the Emotional Speech Dataset (ESD). The dataset will be split into training, validation, and test sets as follows:
Training Set

500 hours of speech from 2000 speakers
Sourced from LibriTTS
Includes text transcripts and speaker IDs

Validation Set

50 hours of speech from 200 speakers (not in training set)
Sourced from LibriTTS
Includes text transcripts and speaker IDs

Test Set

20 speakers (10 male, 10 female) not in training/validation sets
50 utterances per speaker
Mixture of neutral and emotional speech from ESD
Only input text provided, no audio or speaker info

The training and validation sets will be made available with both input text and corresponding audio files to allow model training and validation. The test set will only include input text, to be used for final model evaluation.
All datasets can be accessed at:
[Link to data repository]
We encourage participants to experiment with data augmentation and additional publicly available datasets to improve their models.
