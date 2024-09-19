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

1. Generate speech with natural prosody and expressiveness
2. Accurately capture and reproduce various speaking styles of a single speaker
3. Transfer speaking styles (e.g., emotional states, prosody) within the same speaker's voice
4. Improve the overall quality and naturalness of synthesized speech

By developing such a system, we hope to enable more engaging and versatile voice applications, improve the naturalness of synthesized speech, and advance the state-of-the-art in expressive speech synthesis.

## III. Dataset Description

We will use the Emotional Speech Dataset (ESD) for this project, specifically focusing on 10 English speakers. The dataset will be split into training, validation, and test sets as follows:

### Training Set
- 10 speakers from ESD
- 5 emotions per speaker (angry, happy, neutral, sad, surprise)
- 80% of available utterances for each speaker-emotion combination

### Validation Set  
- Same 10 speakers as in the training set
- 5 emotions per speaker (angry, happy, neutral, sad, surprise)
- 10% of available utterances for each speaker-emotion combination

### Test Set
- Same 10 speakers as in the training and validation sets
- 5 emotions per speaker (angry, happy, neutral, sad, surprise)
- Remaining 10% of available utterances for each speaker-emotion combination
- Only input text and reference style audio provided for testing

All datasets can be accessed at: [Link to data repository]

We encourage participants to focus on developing models that can effectively capture and transfer expressive styles within a speaker's voice, demonstrating high-quality and natural-sounding results across different emotions.
