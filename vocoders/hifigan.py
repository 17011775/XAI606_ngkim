import glob
import json
import os
import re

import librosa
import torch

import utils
from librosa.filters import mel as librosa_mel_fn
from modules.hifigan.hifigan import HifiGanGenerator
from utils.hparams import hparams, set_hparams
from vocoders.base_vocoder import register_vocoder
from vocoders.pwg import PWG
from vocoders.vocoder_utils import denoise

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, hparams, center=False, complex=False):
    # hop_size: 512  # For 22050Hz, 275 ~= 12.5 ms (0.0125 * sample_rate)
    # win_size: 2048  # For 22050Hz, 1100 ~= 50 ms (If None, win_size: fft_size) (0.05 * sample_rate)
    # fmin: 55  # Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
    # fmax: 10000  # To be increased/reduced depending on data.
    # fft_size: 2048  # Extra window size is filled with 0 paddings to match this parameter
    # n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax,
    n_fft = hparams['fft_size'] # 1024
    num_mels = hparams['audio_num_mel_bins'] # 80
    sampling_rate = hparams['audio_sample_rate'] # 16000
    hop_size = hparams['hop_size'] # 256
    win_size = hparams['win_size'] # 1024 
    fmin = hparams['fmin'] # 80
    fmax = hparams['fmax'] # 7600
    print('Wav To mel')
    print(f"WAV min: {y.min()}, max: {y.max()}, mean: {y.mean()}, std: {y.std()}")
    #y = y.clamp(min=-1., max=1.)
    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + '_' + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
                                mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    if not complex:
        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
        spec = torch.matmul(mel_basis[str(fmax) + '_' + str(y.device)], spec)
        spec = spectral_normalize_torch(spec)
    else:
        B, C, T, _ = spec.shape
        spec = spec.transpose(1, 2)  # [B, T, n_fft, 2]
    return spec


def load_model(config_path, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dict = torch.load(checkpoint_path, map_location="cpu")
    if '.yaml' in config_path:
        config = set_hparams(config_path, global_hparams=False)
        state = ckpt_dict["state_dict"]["model_gen"]
    elif '.json' in config_path:
        config = json.load(open(config_path, 'r'))
        state = ckpt_dict["generator"]
    print("Config path")
    print(config_path)
    print("HIFIGAN CONFIG")
    print(config)
    model = HifiGanGenerator(config)
    model.load_state_dict(state, strict=True)
    model.remove_weight_norm()
    model = model.eval().to(device)
    print(f"| Loaded model parameters from {checkpoint_path}.")
    print(f"| HifiGAN device: {device}.")
    return model, config, device


total_time = 0


@register_vocoder
class HifiGAN(PWG):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Default device
        base_dir ='/workspace/ng/data/checkpoints'
        config_path = f'{base_dir}/config.yaml'
        if os.path.exists(config_path):
            ckpt = sorted(glob.glob(f'{base_dir}/model_ckpt_steps_*.ckpt'), key=
            lambda x: int(re.findall(f'{base_dir}/model_ckpt_steps_(\d+).ckpt', x)[0]))[-1]
            print('| load HifiGAN: ', ckpt)
            self.model, self.config, self.device = load_model(config_path=config_path, checkpoint_path=ckpt)
        else:
            config_path = f'{base_dir}/config.json'
            ckpt = f'{base_dir}/generator_v1'
            if os.path.exists(config_path):
                self.model, self.config, self.device = load_model(config_path=config_path, checkpoint_path=ckpt)

    def spec2wav(self, mel, **kwargs):
        device = self.device
        with torch.no_grad():
            #print('spec2wav')
            #print(mel.shape) # (T, 80)
            c = torch.FloatTensor(mel).unsqueeze(0).transpose(2, 1).to(device) # (1, 80, T)
            f0 = kwargs.get('f0')
            if f0 is not None and hparams.get('use_nsf'):
                f0 = torch.FloatTensor(f0[None, :]).to(device)
                y = self.model(c, f0).view(-1)
            else:
                y = self.model(c).view(-1)
        wav_out = y.cpu().numpy()
        if hparams.get('vocoder_denoise_c', 0.0) > 0: # denoise 안함 
            wav_out = denoise(wav_out, v=hparams['vocoder_denoise_c'])
        #print(wav_out) -10~10 
        return wav_out

    #@staticmethod
    #def wav2spec(wav_fn, **kwargs):
    #    print(wav_fn)
    #    wav, _ = librosa.core.load(wav_fn, sr=hparams['audio_sample_rate'])
    #    wav_torch = torch.FloatTensor(wav)[None, :]
    #    mel = mel_spectrogram(wav_torch, hparams).numpy()[0]
    #    return wav, mel.T
