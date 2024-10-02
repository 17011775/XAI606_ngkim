import os

from data_gen.tts.base_preprocess import BasePreprocessor
import glob
import re


class EmoPreAlign(BasePreprocessor):

    def meta_data(self):
        spks = ['0012', '0011', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020']
        for spk in spks:
            with open(f"{self.raw_data_dir}/{spk}/{spk}.txt", 'r', encoding='utf-8', errors='ignore') as file:  # 파일 열기
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    split_ = line.split('\t')
                    if len(split_) < 3:
                        continue
                    item_name, txt, emotion = split_
                    wav_fn = f'{self.raw_data_dir}/{spk}/{emotion}/{item_name}.wav'
                    yield {
                        'item_name': item_name,
                        'wav_fn': wav_fn,
                        'txt': txt,
                        'spk_name': spk,
                        'emotion': emotion
                    }

if __name__ == "__main__":
    EmoPreAlign().process()
