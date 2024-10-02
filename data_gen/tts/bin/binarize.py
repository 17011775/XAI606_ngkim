import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import importlib
from utils.hparams import set_hparams, hparams
from data_gen.tts.base_binarizer_emotion import EmotionBinarizer

def binarize():
    binarizer_cls = hparams.get("binarizer_cls", 'EmotionBinarizer')
    pkg = ".".join(binarizer_cls.split(".")[:-1])
    cls_name = binarizer_cls.split(".")[-1]
    binarizer_cls = getattr(importlib.import_module(pkg), cls_name)
    print("| Binarizer: ", binarizer_cls)
    binarizer_cls().process()


if __name__ == '__main__':
    set_hparams()
    binarize()
