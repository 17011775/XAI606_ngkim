import re
import unicodedata
from g2p_en import G2p
#from data_gen.tts.txt_processors.en import TxtProcessor
#from data_gen.tts.data_gen_utils import is_sil_phoneme
from g2p_en import G2p
from g2p_en.expand import normalize_numbers
from nltk import pos_tag
from nltk.tokenize import TweetTokenizer
from phonemizer import phonemize
PUNCS = '!,.?;:'
REGISTERED_TEXT_PROCESSORS = {}
def register_txt_processors(name):
    def _f(cls):
        REGISTERED_TEXT_PROCESSORS[name] = cls
        return cls

    return _f
def is_sil_phoneme(p):
    return len(p) == 0 or not p[0].isalpha()
class BaseTxtProcessor:
    @staticmethod
    def sp_phonemes():
        return ['|']

    @classmethod
    def process(cls, txt, preprocess_args):
        raise NotImplementedError

    @classmethod
    def postprocess(cls, txt_struct, preprocess_args):
        # remove sil phoneme in head and tail
        while len(txt_struct) > 0 and is_sil_phoneme(txt_struct[0][0]):
            txt_struct = txt_struct[1:]
        while len(txt_struct) > 0 and is_sil_phoneme(txt_struct[-1][0]):
            txt_struct = txt_struct[:-1]
        if preprocess_args['with_phsep']:
            txt_struct = cls.add_bdr(txt_struct)
        if preprocess_args['add_eos_bos']:
            txt_struct = [["<BOS>", ["<BOS>"]]] + txt_struct + [["<EOS>", ["<EOS>"]]]
        return txt_struct

    @classmethod
    def add_bdr(cls, txt_struct):
        txt_struct_ = []
        for i, ts in enumerate(txt_struct):
            txt_struct_.append(ts)
            if i != len(txt_struct) - 1 and \
                    not is_sil_phoneme(txt_struct[i][0]) and not is_sil_phoneme(txt_struct[i + 1][0]):
                txt_struct_.append(['|', ['|']])
        return txt_struct_
class EnG2p(G2p):
    word_tokenize = TweetTokenizer().tokenize

    def __call__(self, text):
        # preprocessing
        words = EnG2p.word_tokenize(text)
        tokens = pos_tag(words)  # tuples of (word, tag)

        # steps
        prons = []
        for word, pos in tokens:
            if re.search("[a-z]", word) is None:
                pron = [word]

            elif word in self.homograph2features:  # Check homograph
                pron1, pron2, pos1 = self.homograph2features[word]
                if pos.startswith(pos1):
                    pron = pron1
                else:
                    pron = pron2
            elif word in self.cmu:  # lookup CMU dict
                pron = self.cmu[word][0]
            else:  # predict for oov
                pron = self.predict(word)

            prons.extend(pron)
            prons.extend([" "])

        return prons[:-1]

class BaseTxtProcessor:
    @staticmethod
    def sp_phonemes():
        return ['|']

    @classmethod
    def process(cls, txt, preprocess_args):
        raise NotImplementedError

    @classmethod
    def postprocess(cls, txt_struct, preprocess_args):
        # remove sil phoneme in head and tail
        while len(txt_struct) > 0 and is_sil_phoneme(txt_struct[0][0]):
            txt_struct = txt_struct[1:]
        while len(txt_struct) > 0 and is_sil_phoneme(txt_struct[-1][0]):
            txt_struct = txt_struct[:-1]
        if preprocess_args['with_phsep']:
            txt_struct = cls.add_bdr(txt_struct)
        if preprocess_args['add_eos_bos']:
            txt_struct = [["<BOS>", ["<BOS>"]]] + txt_struct + [["<EOS>", ["<EOS>"]]]
        return txt_struct

    @classmethod
    def add_bdr(cls, txt_struct):
        txt_struct_ = []
        for i, ts in enumerate(txt_struct):
            txt_struct_.append(ts)
            if i != len(txt_struct) - 1 and \
                    not is_sil_phoneme(txt_struct[i][0]) and not is_sil_phoneme(txt_struct[i + 1][0]):
                txt_struct_.append(['|', ['|']])
        return txt_struct_

@register_txt_processors('en')
class TxtProcessor(BaseTxtProcessor):
    g2p = EnG2p()

    @staticmethod
    def preprocess_text(text):
        text = normalize_numbers(text)
        text = ''.join(char for char in unicodedata.normalize('NFD', text)
                       if unicodedata.category(char) != 'Mn')  # Strip accents
        text = text.lower()
        text = re.sub("[\'\"()]+", "", text)
        text = re.sub("[-]+", " ", text)
        text = re.sub(f"[^ a-z{PUNCS}]", "", text)
        text = re.sub(f" ?([{PUNCS}]) ?", r"\1", text)  # !! -> !
        text = re.sub(f"([{PUNCS}])+", r"\1", text)  # !! -> !
        text = text.replace("i.e.", "that is")
        text = text.replace("i.e.", "that is")
        text = text.replace("etc.", "etc")
        text = re.sub(f"([{PUNCS}])", r" \1 ", text)
        text = re.sub(rf"\s+", r" ", text)
        return text

    @classmethod
    def process(cls, txt, preprocess_args):
        txt = cls.preprocess_text(txt).strip()
        phs = cls.g2p(txt)
        txt_struct = [[w, []] for w in txt.split(" ")]
        i_word = 0
        for p in phs:
            if p == ' ':
                i_word += 1
            else:
                txt_struct[i_word][1].append(p)
        txt_struct = cls.postprocess(txt_struct, preprocess_args)
        return txt_struct, txt

# G2P 모델 초기화
g2p_model = G2p()

# TxtProcessor 초기화
txt_processor = TxtProcessor()

# 간투사 목록
interjections = ['em', 'um', 'ah', 'hm', 'mn', 'en', 'et', 'uh', 'ya', 'ee', 'uu', 'ha', 'hh', 'oh', 'eh', 'mm', 'er', 'ern', 'uhh', 'uum', 'ehm', 'huh', 'err', 'mhm', 'erm', 'yah', 'hmm', 'aum', 'ahh', 'mmm', 'aha', 'umm', 'ehh', 'hehe', 'yeah', 'haha', 'ahhh', 'hmmm']

def convert_interjection(interjection):
    # G2P 변환
    g2p_result = g2p_model(interjection)
    g2p_result = ' '.join(g2p_result)
    
    # TxtProcessor를 사용한 변환
    #txt_processor_result, _ = txt_processor.process(interjection, {'with_phsep': False, 'add_eos_bos': False})
    #txt_processor_result = ' '.join([ph for word, phs in txt_processor_result for ph in phs if not is_sil_phoneme(ph)])
    phonemizer_result = phonemize(interjection, language='en-us', backend='espeak', strip=True)
    return {
        'interjection': interjection,
        'g2p': g2p_result,
        'phonemizer': phonemizer_result
    }

# 모든 간투사에 대해 변환 수행
results = [convert_interjection(interjection) for interjection in interjections]
# 결과를 파일로 저장
with open('interjection_conversion_results.txt', 'w', encoding='utf-8') as f:
    for result in results:
        f.write(f"Interjection: {result['interjection']}\n")
        f.write(f"G2P: {result['g2p']}\n")
        f.write(f"TxtProcessor: {result['txt_processor']}\n")
        f.write("\n")

print("결과가 'interjection_conversion_results.txt' 파일로 저장되었습니다.")
# 결과 출력
for result in results:
    print(f"Interjection: {result['interjection']}")
    print(f"G2P: {result['g2p']}")
    print(f"TxtProcessor: {result['txt_processor']}")
    print()