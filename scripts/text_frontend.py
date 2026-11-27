#!/usr/bin/env python3
"""
AetherTTS 无 PyTorch 依赖的纯 Python 中英双语与混读音系前端。

与 scripts/aethertts_lib.py 中的 BilingualPhonemeFrontend 保持 100% 词表与音素映射一致，
但仅依赖标准库 + pypinyin（可选 jieba / g2p_en），供 ONNX Runtime 轻量纯 CPU 推理脚本直接调用，
无需安装 torch / torchaudio / librosa / scipy。
"""
import re
import socket
import unicodedata

from pypinyin import Style, pinyin

_EN_G2P = None
_EN_G2P_INIT_TRIED = False
_ZH_WORD_SEG = None
_ZH_WORD_SEG_TRIED = False

# 结构助词 / 体标记 / 复数标记永远附着在前一个词上，不自成韵律词
_ENCLITIC_PARTICLES = {"的", "地", "得", "了", "着", "过", "们"}
# 标贝韵律符 -> 控制字符占位符：解析前先替换，保护 #1 的数字不被 normalize_text 转成汉字
_BZ_MARK = {"#1": "\x01", "#2": "\x02", "#3": "\x03", "#4": "\x04"}
_BZ_MARK_CHARS = "".join(_BZ_MARK.values())


def _get_en_g2p():
    global _EN_G2P, _EN_G2P_INIT_TRIED
    if _EN_G2P_INIT_TRIED:
        return _EN_G2P
    _EN_G2P_INIT_TRIED = True
    _prev_to = socket.getdefaulttimeout()
    socket.setdefaulttimeout(30.0)
    try:
        from g2p_en import G2p
        import nltk

        def _has_nltk(path):
            try:
                nltk.data.find(path)
                return True
            except LookupError:
                return False

        _tried_dl = set()

        def _dl_nltk(name):
            if name in _tried_dl:
                return False
            _tried_dl.add(name)
            try:
                ok = nltk.download(name, quiet=True)
            except Exception as _e_dl:
                print(f"[Warn] nltk.download({name}) 异常: {_e_dl}")
                return False
            if not ok:
                print(f"[Warn] nltk.download({name}) 失败")
            return bool(ok)

        if not _has_nltk("corpora/cmudict") and not _dl_nltk("cmudict"):
            raise RuntimeError("nltk 数据缺失 cmudict，请手动 nltk.download('cmudict') 或设置 NLTK_DATA")
        _taggers = [
            ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
            ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ]
        if not any(_has_nltk(p) for p, _ in _taggers):
            for _p, _n in _taggers:
                _dl_nltk(_n)
        _g2p = G2p()
        try:
            _g2p("test")
        except LookupError:
            for _p, _n in _taggers:
                if not _has_nltk(_p):
                    _dl_nltk(_n)
            _g2p("test")
        _EN_G2P = _g2p
    except Exception as e:
        print(f"[Warn] g2p_en unavailable ({e})：英文词将退化为逐字母拼读，仅适合没有英文训练数据的场景。")
        _EN_G2P = None
    finally:
        socket.setdefaulttimeout(_prev_to)
    return _EN_G2P


def _get_zh_word_segmenter():
    """推理期韵律词边界近似来源：jieba 词典分词取词首，无 jieba 时退化为每 2 音节一界。"""
    global _ZH_WORD_SEG, _ZH_WORD_SEG_TRIED
    if _ZH_WORD_SEG_TRIED:
        return _ZH_WORD_SEG
    _ZH_WORD_SEG_TRIED = True
    try:
        import logging
        import jieba

        jieba.setLogLevel(logging.WARNING)
        jieba.lcut("预热")
        _ZH_WORD_SEG = jieba.lcut
    except Exception as e:
        print(f"[Warn] jieba unavailable ({e})：推理期韵律词边界退化为每 2 个音节一个（建议 pip install jieba）。")
        _ZH_WORD_SEG = None
    return _ZH_WORD_SEG


ZH_INITIALS = [
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s", "y", "w",
]
ZH_FINALS_BASE = [
    "a", "o", "e", "i", "ii", "iii", "u", "v", "er",
    "ai", "ei", "ao", "ou", "ia", "ie", "ua", "uo", "ve",
    "iao", "iou", "uai", "uei",
    "an", "en", "in", "vn", "ang", "eng", "ing", "ong",
    "ian", "uan", "van", "uen", "iang", "uang", "ueng", "iong",
]
EN_ARPABET = [
    "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0", "AO1", "AO2",
    "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "B", "CH", "D", "DH",
    "EH0", "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2", "F", "G", "HH",
    "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N", "NG",
    "OW0", "OW1", "OW2", "OY0", "OY1", "OY2", "P", "R", "S", "SH", "T", "TH",
    "UH0", "UH1", "UH2", "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH",
]
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", ",", ".", "?", "!", ";", ":"]

_Y_RESTORE = {
    "u": "v", "ue": "ve", "uan": "van", "un": "vn",
    "a": "ia", "e": "ie", "ao": "iao", "ou": "iou",
    "an": "ian", "ang": "iang", "ong": "iong",
}
_W_RESTORE = {
    "a": "ua", "o": "uo", "ai": "uai", "ei": "uei",
    "an": "uan", "en": "uen", "ang": "uang", "eng": "ueng",
}
_ABBREV_RESTORE = {"iu": "iou", "ui": "uei", "un": "uen"}
_NASAL_SYL_MAP = {"hm": ("h", "en"), "hng": ("h", "eng"), "m": ("m", "u"), "n": ("n", "en"), "ng": ("n", "eng")}
_EN_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_EN_ORD_SMALL = {1: "first", 2: "second", 3: "third", 5: "fifth", 8: "eighth", 9: "ninth", 12: "twelfth"}

NON_ERHUA_WORDS = {
    "儿童", "儿子", "儿女", "儿科", "儿歌", "儿媳", "儿孙", "儿戏", "儿时", "儿茶",
    "而且", "而后", "而是", "而今", "而已", "耳朵", "耳垂", "耳机", "耳光", "耳目",
    "女儿", "孤儿", "婴儿", "幼儿", "少儿", "新生儿", "胎儿", "产儿", "健儿", "宠儿",
    "男儿", "小儿", "育儿", "托儿", "哈尔滨", "首尔", "尼泊尔", "华尔街", "诺贝尔",
    "高尔夫", "马尔代夫", "阿尔法", "爱尔兰", "威尔士", "墨尔本", "荷尔蒙", "达尔文",
}


class BilingualPhonemeFrontend:
    """中英双语统一音素与韵律边界前端（与 scripts/aethertts_lib.py 100% 同步）。"""

    def __init__(self, vocab: list = None):
        if vocab is None:
            vocab = list(SPECIAL_TOKENS)
            vocab += [f"zh_{ini}" for ini in ZH_INITIALS]
            for fin in ZH_FINALS_BASE:
                for tone in ("1", "2", "3", "4", "5"):
                    vocab.append(f"zh_{fin}{tone}")
            vocab.append("zh_er_coda")
            vocab += [f"en_{ph}" for ph in EN_ARPABET]
        self.vocab = list(vocab)
        self.tok2id = {t: i for i, t in enumerate(self.vocab)}
        self.id2tok = {i: t for i, t in enumerate(self.vocab)}
        self.pad_id = self.tok2id["<pad>"]
        self.bos_id = self.tok2id["<bos>"]
        self.eos_id = self.tok2id["<eos>"]
        self.unk_id = self.tok2id["<unk>"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @staticmethod
    def _num_to_zh(num_str: str) -> str:
        digits = "零一二三四五六七八九"
        if "." in num_str:
            int_p, dec_p = num_str.split(".", 1)
            return BilingualPhonemeFrontend._num_to_zh(int_p) + "点" + "".join(digits[int(c)] for c in dec_p if c.isdigit())
        n = int(num_str)
        if len(num_str) == 4 and 1900 <= n <= 2099:
            return "".join(digits[int(c)] for c in num_str)
        if n == 0:
            return "零"
        if n < 10:
            return digits[n]
        if n < 20:
            return "十" + ("" if n == 10 else digits[n % 10])
        if n < 100:
            return digits[n // 10] + "十" + ("" if n % 10 == 0 else digits[n % 10])
        if n < 1000:
            rem = n % 100
            if rem == 0:
                return digits[n // 100] + "百"
            if rem < 10:
                return digits[n // 100] + "百零" + digits[rem]
            return digits[n // 100] + "百" + digits[rem // 10] + "十" + ("" if rem % 10 == 0 else digits[rem % 10])
        return "".join(digits[int(c)] for c in num_str)

    @staticmethod
    def _int_to_en(n: int) -> str:
        if n == 0:
            return "zero"
        if n >= 10 ** 12:
            return " ".join(_EN_ONES[int(c)] for c in str(n))

        def _three(v):
            if v < 20:
                return _EN_ONES[v]
            if v < 100:
                return _EN_TENS[v // 10] + ("" if v % 10 == 0 else " " + _EN_ONES[v % 10])
            return _EN_ONES[v // 100] + " hundred" + ("" if v % 100 == 0 else " " + _three(v % 100))

        parts = []
        for i, scale in enumerate(("", " thousand", " million", " billion")):
            v = (n // (1000 ** i)) % 1000
            if v:
                parts.append(_three(v) + scale)
        return " ".join(reversed(parts))

    @classmethod
    def _en_ordinal(cls, n: int) -> str:
        r, o = n % 100, n % 10

        def _with_head(head_num, last):
            head = cls._int_to_en(head_num) + " " if head_num > 0 else ""
            return head + last

        if 0 < r < 13:
            return _with_head(n - r, _EN_ORD_SMALL.get(r, _EN_ONES[r] + "th"))
        if 13 <= r <= 19:
            return _with_head(n - r, _EN_ONES[r] + "th")
        if r % 10 == 0:
            if 20 <= r <= 90:
                return _with_head(n - r, _EN_TENS[r // 10][:-1] + "ieth")
            words = cls._int_to_en(n).split()
            words[-1] += "th"
            return " ".join(words)
        return _with_head(n - o, _EN_ORD_SMALL.get(o, _EN_ONES[o] + "th"))

    @classmethod
    def _en_year(cls, n: int) -> str:
        hi, lo = n // 100, n % 100
        if 2010 <= n <= 2099:
            return f"{cls._int_to_en(hi)} {cls._int_to_en(lo)}"
        if n >= 2000:
            return "two thousand" + ("" if lo == 0 else " " + cls._int_to_en(lo))
        if lo == 0:
            return f"{cls._int_to_en(hi)} hundred"
        if lo < 10:
            return f"{cls._int_to_en(hi)} oh {cls._int_to_en(lo)}"
        return f"{cls._int_to_en(hi)} {cls._int_to_en(lo)}"

    @classmethod
    def _num_to_en(cls, num_str: str) -> str:
        if "." in num_str:
            int_p, dec_p = num_str.split(".", 1)
            return cls._int_to_en(int(int_p)) + " point " + " ".join(_EN_ONES[int(c)] for c in dec_p)
        return cls._int_to_en(int(num_str))

    @classmethod
    def _en_decade(cls, n: int) -> str:
        hi, lo = n // 100, n % 100
        if lo == 0:
            return "two thousands" if n == 2000 else f"{cls._int_to_en(hi)} hundreds"
        if lo == 10:
            return f"{cls._int_to_en(hi)} tens"
        return f"{cls._int_to_en(hi)} {_EN_TENS[lo // 10][:-1]}ies"

    @classmethod
    def _normalize_en_numbers(cls, text: str) -> str:
        text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
        text = re.sub(r"(?<![\w.])(\d+(?:\.\d+)?)%", lambda m: cls._num_to_en(m.group(1)) + " percent", text)
        text = re.sub(r"\b(1[1-9]\d0|20\d0)s\b", lambda m: cls._en_decade(int(m.group(1))), text)
        text = re.sub(r"\b(\d{1,6})(?:st|nd|rd|th)\b", lambda m: cls._en_ordinal(int(m.group(1))), text, flags=re.I)
        text = re.sub(r"\b((?:[1-9]\d*)?[02-9][23]|[23])d\b", lambda m: cls._en_ordinal(int(m.group(1))), text)
        text = re.sub(
            r"\b(in|by|since|until|from|to|before|after|between|and|year)\s+(1[1-9]\d{2}|20\d{2})\b",
            lambda m: f"{m.group(1)} {cls._en_year(int(m.group(2)))}",
            text,
            flags=re.I,
        )
        return re.sub(r"(?<![\w.])(\d+(?:\.\d+)?)(?!\w)", lambda m: cls._num_to_en(m.group(1)), text)

    def normalize_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text).strip()
        punc_map = {"，": ",", "。": ".", "？": "?", "！": "!", "；": ";", "：": ":", "、": ",", "—": ",", "…": "."}
        for k, v in punc_map.items():
            text = text.replace(k, v)
        has_zh = bool(re.search(r"[\u4e00-\u9fff]", text))
        if has_zh:
            text = re.sub(r"(\d+(?:\.\d+)?)%", lambda m: "百分之" + self._num_to_zh(m.group(1)), text)
            text = re.sub(r"\d+(?:\.\d+)?", lambda m: self._num_to_zh(m.group(0)), text)
        else:
            text = self._normalize_en_numbers(text)
        return text

    @staticmethod
    def _split_zh_syllable(syl: str):
        syl = syl.strip().lower().replace("ü", "v").replace("u:", "v").replace("ê", "e")
        if not syl:
            return []
        tone = syl[-1] if syl[-1] in "12345" else "5"
        body = syl[:-1] if syl[-1] in "12345" else syl
        if not body:
            return []
        if body in _NASAL_SYL_MAP:
            ini_n, fin_n = _NASAL_SYL_MAP[body]
            return [f"zh_{ini_n}", f"zh_{fin_n}{tone}"]

        has_erhua = False
        if len(body) > 1 and body.endswith("r") and body != "er":
            has_erhua = True
            body = body[:-1]

        if body == "er":
            return [f"zh_er{tone}"]

        ini = ""
        for cand in sorted(ZH_INITIALS, key=len, reverse=True):
            if body.startswith(cand) and len(body) > len(cand):
                ini = cand
                body = body[len(cand):]
                break

        if body == "i":
            if ini in ("z", "c", "s"):
                body = "ii"
            elif ini in ("zh", "ch", "sh", "r"):
                body = "iii"
        elif ini == "y" and body in _Y_RESTORE:
            body = _Y_RESTORE[body]
        elif ini == "w" and body in _W_RESTORE:
            body = _W_RESTORE[body]
        elif ini in ("j", "q", "x") and body in ("u", "ue", "uan", "un"):
            body = {"u": "v", "ue": "ve", "uan": "van", "un": "vn"}[body]
        elif ini in ("n", "l") and body == "ue":
            body = "ve"
        elif body in _ABBREV_RESTORE:
            body = _ABBREV_RESTORE[body]

        toks = []
        if ini:
            toks.append(f"zh_{ini}")
        toks.append(f"zh_{body}{tone}")
        if has_erhua:
            toks.append("zh_er_coda")
        return toks

    @staticmethod
    def _is_lexical_non_erhua(span: str, idx: int) -> bool:
        if idx <= 0 or span[idx] != "儿":
            return True
        for wlen in (3, 2):
            for st in range(max(0, idx - wlen + 1), min(len(span) - wlen + 1, idx + 1)):
                if span[st : st + wlen] in NON_ERHUA_WORDS:
                    return True
        return False

    @classmethod
    def _apply_zh_sandhi_and_erhua(cls, span: str, raw_pys: list):
        n = min(len(span), len(raw_pys))
        chars, pys = list(span[:n]), list(raw_pys[:n])

        m_chars, m_pys = [], []
        i = 0
        while i < len(chars):
            ch, py = chars[i], pys[i]
            if (
                i + 1 < len(chars)
                and chars[i + 1] == "儿"
                and pys[i + 1] in ("er2", "er5")
                and not py.startswith("er")
                and not cls._is_lexical_non_erhua(span, i + 1)
            ):
                tone = py[-1] if py and py[-1] in "12345" else "5"
                base = py[:-1] if py and py[-1] in "12345" else py
                m_chars.append(ch + "儿")
                m_pys.append(f"{base}r{tone}")
                i += 2
            else:
                m_chars.append(ch)
                m_pys.append(py)
                i += 1

        for k in range(1, len(m_chars) - 1):
            if m_chars[k] in ("一", "不") and m_chars[k - 1] == m_chars[k + 1]:
                m_pys[k] = m_pys[k][:-1] + "5"

        digits_zh = set("零一二三四五六七八九十百千万亿")
        for k in range(len(m_chars)):
            ch = m_chars[k]
            if ch == "不" and m_pys[k].startswith("bu") and k + 1 < len(m_pys):
                nxt_t = m_pys[k + 1][-1] if m_pys[k + 1] else ""
                if nxt_t == "4" and m_pys[k][-1] != "5":
                    m_pys[k] = "bu2"
            elif ch == "一" and m_pys[k].startswith("yi") and m_pys[k][-1] != "5":
                prev_c = m_chars[k - 1][-1] if k > 0 else ""
                next_c = m_chars[k + 1][0] if k + 1 < len(m_chars) else ""
                if (
                    k + 1 >= len(m_pys)
                    or prev_c in ("第", "唯", "统", "初", "十", "零")
                    or next_c in ("月", "号", "楼", "班", "路", "年")
                    or (next_c in digits_zh and next_c not in ("一", "般", "起", "直", "定", "下", "样", "切", "致", "共", "点", "会"))
                ):
                    m_pys[k] = "yi1"
                else:
                    nxt_t = m_pys[k + 1][-1] if m_pys[k + 1] else ""
                    m_pys[k] = "yi2" if nxt_t == "4" else ("yi4" if nxt_t in ("1", "2", "3") else m_pys[k])

        syl_of_char = [idx_s for idx_s, ch_s in enumerate(m_chars) for _ in ch_s]
        word_spans = []
        seg_fn = _get_zh_word_segmenter()
        if seg_fn is not None and syl_of_char:
            c_cur = 0
            for w in seg_fn(span):
                if c_cur < len(syl_of_char):
                    s0 = syl_of_char[c_cur]
                    s1 = syl_of_char[min(c_cur + len(w) - 1, len(syl_of_char) - 1)] + 1
                    if not word_spans or s0 >= word_spans[-1][1]:
                        word_spans.append((s0, s1))
                    else:
                        word_spans[-1] = (word_spans[-1][0], max(word_spans[-1][1], s1))
                c_cur += len(w)

        def _sandhi_run(lo: int, hi: int):
            run_len = hi - lo
            if run_len == 2:
                m_pys[lo] = m_pys[lo][:-1] + "2"
            elif run_len == 3:
                m_pys[lo] = m_pys[lo][:-1] + "2"
                m_pys[lo + 1] = m_pys[lo + 1][:-1] + "2"
            elif run_len >= 4:
                pos = lo
                while hi - pos > 3:
                    m_pys[pos] = m_pys[pos][:-1] + "2"
                    pos += 2
                if hi - pos == 3:
                    m_pys[pos] = m_pys[pos][:-1] + "2"
                    m_pys[pos + 1] = m_pys[pos + 1][:-1] + "2"
                elif hi - pos == 2:
                    m_pys[pos] = m_pys[pos][:-1] + "2"

        locked = [False] * len(m_pys)
        for s0, s1 in word_spans:
            if s1 - s0 >= 2:
                k = s0
                while k < s1:
                    if not m_pys[k].endswith("3"):
                        k += 1
                        continue
                    r_end = k + 1
                    while r_end < s1 and m_pys[r_end].endswith("3"):
                        r_end += 1
                    if r_end - k >= 2:
                        _sandhi_run(k, r_end)
                        for li in range(k, r_end):
                            locked[li] = True
                    k = r_end

        k = 0
        while k < len(m_pys):
            if locked[k] or not m_pys[k].endswith("3"):
                k += 1
                continue
            r_end = k + 1
            while r_end < len(m_pys) and (not locked[r_end]) and m_pys[r_end].endswith("3"):
                r_end += 1
            if r_end - k >= 2:
                _sandhi_run(k, r_end)
            elif r_end < len(m_pys) and m_pys[r_end].endswith("3"):
                m_pys[k] = m_pys[k][:-1] + "2"
            k = r_end

        return m_pys, m_chars

    def _zh_span_to_phonemes_and_bounds(self, span: str, first_bound: int = 4):
        raw_pys = [p[0] for p in pinyin(span, style=Style.TONE3, neutral_tone_with_five=True, errors="ignore") if p]
        m_pys, m_chars = self._apply_zh_sandhi_and_erhua(span, raw_pys)
        syl_of_char = [k for k, ch in enumerate(m_chars) for _ in ch]
        word_start_syls, c0 = set(), 0
        seg_fn = _get_zh_word_segmenter()
        if seg_fn is not None:
            for w in seg_fn(span):
                if c0 < len(syl_of_char) and w not in _ENCLITIC_PARTICLES:
                    if c0 == 0 or syl_of_char[c0] != syl_of_char[c0 - 1]:
                        word_start_syls.add(syl_of_char[c0])
                c0 += len(w)
        else:
            word_start_syls = set(range(0, len(m_pys), 2))

        phrase_start_syls = set()
        n_syls = len(m_pys)
        if n_syls >= 9:
            last_phr = 0
            for ws in sorted(word_start_syls):
                if ws > 0 and (ws - last_phr) >= 5 and (n_syls - ws) >= 3:
                    phrase_start_syls.add(ws)
                    last_phr = ws

        toks, bounds = [], []
        for idx, s in enumerate(m_pys):
            phs = self._split_zh_syllable(s)
            if not phs:
                continue
            if idx == 0:
                b0 = first_bound
            elif idx in phrase_start_syls:
                b0 = 3
            elif idx in word_start_syls:
                b0 = 2
            else:
                b0 = 1
            for pi, ph in enumerate(phs):
                toks.append(ph)
                bounds.append(b0 if pi == 0 else 0)
        return toks, bounds

    def _en_word_to_phonemes_and_bounds(self, word: str, first_bound: int = 4):
        phs = []
        en_g2p = _get_en_g2p()
        if en_g2p is not None:
            for p in en_g2p(word):
                if f"en_{p}" in self.tok2id:
                    phs.append(f"en_{p}")
        if not phs:
            fallback_map = {"a": "AE1", "e": "EH1", "i": "IH1", "o": "AA1", "u": "AH1"}
            for ch in word.lower():
                ap = fallback_map.get(ch, ch.upper())
                if f"en_{ap}" in self.tok2id:
                    phs.append(f"en_{ap}")
        bounds = [(first_bound if i == 0 else 0) for i in range(len(phs))]
        return phs, bounds

    def text_to_ids(self, text: str):
        """将中英/混读文本转换为等长 (phoneme_ids, lang_ids, boundary_ids) 三元组。"""
        tokens = ["<bos>"]
        langs = [0]
        bounds = [0]

        raw = text
        for _mk, _ph in _BZ_MARK.items():
            raw = raw.replace(_mk, _ph)
        norm = self.normalize_text(raw)
        pattern = re.compile(rf"([\u4e00-\u9fff]+|[A-Za-z][A-Za-z'\-]*|[{re.escape(_BZ_MARK_CHARS)}]|[,\.\?!;:])")
        pending_b = 4
        for m in pattern.finditer(norm):
            seg = m.group(0)
            if seg in _BZ_MARK_CHARS:
                if seg != _BZ_MARK["#4"]:
                    pending_b = max(pending_b, 2 + _BZ_MARK_CHARS.index(seg))
            elif re.match(r"^[\u4e00-\u9fff]+$", seg):
                phs, bds = self._zh_span_to_phonemes_and_bounds(seg, first_bound=pending_b)
                for ph, bd in zip(phs, bds):
                    tokens.append(ph if ph in self.tok2id else "<unk>")
                    langs.append(1)
                    bounds.append(bd)
                pending_b = 2
            elif re.match(r"^[A-Za-z]", seg):
                phs, bds = self._en_word_to_phonemes_and_bounds(seg, first_bound=pending_b)
                for ph, bd in zip(phs, bds):
                    tokens.append(ph if ph in self.tok2id else "<unk>")
                    langs.append(2)
                    bounds.append(bd)
                pending_b = 2
            else:
                if seg in self.tok2id:
                    tokens.append(seg)
                    langs.append(0)
                    bounds.append(0)
                pending_b = 4

        tokens.append("<eos>")
        langs.append(0)
        bounds.append(0)
        return [self.tok2id.get(t, self.unk_id) for t in tokens], langs, bounds
