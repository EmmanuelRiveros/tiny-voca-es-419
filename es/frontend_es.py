"""Front end español de Tiny Voca desde cero: texto -> tokens de fonema + acento.

Misma idea que ``tiny_voca.text`` (front end 3) pero solo español:
* eSpeak NG con la voz ``es-419`` es la única fuente de pronunciación.
* Un token es un fonema de eSpeak tal como lo separa ``--sep=_``.
* El acento léxico es una característica paralela (0, 1 primario, 2 secundario).
* Puntuación, límites de palabra y bordes del enunciado son tokens explícitos
  que pueden durar cero cuadros (llevan las pausas).
* Los signos de apertura ¿ ¡ son sintaxis: se quitan antes de eSpeak.

Los primeros 17 símbolos (especiales y puntuación) son idénticos a los del
repo oficial. El notebook 02 lo usa para anotar y el 03 para entrenar e inferir:
no cambies este archivo después de anotar (la huella lo detecta).
"""
from dataclasses import dataclass
import hashlib
import json
import re
import shutil
import subprocess
import unicodedata

VERSION = 1
VOICE = "es-419"
SEPARATOR = "_"

PAD, BOS, EOS, SP, UNK = "<pad>", "<bos>", "<eos>", "<sp>", "<unk>"
SPECIALS = (PAD, BOS, EOS, SP, UNK)
PUNCTUATION = (",", ".", "?", "!", ";", ":", '"', "'", "(", ")", "—", "…")
CONSONANTS = ("p", "b", "t", "d", "k", "ɡ", "f", "s", "z", "θ", "x", "tʃ", "dʒ", "ts", "m", "n", "ɲ", "ŋ",
              "l", "ʎ", "ɾ", "r", "ʝ", "j", "w", "β", "ð", "ɣ")
VOWELS = ("a", "e", "i", "o", "u", "ɛ", "aɪ", "aʊ", "eɪ", "eʊ", "oɪ")
PHONES = CONSONANTS + VOWELS
# Solo se añade al final: cambiar el orden invalida anotaciones y checkpoints.
SYMBOLS = SPECIALS + PUNCTUATION + PHONES
SYMBOL_TO_ID = {symbol: index for index, symbol in enumerate(SYMBOLS)}
PHONE_SET = frozenset(PHONES)
PHONE_IDS = frozenset(SYMBOL_TO_ID[p] for p in PHONES) | {SYMBOL_TO_ID[UNK]}
VOWEL_SET = frozenset(VOWELS)
VOICELESS = frozenset(("p", "t", "k", "f", "s", "θ", "x", "tʃ", "ts"))
FRICATIVES = frozenset(("f", "s", "θ", "x"))
# Misma notación escrita de otra forma (nunca aproximaciones a otro fonema).
EQUIVALENT = {"g": "ɡ", "t͡ʃ": "tʃ", "d͡ʒ": "dʒ", "t͡s": "ts", "pː": "p", "tː": "t", "kː": "k", "bː": "b",
              "dː": "d", "nː": "n", "mː": "m", "lː": "l", "sː": "s"}
STRESS_MARKS = {"ˈ": 1, "ˌ": 2}

_QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "`": "'", "´": "'",
                         "“": '"', "”": '"', "„": '"', "‟": '"', "«": '"', "»": '"',
                         "–": "—", "―": "—", "[": "(", "{": "(", "]": ")", "}": ")"})
_PUNCTUATION_CHARS = frozenset(PUNCTUATION)
_LANGUAGE_TAG = re.compile(r"\((?:_?[a-z0-9]_?){2,3}(?:-(?:_?[a-z0-9]_?)+)*\)")


@dataclass(frozen=True)
class Phonemized:
    symbols: tuple
    ids: tuple
    stress: tuple
    unknown: tuple

    def __len__(self):
        return len(self.ids)


def espeak_executable():
    executable = shutil.which("espeak-ng")
    if executable is None:
        raise RuntimeError("Instala espeak-ng (Colab: apt-get install -y espeak-ng)")
    return executable


def espeak_version():
    return subprocess.run([espeak_executable(), "--version"], capture_output=True, text=True, check=True,
                          timeout=10).stdout.strip()


def identity():
    return {"frontend": "tiny-voca-es-desde-cero", "version": VERSION, "voice": VOICE, "separator": SEPARATOR,
            "symbols": list(SYMBOLS), "equivalent": dict(sorted(EQUIVALENT.items()))}


def fingerprint():
    return hashlib.sha256(json.dumps(identity(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def normalize_text(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("el texto no puede estar vacío")
    text = unicodedata.normalize("NFC", text).translate(_QUOTES)
    if re.search(r"\d", text):
        raise ValueError("escribe los números con letra")
    text = text.replace("¿", " ").replace("¡", " ")
    text = re.sub(r"\.{3,}", "…", text)
    text = re.sub(r"-{2,}|\s-\s", "—", text)
    return re.sub(r"\s+", " ", text).strip()


def chunks(text):
    """``(es_puntuación, trozo)``: cada signo de puntuación es su propio token."""
    start = 0
    for i, char in enumerate(text):
        if char not in _PUNCTUATION_CHARS:
            continue
        if char == "'" and 0 < i < len(text) - 1 and text[i - 1].isalpha() and text[i + 1].isalpha():
            continue
        if text[start:i].strip():
            yield False, text[start:i].strip()
        yield True, char
        start = i + 1
    if text[start:].strip():
        yield False, text[start:].strip()


def parse_espeak(output):
    words = []
    for word in _LANGUAGE_TAG.sub("", output).split():
        units = []
        for unit in word.split(SEPARATOR):
            stress = 0
            for mark, level in STRESS_MARKS.items():
                if mark in unit:
                    stress = stress or level
                    unit = unit.replace(mark, "")
            unit = "".join(c for c in unit if unicodedata.category(c) != "Cf" and c not in "()")
            if unit:
                units.append((unit, stress))
        if units:
            words.append(units)
    return words


def espeak_words(piece):
    # La entrada debe terminar en salto de línea: eSpeak NG 1.51 se come el último carácter si no.
    result = subprocess.run([espeak_executable(), "-q", "--ipa", "--sep=" + SEPARATOR, "-v", VOICE, "--stdin"],
                            input=piece.rstrip("\n") + "\n", encoding="utf-8", capture_output=True,
                            check=True, timeout=30)
    return parse_espeak(result.stdout)


def tokenize(pieces):
    symbols, stress, unknown = [BOS], [0], []
    for punctuation, value in pieces:
        if punctuation:
            if symbols[-1] != value:
                symbols.append(value)
                stress.append(0)
            continue
        for word in value:
            if symbols[-1] in PHONE_SET or symbols[-1] == UNK:
                symbols.append(SP)
                stress.append(0)
            for phone, level in word:
                phone = EQUIVALENT.get(phone, phone)
                if phone not in PHONE_SET:
                    unknown.append(phone)
                    phone = UNK
                symbols.append(phone)
                stress.append(level)
    symbols.append(EOS)
    stress.append(0)
    if not any(s in PHONE_SET or s == UNK for s in symbols):
        raise ValueError("el texto no tiene nada que pronunciar")
    return Phonemized(tuple(symbols), tuple(SYMBOL_TO_ID[s] for s in symbols), tuple(stress), tuple(unknown))


def phonemize(text):
    return tokenize([(True, piece) if punctuation else (False, espeak_words(piece))
                     for punctuation, piece in chunks(normalize_text(text))])


def words_of(phonemized):
    """Palabras de eSpeak como tuplas de fonemas (para el diccionario de MFA)."""
    words, current = [], []
    for symbol in phonemized.symbols:
        if symbol in PHONE_SET or symbol == UNK:
            current.append(symbol)
        elif current:
            words.append(tuple(current))
            current = []
    return words
