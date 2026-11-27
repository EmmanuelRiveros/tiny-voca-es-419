"""tiny_voca.text para Tiny Voca en español (lo instala el notebook 03 en lugar del original).

Mismos nombres que el front end inglés del repo oficial, pero todo sale de ``frontend_es.py``
(eSpeak NG es-419, 56 símbolos): el entrenamiento, la inferencia y la exportación del repo oficial
funcionan sin cambios y comprueban la misma huella con la que el notebook 02 anotó.
"""
import frontend_es as _fe
from frontend_es import (BOS, CONSONANTS, EOS, PAD, PHONE_IDS, PHONE_SET, PHONES, PUNCTUATION, SEPARATOR, SP,
                         SPECIALS, STRESS_MARKS, SYMBOL_TO_ID, SYMBOLS, UNK, VOICELESS, VOWEL_SET, VOWELS,
                         Phonemized, chunks, espeak_executable, espeak_version, espeak_words, normalize_text,
                         parse_espeak, tokenize)

FRONTEND_VERSION = _fe.VERSION
ESPEAK_VOICE = _fe.VOICE
FRONTENDS = ("espeak",)
FRONTEND_VARIABLE = "TINY_VOCA_FRONTEND"
FALLBACK = {}


class NothingToPronounce(ValueError):
    """El texto no tiene nada que decir: solo signos, emojis y similares."""


def frontend():
    return "espeak"


def set_frontend(name):
    if name not in FRONTENDS:
        raise ValueError(f"este modelo solo tiene el front end español ({FRONTENDS}), no {name!r}")


def frontend_identity():
    return _fe.identity()


def frontend_fingerprint():
    return _fe.fingerprint()


def frontend_runtime():
    return {"espeak_version": espeak_version()}


def canonical(phone):
    phone = _fe.EQUIVALENT.get(phone, phone)
    return (phone,) if phone in PHONE_SET else None


def phonemize(text):
    try:
        return _fe.phonemize(text)
    except ValueError as error:
        if "nada que pronunciar" in str(error):
            raise NothingToPronounce(str(error)) from error
        raise


def is_phone(token_ids):
    return [int(i) in PHONE_IDS for i in token_ids]
