"""Genera 01_generar_dataset.ipynb (Colab). Ejecutar desde la raíz del repo:
    python notebooks/es-desde-cero/build_01_generar_dataset.py
"""
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md(r'''
# 01 · Generar el dataset español de F2 con Qwen3-TTS

Primer paso de la receta de Tiny Voca rehecha 100 % en español, con una sola voz (F2).
Clona la voz `F2` del dataset oficial [`n4ze3m/tiny-voca`](https://huggingface.co/datasets/n4ze3m/tiny-voca)
con `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, igual que la receta original (referencia `references/F2.wav` +
su transcripción de `voices.json`, `x_vector_only_mode=False`, lotes de 8, `max_new_tokens=600`,
control de calidad con Whisper y reintento).

**Modo automático (para dejarlo corriendo).** Con `AUTO_SELECT = True` en la celda 5 basta con
*Entorno de ejecución › Ejecutar todo*. La celda 4b carga un «juez» que escucha por ti: un reconocedor de
fonemas compara lo que se oye con los fonemas que eSpeak NG espera en español (es-419), castiga los
sonidos típicos del inglés (ou, ɹ, ə, ɪ…) y mide el parecido de timbre con la F2 original. La celda 6c
usa ese juez para elegir sola la mejor referencia (rondas de arranque), el modo y la temperatura, y la
celda 9 rechaza y repite los clips con acento. Si Colab se corta, *Ejecutar todo* otra vez: la elección
queda guardada y la generación sigue donde iba.

**Orden de uso (manual)**

1. Celdas 1–4b: configuración, Drive e instalación, referencia F2, modelos y juez de acento.
2. Celda 5 elige la referencia activa.
3. Celda 6: **prueba rápida** de 15 frases difíciles. Escúchalas antes de generar nada grande.
4. Celda 7 (opcional): si el acento suena extranjero, crea candidatas con **VoiceDesign**, elige una en
   la celda 5 y repite la prueba de la celda 6. La celda 5 también permite usar una frase de la prueba
   como referencia española de F2 («arranque»).
5. Celda 8: corpus de texto limpio (se congela en Drive la primera vez).
6. Celda 9: generación de ~10 h. **Incremental**: si Colab se corta, vuelve a ejecutar 1–5, 8 y 9 y
   salta los clips ya hechos.
7. Celda 10: resumen y escucha de muestras.

Todo se guarda en Google Drive (`DRIVE_DIR`). Los clips aceptados quedan en `audio/<id>.flac`
(24 kHz, mono, PCM16, pico 0,95) y cada intento en `manifest.jsonl`, con las columnas del dataset
original. El notebook 02 lee ese manifiesto.

**Créditos.** Voz de referencia F2: *tiny-voca dataset, Muhammed Nazeem*
(https://huggingface.co/datasets/n4ze3m/tiny-voca, CC BY 4.0); aquí se usa para clonar la voz en
español. Qwen3-TTS: Apache-2.0. Textos: Common Voice (CC0) y Tatoeba (CC BY 2.0 FR, con enlace por
frase en el manifiesto).

**GPU.** Funciona en T4. Con L4 o A100 va bastante más rápido y usa bf16.
''')

code(r'''
# @title 1 · Configuración
DRIVE_DIR = "/content/drive/MyDrive/tiny-voca-es-desde-cero/01_dataset"
# v2: la v1 cortaba la última vocal de muchas frases (lo detectó el 02). La prueba 6d mostró que se arregla
# añadiendo silencio al final de la referencia. La v2 va a manifest_v2.jsonl, audio_v2/ y rechazados_v2/;
# la v1 (manifest.jsonl, audio/, rechazados/) no se toca. "" = volver a la v1.
DATASET_VERSION = "v2"
REF_END_PAD_SECONDS = 0.6    # silencio añadido al final de la referencia activa (0 = como en la v1)
END_DB_MAX = -12.0           # rechaza (y repite) clips que acaban en seco: últimos 25 ms casi tan fuertes como la voz

VOICE_ID = "F2"              # sufijo de los id (<text_id>_F2), como en el dataset original
TARGET_HOURS = 10.0          # horas ACEPTADAS por el control de calidad
CORPUS_MARGIN = 1.35         # texto de sobra para cubrir rechazos y la estimación de duración
SOURCE_SHARES = {            # reparto aproximado del corpus por fuente (cv_wiki absorbe lo que falte)
    "tatoeba": 0.25,         # frases cortas y conversacionales, muchas preguntas
    "cv_sentence_collector": 0.45,
    "cv_wiki": 0.30,         # frases de Wikipedia extraídas por Common Voice
}
SPLIT_FRACTIONS = {"validation": 0.05, "test": 0.05}   # el resto, train

TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DTYPE = "auto"               # "auto" = bf16 en L4/A100, float16 en T4; usa "float32" si salen audios vacíos o ruido
BATCH_SIZE = 8
MAX_NEW_TOKENS = 600         # 12 tokens/s -> 50 s como máximo
BASE_SEED = 20261121

ASR_MODEL = "small"          # Whisper multilingüe para el control de calidad
WER_MAX = 0.15               # mismo umbral que la receta original
MAX_ATTEMPTS = 2             # 1 intento + 1 reintento
''')

code(r'''
# @title 2 · Drive, instalación y carpetas
from google.colab import drive
drive.mount("/content/drive")

!pip -q install -U qwen-tts num2words
!apt-get -qq install -y espeak-ng > /dev/null

import gc, hashlib, json, math, os, random, re, shutil, time, unicodedata
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from IPython.display import Audio, Markdown, display

ROOT = Path(DRIVE_DIR)
REF_DIR = ROOT / "referencia"
DESIGN_DIR = REF_DIR / "voicedesign"
TEST_DIR = ROOT / "prueba"
CORPUS_DIR = ROOT / "corpus"
SUFFIX = f"_{DATASET_VERSION}" if DATASET_VERSION else ""
AUDIO_DIR = ROOT / f"audio{SUFFIX}"
REJECT_DIR = ROOT / f"rechazados{SUFFIX}"
MANIFEST = ROOT / f"manifest{SUFFIX}.jsonl"
CORPUS_PATH = CORPUS_DIR / "corpus.jsonl"
for folder in (REF_DIR, DESIGN_DIR, TEST_DIR, CORPUS_DIR / "fuentes", AUDIO_DIR, REJECT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

assert torch.cuda.is_available(), "Activa una GPU: Entorno de ejecución > Cambiar tipo de entorno"
CAPABILITY = torch.cuda.get_device_capability()
if DTYPE == "auto":
    TORCH_DTYPE = torch.bfloat16 if CAPABILITY[0] >= 8 else torch.float16
else:
    TORCH_DTYPE = getattr(torch, DTYPE)
print("GPU:", torch.cuda.get_device_name(), "| dtype:", TORCH_DTYPE, "| Drive:", ROOT, "| manifiesto:", MANIFEST.name)
''')

code(r'''
# @title 3 · Referencia F2 original (audio inglés + transcripción) desde Hugging Face
from huggingface_hub import hf_hub_download

F2_WAV = REF_DIR / "F2_en.wav"
F2_JSON = REF_DIR / "F2_en.json"
if not F2_WAV.exists() or not F2_JSON.exists():
    wav = hf_hub_download("n4ze3m/tiny-voca", "references/F2.wav", repo_type="dataset")
    voices = json.loads(Path(hf_hub_download("n4ze3m/tiny-voca", "voices.json", repo_type="dataset")).read_text())
    shutil.copy(wav, F2_WAV)
    F2_JSON.write_text(json.dumps(voices["F2"], ensure_ascii=False, indent=2), encoding="utf-8")
F2_INFO = json.loads(F2_JSON.read_text(encoding="utf-8"))
print(F2_INFO["label"], "|", F2_INFO["prompt"])
print("Texto:", F2_INFO["text"])
display(Audio(str(F2_WAV)))
''')

code(r'''
# @title 4 · Cargar Qwen3-TTS Base y Whisper, funciones comunes
from qwen_tts import Qwen3TTSModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from num2words import num2words
from scipy.signal import resample_poly

tts = Qwen3TTSModel.from_pretrained(TTS_MODEL, device_map="cuda:0", dtype=TORCH_DTYPE, attn_implementation="sdpa")
try:
    LANGUAGE = next(l for l in tts.get_supported_languages() if str(l).lower() == "spanish")
except Exception:
    LANGUAGE = "Spanish"
print("Idioma Qwen:", LANGUAGE)

# Whisper con transformers (misma CUDA que PyTorch). faster-whisper/CTranslate2 pide libcublas.so.12
# y falla en los Colab con otra versión de CUDA.
ASR_NAME = f"openai/whisper-{ASR_MODEL}"
ASR_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ASR_DTYPE = torch.float16 if ASR_DEVICE == "cuda" else torch.float32
asr_processor = WhisperProcessor.from_pretrained(ASR_NAME)
asr = WhisperForConditionalGeneration.from_pretrained(ASR_NAME, torch_dtype=ASR_DTYPE).to(ASR_DEVICE).eval()


def to_numpy(wave):
    if torch.is_tensor(wave):
        wave = wave.detach().float().cpu().numpy()
    return np.asarray(wave, dtype=np.float32).reshape(-1)


def save_audio(path, wave, sr):
    peak = float(np.max(np.abs(wave))) if wave.size else 0.0
    if peak > 0:
        wave = wave * (0.95 / peak)
    path = Path(path)
    sf.write(path, wave, sr, subtype="PCM_16", format="FLAC" if path.suffix == ".flac" else "WAV")


def synthesize(texts, prompt, seed, sampling=None):
    """Lote de clonación; si no cabe en memoria lo parte en dos, como la receta original.

    ``sampling`` (p. ej. {"temperature": 0.6}) va a generate_voice_clone; por defecto SAMPLING de la celda 5.
    """
    sampling = SAMPLING if sampling is None else sampling
    torch.manual_seed(seed)
    try:
        wavs, sr = tts.generate_voice_clone(text=list(texts), language=[LANGUAGE] * len(texts),
                                            voice_clone_prompt=[prompt] * len(texts),
                                            max_new_tokens=MAX_NEW_TOKENS, **sampling)
    except torch.cuda.OutOfMemoryError:
        if len(texts) == 1:
            raise
        torch.cuda.empty_cache()
        half = len(texts) // 2
        first, sr = synthesize(texts[:half], prompt, seed, sampling)
        second, _ = synthesize(texts[half:], prompt, seed + 1, sampling)
        return first + second, sr
    return [to_numpy(w) for w in wavs], sr


WORD = re.compile(r"[a-záéíóúüñ]+")


def normalize_es(text):
    """Normalización solo para medir el WER: minúsculas, números a letra, sin tildes (la ñ se conserva)."""
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"\d+", lambda m: " " + num2words(int(m.group()), lang="es") + " ", text)
    text = text.replace("ñ", "\0")
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    text = unicodedata.normalize("NFC", text).replace("\0", "ñ")
    return " ".join(WORD.findall(text))


def wer(reference, hypothesis):
    ref, hyp = reference.split(), hypothesis.split()
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        previous, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (r != h))
    return row[-1] / max(len(ref), 1)


def transcribe(wave, sr):
    divisor = math.gcd(16000, sr)
    audio = resample_poly(wave, 16000 // divisor, sr // divisor).astype(np.float32)
    # Ventana de 30 s de Whisper: las frases del corpus duran mucho menos.
    features = asr_processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(ASR_DEVICE, ASR_DTYPE)
    with torch.inference_mode():
        ids = asr.generate(features, language="es", task="transcribe", num_beams=3, max_new_tokens=220)
    return asr_processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def end_db(wave, sr, seconds=0.025):
    """Energía de los últimos 25 ms respecto a la voz normal del clip (percentil 95), en dB.
    Un final natural se apaga (< -25 dB); un corte a mitad de vocal queda cerca de 0 dB. Igual que en el 02."""
    wave = np.asarray(wave, dtype=np.float32).reshape(-1)
    size = int(seconds * sr)
    count = len(wave) // size
    if count < 2:
        return 0.0
    rms = np.sqrt((wave[len(wave) - count * size:].reshape(count, size) ** 2).mean(1)) + 1e-8
    return float(20 * np.log10(rms[-1] / np.percentile(rms, 95)))


def quality(text, wave, sr):
    """Mismas reglas que la receta original, con Whisper en español, más el final cortado (END_DB_MAX)."""
    if wave.size == 0 or not np.isfinite(wave).all() or float(np.max(np.abs(wave))) < 1e-5:
        return ["invalid_audio"], "", 1.0, len(wave) / sr
    reasons, duration = [], len(wave) / sr
    if duration < 0.5 or duration > 35:
        reasons.append("duration")
    words = len(normalize_es(text).split())
    if not 0.5 <= words / duration <= 7:
        reasons.append("speaking_rate")
    if float(np.mean(np.abs(wave) < 1e-4)) > 0.8:
        reasons.append("excessive_silence")
    if END_DB_MAX is not None and end_db(wave, sr) > END_DB_MAX:
        reasons.append("cut_ending")
    heard = transcribe(wave, sr)
    error = wer(normalize_es(text), normalize_es(heard) or "<vacio>")
    if error > WER_MAX:
        reasons.append("asr_mismatch")
    return reasons, heard, round(error, 4), round(duration, 3)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_manifest():
    records = []
    if MANIFEST.exists():
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print("Línea incompleta del manifiesto ignorada (corte anterior)")
    return records


def load_active_reference():
    info = json.loads((REF_DIR / "activa.json").read_text(encoding="utf-8"))
    info["path"] = str(REF_DIR / "activa.wav")
    return info


print("Listo. Memoria GPU usada: %.1f GB" % (torch.cuda.memory_allocated() / 1e9))
''')

code(r'''
# @title 4b · Juez automático de acento (reconocedor de fonemas + parecido con F2)
# Escucha por ti. Para cada audio mide:
#   accent   = coste medio por fonema de los fonemas españoles esperados (eSpeak NG es-419) según un
#              reconocedor de fonemas multilingüe + ANGLICISM_WEIGHT × proporción de fonemas típicos del inglés.
#              Menor = más nativo. Es la cifra que se usa para elegir.
#   anglicism, per (errores de fonemas frente a lo esperado), speaker (coseno con la huella de voz de F2).
import subprocess
from functools import lru_cache
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

PHONE_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
ANGLICISM_WEIGHT = 4.0
# Fonemas que el español nunca produce y que delatan acento inglés («perrou», «caro» con ɹ, vocales relajadas…).
ENGLISH_PHONES = ("oʊ", "əʊ", "ɹ", "ə", "ɪ", "ʊ", "æ", "ʌ", "ɚ", "ɜː", "ɑː", "ɔː", "ɑːɹ", "ɔːɹ", "oːɹ", "ɪɹ",
                  "ʊɹ", "ɛɹ", "aɪə", "aɪɚ", "ʊə", "eə", "iə", "tʰ", "pʰ", "kʰ", "ɒ", "əl", "ɪː", "ʊː")
SPANISH_EQUIVALENT = {"g": "ɡ", "t͡ʃ": "tʃ", "d͡ʒ": "dʒ", "t͡s": "ts"}

phone_extractor = Wav2Vec2FeatureExtractor.from_pretrained(PHONE_MODEL)
phone_model = Wav2Vec2ForCTC.from_pretrained(PHONE_MODEL).to(ASR_DEVICE).eval()   # float32: CTC estable
PHONE_VOCAB = json.loads(Path(hf_hub_download(PHONE_MODEL, "vocab.json")).read_text(encoding="utf-8"))
BLANK = PHONE_VOCAB["<pad>"]
SPECIAL_IDS = {PHONE_VOCAB[t] for t in ("<pad>", "<s>", "</s>", "<unk>") if t in PHONE_VOCAB}
ENGLISH_IDS = {PHONE_VOCAB[p] for p in ENGLISH_PHONES if p in PHONE_VOCAB}


@lru_cache(maxsize=4096)
def expected_phones(text):
    """Ids del reconocedor para los fonemas que eSpeak NG (es-419) espera en ``text``."""
    plain = re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).replace("¿", " ").replace("¡", " ")).strip()
    # La entrada termina en salto de línea: eSpeak NG se come el último carácter si no.
    output = subprocess.run(["espeak-ng", "-q", "--ipa", "--sep=_", "-v", "es-419", "--stdin"], input=plain + "\n",
                            encoding="utf-8", capture_output=True, check=True, timeout=30).stdout
    ids = []
    for word in re.sub(r"\([a-z-]+\)", " ", output).split():
        for unit in word.split("_"):
            unit = "".join(c for c in unit if c not in "ˈˌ" and unicodedata.category(c) != "Cf")
            unit = SPANISH_EQUIVALENT.get(unit, unit)
            if unit and unit not in PHONE_VOCAB:
                unit = unit.replace("ː", "")
            if unit in PHONE_VOCAB:
                ids.append(PHONE_VOCAB[unit])
            else:
                ids.extend(PHONE_VOCAB[c] for c in unit if c in PHONE_VOCAB)
    return tuple(ids)


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        previous, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (x != y))
    return row[-1]


def mono(wave):
    wave = np.asarray(wave, dtype=np.float32)
    return wave.mean(axis=1) if wave.ndim == 2 else wave


def resample(wave, sr, target):
    if sr == target:
        return wave.astype(np.float32)
    divisor = math.gcd(target, sr)
    return resample_poly(wave, target // divisor, sr // divisor).astype(np.float32)


def speaker_embedding(wave, sr):
    embedding = tts.model.extract_speaker_embedding(audio=resample(mono(wave), sr, 24000), sr=24000)
    return F.normalize(embedding.detach().float().reshape(-1), dim=0).cpu()


F2_EMBEDDING = speaker_embedding(*sf.read(F2_WAV, dtype="float32"))
BAD_SCORE = {"accent": 99.0, "nll": 99.0, "anglicism": 1.0, "per": 1.0, "speaker": 0.0}


def judge(text, wave, sr):
    wave = mono(wave)
    target = expected_phones(text)
    if wave.size < 0.3 * sr or not np.isfinite(wave).all() or float(np.max(np.abs(wave))) < 1e-5 or not target:
        return dict(BAD_SCORE)
    audio = resample(wave, sr, 16000)
    inputs = phone_extractor(audio, sampling_rate=16000, return_tensors="pt").input_values.to(ASR_DEVICE)
    with torch.inference_mode():
        log_probs = phone_model(inputs).logits[0].float().log_softmax(-1).cpu()
    frames = log_probs.shape[0]
    loss = F.ctc_loss(log_probs[:, None, :], torch.tensor([target]), [frames], [len(target)], blank=BLANK,
                      reduction="sum") if frames >= len(target) else torch.tensor(float("inf"))
    nll = float(loss) / len(target) if torch.isfinite(loss) else 99.0
    greedy, previous = [], None
    for index in log_probs.argmax(-1).tolist():
        if index != previous and index not in SPECIAL_IDS:
            greedy.append(index)
        previous = index
    anglicism = sum(i in ENGLISH_IDS for i in greedy) / max(len(greedy), 1)
    with torch.inference_mode():
        speaker = float(torch.dot(F2_EMBEDDING, speaker_embedding(wave, sr)))
    return {"accent": round(min(nll + ANGLICISM_WEIGHT * anglicism, 99.0), 4), "nll": round(min(nll, 99.0), 4),
            "anglicism": round(anglicism, 4), "per": round(edit_distance(greedy, list(target)) / len(target), 4),
            "speaker": round(speaker, 4)}


def describe(item):
    line = (f"Whisper: _{item['whisper']}_ · WER {item['wer']:.2f} · {item['seconds']:.1f} s · "
            f"acento **{item['accent']:.3f}** (inglés {item['anglicism']:.0%}, PER {item['per']:.2f}) · "
            f"parecido F2 {item['speaker']:.3f}")
    return line + (f" · ⚠️ {', '.join(item['qc_reasons'])}" if item["qc_reasons"] else "")


def run_variant(folder, texts, prompt, sampling, seed, title=None, show=True, force=False):
    """Genera lo que falte en prueba/<folder>, lo puntúa (Whisper + juez) y guarda resultados.json.

    Reutiliza audios y puntuaciones ya hechos: repetir es casi gratis.
    """
    out = TEST_DIR / folder
    out.mkdir(parents=True, exist_ok=True)
    files = [f"{i:02d}.wav" for i in range(1, len(texts) + 1)]
    missing = [i for i, name in enumerate(files) if force or not (out / name).exists()]
    generated, started = 0.0, time.time()
    for start in range(0, len(missing), BATCH_SIZE):
        chunk = missing[start:start + BATCH_SIZE]
        # Con textos iguales (tomas) cada lote usa otra semilla, y dentro del lote cada fila muestrea distinto.
        waves, sr = synthesize([texts[i] for i in chunk], prompt, seed + chunk[0], sampling)
        for i, wave in zip(chunk, waves):
            save_audio(out / files[i], wave, sr)
            generated += len(wave) / sr
    elapsed = time.time() - started
    cache_path = out / "resultados.json"
    cached = {}
    if cache_path.exists():
        cached = {r["file"]: r for r in json.loads(cache_path.read_text(encoding="utf-8")) if "accent" in r}
    fresh = {files[i] for i in missing}
    results = []
    for i, (text, name) in enumerate(zip(texts, files), 1):
        item = cached.get(name)
        if item is None or name in fresh or item["text"] != text:
            wave, sr = sf.read(out / name, dtype="float32")
            reasons, heard, error, duration = quality(text, wave, sr)
            item = {"n": i, "file": name, "text": text, "whisper": heard, "wer": error, "seconds": duration,
                    "qc_reasons": reasons, **judge(text, wave, sr)}
        results.append(item)
    cache_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    if show:
        display(Markdown(f"## {title or folder}\nCarpeta `prueba/{folder}`"))
        for item in results:
            display(Markdown(f"**{item['n']}.** {item['text']}  \n" + describe(item)))
            display(Audio(str(out / item["file"])))
        summary = summarize(results)
        print(f"Pasan el control: {summary['passed']}/{len(results)} · acento medio {summary['accent']:.3f} · "
              f"parecido F2 {summary['speaker']:.3f}")
    return results, generated, elapsed


def summarize(results):
    return {"passed": sum(not r["qc_reasons"] for r in results),
            "pass_rate": sum(not r["qc_reasons"] for r in results) / max(len(results), 1),
            "accent": float(np.mean([r["accent"] for r in results])),
            "speaker": float(np.mean([r["speaker"] for r in results])),
            "accents": [r["accent"] for r in results]}


print("Juez listo. Fonemas esperados de «Pero el perro no es caro»:",
      " ".join(next(p for p, i in PHONE_VOCAB.items() if i == j) for j in expected_phones("Pero el perro no es caro.")))
''')

code(r'''
# @title 5 · Elegir la referencia activa
# "f2_en"                -> audio inglés original de F2 con su transcripción (receta original)
# "design:K"             -> candidata K de VoiceDesign (celda 7)
# "prueba:ETIQUETA:N"    -> frase N de la prueba hecha con la referencia ETIQUETA (arranque: F2 ya en español)
# "mezcla:ETIQUETA:N,M + ETIQUETA2:K" -> une varias frases de prueba en una sola referencia más larga,
#                           en ese orden, con MIX_GAP_SECONDS de silencio entre ellas
#
# Con AUTO_SELECT = True esta es solo la referencia de PARTIDA: la celda 6c prueba variantes y deja activa la mejor.
AUTO_SELECT = True
REFERENCE_CHOICE = "prueba:prueba-f2_en-08:10"   # tu mejor referencia hasta ahora; "f2_en" para empezar de cero
X_VECTOR_ONLY = False    # True = clonar solo el timbre (variante «xvec» de la celda 6b)
SAMPLING = {}            # p. ej. {"temperature": 0.6} (variante «t0.6» de la celda 6b); {} = valores de Qwen
ALLOW_REFERENCE_CHANGE = False   # True solo si quieres mezclar clips de dos referencias en el dataset
MIX_GAP_SECONDS = 0.3


def variant_suffix(x_vector_only, sampling):
    """Sufijo de la etiqueta: cada modo de generación tiene su propia carpeta de prueba."""
    suffix = "-xvec" if x_vector_only else ""
    for key, value in sorted(sampling.items()):
        suffix += f"-t{value:g}" if key == "temperature" else f"-{key}{value:g}"
    return suffix


def mix_reference(spec):
    waves, texts, rate = [], [], None
    for part in spec.split("+"):
        label, _, numbers = part.strip().rpartition(":")
        results = json.loads((TEST_DIR / label / "resultados.json").read_text(encoding="utf-8"))
        for number in numbers.split(","):
            item = results[int(number) - 1]
            wave, sr = sf.read(TEST_DIR / label / item["file"], dtype="float32")
            if rate is not None and sr != rate:
                raise ValueError("las frases de la mezcla tienen distinta frecuencia de muestreo")
            rate = sr
            waves.append(wave)
            texts.append(item["text"].strip())
            print(f"  + {label} #{int(number)}: {item['text']} ({len(wave) / sr:.1f} s)")
    gap = np.zeros(int(MIX_GAP_SECONDS * rate), dtype=np.float32)
    joined = np.concatenate([piece for wave in waves for piece in (wave, gap)][:-1])
    label = "mezcla-" + hashlib.sha1(spec.replace(" ", "").encode()).hexdigest()[:8]
    path = REF_DIR / "mezclas" / f"{label}.wav"
    path.parent.mkdir(exist_ok=True)
    save_audio(path, joined, rate)
    (path.with_suffix(".json")).write_text(json.dumps({"spec": spec, "texts": texts}, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    return path, " ".join(texts), label


def resolve_reference(choice):
    if choice == "f2_en":
        return F2_WAV, F2_INFO["text"], "f2_en"
    kind, _, rest = choice.partition(":")
    if kind == "mezcla":
        return mix_reference(rest)
    if kind == "design":
        meta = json.loads((DESIGN_DIR / f"candidata_{int(rest)}.json").read_text(encoding="utf-8"))
        return DESIGN_DIR / f"candidata_{int(rest)}.wav", meta["text"], f"design-{int(rest)}"
    if kind == "prueba":
        label, _, number = rest.rpartition(":")
        results = json.loads((TEST_DIR / label / "resultados.json").read_text(encoding="utf-8"))
        item = results[int(number) - 1]
        return TEST_DIR / label / item["file"], item["text"], f"prueba-{label}-{int(number):02d}"
    raise ValueError(f"REFERENCE_CHOICE desconocido: {choice}")


def activate(choice, x_vector_only=False, sampling=None, check=True):
    """Deja activa (activa.wav/.json, REFERENCE, PROMPT, X_VECTOR_ONLY, SAMPLING) una referencia y modo."""
    global REFERENCE, PROMPT, X_VECTOR_ONLY, SAMPLING
    sampling = dict(sampling or {})
    source_path, ref_text, ref_label = resolve_reference(choice)
    ref_label += variant_suffix(x_vector_only, sampling)
    base_label = ref_label                       # etiqueta sin el silencio (la de la elección automática)
    if REF_END_PAD_SECONDS:
        ref_label += f"-sil{REF_END_PAD_SECONDS:g}"
    ref_sha = sha256(source_path)
    used = {(r.get("reference_sha256"), r.get("reference_audio")) for r in read_manifest()} - {(None, None)}
    if check and used and used != {(ref_sha, ref_label)} and not ALLOW_REFERENCE_CHANGE:
        raise RuntimeError(f"El manifiesto ya tiene clips con otra referencia o modo ({sorted(map(str, used))}). "
                           "Vuelve a la configuración anterior o pon ALLOW_REFERENCE_CHANGE = True.")
    wave, sr = sf.read(source_path, dtype="float32")
    if REF_END_PAD_SECONDS:
        # Qwen copia cómo termina la referencia: con silencio detrás deja de cortar la última vocal (prueba 6d).
        wave = np.concatenate([mono(wave), np.zeros(int(REF_END_PAD_SECONDS * sr), np.float32)])
    sf.write(REF_DIR / "activa.wav", wave, sr)
    (REF_DIR / "activa.json").write_text(json.dumps(
        {"choice": choice, "label": ref_label, "base_label": base_label, "end_pad_seconds": REF_END_PAD_SECONDS,
         "text": ref_text, "source": str(source_path),
         "sha256": ref_sha, "seconds": round(len(wave) / sr, 2), "x_vector_only": x_vector_only,
         "sampling": sampling}, ensure_ascii=False, indent=2), encoding="utf-8")
    X_VECTOR_ONLY, SAMPLING = x_vector_only, sampling
    REFERENCE = load_active_reference()
    PROMPT = tts.create_voice_clone_prompt(ref_audio=REFERENCE["path"], ref_text=REFERENCE["text"],
                                           x_vector_only_mode=x_vector_only)[0]
    print(f"Referencia activa: {ref_label} ({REFERENCE['seconds']} s) sha256 {ref_sha[:12]}…"
          + (" · solo timbre" if x_vector_only else "") + (f" · muestreo {sampling}" if sampling else ""))
    print("Texto:", ref_text)
    return REFERENCE


# En modo automático la comprobación del manifiesto se hace en la celda 6c, con la referencia elegida.
activate(REFERENCE_CHOICE, X_VECTOR_ONLY, SAMPLING, check=not AUTO_SELECT)
START_CONFIG = {"choice": REFERENCE_CHOICE, "x_vector_only": X_VECTOR_ONLY, "sampling": dict(SAMPLING)}
display(Audio(REFERENCE["path"]))
''')

code(r'''
# @title 6 · Prueba rápida: 15 frases difíciles con la referencia activa
TEST_SENTENCES = [
    "El niño soñaba con una montaña llena de castañas.",
    "Mañana temprano iremos al teatro con la señora Núñez.",
    "El perro de San Roque no tiene rabo porque Ramón Ramírez se lo ha robado.",
    "Tres tristes tigres tragaban trigo en un trigal.",
    "Pero el perro no es caro; el carro sí es caro.",
    "El carro corrió por la carretera rumbo a Oaxaca, en México.",
    "¿Ya leíste el poema que te recomendé el año pasado?",
    "¿Cuántos años tiene tu abuela, y dónde vive ahora?",
    "El héroe del país aún creía en la poesía.",
    "Raúl y Rocío reían en el río mientras caía la lluvia.",
    "Desde la azotea se veía el océano, y el aire olía a sal.",
    "¡Qué maravilla! No esperaba verte por aquí.",
    "¿Me puedes pasar la sal, por favor? Gracias, eres muy amable.",
    "La reunión empieza a las ocho y media, ¿verdad?",
    "Oye, ¿y si mejor nos quedamos en casa a ver una película?",
]
FORCE_TEST = False   # True para regenerar aunque ya existan

display(Markdown(f"### Referencia `{REFERENCE['label']}`"))
display(Audio(REFERENCE["path"]))
results, generated_audio, elapsed = run_variant(REFERENCE["label"], TEST_SENTENCES, PROMPT, SAMPLING, BASE_SEED,
                                                "Prueba de 15 frases", force=FORCE_TEST)
ok = sum(not r["qc_reasons"] for r in results)
print(f"\nPasan el control: {ok}/{len(results)}. Whisper corrige mucho (p. ej. «anos» -> «años»): juzga la ñ y la rr de oído.")
if generated_audio:
    rtf = elapsed / generated_audio
    print(f"Velocidad: {rtf:.2f} s de cómputo por s de audio -> {TARGET_HOURS} h ≈ "
          f"{TARGET_HOURS * rtf * 1.2:.1f} h de GPU (incluye ~20 % de rechazos y Whisper aparte).")
print("Para usar una de estas frases como referencia española: REFERENCE_CHOICE = "
      f"\"prueba:{REFERENCE['label']}:N\" en la celda 5.")
''')

code(r'''
# @title 6b · Explorar variantes a mano (tomas, solo timbre, temperatura)
# Cada variante va a su propia carpeta de prueba/ y no toca nada de lo anterior. Ejecuta antes las celdas 5 y 6.
# Con AUTO_SELECT = True no hace nada: la celda 6c hace esta exploración sola (y más).
EXPLORE_TAKES = 8                   # 1 · tomas nuevas de la PROPIA frase de la referencia (0 = no)
EXPLORE_X_VECTOR = True             # 2 · las 15 frases clonando solo el timbre
EXPLORE_TEMPERATURES = [0.6, 0.3]   # 3 · las 15 frases con menos azar (Qwen usa 0,9); [] = no
EXPLORE_FORCE = False               # True para regenerar aunque ya existan

if AUTO_SELECT:
    print("AUTO_SELECT = True: se salta; la celda 6c explora y elige sola.")
else:
    base = REFERENCE["label"]
    display(Markdown(f"### Referencia activa `{base}`"))
    display(Audio(REFERENCE["path"]))
    choices = []
    if EXPLORE_TAKES:
        run_variant(f"{base}-tomas", [REFERENCE["text"]] * EXPLORE_TAKES, PROMPT, SAMPLING, BASE_SEED + 5000,
                    f"1 · {EXPLORE_TAKES} tomas nuevas de la frase de la referencia", force=EXPLORE_FORCE)
        choices.append(f'Toma K como nueva referencia: REFERENCE_CHOICE = "prueba:{base}-tomas:K"')
    if EXPLORE_X_VECTOR and not REFERENCE.get("x_vector_only"):
        xvec_prompt = tts.create_voice_clone_prompt(ref_audio=REFERENCE["path"], ref_text=REFERENCE["text"],
                                                    x_vector_only_mode=True)[0]
        run_variant(f"{base}{variant_suffix(True, SAMPLING)}", TEST_SENTENCES, xvec_prompt, SAMPLING, BASE_SEED,
                    "2 · Solo timbre (x_vector_only_mode=True)", force=EXPLORE_FORCE)
        choices.append("Solo timbre: X_VECTOR_ONLY = True (misma REFERENCE_CHOICE)")
    for temperature in EXPLORE_TEMPERATURES:
        sampling = {**SAMPLING, "temperature": temperature}
        run_variant(f"{base}{variant_suffix(bool(REFERENCE.get('x_vector_only')), sampling)}", TEST_SENTENCES,
                    PROMPT, sampling, BASE_SEED, f"3 · Temperatura {temperature:g} (Qwen usa 0,9)", force=EXPLORE_FORCE)
        choices.append(f'Temperatura {temperature:g}: SAMPLING = {{"temperature": {temperature:g}}} (misma REFERENCE_CHOICE)')
    print("\nCompara de oído (y con la cifra de acento: menor = más nativo) y aplica la ganadora en la celda 5:")
    for line in choices:
        print(" ·", line)
''')

code(r'''
# @title 6c · Elección automática de referencia, modo y temperatura (para dejarlo corriendo)
# Parte de la referencia de la celda 5 y, en cada ronda:
#   1. Candidatas = las 15 frases de su prueba + AUTO_TAKES tomas nuevas de su propia frase.
#   2. El juez puntúa cada candidata como audio; las AUTO_FINALISTS mejores (que pasan Whisper, duran
#      2,5–15 s y no se alejan del timbre de F2) se prueban de verdad: 15 frases generadas con cada una.
#   3. Gana la de menor acento medio si pasa Whisper casi igual de bien y no pierde parecido con F2.
#      Si ninguna mejora en al menos AUTO_MIN_GAIN, se acaban las rondas.
# Después prueba con la ganadora «solo timbre» y temperaturas más bajas, y deja activa la mejor combinación.
# Todo queda en prueba/ y referencia/auto_eleccion.json; repetir la celda reutiliza lo ya generado.
# Coste aproximado: 10–20 min de GPU la primera vez.
AUTO_ROUNDS = 2
AUTO_TAKES = 8
AUTO_FINALISTS = 3
AUTO_TRY_X_VECTOR = True
AUTO_TEMPERATURES = [0.6, 0.3]
AUTO_MIN_GAIN = 0.02        # mejora relativa mínima del acento para cambiar (evita cambiar por ruido)
PASS_TOLERANCE = 0.14       # puede pasar Whisper hasta ~2 frases de 15 menos que la mejor
SPEAKER_TOLERANCE = 0.03    # parecido con F2 como mucho 0,03 por debajo de la referencia de partida
ACCENT_QUANTILE = 0.9       # umbral de la celda 9: percentil 90 del acento de la ganadora…
ACCENT_MARGIN = 0.15        # … más este margen
AUTO_FILE = REF_DIR / "auto_eleccion.json"
REDO_AUTO_SELECT = False    # True = repetir la elección aunque ya haya una guardada (solo si el manifiesto está vacío)


def test_config(choice, x_vector_only, sampling):
    source, text, label = resolve_reference(choice)
    label += variant_suffix(x_vector_only, sampling)
    prompt = tts.create_voice_clone_prompt(ref_audio=str(source), ref_text=text, x_vector_only_mode=x_vector_only)[0]
    results, _, _ = run_variant(label, TEST_SENTENCES, prompt, sampling, BASE_SEED, show=False)
    summary = {**summarize(results), "choice": choice, "x_vector_only": x_vector_only, "sampling": dict(sampling),
               "label": label, "text": text}
    print(f"  {label}: acento {summary['accent']:.3f} · pasan {summary['passed']}/{len(results)} · "
          f"parecido F2 {summary['speaker']:.3f}")
    return summary


def pick(incumbent, contenders, baseline):
    pool = [incumbent] + contenders
    best_pass = max(c["pass_rate"] for c in pool)
    eligible = [c for c in pool if c["pass_rate"] >= best_pass - PASS_TOLERANCE
                and c["speaker"] >= baseline["speaker"] - SPEAKER_TOLERANCE]
    if not eligible:
        return max(pool, key=lambda c: (c["pass_rate"], -c["accent"]))
    winner = min(eligible, key=lambda c: c["accent"])
    if incumbent in eligible and winner is not incumbent and winner["accent"] > incumbent["accent"] * (1 - AUTO_MIN_GAIN):
        return incumbent
    return winner


def auto_select():
    print("Partida:")
    best = baseline = test_config(START_CONFIG["choice"], START_CONFIG["x_vector_only"], START_CONFIG["sampling"])
    history = [best]
    for round_number in range(1, AUTO_ROUNDS + 1):
        print(f"\nRonda {round_number}: candidatas salidas de {best['label']}")
        source, text, _ = resolve_reference(best["choice"])
        prompt = tts.create_voice_clone_prompt(ref_audio=str(source), ref_text=text,
                                               x_vector_only_mode=best["x_vector_only"])[0]
        pool = [(f"prueba:{best['label']}:{r['n']}", r)
                for r in json.loads((TEST_DIR / best["label"] / "resultados.json").read_text(encoding="utf-8"))]
        if AUTO_TAKES:
            takes, _, _ = run_variant(f"{best['label']}-tomas", [text] * AUTO_TAKES, prompt, best["sampling"],
                                      BASE_SEED + 5000, show=False)
            pool += [(f"prueba:{best['label']}-tomas:{r['n']}", r) for r in takes]
        good = sorted((r["accent"], choice) for choice, r in pool
                      if not r["qc_reasons"] and 2.5 <= r["seconds"] <= 15
                      and r["speaker"] >= baseline["speaker"] - SPEAKER_TOLERANCE)
        tested = {c["choice"] for c in history}
        finalists = [choice for _, choice in good if choice not in tested][:AUTO_FINALISTS]
        if not finalists:
            print("  Ninguna candidata válida.")
            break
        contenders = [test_config(choice, best["x_vector_only"], best["sampling"]) for choice in finalists]
        history += contenders
        winner = pick(best, contenders, baseline)
        if winner is best:
            print(f"  Ninguna mejora a {best['label']}: fin de las rondas.")
            break
        best = winner
        print(f"  Nueva referencia: {best['label']}")
    print(f"\nModos con {best['label']}:")
    variants = []
    if AUTO_TRY_X_VECTOR and not best["x_vector_only"]:
        variants.append((True, best["sampling"]))
    for temperature in AUTO_TEMPERATURES:
        variants.append((best["x_vector_only"], {**best["sampling"], "temperature": temperature}))
    contenders = [test_config(best["choice"], x, sampling) for x, sampling in variants]
    history += contenders
    best = pick(best, contenders, baseline)
    accent_max = float(np.quantile(best["accents"], ACCENT_QUANTILE)) + ACCENT_MARGIN
    chosen = {"choice": best["choice"], "x_vector_only": best["x_vector_only"], "sampling": best["sampling"],
              "label": best["label"], "accent": round(best["accent"], 4), "speaker": round(best["speaker"], 4),
              "passed": best["passed"], "accent_max": round(accent_max, 4), "start": START_CONFIG,
              "date": time.strftime("%Y-%m-%d %H:%M"),
              "tested": [{k: c[k] for k in ("label", "accent", "speaker", "passed")} for c in history]}
    AUTO_FILE.write_text(json.dumps(chosen, ensure_ascii=False, indent=2), encoding="utf-8")
    return chosen


def show_choice(chosen):
    display(Markdown(f"### Elegida: `{chosen['label']}` · acento {chosen['accent']:.3f} · "
                     f"parecido F2 {chosen['speaker']:.3f} · pasan {chosen['passed']}/{len(TEST_SENTENCES)}  \n"
                     f"Umbral de acento para la celda 9: {chosen['accent_max']:.3f}"))
    display(Audio(REFERENCE["path"]))
    folder = TEST_DIR / chosen["label"]
    results = json.loads((folder / "resultados.json").read_text(encoding="utf-8"))
    ordered = sorted(results, key=lambda r: r["accent"])
    for title, item in (("Más nativa", ordered[0]), ("Mediana", ordered[len(ordered) // 2]),
                        ("Con más acento", ordered[-1])):
        display(Markdown(f"**{title}:** {item['text']}  \n" + describe(item)))
        display(Audio(str(folder / item["file"])))


used = {(r.get("reference_sha256"), r.get("reference_audio")) for r in read_manifest()} - {(None, None)}
CHOSEN = None
if not AUTO_SELECT:
    print("AUTO_SELECT = False: se usa la referencia de la celda 5.")
    activate(START_CONFIG["choice"], START_CONFIG["x_vector_only"], START_CONFIG["sampling"])
elif AUTO_FILE.exists() and (used or not REDO_AUTO_SELECT):
    # Ya hay una elección (de la v1 o de esta versión): se reutiliza. Con clips ya hechos nunca se vuelve a
    # elegir (el dataset debe salir de una sola referencia); sin clips, REDO_AUTO_SELECT = True la repite.
    CHOSEN = json.loads(AUTO_FILE.read_text(encoding="utf-8"))
    print(f"Se mantiene la elección guardada ({CHOSEN['date']}): {CHOSEN['label']}"
          + (f" + {REF_END_PAD_SECONDS:g} s de silencio al final" if REF_END_PAD_SECONDS else ""))
    activate(CHOSEN["choice"], CHOSEN["x_vector_only"], CHOSEN["sampling"])
elif used:
    print("Ya hay clips generados sin elección automática guardada: se usa la referencia de la celda 5.")
    activate(START_CONFIG["choice"], START_CONFIG["x_vector_only"], START_CONFIG["sampling"])
else:
    CHOSEN = auto_select()
    activate(CHOSEN["choice"], CHOSEN["x_vector_only"], CHOSEN["sampling"])
if CHOSEN:
    show_choice(CHOSEN)
''')

code(r'''
# @title 6d · Prueba de finales: ¿se come Qwen la última vocal? (no toca el dataset)
# El 02 mostró que muchos clips acaban con la última vocal recortada o sin decir. Esta celda genera las
# mismas 8 frases de cuatro formas, mide el final de cada audio y las pone juntas para comparar de oído:
#   tal_cual      como en la celda 9 (para comparar)
#   suspensivos   la frase acaba en «...» para que Qwen alargue el cierre
#   relleno       la frase + «... Gracias.»; después se recorta en la pausa de antes de «Gracias»
#   ref_silencio  la frase tal cual, pero la referencia lleva REF_PAD_SECONDS de silencio al final
# Todo va a prueba/finales/ (manifiesto, audio/ y referencia activa no se tocan). ~32 clips: unos minutos de GPU.
# Para usarla: clic en esta celda › Entorno de ejecución › «Ejecutar anteriores», y después ejecuta esta.
RUN_END_TEST = False         # True para (re)hacer la prueba; ya hecha: ganaron relleno y ref_silencio
END_FORCE = False            # True para regenerar aunque ya existan
END_SENTENCES = [
    "Le escribe con enorme sencillez, a su amada.",
    "Formaba parte de una base común para la enseñanza en informática.",
    "Estaba viendo la televisión cuando sonó el teléfono.",
    "Fue suave, limpio, honesto, sin tacto e inconsistente.",
    "Para eso nos han hecho.",
    "Sentí un dolor agudo en el estómago.",
    "La casa de Tom tiene una cocina pequeña.",
    "¿No está sucia la mesa?",
]
FILLER = " ... Gracias."
REF_PAD_SECONDS = 0.6
SILENCE_DB = -35.0           # por debajo = silencio (respecto a la voz normal del clip)
MIN_GAP_SECONDS = 0.15       # pausa mínima donde se puede recortar el relleno
END_DIR = TEST_DIR / "finales"


def window_db(wave, sr, seconds):
    size = max(int(seconds * sr), 1)
    count = len(wave) // size
    blocks = wave[len(wave) - count * size:].reshape(count, size)       # alineado al FINAL del audio
    rms = np.sqrt((blocks ** 2).mean(1)) + 1e-8
    return 20 * np.log10(rms / np.percentile(rms, 95)), size


def end_metrics(wave, sr):
    """end_db: energía de los últimos 25 ms (≈0 dB = corte en seco). tail_ms: silencio tras la última voz."""
    wave = mono(wave)
    last, _ = window_db(wave, sr, 0.025)
    db, _ = window_db(wave, sr, 0.01)
    voiced = np.flatnonzero(db > SILENCE_DB)
    tail = (len(db) - 1 - voiced[-1]) * 10 if voiced.size else 0
    return {"end_db": round(float(last[-1]), 1), "tail_ms": int(tail)}


def final_phone_prob(text, wave, sr):
    """Probabilidad máxima del último fonema esperado (eSpeak) en el último cuarto del audio."""
    target = expected_phones(text)
    if not target or len(wave) < 0.3 * sr:
        return 0.0
    audio = resample(mono(wave), sr, 16000)
    inputs = phone_extractor(audio, sampling_rate=16000, return_tensors="pt").input_values.to(ASR_DEVICE)
    with torch.inference_mode():
        probs = phone_model(inputs).logits[0].float().softmax(-1).cpu()
    tail = probs[-max(5, probs.shape[0] // 4):]
    return round(float(tail[:, target[-1]].max()), 3)


def trim_filler(wave, sr):
    """Corta en la última pausa seguida de voz (la de antes de «Gracias»), dejando un poco de silencio."""
    wave = mono(wave)
    size = int(0.01 * sr)
    count = len(wave) // size
    rms = np.sqrt((wave[:count * size].reshape(count, size) ** 2).mean(1)) + 1e-8
    quiet = 20 * np.log10(rms / np.percentile(rms, 95)) < SILENCE_DB
    gaps, start = [], None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if start > 0 and i - start >= MIN_GAP_SECONDS * 100:
                gaps.append((start, i))
            start = None
    if not gaps:
        return None
    first, end = gaps[-1]
    cut = (first + min(25, (end - first) // 2)) * size
    if cut < 0.4 * len(wave):
        return None
    piece = wave[:cut].copy()
    fade = min(int(0.01 * sr), len(piece))
    piece[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    return piece


def with_ellipsis(text):
    text = text.rstrip()
    return text[:-1] + "..." if text.endswith(".") else text + "..."


if not RUN_END_TEST:
    print("RUN_END_TEST = False: se salta.")
else:
    END_DIR.mkdir(parents=True, exist_ok=True)
    ref_wave, ref_sr = sf.read(REFERENCE["path"], dtype="float32")
    ref_end = end_metrics(ref_wave, ref_sr)
    print(f"Referencia activa {REFERENCE['label']}: final {ref_end['end_db']} dB · silencio final {ref_end['tail_ms']} ms")
    padded = END_DIR / "ref_silencio.wav"
    sf.write(str(padded), np.concatenate([mono(ref_wave), np.zeros(int(REF_PAD_SECONDS * ref_sr), np.float32)]), ref_sr)
    padded_prompt = tts.create_voice_clone_prompt(ref_audio=str(padded), ref_text=REFERENCE["text"],
                                                  x_vector_only_mode=X_VECTOR_ONLY)[0]
    VARIANTS = {
        "tal_cual": (END_SENTENCES, PROMPT),
        "suspensivos": ([with_ellipsis(t) for t in END_SENTENCES], PROMPT),
        "relleno": ([t + FILLER for t in END_SENTENCES], PROMPT),
        "ref_silencio": (END_SENTENCES, padded_prompt),
    }
    END_RESULTS = {}
    for name, (texts, prompt) in VARIANTS.items():
        out = END_DIR / name
        out.mkdir(exist_ok=True)
        raw = [out / f"{i:02d}_bruto.wav" for i in range(1, len(texts) + 1)]
        if END_FORCE or not all(p.exists() for p in raw):
            print(f"Generando {name}…")
            waves, sr = synthesize(texts, prompt, BASE_SEED + 777)       # misma semilla en las cuatro
            for path, wave in zip(raw, waves):
                save_audio(path, wave, sr)
        rows = []
        for i, (sentence, path) in enumerate(zip(END_SENTENCES, raw), 1):
            wave, sr = sf.read(str(path), dtype="float32")
            note = ""
            if name == "relleno":
                piece = trim_filler(wave, sr)
                if piece is None:
                    note = "sin pausa antes de «Gracias»: no se pudo recortar"
                else:
                    wave = piece
            final = out / f"{i:02d}.wav"
            sf.write(str(final), wave, sr)
            reasons, heard, error, seconds = quality(sentence, wave, sr)
            rows.append({"n": i, "text": sentence, "file": final.name, "whisper": heard, "wer": error,
                         "seconds": seconds, "qc_reasons": reasons, "note": note,
                         "final_phone": final_phone_prob(sentence, wave, sr),
                         "speaker": judge(sentence, wave, sr)["speaker"], **end_metrics(wave, sr)})
        (out / "resultados.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        END_RESULTS[name] = rows

    display(Markdown("### Resumen (vocal final: probabilidad del último fonema; más alto = se oye; "
                     "silencio final: más alto = la voz se apaga antes de acabar el audio)"))
    print(f"{'variante':14} {'Whisper':>8} {'vocal final':>12} {'silencio final':>15} {'final dB':>9} {'parecido F2':>12}")
    for name, rows in END_RESULTS.items():
        print(f"{name:14} {sum(not r['qc_reasons'] for r in rows):>5}/{len(rows)} "
              f"{np.mean([r['final_phone'] for r in rows]):>12.2f} {np.mean([r['tail_ms'] for r in rows]):>12.0f} ms "
              f"{np.mean([r['end_db'] for r in rows]):>9.1f} {np.mean([r['speaker'] for r in rows]):>12.3f}")
    for i, sentence in enumerate(END_SENTENCES):
        display(Markdown(f"---\n### {i + 1}. {sentence}"))
        for name, rows in END_RESULTS.items():
            r = rows[i]
            warn = " · ⚠️ " + ", ".join(r["qc_reasons"] + ([r["note"]] if r["note"] else [])) if r["qc_reasons"] or r["note"] else ""
            display(Markdown(f"**{name}** · vocal final {r['final_phone']:.2f} · silencio final {r['tail_ms']} ms · "
                             f"{r['end_db']} dB · Whisper: _{r['whisper']}_{warn}"))
            display(Audio(str(END_DIR / name / r["file"])))
    print("\nEscucha sobre todo la última vocal de cada frase y dime qué variante suena completa.")
''')

code(r'''
# @title 7 · (Opcional) VoiceDesign: F2 con acento mexicano nativo
# Úsalo solo si la clonación desde el audio inglés arrastra acento extranjero.
# OJO: VoiceDesign crea una voz nueva a partir de la descripción; el timbre no será idéntico a F2.
RUN_VOICE_DESIGN = False
FORCE_DESIGN = False
DESIGN_INSTRUCT = (
    "A young adult Mexican woman in her mid twenties with a bright, clear, slightly higher-pitched voice. "
    "Native Mexican Spanish accent from Mexico City, neutral and clear, energetic but relaxed and believable, "
    "playful warmth, expressive natural intonation."
)
DESIGN_TEXT = ("Hola, qué gusto saber de ti. Estaba pensando que podríamos ir por un café este sábado, "
               "como a las diez y media. ¿Te parece bien? Ah, y por fin terminé ese proyecto. "
               "¡Estoy muy contenta con cómo quedó!")
DESIGN_SEEDS = [31416, 31417, 31418, 31419]   # 31416 es la semilla original de F2

if RUN_VOICE_DESIGN:
    designer = Qwen3TTSModel.from_pretrained(DESIGN_MODEL, device_map="cuda:0", dtype=TORCH_DTYPE,
                                             attn_implementation="sdpa")
    for k, seed in enumerate(DESIGN_SEEDS, 1):
        wav_path, meta_path = DESIGN_DIR / f"candidata_{k}.wav", DESIGN_DIR / f"candidata_{k}.json"
        if FORCE_DESIGN or not wav_path.exists():
            torch.manual_seed(seed)
            waves, sr = designer.generate_voice_design(text=DESIGN_TEXT, language=LANGUAGE, instruct=DESIGN_INSTRUCT)
            wave = to_numpy(waves[0])
            save_audio(wav_path, wave, sr)
            meta_path.write_text(json.dumps({"instruct": DESIGN_INSTRUCT, "text": DESIGN_TEXT, "seed": seed,
                                             "model": DESIGN_MODEL, "seconds": round(len(wave) / sr, 2)},
                                            ensure_ascii=False, indent=2), encoding="utf-8")
    del designer
    gc.collect()
    torch.cuda.empty_cache()

for k in range(1, len(DESIGN_SEEDS) + 1):
    if (DESIGN_DIR / f"candidata_{k}.wav").exists():
        display(Markdown(f"**Candidata {k}** → en la celda 5: `REFERENCE_CHOICE = \"design:{k}\"`, y luego repite la celda 6"))
        display(Audio(str(DESIGN_DIR / f"candidata_{k}.wav")))
''')

code(r'''
# @title 8 · Corpus de texto limpio en español (se congela en Drive)
import bz2, urllib.request

REBUILD_CORPUS = False   # los id son hash del texto: reconstruir no invalida los clips ya hechos
EXTRA_TEXT = CORPUS_DIR / "extra.txt"   # opcional: tus frases, una por línea; entran todas primero
CHARS_PER_SECOND = 14.0                 # estimación de habla; se corrige con lo medido en la celda 6
SOURCES = {
    "cv_sentence_collector": ("https://raw.githubusercontent.com/common-voice/common-voice/main/server/data/es/sentence-collector.txt",
                              "CC0-1.0"),
    "cv_wiki": ("https://raw.githubusercontent.com/common-voice/common-voice/main/server/data/es/wiki.es.txt", "CC0-1.0"),
    "tatoeba": ("https://downloads.tatoeba.org/exports/per_language/spa/spa_sentences.tsv.bz2", "CC-BY-2.0-FR"),
}


def fetch(name):
    url = SOURCES[name][0]
    path = CORPUS_DIR / "fuentes" / url.rsplit("/", 1)[1]
    if not path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "tiny-voca-es-colab"})
        with urllib.request.urlopen(request, timeout=120) as response:
            path.write_bytes(response.read())
    data = path.read_bytes()
    text = bz2.decompress(data).decode("utf-8") if path.suffix == ".bz2" else data.decode("utf-8")
    if name == "tatoeba":   # id \t spa \t frase
        return [(parts[2], parts[0]) for parts in (line.split("\t", 2) for line in text.splitlines()) if len(parts) == 3]
    return [(line, None) for line in text.splitlines()]


LETTERS = "A-Za-zÁÉÍÓÚÜÑáéíóúüñ"
ALLOWED = re.compile(rf"^[{LETTERS}¿?¡!,.;: ]+$")
FILTER_VOSOTROS = True   # quita «vosotros», «os», «hacéis»… (español de España) para un F2 mexicano
VOSOTROS = re.compile(r"\b(?:vosotr[oa]s|os)\b|[a-záéíóúñ]+(?:áis|éis)\b", re.I)
ABBREVIATION = re.compile(r"\b(?:sr|sra|srta|dr|dra|lic|ing|etc|ej|pág|núm|art|av|ud|uds|vd|vds|aprox|tel|"
                          r"depto|dpto|cía|gral|prof|ee|uu|sto|sta|mr|mrs|var|subsp|sp|spp|cap|vol|ed|pp|op|cit|fig|"
                          r"km|kg|cm|mm|ml|gr|hab|aprox|cf|ibid|vs)\b\.?", re.I)
FOREIGN = re.compile(r"[wk]|sh|th|ph|ck|tz|ss|ff|q(?!u)|([a-zñ])\1\1", re.I)
ROMAN_OR_INITIAL = re.compile(r"\b(?:[IVXLCDM]{2,}|[B-DF-HJ-NP-TV-XZI])\b")
ACRONYM = re.compile(r"\b[A-ZÁÉÍÓÚÑ]{2,}\b")
SPANISH_WORD = re.compile(rf"[{LETTERS}]+")


def number_to_words(match):
    value = int(match.group())
    return num2words(value, lang="es") if value < 10000 else match.group()


def clean(raw):
    """Devuelve la frase lista para leer o None. Conservadora: mejor perder frases que meter basura."""
    text = unicodedata.normalize("NFC", raw).strip()
    text = re.sub(r"[«»“”\"]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    text = re.sub(r"([¿¡])\s+", r"\1", text)
    text = re.sub(r"(?<!\d)(?<!\d[.,])\d{1,4}(?!\d|[.,:]\d|[%ºª])", number_to_words, text)
    if not ALLOWED.match(text) or ".." in text or ABBREVIATION.search(text):
        return None
    if ROMAN_OR_INITIAL.search(text) or ACRONYM.search(text) or (FILTER_VOSOTROS and VOSOTROS.search(text)):
        return None
    if not re.match(rf"^[¿¡]?[A-ZÁÉÍÓÚÑ]", text) or text[-1] not in ".?!":
        return None
    for opening, closing in (("¿", "?"), ("¡", "!")):
        if text.count(opening) != text.count(closing):
            return None
        if text.count(opening) and text.index(opening) > text.index(closing):
            return None
    words = SPANISH_WORD.findall(text)
    if not 4 <= len(words) <= 28 or len(text) > 220:
        return None
    if any(FOREIGN.search(w) or not re.search(r"[aeiouáéíóúüy]", w, re.I) for w in words):
        return None
    tokens = text.split()
    inner_capitals = sum(1 for previous, token in zip(tokens, tokens[1:])
                         if token.lstrip("¿¡")[:1].isupper() and previous[-1] not in ".?!")
    if inner_capitals > 2:
        return None
    return text


def dedupe_key(text):
    return " ".join(re.findall(r"[a-záéíóúüñ]+", text.casefold()))


def category(text):
    return "question" if "¿" in text else "exclamation" if "¡" in text else "statement"


def estimate_seconds(text):
    return len(text) / CHARS_PER_SECOND + 0.6


def text_split(text_id):
    position = int(text_id[:8], 16) / 0xFFFFFFFF
    cut = 0.0
    for name, fraction in SPLIT_FRACTIONS.items():
        cut += fraction
        if position < cut:
            return name
    return "train"


if REBUILD_CORPUS or not CORPUS_PATH.exists():
    rng, seen, selected = random.Random(BASE_SEED), set(), []
    target = TARGET_HOURS * 3600 * CORPUS_MARGIN

    def take(candidates, origin, license_name, budget):
        used = 0.0
        for raw, source_id in candidates:
            if used >= budget:
                break
            text = clean(raw)
            if text is None or dedupe_key(text) in seen:
                continue
            seen.add(dedupe_key(text))
            text_id = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
            url = f"https://tatoeba.org/es/sentences/show/{source_id}" if origin == "tatoeba" else SOURCES.get(origin, ("",))[0]
            selected.append({"text_id": text_id, "text": text, "split": text_split(text_id), "category": category(text),
                             "origin": origin, "source_id": source_id, "source_license": license_name, "source_url": url})
            used += estimate_seconds(text)
        return used

    report = {}
    if EXTRA_TEXT.exists():
        report["extra"] = take([(l, None) for l in EXTRA_TEXT.read_text(encoding="utf-8").splitlines()],
                               "extra", "usuario", float("inf"))
    carry = 0.0
    for name in ("tatoeba", "cv_sentence_collector", "cv_wiki"):
        rows = fetch(name)
        rng.shuffle(rows)
        remaining = target - sum(report.values())
        budget = remaining if name == "cv_wiki" else target * SOURCE_SHARES[name] + carry
        report[name] = take(rows, name, SOURCES[name][1], budget)
        carry = max(0.0, budget - report[name])
        print(f"{name}: {len(rows)} líneas, {report[name] / 3600:.2f} h estimadas tomadas")
    rng.shuffle(selected)
    with CORPUS_PATH.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

CORPUS = [json.loads(line) for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines()]
from collections import Counter
print(f"Corpus: {len(CORPUS)} frases, ≈{sum(estimate_seconds(r['text']) for r in CORPUS) / 3600:.1f} h estimadas")
print("Por fuente:", dict(Counter(r["origin"] for r in CORPUS)))
print("Por tipo:", dict(Counter(r["category"] for r in CORPUS)), "| Por split:", dict(Counter(r["split"] for r in CORPUS)))
for row in random.Random(1).sample(CORPUS, 12):
    print(" ·", row["text"])
''')

code(r'''
# @title 9 · Generar ~10 h (incremental: vuelve a ejecutar tras un corte y sigue donde iba)
WINDOW = 64            # se ordenan por longitud de 64 en 64 para que los lotes tengan poco relleno
ACCENT_FILTER = True   # rechaza (y repite, hasta MAX_ATTEMPTS) los clips cuyo acento supere el umbral de la celda 6c
ACCENT_MAX = None      # None = umbral automático de la celda 6c; o un número fijo (menor = más estricto)
GEN_TAKES = 1          # 2 = genera cada frase dos veces y se queda con la mejor (doble de GPU, menos acento)
REFERENCE = load_active_reference()
if ACCENT_FILTER and ACCENT_MAX is None:
    auto_file = REF_DIR / "auto_eleccion.json"
    saved = json.loads(auto_file.read_text(encoding="utf-8")) if auto_file.exists() else {}
    if saved.get("label") in (REFERENCE["label"], REFERENCE.get("base_label")):
        ACCENT_MAX = saved["accent_max"]
    else:
        # Sin elección automática: umbral sacado de la prueba de 15 frases (celda 6) de la referencia activa.
        tested = TEST_DIR / REFERENCE["label"] / "resultados.json"
        scored = [r["accent"] for r in json.loads(tested.read_text(encoding="utf-8")) if "accent" in r] if tested.exists() else []
        ACCENT_MAX = float(np.quantile(scored, 0.9)) + 0.15 if scored else None
print("Filtro de acento:", f"umbral {ACCENT_MAX:.3f}" if ACCENT_FILTER and ACCENT_MAX else "desactivado",
      f"· mejor de {GEN_TAKES} tomas" if GEN_TAKES > 1 else "")
records = read_manifest()
used = {(r.get("reference_sha256"), r.get("reference_audio")) for r in records} - {(None, None)}
if used and used != {(REFERENCE["sha256"], REFERENCE["label"])} and not ALLOW_REFERENCE_CHANGE:
    raise RuntimeError("La referencia activa no es la usada en los clips existentes; revisa la celda 5.")

last = {}
for record in records:
    last[record["id"]] = record
clip_rows = {f"{row['text_id']}_{VOICE_ID}": row for row in CORPUS}


def needs_work(record):
    return record is None or (record["qc_reasons"] and record["attempt"] < MAX_ATTEMPTS)


def accepted_seconds():
    return sum(r["duration_seconds"] for r in last.values() if not r["qc_reasons"])


def append_record(record):
    with MANIFEST.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    last[record["id"]] = record


target, started, batches = TARGET_HOURS * 3600, time.time(), 0
print(f"Ya aceptadas: {accepted_seconds() / 3600:.2f} h de {TARGET_HOURS} h")
while accepted_seconds() < target:
    todo = [clip for clip in clip_rows if needs_work(last.get(clip))][:WINDOW]
    if not todo:
        print("Se acabó el corpus. Sube CORPUS_MARGIN y ejecuta la celda 8 con REBUILD_CORPUS = True.")
        break
    todo.sort(key=lambda clip: len(clip_rows[clip]["text"]))
    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo[start:start + BATCH_SIZE]
        attempts = [(last[c]["attempt"] + 1) if c in last else 1 for c in batch]
        seed = BASE_SEED + int(batch[0][:8], 16) % 1_000_000 + max(attempts) * 100_000
        tick = time.time()
        takes = []
        for k in range(GEN_TAKES):
            waves, sr = synthesize([clip_rows[c]["text"] for c in batch], PROMPT, seed + k * 7919)
            takes.append(waves)
        generation_seconds = round(time.time() - tick, 2)
        for index, (clip, attempt) in enumerate(zip(batch, attempts)):
            row = clip_rows[clip]
            options = []
            for k in range(GEN_TAKES):
                wave = takes[k][index]
                reasons, heard, error, duration = quality(row["text"], wave, sr)
                scores = judge(row["text"], wave, sr)
                if ACCENT_FILTER and ACCENT_MAX and scores["accent"] > ACCENT_MAX:
                    reasons.append("accent")
                options.append((bool(reasons), scores["accent"], k, wave, reasons, heard, error, duration, scores))
            _, _, take, wave, reasons, heard, error, duration, scores = min(options, key=lambda o: o[:3])
            target_path = AUDIO_DIR / f"{clip}.flac" if not reasons else REJECT_DIR / f"{clip}_a{attempt}.flac"
            save_audio(target_path, wave, sr)
            append_record({
                "id": clip, "text_id": row["text_id"], "voice_id": VOICE_ID, "text": row["text"],
                "split": row["split"], "category": row["category"], "origin": row["origin"],
                "source_id": row["source_id"], "source_license": row["source_license"], "source_url": row["source_url"],
                "duration_seconds": duration, "sample_rate": sr, "asr_transcript": heard, "normalized_wer": error,
                "qc_reasons": reasons, "attempt": attempt, "seed": seed + take * 7919, "language": LANGUAGE,
                "accent_score": scores["accent"], "accent_nll": scores["nll"], "anglicism": scores["anglicism"],
                "phone_error_rate": scores["per"], "speaker_similarity": scores["speaker"], "takes": GEN_TAKES,
                "generation_seconds_batch": generation_seconds, "teacher_model": TTS_MODEL,
                "reference_audio": REFERENCE["label"], "reference_sha256": REFERENCE["sha256"],
                "audio_path": str(target_path.relative_to(ROOT)),
            })
        batches += 1
        if batches % 10 == 0:
            done = accepted_seconds()
            rate = (time.time() - started) / max(batches, 1)
            rejected = sum(1 for r in last.values() if r["qc_reasons"])
            print(f"{time.strftime('%H:%M:%S')} · aceptadas {done / 3600:.2f}/{TARGET_HOURS} h · "
                  f"clips {len(last)} · rechazados ahora {rejected} · {rate:.1f} s/lote")
        if accepted_seconds() >= target:
            break
print(f"Fin de la sesión: {accepted_seconds() / 3600:.2f} h aceptadas.")
''')

code(r'''
# @title 10 · Resumen, escucha de muestras y dataset_info.json
from collections import Counter
records = read_manifest()
last = {}
for record in records:
    last[record["id"]] = record
accepted = [r for r in last.values() if not r["qc_reasons"]]
rejected = [r for r in last.values() if r["qc_reasons"]]
hours = {split: round(sum(r["duration_seconds"] for r in accepted if r["split"] == split) / 3600, 3)
         for split in ("train", "validation", "test")}
info = {
    "voice_id": VOICE_ID, "language": "es (México, es-419)", "teacher_model": TTS_MODEL,
    "reference": load_active_reference(), "sample_rate": 24000, "format": "FLAC PCM16, pico 0,95",
    "clips_accepted": len(accepted), "clips_rejected_final": len(rejected), "attempts_total": len(records),
    "hours_accepted": round(sum(r["duration_seconds"] for r in accepted) / 3600, 3), "hours_by_split": hours,
    "clips_by_split": dict(Counter(r["split"] for r in accepted)),
    "reject_reasons": dict(Counter(reason for r in rejected for reason in r["qc_reasons"])),
    "config": {"batch_size": BATCH_SIZE, "max_new_tokens": MAX_NEW_TOKENS, "base_seed": BASE_SEED,
               "asr_model": ASR_MODEL, "wer_max": WER_MAX, "max_attempts": MAX_ATTEMPTS,
               "phone_model": PHONE_MODEL, "accent_filter": globals().get("ACCENT_FILTER"),
               "accent_max": globals().get("ACCENT_MAX"), "gen_takes": globals().get("GEN_TAKES"),
               "auto_selection": globals().get("CHOSEN")},
    "accent_mean_accepted": round(float(np.mean([r["accent_score"] for r in accepted if "accent_score" in r] or [0])), 4),
    "credits": "Voz de referencia F2: tiny-voca dataset, Muhammed Nazeem, "
               "https://huggingface.co/datasets/n4ze3m/tiny-voca (CC BY 4.0). Audio generado con Qwen3-TTS "
               "(Apache-2.0). Textos: Common Voice (CC0) y Tatoeba (CC BY 2.0 FR, URL por frase).",
}
(ROOT / f"dataset_info{SUFFIX}.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in info.items() if k not in ("reference", "credits")}, ensure_ascii=False, indent=2))

rng = random.Random()
with_accent = sorted((r for r in accepted if "accent_score" in r), key=lambda r: r["accent_score"])
for title, group, count in (("Aceptados al azar", accepted, 6), ("Aceptados con más acento", with_accent[-3:], 3),
                            ("Rechazados al azar", rejected, 3)):
    display(Markdown(f"### {title}"))
    for record in rng.sample(group, min(count, len(group))):
        display(Markdown(f"{record['text']}  \nWhisper: _{record['asr_transcript']}_ · WER {record['normalized_wer']}"
                         + (f" · acento {record['accent_score']:.3f}" if "accent_score" in record else "")
                         + (f" · {', '.join(record['qc_reasons'])}" if record["qc_reasons"] else "")))
        display(Audio(str(ROOT / record["audio_path"])))
''')

md(r'''
**Siguiente:** `02_fonemas_y_alineacion.ipynb` leerá `manifest_v2.jsonl` (`MANIFEST_NAME` en su celda 1;
`manifest.jsonl` es la v1) (último registro por `id`, solo los
que tienen `qc_reasons` vacío) y `audio/`.
''')

notebook = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {"colab": {"provenance": [], "gpuType": "T4"}, "accelerator": "GPU",
                 "kernelspec": {"name": "python3", "display_name": "Python 3"},
                 "language_info": {"name": "python"}},
    "cells": [],
}
for cell in cells:
    source = cell["source"]
    if cell["cell_type"] == "code":
        ast.parse("\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("!", "%"))))
    cell["source"] = source.splitlines(keepends=True)
    notebook["cells"].append(cell)
target = HERE / "01_generar_dataset.ipynb"
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(target, len(notebook["cells"]), "celdas")
