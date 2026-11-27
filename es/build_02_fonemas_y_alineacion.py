"""Genera 02_fonemas_y_alineacion.ipynb (Colab). Ejecutar desde cualquier carpeta:
    python notebooks/es-desde-cero/build_02_fonemas_y_alineacion.py

Incrusta frontend_es.py y alineacion_es.py (misma carpeta) en el notebook: el notebook los escribe en
Drive y el 03 los reutiliza desde ahí.
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


def embedded(name):
    source = (HERE / name).read_text(encoding="utf-8")
    if "'''" in source:
        raise ValueError(f"{name} no puede contener comillas triples simples")
    return source


md(r'''
# 02 · Fonemas es-419 y alineación forzada (MFA) para Tiny Voca

Segundo paso de Tiny Voca desde cero en español (una voz, F2). Convierte el audio del notebook 01 en
lo que lee el entrenamiento del repo oficial [`n4ze3m/tiny-voca`](https://github.com/n4ze3m/tiny-voca):

| Capa | Quién la escribe | Archivos |
| --- | --- | --- |
| Acústica | `python -m tiny_voca.prepare` **del repo oficial, sin cambios** | `features/<split>.jsonl`, `features/<split>/<sha256(id)>.pt` (`wave`, `mel`, `prosody`, `speaker`), `features.json`, `<split>.preparation.json` |
| Anotación | este notebook (mismo formato que `tiny_voca.annotate`) | `anotaciones/<split>.jsonl` (`symbols`, `tokens`, `stress`, `durations`…), `annotations.json`, `summary.json`, `phonemes.jsonl` |

**Diferencias con el repo oficial (a propósito):**
* Fonemas: eSpeak NG `es-419` con un inventario solo español (`frontend_es.py`). Los 17 primeros
  símbolos (especiales y puntuación) son los del repo oficial; después vienen 39 fonemas españoles
  (56 en total; el inglés tiene 76).
* Alineación: **Montreal Forced Aligner** entrenado sobre este mismo corpus, en vez del CTC
  `wav2vec2-lv-60-espeak-cv-ft` del repo oficial. Las duraciones siguen sus reglas: rejilla de 256
  muestras, cada fonema ≥ 1 cuadro, pausas ≥ 7 cuadros (~75 ms) en la puntuación o `<sp>`.
* El notebook 03 tiene que cargar `frontend_es.py` en lugar de `tiny_voca.text` (otro inventario y
  otra huella `frontend`) y usar `vocab_size = 56`, `speakers = ("F2",)`.

**No necesita GPU.** Usa un entorno de CPU (si puedes, con RAM alta: más núcleos = más rápido).

**Incremental:** todo vive en Drive. Si Colab se corta, vuelve a ejecutar las celdas en orden; cada paso
retoma donde quedó (fonemas en caché, `prepare --resume`, MFA marcado como hecho, anotación rehecha
en minutos).
''')

code(r'''
#@title 1 · Configuración
import os

DATASET_DIR = "/content/drive/MyDrive/tiny-voca-es-desde-cero/01_dataset"     # salida del notebook 01
MANIFEST_NAME = "manifest_v2.jsonl"         # v2 del 01 (referencia con silencio al final); "manifest.jsonl" = v1
OUT_DIR = "/content/drive/MyDrive/tiny-voca-es-desde-cero/02_alineacion_v2"   # la v1 queda en 02_alineacion
VOICE_ID = "F2"
DATASET_REVISION = "tiny-voca-es-f2-v2"     # cámbialo si regeneras el dataset

TINY_VOCA_REPO = "https://github.com/n4ze3m/tiny-voca.git"
TINY_VOCA_COMMIT = ""                       # vacío = la punta actual; se anota el hash usado

# Alineación y control de calidad (mismos nombres y valores por defecto que tiny_voca.annotate).
MIN_PAUSE_FRAMES = 7
LIMITS = {
    "max_phone_frames": 70,             # ~0,75 s en un solo fonema = alineación rota
    "min_vowel_voiced": 0.5,            # fracción de cuadros de vocal con sonoridad según pyin
    "min_phones_per_second": 4.0,
    "max_phones_per_second": 30.0,
    "worst_likelihood_fraction": 0.02,  # descarta el 2 % con peor verosimilitud de MFA
    "max_end_db": -12.0,                # final cortado: los últimos 25 ms suenan casi tan fuerte como la voz
                                        # (Qwen a veces corta la última vocal); None = sin este filtro
    "min_final_vowel_frames": 5,        # si la frase acaba en vocal, MFA le da ≥5 cuadros (~55 ms); 1–4 = Qwen
                                        # no llegó a decirla (comprobado de oído); None = sin este filtro
    "max_edge_silence_frames": 94,      # ~1 s de silencio antes de hablar o después de acabar = Qwen tardó en
                                        # arrancar o se quedó callado; None = sin este filtro
}
MFA_BEAM, MFA_RETRY_BEAM = 100, 400
WORKERS = os.cpu_count() or 2
SYNC_EVERY_SECONDS = 300                # copia de features locales a Drive durante prepare
print("núcleos de CPU:", WORKERS)
''')

code(r'''
#@title 2 · Drive, eSpeak NG, repo oficial y Montreal Forced Aligner
import json, os, shutil, subprocess, sys
from pathlib import Path
from google.colab import drive

drive.mount("/content/drive")
DATA = Path(DATASET_DIR)
OUT = Path(OUT_DIR)
if not (DATA / MANIFEST_NAME).is_file():
    raise FileNotFoundError(f"No encuentro {DATA / MANIFEST_NAME}: ejecuta antes el notebook 01")
for folder in ("codigo", "prepare_inputs", "features", "mfa", "anotaciones"):
    (OUT / folder).mkdir(parents=True, exist_ok=True)
FEATURES_LOCAL = Path("/content/features")
FEATURES_DRIVE = OUT / "features"
ANNOTATIONS = OUT / "anotaciones"

!apt-get -qq install -y espeak-ng > /dev/null

REPO = Path("/content/tiny-voca")
if not REPO.exists():
    !git clone -q {TINY_VOCA_REPO} {REPO}
if TINY_VOCA_COMMIT:
    !git -C {REPO} checkout -q {TINY_VOCA_COMMIT}
REPO_COMMIT = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
REPO_SRC = next(p.parent.parent for p in REPO.rglob("tiny_voca/prepare.py"))
sys.path.insert(0, str(REPO_SRC))
print("tiny-voca", REPO_COMMIT, "en", REPO_SRC)

# MFA necesita Kaldi: solo está en conda-forge. micromamba no reinicia el kernel de Colab.
MAMBA = Path("/content/bin/micromamba")
MFA_ENV = Path("/content/mfa_env")
if not (MFA_ENV / "bin" / "mfa").exists():
    if not MAMBA.exists():
        !curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /content bin/micromamba
    !{MAMBA} create -q -y -p {MFA_ENV} -c conda-forge montreal-forced-aligner python=3.11 > /dev/null
MFA_RUN_ENV = {**os.environ, "PATH": f"{MFA_ENV / 'bin'}:{os.environ['PATH']}", "MFA_ROOT_DIR": "/content/mfa_root"}


def mfa(*args, check=True):
    result = subprocess.run([str(MFA_ENV / "bin" / "mfa"), *map(str, args)], env=MFA_RUN_ENV, text=True,
                            capture_output=True)
    if check and result.returncode:
        print(result.stdout[-4000:], result.stderr[-4000:])
        raise RuntimeError(f"mfa {args[0]} falló ({result.returncode})")
    return result


MFA_VERSION = mfa("version").stdout.strip()
mfa("configure", "--disable_use_postgres", check=False)   # SQLite: sin servidor postgres como root
print("MFA", MFA_VERSION)
print(subprocess.run(["espeak-ng", "--version"], capture_output=True, text=True).stdout.strip())
''')

code(r'''
#@title 3 · Front end español (`frontend_es.py`) y utilidades de alineación
import importlib, hashlib

SOURCES = {
    "frontend_es.py": r''' + "'''" + embedded("frontend_es.py") + "'''" + r''',
    "alineacion_es.py": r''' + "'''" + embedded("alineacion_es.py") + "'''" + r''',
}
for name, source in SOURCES.items():
    target = OUT / "codigo" / name
    if target.exists() and target.read_text(encoding="utf-8") != source:
        # El 03 lee este archivo: si cambia, las anotaciones ya hechas no corresponden al front end.
        raise RuntimeError(f"{target} existe y es distinto. Usa otro OUT_DIR o bórralo a mano si sabes lo que haces")
    if not target.exists():
        target.write_text(source, encoding="utf-8")
sys.path.insert(0, str(OUT / "codigo"))
import frontend_es as fe
import alineacion_es as al
importlib.reload(fe); importlib.reload(al)

ESPEAK_VERSION = fe.espeak_version()
inventory = {"fingerprint": fe.fingerprint(), "identity": fe.identity(), "vocab_size": len(fe.SYMBOLS),
             "espeak": ESPEAK_VERSION}
(OUT / "inventario.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
print(len(fe.SYMBOLS), "símbolos · huella", fe.fingerprint()[:16])
for text in ("¿El niño soñaba con una montaña?", "Raúl y Rocío reían en el río.", "Pero el perro no es caro."):
    print(text, "→", " ".join(fe.phonemize(text).symbols))
''')

code(r'''
#@title 4 · Clips aceptados del notebook 01 → entradas de `tiny_voca.prepare`
from collections import Counter
import soundfile as sf

SPLITS = ("train", "validation", "test")
PROVENANCE = ("text_id", "origin", "source_id", "source_license", "source_url", "teacher_model", "reference_audio")


def text_hash(text):
    return hashlib.sha256(text.strip().encode()).hexdigest()


last = {}
with (DATA / MANIFEST_NAME).open(encoding="utf-8") as stream:
    for line in stream:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue                                  # línea cortada por una desconexión
        last[record["id"]] = record
accepted = sorted((r for r in last.values() if not r["qc_reasons"] and r.get("voice_id") == VOICE_ID),
                  key=lambda r: r["id"])
print(len(accepted), "clips aceptados ·", round(sum(r["duration_seconds"] for r in accepted) / 3600, 2), "h")

hashes = {split: {text_hash(r["text"]) for r in accepted if r["split"] == split} for split in SPLITS}
for i, a in enumerate(SPLITS):
    for b in SPLITS[i + 1:]:
        if hashes[a] & hashes[b]:
            raise ValueError(f"la misma frase está en {a} y {b}")

for split in SPLITS:
    target = OUT / "prepare_inputs" / f"{split}.jsonl"
    rows = [r for r in accepted if r["split"] == split]
    if target.exists():
        # Congelado: prepare --resume exige exactamente el mismo archivo de entrada.
        frozen = {json.loads(line)["id"] for line in target.read_text(encoding="utf-8").splitlines() if line.strip()}
        extra = sum(r["id"] not in frozen for r in rows)
        print(f"{split}: {len(frozen)} filas congeladas" + (f" · {extra} clips nuevos del 01 NO incluidos" if extra else ""))
        continue
    for r in rows[:3]:
        if sf.info(str(DATA / r["audio_path"])).samplerate != 24000:
            raise ValueError("el audio del 01 debe estar a 24 kHz")
    lines = []
    for r in rows:
        row = {"id": r["id"], "split": split, "text": r["text"], "voice_id": r["voice_id"],
               "audio": str(DATA / r["audio_path"]), **{k: r[k] for k in PROVENANCE if r.get(k) is not None}}
        lines.append(json.dumps(row, ensure_ascii=False) + "\n")
    temporary = target.with_suffix(".tmp")
    temporary.write_text("".join(lines), encoding="utf-8")
    temporary.replace(target)
    print(f"{split}: {len(rows)} filas escritas")
INPUT_ROWS = {split: [json.loads(line) for line in (OUT / "prepare_inputs" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
                      if line.strip()] for split in SPLITS}
''')

code(r'''
#@title 5 · Fonemización es-419 (caché incremental `phonemes.jsonl`)
from concurrent.futures import ThreadPoolExecutor
import time

PHONEMES_PATH = ANNOTATIONS / "phonemes.jsonl"
phonemes = {}
if PHONEMES_PATH.exists():
    for line in PHONEMES_PATH.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        phonemes[record["text_hash"]] = fe.Phonemized(tuple(record["symbols"]),
                                                      tuple(fe.SYMBOL_TO_ID[s] for s in record["symbols"]),
                                                      tuple(record["stress"]), tuple(record["unknown"]))
todo = {}
for rows in INPUT_ROWS.values():
    for row in rows:
        key = text_hash(row["text"])
        if key not in phonemes:
            todo[key] = row["text"]
print(len(phonemes), "en caché ·", len(todo), "por fonemizar")
started = time.time()
with PHONEMES_PATH.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(WORKERS * 4) as pool:
    for done, (key, result) in enumerate(zip(todo, pool.map(fe.phonemize, todo.values())), 1):
        phonemes[key] = result
        stream.write(json.dumps({"text_hash": key, "text": todo[key], "symbols": list(result.symbols),
                                 "stress": list(result.stress), "unknown": list(result.unknown)}, ensure_ascii=False) + "\n")
        if done % 500 == 0:
            stream.flush(); os.fsync(stream.fileno())
            print(f"{done}/{len(todo)} · {time.time() - started:.0f} s")
counts = Counter(s for p in phonemes.values() for s in p.symbols if s in fe.PHONE_SET)
unknown = Counter(u for p in phonemes.values() for u in p.unknown)
print("fonemas usados:", len(counts), "de", len(fe.PHONES))
print("poco frecuentes:", [(p, counts[p]) for p in fe.PHONES if counts[p] < 30])
print("desconocidos (esas frases se rechazan):", dict(unknown))
''')

code(r'''
#@title 6 · Rasgos acústicos con `tiny_voca.prepare` del repo oficial (reanudable)
import time
from tiny_voca.config import Config

ACOUSTIC_CONFIG = Config(speakers=(VOICE_ID,))
config_path = OUT / "config_acustica.json"
config_path.write_text(json.dumps(ACOUSTIC_CONFIG.to_dict(), indent=2), encoding="utf-8")
print("huella acústica", ACOUSTIC_CONFIG.acoustic_fingerprint[:16])


def sync(source, target, required=True, attempts=4):
    """rsync con reintentos: la conexión de Colab con Drive falla a ratos. --update nunca pisa un archivo más nuevo."""
    for attempt in range(1, attempts + 1):
        result = subprocess.run(["rsync", "-a", "--update", f"{source}/", f"{target}/"], capture_output=True, text=True)
        if result.returncode == 0:
            return True
        print(f"  ⚠ copia a/desde Drive falló (intento {attempt}/{attempts}, código {result.returncode}): "
              f"{result.stderr.strip()[-300:]}")
        time.sleep(30 * attempt)
    if required:
        raise RuntimeError("No se pudo copiar con Drive. Comprueba que Drive tiene espacio libre y vuelve a ejecutar "
                           "la celda: sigue donde iba.")
    return False


def drop_missing(manifest, root):
    """Una copia a Drive cortada puede dejar líneas cuyo .pt no llegó: se rehacen."""
    if not manifest.exists():
        return 0
    lines = manifest.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = []
    for line in lines:
        try:
            if (root / json.loads(line)["path"]).is_file():
                kept.append(line)
        except (json.JSONDecodeError, KeyError):
            pass
    if len(kept) != len(lines):
        manifest.write_text("".join(kept), encoding="utf-8")
    return len(lines) - len(kept)


# Si una ejecución anterior de esta celda se cortó, su prepare puede seguir vivo en segundo plano: se para
# antes de lanzar otro (dos escribiendo en la misma carpeta la estropearían). Lo ya hecho se conserva.
if subprocess.run(["pkill", "-f", "tiny_voca.prepare"]).returncode == 0:
    print("Había un prepare anterior en marcha: parado; se reanuda desde lo ya preparado.")
    time.sleep(5)
FEATURES_LOCAL.mkdir(exist_ok=True)
sync(FEATURES_DRIVE, FEATURES_LOCAL)                   # reanudar desde Drive (sin pisar lo local más nuevo)
for split in SPLITS:
    metadata = FEATURES_LOCAL / f"{split}.preparation.json"
    if metadata.exists() and json.loads(metadata.read_text())["complete"]:
        print(split, "ya preparado:", json.loads(metadata.read_text())["accepted"], "clips")
        continue
    dropped = drop_missing(FEATURES_LOCAL / f"{split}.jsonl", FEATURES_LOCAL)
    if dropped:
        print(split, ":", dropped, "líneas sin archivo se rehacen")
    command = [sys.executable, "-m", "tiny_voca.prepare", "--inputs", str(OUT / "prepare_inputs" / f"{split}.jsonl"),
               "--output", str(FEATURES_LOCAL), "--split", split, "--config", str(config_path)]
    if metadata.exists():
        command.append("--resume")
    # prepare es de un solo proceso: un proceso por trozo no cabe en su formato, así que se lanza tal cual.
    print(split, "→", len(INPUT_ROWS[split]), "clips; pyin tarda ~0,25 s por segundo de audio")
    process = subprocess.Popen(command, env={**os.environ, "PYTHONPATH": str(REPO_SRC)})
    while process.poll() is None:
        try:
            process.wait(timeout=SYNC_EVERY_SECONDS)
        except subprocess.TimeoutExpired:
            sync(FEATURES_LOCAL, FEATURES_DRIVE, required=False, attempts=1)   # copia de seguridad: si falla, sigue
            done = sum(1 for _ in (FEATURES_LOCAL / f"{split}.jsonl").open()) if (FEATURES_LOCAL / f"{split}.jsonl").exists() else 0
            print(f"  {split}: {done}/{len(INPUT_ROWS[split])}")
    sync(FEATURES_LOCAL, FEATURES_DRIVE)
    if process.returncode:
        raise RuntimeError(f"prepare {split} falló; mira {FEATURES_LOCAL / (split + '.rejected.jsonl')}")
for split in SPLITS:
    meta = json.loads((FEATURES_LOCAL / f"{split}.preparation.json").read_text())
    print(split, {k: meta[k] for k in ("accepted", "rejected")})
''')

code(r'''
#@title 7 · Montreal Forced Aligner: corpus, diccionario y entrenamiento (se salta si ya está)
import csv
import librosa

MFA_DRIVE = OUT / "mfa"
MFA_DONE = MFA_DRIVE / "alineado.json"
TO_ALIAS, FROM_ALIAS = al.phone_aliases(fe.PHONES)


def feature_rows(split):
    return [json.loads(line) for line in (FEATURES_LOCAL / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


FEATURE_ROWS = {split: feature_rows(split) for split in SPLITS}
alignable = [row for split in SPLITS for row in FEATURE_ROWS[split] if not phonemes[row["text_hash"]].unknown]
audio_of = {row["id"]: row["audio"] for rows in INPUT_ROWS.values() for row in rows}

if MFA_DONE.exists():
    print("MFA ya hecho:", json.loads(MFA_DONE.read_text()))
else:
    corpus, work = Path("/content/mfa_corpus"), Path("/content/mfa_work")
    shutil.rmtree(corpus, ignore_errors=True); shutil.rmtree(work, ignore_errors=True)
    (corpus / VOICE_ID).mkdir(parents=True); work.mkdir()
    # Una «pseudopalabra» por secuencia de fonemas de eSpeak: el diccionario da exactamente
    # nuestra pronunciación y MFA solo decide tiempos y silencios entre palabras.
    words = {}
    for row in alignable:
        labels = [words.setdefault(w, f"w{len(words):06d}") for w in fe.words_of(phonemes[row["text_hash"]])]
        (corpus / VOICE_ID / f"{row['id']}.lab").write_text(" ".join(labels) + "\n", encoding="utf-8")

    def to_wav16(identifier):
        # MFA trabaja a 16 kHz; WAV PCM16 evita depender de sox/ffmpeg para leer FLAC.
        wave, rate = sf.read(audio_of[identifier], dtype="float32")
        sf.write(corpus / VOICE_ID / f"{identifier}.wav", librosa.resample(wave, orig_sr=rate, target_sr=16000), 16000,
                 subtype="PCM_16")

    with ThreadPoolExecutor(WORKERS * 2) as pool:
        list(pool.map(to_wav16, [row["id"] for row in alignable]))
    dictionary = work / "diccionario.txt"
    dictionary.write_text("".join(f"{label}\t{' '.join(TO_ALIAS[p] for p in phones)}\n" for phones, label in words.items()),
                          encoding="utf-8")
    print(len(alignable), "clips ·", len(words), "pseudopalabras")
    common = ["--clean", "--single_speaker", "-j", WORKERS, "--use_mp", "--beam", MFA_BEAM,
              "--retry_beam", MFA_RETRY_BEAM, "-t", work / "tmp"]
    started = time.time()
    mfa("train", corpus, dictionary, work / "modelo_acustico.zip", "--output_directory", work / "alineaciones",
        "--output_format", "json", *common)
    output = work / "alineaciones"
    aligned = list(output.rglob("*.json"))
    if not aligned or not any(output.rglob("alignment_analysis.csv")):
        # Algunas versiones no escriben las alineaciones ni alignment_analysis.csv (la verosimilitud que usa
        # el filtro low_confidence de la celda 8) al entrenar: se alinea otra vez con el modelo entrenado.
        print("entrenado en", round((time.time() - started) / 60, 1), "min; alineando con el modelo entrenado…")
        output = work / "alineaciones_align"
        mfa("align", corpus, dictionary, work / "modelo_acustico.zip", output, "--output_format", "json", *common)
        aligned = list(output.rglob("*.json"))
    print(len(aligned), "alineaciones en", round((time.time() - started) / 60, 1), "min ·",
          "con" if any(output.rglob("alignment_analysis.csv")) else "SIN", "alignment_analysis.csv")
    target = MFA_DRIVE / "alineaciones"
    shutil.rmtree(target, ignore_errors=True); target.mkdir()
    for path in aligned:
        shutil.copyfile(path, target / path.name)
    for path in [work / "modelo_acustico.zip", dictionary, *output.rglob("alignment_analysis.csv")]:
        if path.exists():
            shutil.copyfile(path, MFA_DRIVE / path.name)
    MFA_DONE.write_text(json.dumps({"mfa": MFA_VERSION, "clips": len(alignable), "aligned": len(aligned),
                                    "pseudowords": len(words), "beam": MFA_BEAM, "retry_beam": MFA_RETRY_BEAM,
                                    "aliases": TO_ALIAS}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("guardado en", MFA_DRIVE)
''')

code(r'''
#@title 8 · Duraciones, control de calidad y manifiestos en el formato de `tiny_voca.annotate`
import numpy as np
import torch

ALIGN_FILES = {p.stem: p for p in (MFA_DRIVE / "alineaciones").iterdir()}
likelihood = {}
analysis = MFA_DRIVE / "alignment_analysis.csv"
if analysis.exists():
    with analysis.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                likelihood[Path(row["file"]).stem] = float(row["speech_log_likelihood"])
            except (KeyError, ValueError):
                pass
cutoff = (float(np.percentile(list(likelihood.values()), 100 * LIMITS["worst_likelihood_fraction"]))
          if likelihood and LIMITS["worst_likelihood_fraction"] else None)
c = ACOUSTIC_CONFIG
identity = {
    "annotation_version": 3, "acoustic": c.acoustic_fingerprint, "frontend": fe.fingerprint(),
    "frontend_runtime": {"espeak": ESPEAK_VERSION, "voice": fe.VOICE},
    "aligner": {"model": "montreal-forced-aligner", "revision": MFA_VERSION, "beam": MFA_BEAM, "retry_beam": MFA_RETRY_BEAM,
                "training": "mfa train sobre este corpus, una voz"},
    "alignment": {"min_pause_frames": MIN_PAUSE_FRAMES}, "dataset_revision": DATASET_REVISION, "limit": None,
    "sources": {split: hashlib.sha256((FEATURES_LOCAL / f"{split}.jsonl").read_bytes()).hexdigest() for split in SPLITS},
    "tiny_voca_commit": REPO_COMMIT,
}
fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
metadata_path = ANNOTATIONS / "annotations.json"
metadata_path.write_text(json.dumps({"identity": identity, "fingerprint": fingerprint, "complete": False}, indent=2))


def end_db(wave, window_seconds=0.025):
    """Energía de los últimos 25 ms respecto a la voz normal del clip (percentil 95), en dB.
    Un final natural se apaga (< -25 dB); un corte a mitad de vocal queda cerca de 0 dB."""
    wave = wave.float().numpy().ravel()
    window = int(window_seconds * c.sample_rate)
    count = len(wave) // window
    rms = np.sqrt((wave[len(wave) - count * window:].reshape(count, window) ** 2).mean(1)) + 1e-8
    return round(float(20 * np.log10(rms[-1] / np.percentile(rms, 95))), 2)


def qc_of(row, symbols, durations, voiced, extra):
    seconds = row["frames"] * c.hop / c.sample_rate
    phone_frames = [d for s, d in zip(symbols, durations) if s in fe.PHONE_SET]
    vowel = np.repeat([s in fe.VOWEL_SET for s in symbols], durations)
    last = next(((s, d) for s, d in zip(reversed(symbols), reversed(durations)) if s in fe.PHONE_SET), (None, None))
    return {"final_vowel_frames": last[1] if last[0] in fe.VOWEL_SET else None, "phones_per_second": round(len(phone_frames) / seconds, 3), "max_phone_frames": max(phone_frames),
            "one_frame_phone_fraction": round(float(np.mean(np.asarray(phone_frames) == 1)), 4),
            "vowel_voiced": round(float(voiced[vowel].mean()), 4) if vowel.any() else None,
            "mfa_speech_log_likelihood": likelihood.get(row["id"]), **extra}


def gate(entry):
    qc, reasons = entry["qc"], []
    if entry["unknown"]:
        reasons.append("unknown_phoneme")
    if qc["max_phone_frames"] > LIMITS["max_phone_frames"]:
        reasons.append("overlong_phone")
    if qc["vowel_voiced"] is not None and qc["vowel_voiced"] < LIMITS["min_vowel_voiced"]:
        reasons.append("vowels_not_voiced")
    if not LIMITS["min_phones_per_second"] <= qc["phones_per_second"] <= LIMITS["max_phones_per_second"]:
        reasons.append("implausible_rate")
    if cutoff is not None and qc["mfa_speech_log_likelihood"] is not None and qc["mfa_speech_log_likelihood"] < cutoff:
        reasons.append("low_confidence")
    if LIMITS.get("max_end_db") is not None and qc["end_db"] > LIMITS["max_end_db"]:
        reasons.append("cut_ending")
    if (LIMITS.get("min_final_vowel_frames") is not None and qc.get("final_vowel_frames") is not None
            and qc["final_vowel_frames"] < LIMITS["min_final_vowel_frames"]):
        reasons.append("short_final_vowel")
    edge = LIMITS.get("max_edge_silence_frames")
    if edge is not None and qc.get("leading_frames", 0) > edge:
        reasons.append("long_leading_silence")
    if edge is not None and qc.get("trailing_frames", 0) > edge:
        reasons.append("long_trailing_silence")
    return reasons


summary = {"limits": LIMITS, "stage": "mfa", "splits": {}}
phone_frames, pause_tokens, pause_nonzero = {}, Counter(), Counter()
for split in SPLITS:
    accepted_rows, rejected_rows = [], []
    for row in FEATURE_ROWS[split]:
        result = phonemes[row["text_hash"]]
        base = {"id": row["id"], "speaker": row["speaker"], "text": row["text"]}
        try:
            if result.unknown:
                raise ValueError("unknown_phoneme")
            if row["id"] not in ALIGN_FILES:
                raise ValueError("mfa_sin_alineacion")
            intervals = al.read_phone_intervals(ALIGN_FILES[row["id"]], FROM_ALIAS)
            durations, extra = al.durations_from_intervals(result.symbols, intervals, row["frames"], c.hop, c.sample_rate,
                                                           MIN_PAUSE_FRAMES, fe.PHONE_SET, fe.PUNCTUATION, fe.SP)
            stored = torch.load(FEATURES_LOCAL / row["path"], map_location="cpu", weights_only=True)
            qc = qc_of(row, result.symbols, durations, stored["prosody"][:, 1].numpy() > 0.5,
                       {**extra, "end_db": end_db(stored["wave"])})
        except ValueError as error:
            rejected_rows.append({**base, "reasons": [str(error)]})
            continue
        entry = {"id": row["id"], "split": split, "speaker": row["speaker"], "text": row["text"], "text_hash": row["text_hash"],
                 "path": row["path"], "frames": row["frames"], "symbols": " ".join(result.symbols), "tokens": list(result.ids),
                 "stress": list(result.stress), "durations": durations, "unknown": list(result.unknown), "qc": qc,
                 "provenance": {**row.get("provenance", {}), "dataset_revision": DATASET_REVISION},
                 "acoustic": c.acoustic_fingerprint, "frontend": fe.fingerprint(), "annotation": fingerprint}
        if sum(durations) != row["frames"] or len(durations) != len(result.ids):
            raise RuntimeError(f"las duraciones de {row['id']} no cuadran")
        reasons = gate(entry)
        if reasons:
            rejected_rows.append({**base, "reasons": reasons, "qc": qc})
            continue
        accepted_rows.append(entry)
        for symbol, frames in zip(result.symbols, durations):
            if symbol in fe.PHONE_SET:
                phone_frames.setdefault(symbol, []).append(frames)
            else:
                pause_tokens[symbol] += 1
                pause_nonzero[symbol] += frames > 0
    for name, payload in ((f"{split}.jsonl", accepted_rows), (f"{split}.rejected.jsonl", rejected_rows)):
        temporary = ANNOTATIONS / (name + ".tmp")
        temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in payload), encoding="utf-8")
        temporary.replace(ANNOTATIONS / name)
    summary["splits"][split] = {
        "accepted": len(accepted_rows), "rejected": len(rejected_rows), "accepted_by_voice": {VOICE_ID: len(accepted_rows)},
        "accepted_hours": round(sum(r["frames"] for r in accepted_rows) * c.hop / c.sample_rate / 3600, 3),
        "rejection_reasons": dict(Counter(reason for r in rejected_rows for reason in r["reasons"])),
    }
summary["phone_duration_frames"] = {VOICE_ID: {
    symbol: {"count": len(v), "mean": round(float(np.mean(v)), 2), "median": float(np.median(v)),
             "one_frame": round(float(np.mean(np.asarray(v) == 1)), 3)}
    for symbol, v in sorted(phone_frames.items(), key=lambda kv: -len(kv[1]))}}
summary["pause_tokens"] = {s: {"count": pause_tokens[s], "with_frames": round(pause_nonzero[s] / pause_tokens[s], 4)} for s in pause_tokens}
(ANNOTATIONS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
counts = {split: summary["splits"][split]["accepted"] for split in SPLITS}
metadata_path.write_text(json.dumps({"identity": identity, "fingerprint": fingerprint, "complete": all(counts.values()),
                                     "stage": "mfa", "acoustic_root": str(FEATURES_DRIVE), "limits": LIMITS,
                                     "counts": counts}, indent=2))
print(json.dumps(summary["splits"], ensure_ascii=False, indent=2))
''')

code(r'''
#@title 9 · Revisión: duraciones por fonema y un clip con sus fronteras
import random
import matplotlib.pyplot as plt
from IPython.display import Audio, Markdown, display

table = summary["phone_duration_frames"][VOICE_ID]
print("fonema  n      media  mediana  1 cuadro")
for symbol, stats in table.items():
    print(f"{symbol:6} {stats['count']:6d} {stats['mean']:7.2f} {stats['median']:7.1f} {stats['one_frame']:8.3f}")
print("pausas:", summary["pause_tokens"])

rows = [json.loads(line) for line in (ANNOTATIONS / "validation.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
row = random.choice(rows)
stored = torch.load(FEATURES_LOCAL / row["path"], map_location="cpu", weights_only=True)
symbols, durations = row["symbols"].split(), row["durations"]
fig, ax = plt.subplots(figsize=(min(24, 4 + row["frames"] / 40), 4))
ax.imshow(stored["mel"].T.numpy(), origin="lower", aspect="auto", cmap="magma")
edge = 0
for symbol, frames in zip(symbols, durations):
    if frames:
        ax.axvline(edge - 0.5, color="white", lw=0.6)
        ax.text(edge + frames / 2 - 0.5, 82, symbol, ha="center", va="bottom", fontsize=9)
    edge += frames
ax.set_ylim(0, 92); ax.set_xlabel("cuadros (256 muestras)"); ax.set_title(row["text"])
plt.show()
display(Markdown(f"`{row['id']}` · {row['frames']} cuadros · qc {row['qc']}"))
display(Audio(stored["wave"].numpy(), rate=c.sample_rate))
''')

md(r'''
**Siguiente:** `03_entrenar_tiny_voca.ipynb` leerá

* `02_alineacion_v2/anotaciones/{train,validation,test}.jsonl` + `annotations.json`
  (`acoustic_root` = `02_alineacion_v2/features`; cópialos a disco local para entrenar rápido),
* `02_alineacion_v2/codigo/frontend_es.py` (inventario de 56 símbolos y huella `frontend`) y
  `inventario.json`,
* `02_alineacion_v2/config_acustica.json` (misma huella acústica).

`tiny_voca.data.AnnotatedDataset` compara la huella `frontend` con la de `tiny_voca.text` (inglés): el 03
debe sustituir ese módulo por `frontend_es` antes de cargar los datos y usar `vocab_size = 56`.
''')

notebook = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
                 "language_info": {"name": "python"}},
    "cells": [],
}
for cell in cells:
    source = cell["source"]
    if cell["cell_type"] == "code":
        # Las líneas «!» de Colab valen dentro de bloques: para comprobar la sintaxis se cambian por «pass».
        ast.parse("\n".join(line[:len(line) - len(line.lstrip())] + "pass" if line.lstrip().startswith(("!", "%"))
                            else line for line in source.splitlines()))
    cell["source"] = source.splitlines(keepends=True)
    notebook["cells"].append(cell)
target = HERE / "02_fonemas_y_alineacion.ipynb"
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(target, len(notebook["cells"]), "celdas")
