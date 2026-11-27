"""Genera 03_entrenar.ipynb (Colab). Ejecutar desde cualquier carpeta:
    python notebooks/es-desde-cero/build_03_entrenar.py

Incrusta text_es.py (misma carpeta): sustituye a tiny_voca/text.py en la copia del repo oficial que clona
el notebook, para que el entrenamiento y la inferencia usen el front end español de frontend_es.py.
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
# 03 · Entrenar Tiny Voca en español desde cero (voz F2)

Tercer y último paso. Entrena el modelo con `python -m tiny_voca.train` **del repo oficial, sin cambiar su
código de entrenamiento**, sobre las anotaciones del notebook 02 (`02_alineacion_v2/`). Lo único que se
cambia es `tiny_voca/text.py` (el front end inglés) por un adaptador al front end español
`frontend_es.py` que escribió el 02: mismo inventario de 56 símbolos y misma huella con la que se anotó.

**Necesita GPU** (*Entorno de ejecución → Cambiar tipo de entorno*):

| GPU | Velocidad aproximada | Comentario |
| --- | --- | --- |
| **A100** | la más rápida (~8–10 h para 100 000 pasos) | gasta más unidades por hora |
| **L4** | ~2× más lenta que la A100 | buen equilibrio |
| T4 | ~4× más lenta | barata pero muy lenta |

La celda 3 mide la velocidad real a los pocos minutos y calcula cuánto falta.

**Reanudable:** todo se guarda en Drive (`03_entrenamiento/<RUN_NAME>/`). Si Colab se corta o paras la celda 3,
vuelve a *Ejecutar todo* y sigue en el paso donde iba. **Puedes escuchar el modelo cuando quieras**: para la
celda 3 (botón ■; guarda antes de salir) y ejecuta la 4. Para continuar, ejecuta otra vez la 3.

**Qué esperar:** hasta el paso ~8 000 suena a robot/ruido (aprende fonemas y duraciones). Desde ahí entran
los discriminadores y la voz se vuelve natural poco a poco. A partir de ~40 000 pasos ya debería sonar bien.
''')

code(r'''
#@title 1 · Configuración
ALIGN_DIR = "/content/drive/MyDrive/tiny-voca-es-desde-cero/02_alineacion_v2"   # salida del notebook 02
RUNS_DIR = "/content/drive/MyDrive/tiny-voca-es-desde-cero/03_entrenamiento"
RUN_NAME = "f2-v2"          # cambia el nombre para empezar otro entrenamiento desde cero (el anterior se queda)

STEPS = 100000              # pasos totales; se puede subir después y seguir (el repo oficial usó 104 000)
SYNC_MINUTES = 10           # copia a Drive cada tanto (además de al terminar o parar)

# Receta del repo oficial (README). Si cambias algo de esto, el entrenamiento ya empezado no se puede
# reanudar: usa otro RUN_NAME.
RECIPE = {"--gan-start": 8000, "--gan-ratio": 0.7, "--teacher-final": 0, "--batch-frames": 9000,
          "--seed": 20261113}
SAVE_EVERY = 1000           # guarda inference.pt (el modelo para escuchar) cada estos pasos
KEEP_EVERY = 20000          # además conserva una copia completa numerada cada estos pasos
RUIDO_FRICATIVAS = True     # celda 4: «s», «f» y respiraciones con ruido en vez de pulsos (menos estática; solo inferencia)

LISTEN_SENTENCES = [        # frases de la celda 4 (números con letra)
    "Hola, soy una voz sintética entrenada desde cero en español.",
    "¿Te gustaría dar un paseo por el parque esta tarde?",
    "La ciudad de México amaneció con lluvia y mucho tráfico.",
    "¡Qué alegría verte de nuevo después de tanto tiempo!",
]
''')

code(r'''
#@title 2 · Drive, GPU, repo oficial con el front end español y datos en disco local
import json, os, shutil, subprocess, sys, time
from pathlib import Path
from google.colab import drive

drive.mount("/content/drive")
import torch
if not torch.cuda.is_available():
    raise RuntimeError("No hay GPU: Entorno de ejecución → Cambiar tipo de entorno → GPU (A100, L4 o T4)")
print("GPU:", torch.cuda.get_device_name(0), "·", round(torch.cuda.get_device_properties(0).total_memory / 2**30), "GB")

ALIGN = Path(ALIGN_DIR)
ANNOTATIONS = ALIGN / "anotaciones"
metadata = json.loads((ANNOTATIONS / "annotations.json").read_text())
if not metadata.get("complete"):
    raise RuntimeError("Las anotaciones del 02 no están completas: termina antes el notebook 02 (celda 8)")
RUN = Path(RUNS_DIR) / RUN_NAME
RUN.mkdir(parents=True, exist_ok=True)
RUN_LOCAL = Path("/content/run")
DATA_LOCAL = Path("/content/datos")

# Repo oficial en el mismo commit con el que el 02 preparó los rasgos acústicos.
COMMIT = metadata["identity"]["tiny_voca_commit"]
REPO = Path("/content/tiny-voca")
if not REPO.exists():
    subprocess.run(["git", "clone", "-q", "https://github.com/n4ze3m/tiny-voca.git", str(REPO)], check=True)
subprocess.run(["git", "-C", str(REPO), "checkout", "-q", COMMIT], check=True)
REPO_SRC = next(p.parent.parent for p in REPO.rglob("tiny_voca/train.py"))
subprocess.run(["apt-get", "-qq", "install", "-y", "espeak-ng"], check=True, stdout=subprocess.DEVNULL)

# Front end español: frontend_es.py tal cual lo dejó el 02 en Drive + adaptador en lugar de tiny_voca/text.py.
TEXT_ES = r''' + "'''" + embedded("text_es.py") + "'''" + r'''
RUIDO_FRICATIVAS_PY = r''' + "'''" + embedded("ruido_fricativas.py") + "'''" + r'''
(REPO_SRC / "ruido_fricativas.py").write_text(RUIDO_FRICATIVAS_PY, encoding="utf-8")
shutil.copyfile(ALIGN / "codigo" / "frontend_es.py", REPO_SRC / "frontend_es.py")
(REPO_SRC / "tiny_voca" / "text.py").write_text(TEXT_ES, encoding="utf-8")
# model.py crea un Config() por defecto al importarse: su vocab_size por defecto (76, el inglés) debe ser el nuestro.
config_py = REPO_SRC / "tiny_voca" / "config.py"
source = config_py.read_text(encoding="utf-8")
if "vocab_size: int = 76" in source:
    config_py.write_text(source.replace("vocab_size: int = 76", "vocab_size: int = 56  # español (notebook 03)"),
                         encoding="utf-8")
elif "vocab_size: int = 56" not in source:
    raise RuntimeError("No encuentro vocab_size en tiny_voca/config.py: el repo oficial ha cambiado")
(RUN / "codigo").mkdir(exist_ok=True)
for source in (REPO_SRC / "frontend_es.py", REPO_SRC / "tiny_voca" / "text.py", REPO_SRC / "ruido_fricativas.py"):
    shutil.copyfile(source, RUN / "codigo" / ("text_es.py" if source.name == "text.py" else source.name))
sys.path.insert(0, str(REPO_SRC))
for name in [m for m in sys.modules if m in ("frontend_es", "ruido_fricativas") or m.startswith("tiny_voca")]:
    del sys.modules[name]
import frontend_es as fe
from tiny_voca.config import Config
from tiny_voca import text as tv_text

first = json.loads(next(line for line in (ANNOTATIONS / "validation.jsonl").open(encoding="utf-8") if line.strip()))
if first["frontend"] != tv_text.frontend_fingerprint():
    raise RuntimeError("El front end no coincide con el de las anotaciones del 02")
CONFIG = Config(speakers=(first["speaker"],), vocab_size=len(fe.SYMBOLS))
if first["acoustic"] != CONFIG.acoustic_fingerprint:
    raise RuntimeError("Los rasgos acústicos del 02 no coinciden con esta versión del repo")
CONFIG_PATH = RUN / "config_modelo.json"
if CONFIG_PATH.exists():
    if Config.from_dict(json.loads(CONFIG_PATH.read_text())) != CONFIG:   # reanudar: misma configuración
        raise RuntimeError(f"{CONFIG_PATH} es de otra configuración: usa otro RUN_NAME")
else:
    CONFIG_PATH.write_text(json.dumps(CONFIG.to_dict(), indent=2))
print(f"tiny-voca {COMMIT[:10]} · {len(fe.SYMBOLS)} símbolos · voz {CONFIG.speakers[0]} · "
      f"huella {tv_text.frontend_fingerprint()[:12]}")

# Anotaciones y rasgos a disco local (leer de Drive durante el entrenamiento sería muy lento).
# Se copian solo los .pt de los clips aceptados; el manifiesto se copia byte a byte (su huella cuenta).
(DATA_LOCAL / "anotaciones").mkdir(parents=True, exist_ok=True)
paths = set()
for split in ("train", "validation", "test"):
    shutil.copyfile(ANNOTATIONS / f"{split}.jsonl", DATA_LOCAL / "anotaciones" / f"{split}.jsonl")
    for line in (ANNOTATIONS / f"{split}.jsonl").open(encoding="utf-8"):
        if line.strip():
            paths.add(json.loads(line)["path"])
local_meta = {**metadata, "acoustic_root": str(DATA_LOCAL / "features")}
(DATA_LOCAL / "anotaciones" / "annotations.json").write_text(json.dumps(local_meta, indent=2))
(DATA_LOCAL / "features").mkdir(exist_ok=True)
(DATA_LOCAL / "lista.txt").write_text("\n".join(sorted(paths)) + "\n")
started = time.time()
print(f"copiando {len(paths)} clips de Drive a disco local…")
subprocess.run(["rsync", "-a", "--files-from", str(DATA_LOCAL / "lista.txt"), f"{metadata['acoustic_root']}/",
                f"{DATA_LOCAL / 'features'}/"], check=True)
missing = [p for p in paths if not (DATA_LOCAL / "features" / p).is_file()]
if missing:
    raise RuntimeError(f"faltan {len(missing)} archivos de rasgos, p. ej. {missing[:3]}")
print(f"listo en {round(time.time() - started)} s")
''')

code(r'''
#@title 3 · Entrenar (reanudable; para con ■ y vuelve a ejecutar para seguir)
import threading

STOP_FILE = RUN_LOCAL / "PARAR"


def sync(source, target):
    result = subprocess.run(["rsync", "-a", "--update", "--exclude", "*.tmp", "--exclude", "PARAR", f"{source}/", f"{target}/"],
                            capture_output=True, text=True)
    if result.returncode:
        print("  ⚠ no se pudo copiar a Drive (se reintenta luego):", result.stderr.strip()[-200:])
    return result.returncode == 0


# Retomar: lo de Drive a disco local (sin pisar lo local más nuevo).
RUN_LOCAL.mkdir(exist_ok=True)
STOP_FILE.unlink(missing_ok=True)
sync(RUN, RUN_LOCAL)
resume = RUN_LOCAL / "last.pt"
if not resume.exists():
    # Empezar de cero: el entrenador exige una carpeta de salida vacía (lo de Drive se queda como está).
    shutil.rmtree(RUN_LOCAL)
    RUN_LOCAL.mkdir()
done_steps = torch.load(resume, map_location="cpu", weights_only=True)["step"] if resume.exists() else 0
if done_steps >= STEPS:
    print(f"Ya está entrenado hasta el paso {done_steps} (STEPS = {STEPS}). Sube STEPS para seguir, o escucha en la celda 4.")
else:
    command = [sys.executable, "-u", "-m", "tiny_voca.train",
               "--train", str(DATA_LOCAL / "anotaciones" / "train.jsonl"),
               "--validation", str(DATA_LOCAL / "anotaciones" / "validation.jsonl"),
               "--output", str(RUN_LOCAL), "--config", str(CONFIG_PATH), "--steps", str(STEPS),
               "--save-every", str(SAVE_EVERY), "--keep-every", str(KEEP_EVERY), "--log-every", "50",
               "--max-seconds", str(30 * 24 * 3600), "--stop-file", str(STOP_FILE),
               *[str(x) for item in RECIPE.items() for x in item]]
    if resume.exists():
        command += ["--resume", str(resume)]
        print(f"Reanudando desde el paso {done_steps}.")
    else:
        print("Empezando desde cero.")
    environment = {**os.environ, "PYTHONPATH": str(REPO_SRC)}
    process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               bufsize=1, start_new_session=True)   # ■ llega al notebook, no al entrenamiento
    last_sync, last_print, first = time.time(), 0.0, None
    try:
        for line in process.stdout:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(line.rstrip())
                continue
            if "inference_parameters" in record:
                print(f"parámetros del modelo: {record['inference_parameters']:,} · clips de train: {record['train_utterances']}")
                continue
            step = record["step"]
            first = first or (step, time.time())
            validation = "validation_predicted_mel" in record
            if validation or time.time() - last_print > 120:
                rate = (step - first[0]) / max(time.time() - first[1], 1e-9)
                left = f"~{(STEPS - step) / rate / 3600:.1f} h" if rate > 0 else "(calculando)"
                text = (f"paso {step:>6}/{STEPS} · pérdida {record['loss']:.3f} · mel {record.get('mel', float('nan')):.3f} · "
                        f"{record['steps_per_second']:.2f} pasos/s · quedan {left}")
                if validation:
                    text += (f"\n   validación: mel {record['validation_predicted_mel']:.3f} · duración "
                             f"{record['validation_duration']:.3f} · tono {record['validation_pitch_error_semitones']:.2f} st"
                             f" → guardado inference.pt")
                print(text, flush=True)
                last_print = time.time()
            if validation and time.time() - last_sync > SYNC_MINUTES * 60:
                threading.Thread(target=sync, args=(RUN_LOCAL, RUN), daemon=True).start()
                last_sync = time.time()
        process.wait()
    except KeyboardInterrupt:
        print("Parando: el entrenamiento guarda en el siguiente paso y sale…")
        STOP_FILE.touch()
        process.wait()
    finally:
        STOP_FILE.unlink(missing_ok=True)
        print("Copiando a Drive…")
        while not sync(RUN_LOCAL, RUN):
            time.sleep(30)
    if process.returncode not in (0, None):
        raise RuntimeError(f"El entrenamiento falló (código {process.returncode}); mira el error de arriba. "
                           "Si dice «CUDA out of memory», usa otro RUN_NAME con '--batch-frames' más bajo (p. ej. 6000).")
    step = torch.load(resume, map_location="cpu", weights_only=True)["step"] if resume.exists() else 0
    print(f"Guardado en Drive hasta el paso {step}: {RUN}")
''')

code(r'''
#@title 4 · Escuchar el modelo (último inference.pt guardado)
import random
import soundfile as sf
from IPython.display import Audio, Markdown, display
from tiny_voca.infer import load_model, speak_text
import ruido_fricativas
ruido_fricativas.activar() if RUIDO_FRICATIVAS else ruido_fricativas.activar(None)

checkpoint = RUN_LOCAL / "inference.pt" if (RUN_LOCAL / "inference.pt").exists() else RUN / "inference.pt"
if not checkpoint.exists():
    raise FileNotFoundError("Todavía no hay inference.pt: entrena al menos SAVE_EVERY pasos en la celda 3")
model = load_model(checkpoint, "cuda")
step = torch.load(checkpoint, map_location="cpu", weights_only=True)["step"]
samples = RUN / "muestras" / f"paso-{step:06d}"
samples.mkdir(parents=True, exist_ok=True)
test_rows = [json.loads(line) for line in (DATA_LOCAL / "anotaciones" / "test.jsonl").open(encoding="utf-8") if line.strip()]
chosen = random.Random(step).sample(test_rows, min(3, len(test_rows)))
display(Markdown(f"### Paso {step}"))
for number, sentence in enumerate(LISTEN_SENTENCES + [row["text"] for row in chosen], 1):
    with torch.no_grad():
        wave, _ = speak_text(model, sentence, CONFIG.speakers[0])
    sf.write(str(samples / f"{number:02d}.wav"), wave.numpy(), CONFIG.sample_rate)
    display(Markdown(f"**{sentence}**" + (" *(frase de test: no la vio al entrenar)*" if number > len(LISTEN_SENTENCES) else "")))
    display(Audio(wave.numpy(), rate=CONFIG.sample_rate))
print("guardado en", samples)
''')

code(r'''
#@title 5 · Apagar el entorno al terminar (deja de gastar unidades)
# Con «Ejecutar todo» corre después de entrenar y escuchar. Si paraste la celda 3 con ■ no se ejecuta sola.
from google.colab import drive, runtime
drive.flush_and_unmount()   # asegura que todo lo escrito llegó a Drive
runtime.unassign()          # libera la GPU: Colab se desconecta y deja de cobrar
''')

md(r'''
## Qué queda en Drive

`03_entrenamiento/<RUN_NAME>/`:

* `inference.pt`: **el modelo** (≈ 18 MB). Es lo que necesitas para usarlo.
* `last.pt` y `step-XXXXXXXX.pt`: estado completo para reanudar (no hacen falta para usarlo).
* `metrics.jsonl`, `config.json`, `recipe.json`, `config_modelo.json`: registro del entrenamiento.
* `codigo/frontend_es.py` y `codigo/text_es.py`: el front end español. Para usar el modelo fuera de este
  notebook, clona el repo oficial, copia `frontend_es.py` junto a la carpeta `tiny_voca/` (en `src/`) y
  `text_es.py` encima de `tiny_voca/text.py`. Después funciona igual que en inglés:
  `python -m tiny_voca.infer --checkpoint inference.pt --voice F2 --text "Hola." --output hola.wav`
  (escribe los números con letra).
* `codigo/ruido_fricativas.py`: ajuste de inferencia que quita la estática de las «s», «f» y respiraciones
  (portado de stdevMac/tiny-voca, Apache 2.0). Ponlo también en `src/` y, en Python, antes de generar:
  `import ruido_fricativas; ruido_fricativas.activar()`. El comando `python -m tiny_voca.infer` no lo usa.
* `muestras/paso-XXXXXX/`: lo que generó la celda 4 en cada momento.
''')

# Comprobación de sintaxis (las líneas mágicas de IPython no son Python).
for cell in cells:
    if cell["cell_type"] == "code":
        ast.parse("\n".join(line[:len(line) - len(line.lstrip())] + "pass" if line.lstrip().startswith(("!", "%"))
                            else line for line in cell["source"].splitlines()))

notebook = {"cells": [{**c, "source": c["source"].splitlines(keepends=True)} for c in cells],
            "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}
out = HERE / "03_entrenar.ipynb"
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(out, len(cells), "celdas")
