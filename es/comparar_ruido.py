"""Genera las mismas frases con pulsos solos (A, como salió del entrenamiento) y con ruido_fricativas (B).

    py comparar_ruido.py --repo <carpeta src del tiny-voca adaptado> --modelo inference.pt --salida comparacion

<repo> es la carpeta que contiene ``tiny_voca/`` (con text.py = text_es.py y vocab_size 56) y ``frontend_es.py``,
como la prepara la celda 2 del notebook 03. Funciona en CPU.
"""
import argparse, sys
from pathlib import Path

FRASES = [
    "Hola, soy una voz sintética entrenada desde cero en español.",
    "Las sillas y las mesas del salón son de madera fina.",
    "Fernanda fue a la farmacia a buscar unas fresas frescas.",
    "Si sales sin suéter, seguro sientes frío esta semana.",
    "¿Sabes si el sábado hace sol o si se suspende la fiesta?",
    "Después de la función, fuimos a cenar unos tacos sabrosos.",
]

parser = argparse.ArgumentParser()
parser.add_argument("--repo", required=True, type=Path)
parser.add_argument("--modelo", required=True, type=Path)
parser.add_argument("--salida", default=Path("comparacion"), type=Path)
parser.add_argument("--limite", default="2,33", help="armónico límite en tramos sordos,sonoros")
args = parser.parse_args()
sys.path[:0] = [str(args.repo.resolve()), str(Path(__file__).resolve().parent)]

import soundfile as sf
from tiny_voca.infer import load_model, speak_text
import ruido_fricativas

model = load_model(args.modelo, "cpu")
voice = model.config.speakers[0]
args.salida.mkdir(parents=True, exist_ok=True)
limite = tuple(float(x) for x in args.limite.split(","))
for i, texto in enumerate(FRASES, 1):
    for nombre, ajuste in (("A_original", None), ("B_ruido", limite)):
        ruido_fricativas.activar(ajuste)
        wave, _ = speak_text(model, texto, voice, seed=i)
        sf.write(str(args.salida / f"{i:02d}_{nombre}.wav"), wave.cpu().numpy(), model.config.sample_rate)
    print(f"{i:02d}  {texto}")
print("listo en", args.salida.resolve())
