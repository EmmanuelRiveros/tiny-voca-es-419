# tiny-voca es-419: Tiny Voca en español latinoamericano

Modelo de texto a voz **pequeño y rápido en español de Latinoamérica**, con una voz femenina (`F2`).
Es [tiny-voca](https://github.com/n4ze3m/tiny-voca) con la **misma arquitectura y el mismo tamaño**
(≈18 MB, funciona en CPU), entrenado **desde cero** con un dataset sintético en español.

- **Modelo:** [`inference.pt` en el release `es-v1`](https://github.com/EmmanuelRiveros/tiny-voca-es-419/releases/tag/es-v1)
- **Idioma:** español latinoamericano (es-419): seseo («cielo» → «sielo») y yeísmo («ll» = «y»)
- **Voz:** una, `F2`, 24 kHz mono

## Qué cambia respecto a tiny-voca

Solo el texto de entrada. El resto del repo oficial se usa sin tocar:

| | tiny-voca (inglés) | es-419 |
|---|---|---|
| Fonemas | front end inglés | eSpeak NG, voz `es-419` (`frontend_es.py`) |
| Símbolos | 76 | 56 |
| Arquitectura | — | la misma, sin cambios |
| Inferencia | pulsos | pulsos + ruido en «s», «f» y respiraciones (`ruido_fricativas.py`, opcional) |

## Usarlo

Necesitas Python, PyTorch y **eSpeak NG** instalado (en Linux `sudo apt install espeak-ng`; en Windows o Mac,
el instalador de https://github.com/espeak-ng/espeak-ng/releases).

```bash
git clone https://github.com/EmmanuelRiveros/tiny-voca-es-419.git
cd tiny-voca-es-419
# front end español en lugar del inglés
cp es/frontend_es.py es/ruido_fricativas.py src/
cp es/text_es.py src/tiny_voca/text.py
python -c "import pathlib; p = pathlib.Path('src/tiny_voca/config.py'); p.write_text(p.read_text().replace('vocab_size: int = 76', 'vocab_size: int = 56'))"
pip install -e .
```

Descarga `inference.pt` del [release](https://github.com/EmmanuelRiveros/tiny-voca-es-419/releases/tag/es-v1) y:

```python
import sys; sys.path.insert(0, "src")
import soundfile as sf
import ruido_fricativas
from tiny_voca.infer import load_model, speak_text

ruido_fricativas.activar()            # recomendado: quita la estática de las «s» y «f»
model = load_model("inference.pt", "cpu")
wave, _ = speak_text(model, "Hola, ¿cómo estás? Hoy hace un día precioso.", "F2")
sf.write("hola.wav", wave.numpy(), model.config.sample_rate)
```

También funciona el comando del repo oficial, aunque **sin** el ajuste de las «s» y «f»:

```bash
python -m tiny_voca.infer --checkpoint inference.pt --voice F2 --text "Hola." --output hola.wav
```

**Escribe los números con letra** («veinte», no «20»): el front end no los convierte.

## Cómo se hizo

Tres notebooks de Google Colab en esta carpeta (los `build_0*.py` son los scripts que los generan):

1. **`01_generar_dataset.ipynb`**: genera el dataset con **Qwen3-TTS** clonando una voz de referencia.
   Cada clip se comprueba con Whisper (que diga el texto) y con un juez de acento; los que fallan se repiten
   o se descartan.
2. **`02_fonemas_y_alineacion.ipynb`**: pasa los textos a fonemas con eSpeak NG `es-419`, alinea fonemas y
   audio, y filtra los clips con problemas: finales cortados, última vocal demasiado corta, silencios largos
   al principio o al final y alineaciones poco fiables. Quedan unas **8,4 horas**.
3. **`03_entrenar.ipynb`**: entrena con el repo oficial (misma receta del README: `--gan-start 8000`,
   `--gan-ratio 0.7`, `--teacher-final 0`, `--batch-frames 9000`) durante **150 000 pasos** en una GPU.
   Se puede parar y reanudar; todo se guarda en Drive.

`comparar_ruido.py` genera las mismas frases con y sin `ruido_fricativas` para compararlas de oído.

## Limitaciones

- **Datos 100 % sintéticos:** el modelo no puede sonar más natural que Qwen3-TTS, que generó el audio.
- Queda **algo de ruido** de fondo, y a veces **se corta alguna «s» final**.
- Una sola voz, sin clonación ni control de emoción. La entonación es bastante plana.
- Nombres extranjeros y grafías raras pueden leerse mal (eSpeak NG los lee como español).
- Solo 8,4 h de datos. El [tiny-voca-es de España](https://github.com/stdevMac/tiny-voca), por ejemplo,
  usó 24,5 h.

## Créditos y licencias

- [tiny-voca](https://github.com/n4ze3m/tiny-voca) de n4ze3m: código, arquitectura y receta (Apache 2.0).
- `ruido_fricativas.py`: ajuste de inferencia portado de
  [stdevMac/tiny-voca](https://github.com/stdevMac/tiny-voca) (`NOISE_ABOVE`, Apache 2.0).
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS): generó todo el audio de entrenamiento; consulta sus términos.
- [eSpeak NG](https://github.com/espeak-ng/espeak-ng) (GPL-3.0): fonemas en español. Hace falta para usar el modelo.
