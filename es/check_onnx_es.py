"""ONNX (--external-noise) contra PyTorch + ruido_fricativas, muestra a muestra, con el mismo ruido."""
import json, sys
import numpy as np, onnxruntime as ort, torch
from tiny_voca.infer import load_model
from tiny_voca.text import phonemize
import ruido_fricativas

ruido_fricativas.activar()
model = load_model(sys.argv[1], "cpu")
exact = ort.InferenceSession(sys.argv[2], providers=["CPUExecutionProvider"])
hop, max_frames = model.config.hop, model.config.max_frames
for text in ("Hola, soy una voz sintética.", "Si sales sin suéter, seguro sientes frío esta semana.",
             "¿Sabes si el sábado hace sol o si se suspende la fiesta de Fernanda?"):
    phones = phonemize(text)
    tokens, stress = torch.tensor([phones.ids]), torch.tensor([phones.stress])
    ref, lengths, details = model.synthesize(tokens, stress, torch.tensor([len(phones)]), torch.tensor([0]))
    ref = ref[0, :int(lengths[0])].numpy()
    stream = torch.Generator().manual_seed(0)
    draws = [torch.randn(len(ref), generator=stream).numpy() for _ in range(2)]
    one = lambda v: np.array([v], dtype=np.float32)
    feed = {"tokens": np.array([phones.ids], np.int64), "stress": np.array([phones.stress], np.int64), "voice": np.array([0], np.int64),
            "speed": one(1), "pitch_shift": one(0), "pitch_range": one(1)}
    for name, draw in zip(("noise", "breath"), draws):
        feed[name] = np.zeros((1, hop * max_frames), np.float32); feed[name][0, :len(draw)] = draw
    audio, durations = exact.run(None, feed)
    print(json.dumps({"texto": text[:30], "mismas_duraciones": bool((durations == details["durations"].numpy()).all()),
                      "misma_longitud": audio.shape[1] == len(ref),
                      "error_relativo": float(np.abs(audio[0, :len(ref)] - ref).max() / np.abs(ref).max())}, ensure_ascii=False))
