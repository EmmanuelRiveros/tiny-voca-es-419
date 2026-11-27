"""Alineaciones de Montreal Forced Aligner -> duraciones de Tiny Voca.

Mismas convenciones que ``tiny_voca.align.assemble`` del repo oficial:
* El cuadro acústico i está centrado en la muestra ``i * hop``.
* Cada fonema recibe al menos un cuadro y las duraciones suman los cuadros del audio.
* El silencio inicial va a ``<bos>`` y el final a ``<eos>``.
* Un silencio interior de al menos ``min_pause_frames`` cuadros (7 ≈ 75 ms) va al token
  que puede llevar pausa (puntuación con pausa > otra puntuación > ``<sp>``).
* Si un silencio interior es más corto, o no hay ningún token entre los dos fonemas, se reparte
  entre los fonemas vecinos (cierres de oclusivas).
"""
import json
import math
from pathlib import Path
import re

import numpy as np

PAUSING = (",", ".", "?", "!", ";", ":", "—", "…")
SILENCE_LABELS = frozenset(("", "sil", "sp", "spn", "<eps>", "silence", "<sil>"))


def phone_aliases(phones):
    """Nombres ASCII para MFA (``ph01``…): evita que MFA interprete símbolos IPA."""
    to_alias = {phone: f"ph{index + 1:02d}" for index, phone in enumerate(phones)}
    return to_alias, {alias: phone for phone, alias in to_alias.items()}


def pause_carrier(symbols, group, punctuation, sp):
    """Índice del token que recibe una pausa dentro de un tramo de tokens no fonéticos."""
    for wanted in (PAUSING, punctuation, (sp,)):
        for index in group:
            if symbols[index] in wanted:
                return index
    return group[0]


def _textgrid_tiers(text):
    tiers, current = {}, None
    for line in text.splitlines():
        line = line.strip()
        name = re.match(r'name = "(.*)"', line)
        if name:
            current = tiers.setdefault(name.group(1), [])
            continue
        if current is None:
            continue
        for key in ("xmin", "xmax"):
            value = re.match(key + r" = ([0-9.eE+-]+)", line)
            if value:
                if key == "xmin":
                    current.append([float(value.group(1)), None, None])
                elif current and current[-1][1] is None:
                    current[-1][1] = float(value.group(1))
        label = re.match(r'text = "(.*)"', line)
        if label and current and current[-1][2] is None:
            current[-1][2] = label.group(1)
    return {name: [tuple(entry) for entry in entries if None not in entry] for name, entries in tiers.items()}


def _json_tiers(value):
    tiers = value.get("tiers", value)
    result = {}
    for name, tier in tiers.items():
        entries = tier.get("entries", []) if isinstance(tier, dict) else tier
        parsed = []
        for entry in entries:
            if isinstance(entry, dict):
                parsed.append((float(entry.get("start", entry.get("begin"))), float(entry["end"]),
                               str(entry.get("label", entry.get("text", "")))))
            else:
                parsed.append((float(entry[0]), float(entry[1]), str(entry[2])))
        result[name] = parsed
    return result


def read_phone_intervals(path, from_alias):
    """Intervalos ``(inicio, fin, fonema)`` sin silencios, en segundos."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    tiers = _json_tiers(json.loads(text)) if path.suffix == ".json" else _textgrid_tiers(text)
    names = [name for name in tiers if name.lower().endswith("phones")]
    if len(names) != 1:
        raise ValueError(f"se esperaba un único nivel de fonemas en {path.name}: {sorted(tiers)}")
    intervals = []
    for start, end, label in sorted(tiers[names[0]]):
        label = label.strip()
        if label.lower() in SILENCE_LABELS:
            continue
        if label not in from_alias:
            raise ValueError(f"fonema de MFA desconocido: {label!r}")
        intervals.append((start, end, from_alias[label]))
    return intervals


def durations_from_intervals(symbols, intervals, frames, hop, sample_rate, min_pause_frames, phones,
                             punctuation, sp):
    phone_indices = [i for i, s in enumerate(symbols) if s in phones]
    if [symbols[i] for i in phone_indices] != [phone for _, _, phone in intervals] or not phone_indices:
        raise ValueError("mfa_fonemas_distintos")
    count, scale = len(phone_indices), sample_rate / hop
    # El cuadro i pertenece al intervalo [a, b) si su centro i/scale cae dentro.
    starts = [math.ceil(a * scale - 1e-9) for a, _, _ in intervals]
    ends = [math.ceil(b * scale - 1e-9) for _, b, _ in intervals]
    previous = 0
    for k in range(count):
        starts[k] = max(starts[k], previous)
        ends[k] = max(ends[k], starts[k] + 1)
        previous = ends[k]
    following = frames
    for k in reversed(range(count)):
        ends[k] = min(ends[k], following)
        starts[k] = min(starts[k], ends[k] - 1)
        following = starts[k]
    if starts[0] < 0:
        raise ValueError("audio_demasiado_corto")
    token = np.empty(frames, dtype=np.int64)
    token[:starts[0]] = 0
    token[ends[-1]:] = len(symbols) - 1
    for k in range(count):
        token[starts[k]:ends[k]] = phone_indices[k]
    pauses = 0
    for k in range(count - 1):
        gap = starts[k + 1] - ends[k]
        if gap <= 0:
            continue
        left, right = phone_indices[k], phone_indices[k + 1]
        if right > left + 1 and gap >= min_pause_frames:
            token[ends[k]:starts[k + 1]] = pause_carrier(symbols, range(left + 1, right), punctuation, sp)
            pauses += gap
        else:
            middle = ends[k] + (gap + 1) // 2
            token[ends[k]:middle] = left
            token[middle:starts[k + 1]] = right
    if np.any(np.diff(token) < 0):
        raise ValueError("alineacion_no_monotona")
    durations = np.bincount(token, minlength=len(symbols)).astype(np.int64)
    if durations.sum() != frames or np.any(durations[phone_indices] < 1):
        raise ValueError("duraciones_no_cubren_el_audio")
    return [int(d) for d in durations], {"leading_frames": int(starts[0]), "trailing_frames": int(frames - ends[-1]),
                                         "interior_pause_frames": int(pauses)}
