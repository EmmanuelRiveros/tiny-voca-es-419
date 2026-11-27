"""Ajuste de inferencia para Tiny Voca: «s», «f», respiraciones y pausas con ruido en vez de pulsos.

El generador de onda de tiny-voca excita su rama armónica con un tren de pulsos a la frecuencia de la voz.
Todo lo que suena por esa rama llega a golpes, uno por periodo (≈4 ms en una voz femenina), también las
fricativas y las respiraciones, que entonces se oyen ásperas, como estática. Este ajuste, solo de inferencia,
sustituye los pulsos por ruido de la misma potencia por encima de un armónico: el 2 en los tramos sordos
(«s», «f», «j», pausas) y el 33 en los sonoros (solo lo más agudo de las vocales). Los pesos no cambian.

Portado de stdevMac/tiny-voca (Apache 2.0), ``NOISE_ABOVE`` en src/tiny_voca/model.py, commit a4bf2bd
(docs/word-edge-artifact.md, sección 11). Se aplica sobre el tiny_voca oficial sin editar sus archivos:

    import ruido_fricativas
    ruido_fricativas.activar()            # (2, 33) por defecto; activar(None) vuelve a los pulsos solos
    # ...luego load_model / speak_text como siempre
"""
import math

import torch
from torch import nn

from tiny_voca import model as _model

NOISE_ABOVE = (2.0, 33.0)
_estado = {"noise_above": None}
_originales = {}


def _noise_share(self, harmonics, f0, here, following):
    """Reparto pulsos/ruido por sub-frame y bin: sube de 0 a 1 en la octava centrada en el armónico límite."""
    c = self.config
    hertz = torch.arange(c.synth_fft // 2 + 1, device=f0.device) * (c.sample_rate / c.synth_fft)
    octaves = torch.log2(hertz.clamp_min(1e-3) / (harmonics.float() * f0).clamp_min(1e-6)[..., None]).clamp(-0.5, 0.5)
    share = 0.5 * (1 + torch.sin(math.pi * octaves))
    band = (hertz >= 0.5 * f0[..., None]) & (hertz <= c.harmonic_limit * c.sample_rate)
    share, band = (0.5 * (self._take(value, here) + self._take(value, following)) for value in (share, band.float()))
    return share, band


def _forward(self, hidden, f0, lengths, noise=None, phase=None, noise_above=None, breath=None):
    """SourceFilter.forward del repo oficial + la mezcla de stdevMac cuando ``noise_above`` no es None."""
    F, cepstrum_response, length_mask, pulse_train = _model.F, _model.cepstrum_response, _model.length_mask, _model.pulse_train
    c = self.config
    frames = hidden.shape[1]
    samples = frames * c.hop
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        hidden = hidden.float()
        audible = length_mask(lengths * c.hop, samples)
        last = (lengths - 1).clamp_min(0)[:, None]
        h = self.harmonic(hidden)
        split = c.cep_causal + c.cep_anti
        sub = self._subframes(frames, lengths, self.harmonic_steps)
        here, following = sub // self.harmonic_steps, torch.minimum((sub + 1) // self.harmonic_steps, last)
        causal = 0.5 * (self._take(h[..., :c.cep_causal], here) + self._take(h[..., :c.cep_causal], following)) * self.causal_lifter
        anti = 0.5 * (self._take(h[..., c.cep_causal:split], here) + self._take(h[..., c.cep_causal:split], following)) * self.anti_lifter
        causal = torch.cat((causal[..., :1] + h[..., split:].flatten(1).gather(1, sub)[..., None], causal[..., 1:]), dim=-1)
        harmonic_response = cepstrum_response(causal, anti, c.synth_fft)
        n = self.noise(hidden)
        sub = self._subframes(frames, lengths, self.noise_steps)
        shape = self._take(n[..., :c.noise_cep], sub // self.noise_steps) * self.noise_lifter
        shape = torch.cat((shape[..., :1] + n[..., c.noise_cep:].flatten(1).gather(1, sub)[..., None], shape[..., 1:]), dim=-1)
        noise_response = cepstrum_response(shape, None, c.noise_fft)
        f0 = f0.float().gather(1, torch.minimum(torch.arange(frames, device=f0.device)[None], last))
        pulses = pulse_train(f0, c.sample_rate, c.hop, c.harmonic_limit, phase) * audible
        if noise is None:
            noise = torch.randn(hidden.shape[0], samples, device=hidden.device)
        if noise_above is None:
            harmonic = self._shape(pulses, harmonic_response, c.synth_fft, c.synth_hop, self.synth_window)
        else:
            share, band = _noise_share(self, noise_above, f0, here, following)
            if breath is None:
                breath = torch.randn(hidden.shape[0], samples, device=hidden.device)
            harmonic = (self._shape(pulses, harmonic_response * torch.sqrt(1 - share), c.synth_fft, c.synth_hop, self.synth_window)
                        + self._shape(breath * audible * math.sqrt(0.5 / c.harmonic_limit), harmonic_response * torch.sqrt(share) * band,
                                      c.synth_fft, c.synth_hop, self.synth_window))
        wave = harmonic + self._shape(noise * audible, noise_response, c.noise_fft, c.noise_hop, self.noise_window)
        return wave * audible


@torch.inference_mode()
def _synthesize(self, tokens, stress, lengths, speakers, speed=1.0, pitch_shift=0.0, seed=0, pitch_range=1.0, pitch_jitter=0.0,
                voiced_vowels=True):
    """TinyVoca.synthesize del repo oficial; con el ajuste activo calcula el límite por frame (sordo/sonoro)."""
    if not math.isfinite(speed) or not 0.25 <= speed <= 4 or not math.isfinite(pitch_shift) or not 0 <= pitch_range <= 3:
        raise ValueError("speed must be between 0.25 and 4, pitch_shift finite and pitch_range between 0 and 3")
    hidden, mask = self.encode(tokens, stress, lengths, speakers)
    durations = self.predict_durations(hidden, mask, tokens, speed)
    frames, frame_mask, frame_lengths = self.expand(hidden, durations, mask, speakers)
    prosody_hidden, prosody = self.predict_prosody(frames, frame_mask)
    if voiced_vowels:
        ends = durations.cumsum(1)
        frame = torch.arange(frames.shape[1], device=tokens.device)[None].expand(len(lengths), -1).contiguous()
        owner = torch.searchsorted(ends, frame, right=True).clamp(max=tokens.shape[1] - 1)
        vowel = self.vowel_tokens[tokens].gather(1, owner) & frame_mask
        prosody = torch.cat((prosody[..., :1], torch.where(vowel, prosody[..., 1].clamp_min(2.0), prosody[..., 1])[..., None],
                             prosody[..., 2:]), dim=-1)
    prosody = self.widen_pitch(prosody, frame_mask, pitch_range)
    acoustic = self.decode(prosody_hidden, self.hard_prosody(prosody), frame_mask)
    # Mismo ruido que el oficial (mismo generador y semilla); el ruido nuevo sale después del mismo generador.
    streams = [torch.Generator().manual_seed(seed + row) for row in range(len(frame_lengths))]
    noise = nn.utils.rnn.pad_sequence([torch.randn(int(n) * self.config.hop, generator=stream)
                                       for stream, n in zip(streams, frame_lengths)], batch_first=True)
    cutoff = breath = None
    if _estado["noise_above"] is not None:
        unvoiced, voiced = _estado["noise_above"]
        cutoff = torch.where(prosody[..., 1] > 0, float(voiced), float(unvoiced))
        breath = nn.utils.rnn.pad_sequence([torch.randn(int(n) * self.config.hop, generator=stream)
                                            for stream, n in zip(streams, frame_lengths)], batch_first=True).to(hidden.device)
    pitch = prosody[..., 0] + pitch_shift
    if pitch_jitter > 0:
        pitch = pitch + self.pitch_wobble(pitch.shape, pitch_jitter, seed, pitch.device)
    wave = self.vocoder(acoustic, self.frequency(pitch), frame_lengths, noise=noise.to(hidden.device),
                        noise_above=cutoff, breath=breath)
    return wave, frame_lengths * self.config.hop, {"durations": durations, "prosody": prosody, "frame_mask": frame_mask}


def activar(noise_above=NOISE_ABOVE):
    """Activa el ajuste (pareja: armónico límite en tramos sordos, en tramos sonoros). None = pulsos solos."""
    if not _originales:
        _originales.update(forward=_model.SourceFilter.forward, synthesize=_model.TinyVoca.synthesize)
        _model.SourceFilter.forward = _forward
        _model.TinyVoca.synthesize = _synthesize
    _estado["noise_above"] = None if noise_above is None else (float(noise_above[0]), float(noise_above[1]))
