// Ejecuta el front end español (eSpeak NG 1.52.0 en WebAssembly, voz es-419, y luego
// site/frontend.js) y el modelo (ONNX Runtime Web) fuera del hilo principal.
// Basado en la demo de stdevMac/tiny-voca (Apache 2.0).
// A module worker. Messages in: {type: "load", model} and
// {type: "speak", id, text, voice, speed}. Messages out: status, ready,
// sentence (with its audio, as soon as it is made), done (with the whole clip),
// error (plain: true when the message is meant for the reader as it is).
//
// Speaking follows tiny_voca.infer.speak_text: sentences of the finished text,
// each cut after commas, then between words, while it is too long for the model.

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.min.mjs";
import { createEspeak } from "./espeak/espeak.js";
import { NothingToPronounce, phonemize, SYMBOLS, ValueError } from "./frontend.js";
import { finished, sentences, shorter } from "./sentences.js";

const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
// El dataset se fonetizó con eSpeak NG 1.51 (Colab); esta página trae 1.52.0, que da los mismos fonemas
// es-419 en las pruebas (site/test). Otra versión podría escribir distinto algunas palabras.
const ESPEAK_VERSION = "1.52.0";
// Se muestra cuando ninguna frase del texto se pudo leer.
const ONLY_SPANISH = "No encontré nada que leer: este modelo solo lee español.";
// The model sometimes peaks above 1.0, which would clip in the 16-bit file. Scale the whole clip down then.
const PEAK = 0.99;

let espeak, session, config;

// Un error que se muestra tal cual a quien escribe el texto.
class Plain extends Error {}

// fetch() rejects with a bare "Failed to fetch" when the network or CORS fails; name the file instead.
async function get(url, options) {
  let response;
  try {
    response = await fetch(url, options);
  } catch (error) {
    throw new Error(`no se pudo descargar ${url} (${error.message}). ¿Hay conexión?`);
  }
  if (!response.ok) {
    const missing = response.status === 401 || response.status === 404;
    throw new Error(`${url}: ${response.status}${missing ? " (el archivo no existe o aún no está publicado)" : ""}`);
  }
  return response;
}

const status = (text, progress) => postMessage({ type: "status", text, progress });

// Downloaded files are kept in the browser's cache storage: a second visit starts without the network.
// The key is the URL, which names a fixed revision of the model, so a new model is a new URL.
async function download(url, label) {
  let store = null;
  try {
    store = await caches.open("tiny-voca-es-419");
    const hit = await store.match(url);
    if (hit) { status(label, 1); return new Uint8Array(await hit.arrayBuffer()); }
  } catch (error) { store = null; }
  const bytes = await fetchBytes(url, label);
  if (store) store.put(url, new Response(bytes)).catch(() => {});
  return bytes;
}

async function fetchBytes(url, label) {
  const response = await get(url);
  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body || !total) return new Uint8Array(await response.arrayBuffer());
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    status(label, received / total);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  return bytes;
}

async function loadFrontEnd() {
  try {
    espeak = await createEspeak(new URL("./espeak/", import.meta.url).href);
  } catch (error) {
    throw new Error(`no se pudo cargar eSpeak NG: ${error.message}`);
  }
  if (espeak.version !== ESPEAK_VERSION) console.warn(`eSpeak NG ${espeak.version}; la página espera ${ESPEAK_VERSION}`);
  phonemize("Hola.", espeak.phonemize);  // warm up
}

async function loadModel(base) {
  // El JSON se pide siempre a la red (es pequeño); su "step" entra en la URL del ONNX, así un modelo
  // nuevo no se confunde con el que quedó guardado en la caché del navegador.
  const response = await get(base + "tiny-voca-es-419.json", { cache: "no-cache" });
  config = await response.json();
  if (JSON.stringify(config.symbols) !== JSON.stringify(SYMBOLS)) throw new Error("los símbolos del modelo no coinciden con los del front end");
  const bytes = await download(`${base}tiny-voca-es-419.onnx?paso=${config.step}`, "Descargando el modelo");
  status("Iniciando el modelo");
  ort.env.wasm.wasmPaths = ORT;
  ort.env.wasm.numThreads = 1;
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
}

async function synthesize(phones, voice, speed) {
  const count = phones.ids.length;
  const integers = (values) => new ort.Tensor("int64", BigInt64Array.from(values.map(BigInt)), [1, count]);
  const scalar = (value) => new ort.Tensor("float32", Float32Array.from([value]), [1]);
  const output = await session.run({
    tokens: integers(phones.ids), stress: integers(phones.stress),
    voice: new ort.Tensor("int64", BigInt64Array.from([BigInt(voice)]), [1]),
    speed: scalar(speed), pitch_shift: scalar(0), pitch_range: scalar(1),
  });
  let frames = 0;
  for (const value of output.durations.data) frames += Number(value);
  return frames > config.max_frames ? null : output.audio.data;
}

function limitPeak(samples) {
  let loudest = 0;
  for (let i = 0; i < samples.length; i++) loudest = Math.max(loudest, Math.abs(samples[i]));
  if (loudest > PEAK) for (let i = 0; i < samples.length; i++) samples[i] *= PEAK / loudest;
}

// One piece of text -> its utterances, each [phones, audio], in order. Too long for one call of the
// model (max_tokens phonemes or max_frames frames): cut it smaller. Nothing to pronounce: nothing.
async function read(piece, voice, speed, times) {
  let phones;
  let mark = performance.now();
  try {
    phones = phonemize(piece, espeak.phonemize);
  } catch (error) {
    if (error instanceof NothingToPronounce) return [];
    // p. ej. «Escribe los números con letra.»
    if (error instanceof ValueError) throw new Plain(error.message[0].toUpperCase() + error.message.slice(1) + ".");
    throw error;
  } finally {
    times.phonemes += performance.now() - mark;
  }
  if (phones.ids.length <= config.max_tokens - 12) {
    mark = performance.now();
    const audio = await synthesize(phones, voice, speed);
    times.model += performance.now() - mark;
    if (audio) return [[phones, audio]];
  }
  const pieces = shorter(piece);
  if (!pieces.length) throw new Plain(`una palabra es demasiado larga para leerla: ${piece.slice(0, 40)}`);
  const result = [];
  for (const part of pieces) result.push(...await read(part, voice, speed, times));
  return result;
}

const spelled = (phones) => phones.symbols.join(" ");

async function speak({ id, text, voice, speed }) {
  const started = performance.now();
  const list = sentences(finished(text));
  const chunks = [];
  const spoken = [];
  const times = { phonemes: 0, model: 0 };
  for (let index = 0; index < list.length; index++) {
    for (const [phones, audio] of await read(list[index], voice, speed, times)) {
      chunks.push(audio);
      spoken.push(spelled(phones));
      // The page plays this sentence at once, before the whole clip is known: a copy, kept below full scale by itself.
      const heard = audio.slice();
      limitPeak(heard);
      postMessage({ type: "sentence", id, index, total: list.length, phonemes: spelled(phones), audio: heard,
                    sampleRate: config.sample_rate }, [heard.buffer]);
    }
  }
  if (!chunks.length) {
    postMessage({ type: "error", id, message: ONLY_SPANISH, plain: true });
    return;
  }
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const audio = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) { audio.set(chunk, offset); offset += chunk.length; }
  limitPeak(audio);
  postMessage({ type: "done", id, audio, sampleRate: config.sample_rate, phonemes: spoken,
                phonemeTime: times.phonemes, modelTime: times.model, totalTime: performance.now() - started }, [audio.buffer]);
}

onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      await Promise.all([loadFrontEnd(), loadModel(data.model)]);
      postMessage({ type: "ready", voices: config.voices, sampleRate: config.sample_rate, espeak: espeak.version });
    } else if (data.type === "speak") {
      await speak(data);
    }
  } catch (error) {
    postMessage({ type: "error", id: data.id, message: String(error && error.message || error), plain: error instanceof Plain });
  }
};
