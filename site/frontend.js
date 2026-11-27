// Front end español (es-419) de tiny-voca en JavaScript: texto -> ids de fonema + acento.
//
// Copia línea a línea de es/frontend_es.py (normalize_text, chunks, parse_espeak, tokenize,
// phonemize), con la llamada a eSpeak sustituida por el callback `espeak`. Los ids son los
// índices de SYMBOLS, que coinciden con "symbols" del JSON del modelo ONNX.
// site/test/frontend_golden.mjs lo compara con el Python en muchas frases: debe pasar tras
// CUALQUIER cambio en cualquiera de los dos.
//
// Semántica de Python reproducida: str.isalpha (\p{L}), str.split()/strip() y \s de re (el
// conjunto de espacios de Python: tiene \x1c-\x1f y \x85, no U+FEFF), \d de re (\p{Nd}),
// unicodedata.category == "Cf" e iteración por punto de código, no por unidad UTF-16.

const SPECIALS = ["<pad>", "<bos>", "<eos>", "<sp>", "<unk>"];
const [, BOS, EOS, SP, UNK] = SPECIALS;
const PUNCTUATION = [",", ".", "?", "!", ";", ":", '"', "'", "(", ")", "—", "…"];
const CONSONANTS = ["p", "b", "t", "d", "k", "ɡ", "f", "s", "z", "θ", "x", "tʃ", "dʒ", "ts", "m", "n", "ɲ", "ŋ",
  "l", "ʎ", "ɾ", "r", "ʝ", "j", "w", "β", "ð", "ɣ"];
const VOWELS = ["a", "e", "i", "o", "u", "ɛ", "aɪ", "aʊ", "eɪ", "eʊ", "oɪ"];
const PHONES = [...CONSONANTS, ...VOWELS];
export const SYMBOLS = [...SPECIALS, ...PUNCTUATION, ...PHONES];
const SYMBOL_TO_ID = new Map(SYMBOLS.map((s, i) => [s, i]));
const PHONE_SET = new Set(PHONES);
const PUNCTUATION_CHARS = new Set(PUNCTUATION);
const SEPARATOR = "_";
const STRESS_MARKS = [["ˈ", 1], ["ˌ", 2]];
const EQUIVALENT = new Map(Object.entries({
  "g": "ɡ", "t͡ʃ": "tʃ", "d͡ʒ": "dʒ", "t͡s": "ts", "pː": "p", "tː": "t", "kː": "k", "bː": "b",
  "dː": "d", "nː": "n", "mː": "m", "lː": "l", "sː": "s",
}));
const QUOTES = new Map(Object.entries({
  "‘": "'", "’": "'", "‚": "'", "‛": "'", "`": "'", "´": "'",
  "“": '"', "”": '"', "„": '"', "‟": '"', "«": '"', "»": '"',
  "–": "—", "―": "—", "[": "(", "{": "(", "]": ")", "}": ")",
}));
const LANGUAGE_TAG = /\((?:_?[a-z0-9]_?){2,3}(?:-(?:_?[a-z0-9]_?)+)*\)/g;

// Espacios de Python (str.isspace y \s de re): \s de JS no tiene \x1c-\x1f ni \x85 y sí U+FEFF.
const WS = "\\t-\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const WS_RUN = new RegExp(`[${WS}]+`, "gu");
const WS_DASH_WS = new RegExp(`-{2,}|[${WS}]-[${WS}]`, "gu");
const WS_EDGES = new RegExp(`^[${WS}]+|[${WS}]+$`, "gu");
const DIGIT = /\p{Nd}/u;
const IS_ALPHA = /^\p{L}$/u;
const IS_CF = /^\p{Cf}$/u;

const isAlpha = (c) => c !== "" && IS_ALPHA.test(c);
const strip = (s) => s.replace(WS_EDGES, "");
const pySplit = (s) => s.split(WS_RUN).filter((x) => x !== "");

export class ValueError extends Error {
  constructor(message) { super(message); this.name = "ValueError"; }
}
/** El texto no tiene nada que decir: solo signos, emojis y similares. */
export class NothingToPronounce extends ValueError {
  constructor(message) { super(message); this.name = "NothingToPronounce"; }
}

export function normalizeText(text) {
  if (typeof text !== "string" || !strip(text)) throw new ValueError("el texto no puede estar vacío");
  let out = "";
  for (const ch of text.normalize("NFC")) out += QUOTES.get(ch) ?? ch;
  if (DIGIT.test(out)) throw new ValueError("escribe los números con letra");
  out = out.replaceAll("¿", " ").replaceAll("¡", " ");
  out = out.replace(/\.{3,}/gu, "…");
  out = out.replace(WS_DASH_WS, "—");
  return strip(out.replace(WS_RUN, " "));
}

/** [esPuntuación, trozo]: cada signo de puntuación es su propio token. */
export function chunks(text) {
  const out = [];
  const cp = Array.from(text);
  let start = 0;
  for (let i = 0; i < cp.length; i++) {
    const char = cp[i];
    if (!PUNCTUATION_CHARS.has(char)) continue;
    if (char === "'" && i > 0 && i < cp.length - 1 && isAlpha(cp[i - 1]) && isAlpha(cp[i + 1])) continue;
    const piece = strip(cp.slice(start, i).join(""));
    if (piece) out.push([false, piece]);
    out.push([true, char]);
    start = i + 1;
  }
  const rest = strip(cp.slice(start).join(""));
  if (rest) out.push([false, rest]);
  return out;
}

/** Salida de eSpeak -> palabras, cada una una lista de [fonema, acento]. */
export function parseEspeak(output) {
  const words = [];
  for (const word of pySplit(output.replace(LANGUAGE_TAG, ""))) {
    const units = [];
    for (let unit of word.split(SEPARATOR)) {
      let stress = 0;
      for (const [mark, level] of STRESS_MARKS) {
        if (unit.includes(mark)) {
          stress = stress || level;
          unit = unit.split(mark).join("");
        }
      }
      let kept = "";
      for (const c of unit) if (!IS_CF.test(c) && c !== "(" && c !== ")") kept += c;
      if (kept) units.push([kept, stress]);
    }
    if (units.length) words.push(units);
  }
  return words;
}

/** pieces: lista de [esPuntuación, valor]; el valor de un trozo de texto son las palabras de parseEspeak. */
export function tokenize(pieces) {
  const symbols = [BOS], stress = [0], unknown = [];
  for (const [punctuation, value] of pieces) {
    if (punctuation) {
      if (symbols[symbols.length - 1] !== value) {
        symbols.push(value);
        stress.push(0);
      }
      continue;
    }
    for (const word of value) {
      const last = symbols[symbols.length - 1];
      if (PHONE_SET.has(last) || last === UNK) {
        symbols.push(SP);
        stress.push(0);
      }
      for (let [phone, level] of word) {
        phone = EQUIVALENT.get(phone) ?? phone;
        if (!PHONE_SET.has(phone)) {
          unknown.push(phone);
          phone = UNK;
        }
        symbols.push(phone);
        stress.push(level);
      }
    }
  }
  symbols.push(EOS);
  stress.push(0);
  if (!symbols.some((s) => PHONE_SET.has(s) || s === UNK)) {
    throw new NothingToPronounce("el texto no tiene nada que pronunciar");
  }
  return { symbols, ids: symbols.map((s) => SYMBOL_TO_ID.get(s)), stress, unknown };
}

export const ESPEAK_VOICE = "es-419";

/**
 * texto -> {symbols, ids, stress, unknown}. `espeak(trozo)` debe devolver la salida de
 * `espeak-ng -q --ipa --sep=_ -v es-419 --stdin` para `trozo + "\n"`.
 */
export function phonemize(text, espeak) {
  const pieces = [];
  for (const [punctuation, piece] of chunks(normalizeText(text))) {
    pieces.push(punctuation ? [true, piece] : [false, parseEspeak(espeak(piece))]);
  }
  return tokenize(pieces);
}
