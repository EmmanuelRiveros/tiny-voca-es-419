// Compara site/frontend.js con es/frontend_es.py, frase a frase, con eSpeak nativo y con el WebAssembly.
//   node site/test/frontend_golden.mjs [frases.txt]
// Necesita py (o python3) y espeak-ng en el PATH.
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { phonemize } from "../frontend.js";
import { createEspeak } from "../espeak/espeak.js";

const here = fileURLToPath(new URL(".", import.meta.url));
const file = process.argv[2] ?? here + "frases_es.txt";
const python = process.platform === "win32" ? "py" : "python3";
const expected = JSON.parse(execFileSync(python, [here + "frontend_golden.py", file],
  { encoding: "utf8", env: { ...process.env, PYTHONIOENCODING: "utf-8" } }));
const lines = readFileSync(file, "utf8").split("\n").filter((l) => l !== "");

const native = (piece) => execFileSync("espeak-ng", ["-q", "--ipa", "--sep=_", "-v", "es-419", "--stdin"],
  { input: piece.replace(/\n+$/, "") + "\n", encoding: "utf8" }).replace(/\r\n/g, "\n");
const wasm = await createEspeak(new URL("../espeak/", import.meta.url));

function run(text, espeak) {
  try {
    const { symbols, ids, stress, unknown } = phonemize(text, espeak);
    return { symbols, ids, stress, unknown };
  } catch (error) {
    if (error.name !== "ValueError" && error.name !== "NothingToPronounce") throw error;
    return { error: error.message };
  }
}

let failures = 0;
for (const [name, espeak] of [["nativo", native], ["wasm", (p) => wasm.phonemize(p)]]) {
  let bad = 0;
  lines.forEach((line, i) => {
    const got = JSON.stringify(run(line, espeak)), want = JSON.stringify(expected[i]);
    if (got !== want) {
      bad++;
      if (bad <= 5) console.log(`DIFF (${name})`, JSON.stringify(line), "\n  js    ", got, "\n  python", want);
    }
  });
  console.log(`eSpeak ${name}: ${lines.length - bad}/${lines.length} iguales al Python`);
  failures += bad;
}
console.log(`(eSpeak WebAssembly ${wasm.version})`);
process.exit(failures ? 1 : 0);
