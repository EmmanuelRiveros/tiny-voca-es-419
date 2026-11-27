// Compara eSpeak WebAssembly (site/espeak, voz es-419) con el programa nativo, frase a frase.
// node site/test/espeak_vs_nativo.mjs site/espeak site/test/frases_es.txt
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const [dir, file] = process.argv.slice(2);
const { createEspeak } = await import(pathToFileURL(dir + "/espeak.js").href);
const espeak = await createEspeak(pathToFileURL(dir + "/"));
const lines = readFileSync(file, "utf8").split("\n").filter((l) => l.trim());
let bad = 0;
for (const line of lines) {
  const wasm = espeak.phonemize(line);
  const native = execFileSync("espeak-ng", ["-q", "--ipa", "--sep=_", "-v", "es-419", "--stdin"],
                              { input: line + "\n", encoding: "utf8" }).replace(/\r\n/g, "\n");
  if (wasm !== native) { bad++; if (bad <= 5) console.log("DIFF", JSON.stringify(line), "\n wasm  ", JSON.stringify(wasm), "\n nativo", JSON.stringify(native)); }
}
console.log(`eSpeak ${espeak.version}: ${lines.length - bad}/${lines.length} iguales`);
