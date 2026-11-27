// Generator page: calls POST /api/generate and shows the text plus the model's inner workings
// at the selected character: cell diagram, next-character prediction, memory heat map, gates.
import { $, BLUE_TEXT, ORANGE_TEXT, diverging, esc, hideTip, pct, postJSON, scaleLegend, sequential, showChar, showTip } from "./utils.js";
import { drawCell } from "./viz/cell.js";
import { drawMemory } from "./viz/memory.js";
import { drawTimeline } from "./viz/timeline.js";

const PROMPTS = ["My lord, ", "Thou art ", "What light ", "I will not ", "O, "];

let data = null;      // the last /api/generate response
let selected = null;  // index of the selected character
let timer = null;     // playback timer

// ---------------------------------------------------------------- text + colouring
// Colour options: how sure the model was, a gate, or one of the cells found in experiment RQ2.
function colourOptions() {
  const options = [["none", "nothing"], ["confidence", "model confidence"]];
  for (const g of data.gate_names) options.push([`gate:${g}`, `${g} gate`]);
  for (const name of Object.keys(data.neurons)) options.push([`neuron:${name}`, `neuron: ${name}`]);
  return options;
}

function colourFn(mode) {
  if (mode === "confidence") {
    return { fn: (i) => sequential(data.steps[i].p, ...BLUE_TEXT), legend: scaleLegend("blue-text", "unsure", "certain") };
  }
  if (mode.startsWith("gate:")) {
    const g = mode.slice(5);
    const vals = data.steps.map((s) => s.gates[g]);
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    return {
      fn: (i) => sequential((vals[i] - lo) / Math.max(hi - lo, 1e-6), ...ORANGE_TEXT),
      legend: scaleLegend("orange", `${lo.toFixed(2)} (closed)`, `${hi.toFixed(2)} (open)`),
    };
  }
  if (mode.startsWith("neuron:")) {
    const vals = data.neurons[mode.slice(7)];
    return { fn: (i) => diverging(vals[i], 0.75), legend: scaleLegend("diverging", "−1", "+1") };
  }
  return { fn: null, legend: "" };
}

function renderText() {
  const { fn, legend } = colourFn($("gen-colour").value);
  $("gen-scale").innerHTML = legend;
  // characters that belong to an invented word (not in Shakespeare) get a red underline
  const invented = new Set();
  for (const m of data.text.matchAll(/[A-Za-z']+/g)) {
    if (data.unknown_words.includes(m[0])) for (let k = 0; k < m[0].length; k++) invented.add(m.index + k);
  }
  const spans = [...data.text].map((ch, i) => {
    const shown = ch === "\n" ? '<span class="nl">↵</span>\n' : esc(ch);
    const style = fn ? ` style="background:${fn(i)}"` : "";
    const cls = invented.has(i) ? "ch invented" : "ch";
    return `<span class="${cls}" data-i="${i}"${style}>${shown}</span>`;
  });
  $("gen-text").innerHTML = `<span class="prompt">${esc(data.prompt)}</span>${spans.join("")}`;
  if (selected !== null) $("gen-text").querySelector(`[data-i="${selected}"]`)?.classList.add("sel");
}

// share of generated words that the spelling filter knows (the server checks against Shakespeare)
function realWords() {
  const words = data.text.match(/[A-Za-z']+/g) || [];
  if (!words.length) return 1;
  return words.filter((w) => !data.unknown_words.includes(w)).length / words.length;
}

function renderStats() {
  const meanP = data.steps.reduce((a, s) => a + s.p, 0) / data.steps.length;
  const stats = [
    [data.text.length, "characters"],
    [pct(meanP), "avg. probability of chosen char"],
    [pct(realWords()), "real words"],
    [data.style_score === null ? "–" : pct(data.style_score), "Shakespeare-style score"],
    [`${data.cell.toUpperCase()} · ${data.hidden_size}`, "cell · memory units"],
  ];
  $("gen-stats").innerHTML = stats.map(([v, k]) => `<div class="stat"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");
  $("gen-stats").hidden = false;
}

// ---------------------------------------------------------------- inspector for one step
function renderPrediction(step, ch) {
  const rows = step.top.map(([c, p, blocked]) => ({ c, p, blocked, chosen: c === ch }));
  if (!rows.some((r) => r.chosen)) rows.push({ c: ch, p: step.p, blocked: false, chosen: true });
  const max = Math.max(...rows.map((r) => r.p));
  const filterNote = step.blocked > 0.001
    ? ` The spelling filter crossed out characters that would not lead to a real word
        (<b>${(100 * step.blocked).toFixed(1)}%</b> of the probability).`
    : "";
  $("prediction").innerHTML =
    rows
      .map((r) => `<div class="pred-row${r.chosen ? " chosen" : ""}${r.blocked ? " blocked" : ""}">
          <span class="c">${esc(showChar(r.c))}</span>
          <div class="track"><div class="fill" style="width:${(100 * r.p) / max}%"></div></div>
          <span class="p">${(100 * r.p).toFixed(1)}%</span></div>`)
      .join("") +
    `<p class="muted" style="margin-top:8px">The model gave the character it wrote (outlined) a probability of
     <b>${(100 * step.p).toFixed(1)}%</b>; sampling at temperature ${$("gen-temp").value} picked it.${filterNote}</p>`;
}

function select(i) {
  if (!data) return;
  selected = Math.max(0, Math.min(data.steps.length - 1, i));
  const ch = data.text[selected];
  $("gen-text").querySelector(".ch.sel")?.classList.remove("sel");
  const span = $("gen-text").querySelector(`[data-i="${selected}"]`);
  span?.classList.add("sel");
  span?.scrollIntoView({ block: "nearest" });
  $("step-label").innerHTML = `character ${selected + 1} of ${data.steps.length} · <span class="kbd">${esc(showChar(ch))}</span>`;
  drawCell($("cell-diagram"), data.cell, data.steps[selected].gates, ch);
  renderPrediction(data.steps[selected], ch);
  drawMemory($("memory-canvas"), data.memory, data.text, selected, select);
  drawTimeline($("gate-timeline"), data.steps, data.gate_names, data.text, selected, select);
}

// ---------------------------------------------------------------- playback
function stop() {
  clearInterval(timer);
  timer = null;
  $("step-play").textContent = "▶ Play";
}
function togglePlay() {
  if (!data) return;
  if (timer) return stop();
  if (selected === null || selected >= data.steps.length - 1) select(0);
  $("step-play").textContent = "❚❚ Pause";
  timer = setInterval(() => (selected >= data.steps.length - 1 ? stop() : select(selected + 1)), 90);
}

// ---------------------------------------------------------------- generate
async function run() {
  stop();
  const button = $("gen-btn");
  button.disabled = true;
  button.textContent = "Generating…";
  try {
    data = await postJSON("/api/generate", {
      prompt: $("gen-prompt").value,
      temperature: parseFloat($("gen-temp").value),
      length: parseInt($("gen-length").value, 10),
      model: $("gen-model").value,
      spelling: $("gen-spelling").checked,
    });
    const previous = $("gen-colour").value;
    const options = colourOptions();
    $("gen-colour").innerHTML = options.map(([v, l]) => `<option value="${v}">${esc(l)}</option>`).join("");
    $("gen-colour").value = options.some(([v]) => v === previous) ? previous : "confidence";
    $("memory-title").textContent = `Memory over time (${data.memory_name}, ${data.memory[0].length} units)`;
    $("memory-scale").innerHTML = scaleLegend("diverging", "−1", "+1");
    selected = null;
    renderText();
    renderStats();
    select(0);
  } catch (err) {
    $("gen-text").innerHTML = `<p class="error">${esc(err.message)}</p>`;
  } finally {
    button.disabled = false;
    button.textContent = "Generate";
  }
}

export function setupGenerator(models) {
  $("gen-model").innerHTML = models.length
    ? models.map((m) => `<option value="${m.id}">${esc(m.label)}</option>`).join("")
    : '<option value="lstm_long">no generator trained yet</option>';
  $("gen-prompts").innerHTML = PROMPTS.map((p) => `<button class="chip">${esc(p.trim())}</button>`).join("");
  $("gen-prompts").onclick = (e) => {
    const i = [...$("gen-prompts").children].indexOf(e.target);
    if (i >= 0) $("gen-prompt").value = PROMPTS[i];
  };
  $("gen-temp").oninput = () => ($("gen-temp-value").textContent = $("gen-temp").value);
  $("gen-btn").onclick = run;
  $("gen-colour").onchange = () => data && renderText();
  $("step-prev").onclick = () => selected !== null && select(selected - 1);
  $("step-next").onclick = () => selected !== null && select(selected + 1);
  $("step-play").onclick = togglePlay;

  $("gen-text").addEventListener("click", (e) => {
    const span = e.target.closest(".ch");
    if (span) { stop(); select(+span.dataset.i); }
  });
  $("gen-text").addEventListener("mousemove", (e) => {
    const span = e.target.closest(".ch");
    if (!span) return hideTip();
    const step = data.steps[+span.dataset.i];
    showTip(`“${esc(showChar(data.text[+span.dataset.i]))}” had probability <b>${(100 * step.p).toFixed(1)}%</b><br>
             top guess: “${esc(showChar(step.top[0][0]))}” ${(100 * step.top[0][1]).toFixed(0)}%`, e);
  });
  $("gen-text").addEventListener("mouseleave", hideTip);

  document.addEventListener("keydown", (e) => {
    if ($("page-generator").hidden || !data || ["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return;
    if (e.key === "ArrowRight") { stop(); select((selected ?? -1) + 1); e.preventDefault(); }
    if (e.key === "ArrowLeft") { stop(); select((selected ?? 1) - 1); e.preventDefault(); }
    if (e.key === " ") { togglePlay(); e.preventDefault(); }
  });
  window.addEventListener("resize", () => data && selected !== null && select(selected));
}
