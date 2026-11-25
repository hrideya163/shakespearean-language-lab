// Translator tab: calls POST /api/translate and draws the word map.
import { $, esc } from "./utils.js";

const LOW_CONFIDENCE = 0.4; // same value as config.LOW_CONFIDENCE in Python

const EXAMPLE = `Wherefore art thou Romeo?
I know not why I am so sad.
Thou hast betrayed me, Antonio.
'Tis but thy name that is my enemy.`;

const LABELS = {
  kept: "kept",
  swapped: "archaic ↔ modern swap",
  reworded: "reworded",
  added: "added",
  removed: "removed",
};

async function translate() {
  const button = $("translate-btn");
  button.disabled = true;
  button.textContent = "Translating…";
  try {
    const res = await fetch("/api/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: $("passage").value, direction: $("direction").value }),
    });
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    render(await res.json());
  } catch (err) {
    $("results").innerHTML = `<p class="warning">${esc(err.message)}</p>`;
  } finally {
    button.disabled = false;
    button.textContent = "Translate";
  }
}

// ---------------------------------------------------------------- rendering
// Coloured word map. Each output word gets a colour for its change type;
// words the model was unsure about get a dotted red underline.
function renderWordMap(line) {
  let outIndex = 0;
  const spans = [];
  for (const item of line.items) {
    if (item.type === "removed") {
      spans.push(`<span class="w removed" title="removed">${esc(item.src)}</span>`);
      continue;
    }
    for (const word of item.out) {
      const conf = line.confidences ? line.confidences[outIndex] : null;
      let tip = LABELS[item.type];
      if (item.src && item.type !== "kept") tip = `${item.src} → ${item.out.join(" ")} (${tip})`;
      if (conf !== null) tip += ` | confidence ${Math.round(conf * 100)}%`;
      const unsure = conf !== null && conf < LOW_CONFIDENCE ? " unsure" : "";
      spans.push(`<span class="w ${item.type}${unsure}" title="${esc(tip)}">${esc(word)}</span>`);
      outIndex++;
    }
  }
  return spans.join(" ");
}

// Attention heat-map: rows = output words, columns = input words.
function renderAttention(line) {
  const head = line.source_tokens.map((w) => `<th>${esc(w)}</th>`).join("");
  const rows = line.output_tokens
    .map((word, i) => {
      const cells = line.attention[i]
        .map((a) => `<td style="background: rgba(30, 100, 220, ${a.toFixed(2)})"></td>`)
        .join("");
      return `<tr><th class="row-label">${esc(word)}</th>${cells}</tr>`;
    })
    .join("");
  return `<table class="heatmap"><tr><th></th>${head}</tr>${rows}</table>`;
}

// Copy gate: how much of each output word was copied from the input.
function renderCopyGate(line) {
  return line.output_tokens
    .map((word, i) => {
      const p = line.copy_probs[i];
      return `<div class="bar-row">
        <span class="bar-label">${esc(word)}</span>
        <div class="bar" style="width:${Math.round(p * 200)}px"></div>
        <span>${Math.round(p * 100)}% copied</span>
      </div>`;
    })
    .join("");
}

function render(data) {
  const showInside = $("show-inside").checked;

  $("results").innerHTML = data.lines
    .map((line) => {
      const counts = Object.entries(line.counts)
        .filter(([, n]) => n > 0)
        .map(([k, n]) => `${n} ${k}`)
        .join(" · ");
      const warnings = line.warnings
        .map((w) => `<p class="warning">⚠️ Meaning guard: ${esc(w)}</p>`)
        .join("");
      const inside =
        showInside && line.attention
          ? `<details><summary>Look inside the LSTM</summary>
               <h4>Attention: which input word the model looked at for each output word</h4>
               ${renderAttention(line)}
               <h4>Copy gate: did the model copy the word or generate a new one?</h4>
               ${renderCopyGate(line)}
             </details>`
          : "";
      return `<article class="line">
        <p class="original">${esc(line.original)}</p>
        <p class="wordmap">${renderWordMap(line)}</p>
        <p class="counts">${counts}</p>
        ${warnings}${inside}
      </article>`;
    })
    .join("");

  const glossaryRows = Object.entries(data.glossary)
    .sort()
    .map(([archaic, modern]) => `<tr><td>${esc(archaic)}</td><td>${esc(modern)}</td></tr>`)
    .join("");
  $("sidebar").innerHTML = `
    <div class="card">
      <h3>Difficulty</h3>
      <p class="big">${data.difficulty}</p>
      <p>${Math.round(data.archaic_ratio * 100)}% archaic words</p>
    </div>
    ${glossaryRows ? `<div class="card"><h3>Glossary</h3>
      <table class="glossary"><tr><th>archaic</th><th>modern</th></tr>${glossaryRows}</table></div>` : ""}`;
}

export function setupTranslator() {
  $("passage").value = EXAMPLE;
  $("translate-btn").addEventListener("click", translate);
}
