import { $, esc, hideTip, pct, postJSON, segmented, sequential, showTip } from "../utils.js";
import { drawAlignment } from "../viz/alignment.js";

const LOW_CONFIDENCE = 0.4;

const EXAMPLES = {
  to_modern: [
    "Wherefore art thou Romeo?",
    "I know not why I am so sad.",
    "Thou hast betrayed me, and I will not forgive thee.",
    "'Tis but thy name that is my enemy.",
    "What say you, my lord? Shall we go hence?",
  ],
  to_shakes: [
    "Why are you here? I don't know what you want from me.",
    "Do you love him? Yes, more than my own life.",
    "Please, my lord, don't leave me here alone.",
    "Your father is a good man, and I trust him.",
  ],
};

const TYPES = {
  kept: ["kept", "#f3f1ec", "#d6d1c6"],
  swapped: ["archaic ↔ modern swap", "#e3eefb", "#2a78d6"],
  reworded: ["reworded", "#fde9e0", "#eb6834"],
  added: ["added", "#e0f4ec", "#1baf7a"],
};

let getDirection, getDecoding;

function wordMap(line) {
  let k = 0;
  return line.items
    .map((item) => {
      if (item.type === "removed") return `<span class="w removed" title="removed">${esc(item.src)}</span>`;
      return item.out
        .map((word) => {
          const conf = line.confidences ? line.confidences[k] : null;
          k++;
          let tip = TYPES[item.type][0];
          if (item.src && item.type !== "kept") tip = `${item.src} → ${item.out.join(" ")} (${tip})`;
          if (conf !== null) tip += ` · confidence ${pct(conf)}`;
          const unsure = conf !== null && conf < LOW_CONFIDENCE ? " unsure" : "";
          return `<span class="w ${item.type}${unsure}" data-tip="${esc(tip)}">${esc(word)}</span>`;
        })
        .join("");
    })
    .join("");
}

function heatmap(line) {
  const head = line.source_tokens.map((w) => `<th class="col">${esc(w)}</th>`).join("");
  const rows = line.output_tokens
    .map((word, i) => {
      const cells = line.attention[i]
        .map((a, j) => `<td style="background:${sequential(Math.sqrt(a))}" data-tip="${esc(word)} → ${esc(line.source_tokens[j])}: ${pct(a)}"></td>`)
        .join("");
      return `<tr><th style="text-align:right">${esc(word)}</th>${cells}</tr>`;
    })
    .join("");
  return `<div class="align-scroll"><table class="heatmap"><tr><th></th>${head}</tr>${rows}</table></div>`;
}

function beams(line) {
  const rows = line.alternatives.flatMap((piece) =>
    piece.options.map((o, r) => `<div class="beam${r === 0 ? " best" : ""}">
        <span class="rank">${r + 1}</span><span>${esc(o.text)}</span>
        <span class="score" title="average log-probability per word, with the coverage penalty">${o.score.toFixed(3)}</span></div>`),
  );
  return rows.join("");
}

function renderLine(line, i, decoding) {
  const counts = Object.entries(line.counts).filter(([, n]) => n > 0).map(([k, n]) => `${n} ${k}`).join(" · ");
  const warnings = line.warnings.map((w) => `<p class="warning">⚠ Meaning guard: ${esc(w)}</p>`).join("");
  const unsure = line.confidences ? line.confidences.filter((c) => c < LOW_CONFIDENCE).length : 0;
  const explain = line.attention
    ? `<div class="explain">
        <div><h4>Alignment: where the model looked</h4>
          <p class="sub">Each output word (bottom) is linked to the input words (top) its attention focused on.
            The bar under each word is the copy gate.</p>
          <div id="align-${i}"></div></div>
        ${line.alternatives?.length ? `<div><h4>Beam search: the top candidates</h4>
          <p class="sub">${decoding === "beam" ? "The decoder kept the 5 best partial translations; these are the best finished ones. Higher score = more likely."
            : "Greedy decoding keeps only one candidate. Switch to beam search to compare."}</p>
          <div class="beams">${beams(line)}</div></div>` : ""}
        <details><summary class="muted" style="cursor:pointer">Attention as a heat map</summary>
          <div style="margin-top:10px">${heatmap(line)}</div></details>
      </div>`
    : `<p class="muted">The dictionary baseline has no attention to show. Train the LSTM translator first.</p>`;
  return `<article class="card line-card">
      <p class="original">${esc(line.original)}</p>
      <p class="translation">${esc(line.text)}</p>
      <div class="wordmap">${wordMap(line)}</div>
      <p class="counts">${counts}${unsure ? ` · ${unsure} word${unsure > 1 ? "s" : ""} below ${pct(LOW_CONFIDENCE)} confidence (dotted)` : ""}</p>
      ${warnings}${explain}
    </article>`;
}

function renderSide(data) {
  const glossary = Object.entries(data.glossary).sort()
    .map(([a, m]) => `<tr><td><i>${esc(a)}</i></td><td>${esc(m)}</td></tr>`).join("");
  $("tr-side").innerHTML = `
    <div class="card side-card"><span class="muted">DIFFICULTY</span>
      <span class="big">${data.difficulty}</span>
      <span class="muted">${pct(data.archaic_ratio)} archaic words</span></div>
    ${glossary ? `<div class="card"><h3 style="margin-bottom:8px">Glossary</h3>
      <table class="glossary"><tr><th>archaic</th><th>modern</th></tr>${glossary}</table></div>` : ""}
    <div class="card"><h3 style="margin-bottom:8px">How it decides</h3>
      <p class="muted">1. A bidirectional LSTM reads the whole sentence.<br>
      2. For each output word, <b>attention</b> scores every input word.<br>
      3. The <b>copy gate</b> mixes copying an input word with writing a new one.<br>
      4. <b>Beam search</b> keeps 5 candidates, blocks repeated words and prefers candidates
      that covered every input word.</p></div>`;
}

async function translate() {
  const button = $("tr-btn");
  button.disabled = true;
  button.textContent = "Translating…";
  try {
    const decoding = getDecoding();
    const data = await postJSON("/api/translate", { text: $("tr-input").value, direction: getDirection(), decoding });
    $("tr-results").innerHTML = data.lines.length
      ? data.lines.map((l, i) => renderLine(l, i, decoding)).join("")
      : `<p class="empty">Type a sentence to translate.</p>`;
    data.lines.forEach((l, i) => l.attention && drawAlignment($(`align-${i}`), l));
    renderSide(data);
  } catch (err) {
    $("tr-results").innerHTML = `<p class="error">${esc(err.message)}</p>`;
  } finally {
    button.disabled = false;
    button.textContent = "Translate";
  }
}

function showExamples() {
  const list = EXAMPLES[getDirection()];
  $("tr-examples").innerHTML = list.map((e) => `<button class="chip">${esc(e)}</button>`).join("");
}

export function setupTranslator() {
  getDirection = segmented("tr-direction", () => {
    showExamples();
    $("tr-input").value = EXAMPLES[getDirection()].slice(0, 2).join("\n");
  });
  getDecoding = segmented("tr-decoding", () => $("tr-results").children.length && translate());
  $("tr-input").value = EXAMPLES.to_modern.slice(0, 3).join("\n");
  showExamples();
  $("tr-examples").onclick = (e) => {
    if (!e.target.classList.contains("chip")) return;
    $("tr-input").value = e.target.textContent;
    translate();
  };
  $("tr-btn").onclick = translate;
  $("tr-legend").innerHTML =
    Object.values(TYPES).map(([label, fill, line]) =>
      `<span><span class="sw" style="background:${fill};box-shadow:inset 0 -2px 0 ${line}"></span>${label}</span>`).join("") +
    `<span><span class="w removed" style="padding:0">removed</span></span>
     <span><span class="w unsure" style="padding:0;background:none;border:0">unsure</span></span>`;

  $("tr-results").addEventListener("mousemove", (e) => {
    const el = e.target.closest("[data-tip]");
    if (el) showTip(el.dataset.tip, e);
    else if (!e.target.closest("svg")) hideTip();
  });
  $("tr-results").addEventListener("mouseleave", hideTip);
}
