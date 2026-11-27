import { esc, hideTip, pct, showTip } from "../utils.js";

const TYPE_FILL = { kept: "#f3f1ec", swapped: "#e3eefb", reworded: "#fde9e0", added: "#e0f4ec" };
const TYPE_LINE = { kept: "#d6d1c6", swapped: "#2a78d6", reworded: "#eb6834", added: "#1baf7a" };
const GAP = 8;
const SRC_Y = 10;
const OUT_Y = 170;
const CHIP_H = 26;

let measureCtx = null;
function textWidth(text) {
  measureCtx ??= document.createElement("canvas").getContext("2d");
  measureCtx.font = "13px Segoe UI, system-ui, sans-serif";
  return measureCtx.measureText(text).width + 16;
}

function layout(words, total) {
  const widths = words.map(textWidth);
  const rowWidth = widths.reduce((a, b) => a + b, 0) + GAP * (words.length - 1);
  let x = (total - rowWidth) / 2;
  return widths.map((w) => {
    const box = { x, w, cx: x + w / 2 };
    x += w + GAP;
    return box;
  });
}

export function outputTypes(items) {
  return items.filter((it) => it.type !== "removed").flatMap((it) => it.out.map(() => it.type));
}

export function drawAlignment(target, line) {
  const src = line.source_tokens;
  const out = line.output_tokens;
  const types = outputTypes(line.items);
  const total = Math.max(
    ...[src, out].map((ws) => ws.reduce((a, w) => a + textWidth(w), 0) + GAP * (ws.length - 1)),
  ) + 20;
  const S = layout(src, total);
  const O = layout(out, total);

  let curves = "";
  line.attention.forEach((row, i) => {
    row.forEach((a, j) => {
      if (a < 0.05) return;
      const x1 = O[i].cx, y1 = OUT_Y, x2 = S[j].cx, y2 = SRC_Y + CHIP_H;
      curves += `<path d="M${x1},${y1} C${x1},${y1 - 60} ${x2},${y2 + 60} ${x2},${y2}"
        data-o="${i}" data-s="${j}" fill="none" stroke="#2a78d6" stroke-width="${(1 + 4 * a).toFixed(1)}"
        stroke-opacity="${Math.max(0.12, a).toFixed(2)}" stroke-linecap="round"/>`;
    });
  });

  const chip = (b, word, y, fill, stroke, attr) =>
    `<g ${attr} style="cursor:pointer"><rect x="${b.x}" y="${y}" width="${b.w}" height="${CHIP_H}" rx="6" fill="${fill}" stroke="${stroke}"/>
     <text x="${b.cx}" y="${y + 17}" text-anchor="middle" fill="#1f1d1a">${esc(word)}</text></g>`;
  const srcChips = src.map((w, j) => chip(S[j], w, SRC_Y, "#fffefb", "#d6d1c6", `data-s="${j}"`)).join("");
  const outChips = out
    .map((w, i) => {
      const t = types[i] ?? "kept";
      const copy = line.copy_probs[i];
      const bar = `<rect x="${O[i].x}" y="${OUT_Y + CHIP_H + 5}" width="${O[i].w * copy}" height="6" fill="#2a78d6" rx="2"/>
                   <rect x="${O[i].x + O[i].w * copy}" y="${OUT_Y + CHIP_H + 5}" width="${O[i].w * (1 - copy)}" height="6" fill="#eb6834" rx="2"/>`;
      return chip(O[i], w, OUT_Y, TYPE_FILL[t], TYPE_LINE[t], `data-o="${i}"`) + bar;
    })
    .join("");

  const height = OUT_Y + CHIP_H + 20;
  target.innerHTML = `
    <div class="align-scroll"><svg class="align-svg" width="${total}" height="${height}">
      ${curves}${srcChips}${outChips}</svg></div>
    <div class="legend" style="margin-top:6px">
      <span><span class="sw" style="background:#2a78d6"></span>copied from the input</span>
      <span><span class="sw" style="background:#eb6834"></span>generated from the vocabulary</span>
      <span class="muted">· curves = attention · hover a word for details</span>
    </div>`;

  const svg = target.querySelector("svg");
  const paths = [...svg.querySelectorAll("path")];
  const focus = (attr, idx) => paths.forEach((p) => (p.style.opacity = p.dataset[attr] === String(idx) ? 1 : 0.06));
  const reset = () => paths.forEach((p) => (p.style.opacity = ""));
  svg.querySelectorAll("g[data-o]").forEach((g) => {
    const i = +g.dataset.o;
    g.onmouseenter = () => focus("o", i);
    g.onmousemove = (e) => {
      const row = line.attention[i];
      const j = row.indexOf(Math.max(...row));
      showTip(`<b>${esc(out[i])}</b> (${types[i] ?? "kept"})<br>
        copied ${pct(line.copy_probs[i])} · generated ${pct(1 - line.copy_probs[i])}<br>
        confidence ${pct(line.confidences[i])}<br>
        looked mostly at “${esc(src[j])}” (${pct(row[j])})`, e);
    };
    g.onmouseleave = () => { reset(); hideTip(); };
  });
  svg.querySelectorAll("g[data-s]").forEach((g) => {
    const j = +g.dataset.s;
    g.onmouseenter = () => focus("s", j);
    g.onmousemove = (e) => showTip(`input word <b>${esc(src[j])}</b><br>curves show which output words looked at it`, e);
    g.onmouseleave = () => { reset(); hideTip(); };
  });
}
