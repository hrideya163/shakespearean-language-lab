import { esc, showChar } from "../utils.js";

const COLOURS = {
  input: "var(--series-1)",
  forget: "var(--series-2)",
  output: "var(--series-3)",
  update: "var(--series-2)",
  reset: "var(--series-1)",
};
const BOX_W = 66;
const BOX_H = 78;

const line = (pts, extra = "") =>
  `<polyline points="${pts.map((p) => p.join(",")).join(" ")}" fill="none" stroke="var(--ink-2)" stroke-width="1.6" ${extra}/>`;
const arrowLine = (pts) => line(pts, 'marker-end="url(#arrow)"');
const node = (x, y, symbol) =>
  `<circle cx="${x}" cy="${y}" r="12" fill="var(--surface)" stroke="var(--ink-2)" stroke-width="1.6"/>
   <text x="${x}" y="${y + 5}" text-anchor="middle" font-size="15" fill="var(--ink)">${symbol}</text>`;
const label = (x, y, text, size = 14, anchor = "middle", fill = "var(--ink-2)", weight = 400) =>
  `<text x="${x}" y="${y}" text-anchor="${anchor}" font-size="${size}" fill="${fill}" font-weight="${weight}">${text}</text>`;

function gateBox(cx, top, name, symbol, value, colour) {
  const x = cx - BOX_W / 2;
  const filled = value === null ? 0 : value * BOX_H;
  return `
    <rect x="${x}" y="${top}" width="${BOX_W}" height="${BOX_H}" rx="8" fill="var(--surface-2)" stroke="var(--border)"/>
    <rect x="${x}" y="${top + BOX_H - filled}" width="${BOX_W}" height="${filled}" rx="8"
          fill="${colour}" opacity="0.85" style="transition: all 0.25s"/>
    <rect x="${x}" y="${top}" width="${BOX_W}" height="${BOX_H}" rx="8" fill="none" stroke="${colour}" stroke-width="1.4"/>
    ${label(cx, top + 22, symbol, 14, "middle", "var(--ink)", 600)}
    ${value === null ? "" : label(cx, top + BOX_H - 10, value.toFixed(2), 15, "middle", "var(--ink)", 700)}
    ${label(cx, top + BOX_H + 18, name, 14, "middle", "var(--ink-2)", 600)}`;
}

function frame(body, width, height) {
  return `<svg class="cell-svg" viewBox="0 0 ${width} ${height}" role="img">
    <defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="var(--ink-2)"/></marker></defs>${body}</svg>`;
}

function inputBus(gateXs, y, ch) {
  const right = Math.max(...gateXs);
  let s = line([[40, y], [right, y]]);
  for (const x of gateXs) s += arrowLine([[x, y], [x, y - 22]]);
  s += label(36, y + 4, "h(t-1)", 14, "end");
  s += arrowLine([[110, y + 36], [110, y]]);
  s += label(110, y + 52, `x(t) = “${esc(showChar(ch))}”`, 14, "middle", "var(--accent)", 600);
  return s;
}

function lstm(g, ch) {
  const top = 168;
  const cx = { forget: 170, input: 262, cand: 350, output: 448 };
  let s = "";
  s += arrowLine([[40, 50], [158, 50]]) + line([[182, 50], [288, 50]]) + arrowLine([[312, 50], [560, 50]]);
  s += label(36, 54, "c(t-1)", 14, "end") + label(566, 54, "c(t)", 14, "start");
  s += label(300, 28, "cell state: the long-term memory", 13, "middle", "var(--ink-3)");
  s += node(170, 50, "×") + node(300, 50, "+");
  s += arrowLine([[cx.forget, top], [cx.forget, 63]]);
  s += line([[cx.input, top], [cx.input, 112], [288, 112]]) + line([[cx.cand, top], [cx.cand, 112], [312, 112]]);
  s += node(300, 112, "×") + arrowLine([[300, 100], [300, 63]]);
  s += arrowLine([[505, 50], [505, 84]]);
  s += `<ellipse cx="505" cy="98" rx="24" ry="13" fill="var(--surface)" stroke="var(--ink-2)" stroke-width="1.6"/>`;
  s += label(505, 102, "tanh", 11);
  s += arrowLine([[505, 111], [505, 128]]) + node(505, 140, "×");
  s += line([[cx.output, top], [cx.output, 140], [493, 140]]);
  s += arrowLine([[517, 140], [560, 140]]) + label(566, 144, "h(t)", 14, "start");
  s += gateBox(cx.forget, top, "forget gate", "σ", g.forget, COLOURS.forget);
  s += gateBox(cx.input, top, "input gate", "σ", g.input, COLOURS.input);
  s += gateBox(cx.cand, top, "new info", "tanh", null, "var(--ink-3)");
  s += gateBox(cx.output, top, "output gate", "σ", g.output, COLOURS.output);
  s += inputBus(Object.values(cx), 290, ch);
  return frame(s, 610, 350);
}

function gru(g, ch) {
  const top = 168;
  const cx = { reset: 170, update: 290, cand: 410 };
  let s = "";
  s += arrowLine([[40, 50], [278, 50]]) + arrowLine([[302, 50], [560, 50]]);
  s += label(36, 54, "h(t-1)", 14, "end") + label(566, 54, "h(t)", 14, "start");
  s += label(300, 28, "hidden state (memory)", 13, "middle", "var(--ink-3)");
  s += node(290, 50, "⊕");
  s += label(300, 84, "mix old and new memory by the update gate", 13, "start", "var(--ink-3)");
  s += arrowLine([[cx.update, top], [cx.update, 63]]);
  s += line([[cx.reset, top], [cx.reset, 130], [cx.cand - 20, 130], [cx.cand - 20, top]], 'marker-end="url(#arrow)"');
  s += label(cx.reset + 60, 124, "reset scales the old memory", 13, "middle", "var(--ink-3)");
  s += line([[cx.cand, top], [cx.cand, 50]]);
  s += gateBox(cx.reset, top, "reset gate", "σ", g.reset, COLOURS.reset);
  s += gateBox(cx.update, top, "update gate", "σ", g.update, COLOURS.update);
  s += gateBox(cx.cand, top, "new info", "tanh", null, "var(--ink-3)");
  s += inputBus(Object.values(cx), 290, ch);
  return frame(s, 610, 350);
}

function rnn(ch) {
  const top = 168;
  let s = "";
  s += line([[40, 50], [300, 50]], 'stroke-dasharray="4 4"') + label(36, 54, "h(t-1)", 14, "end");
  s += label(170, 40, "the old memory only enters through the box below", 13, "middle", "var(--ink-3)");
  s += arrowLine([[300, top], [300, 120], [560, 120]]) + label(566, 124, "h(t)", 14, "start");
  s += gateBox(300, top, "whole memory rewritten", "tanh", null, "var(--ink-3)");
  s += inputBus([300], 290, ch);
  s += label(300, 340, "No gates: nothing decides what to keep, so old information fades quickly.", 14, "middle", "var(--ink-2)");
  return frame(s, 610, 360);
}

function caption(cell, g) {
  const p = (v) => `<b>${Math.round(v * 100)}%</b>`;
  if (cell === "lstm") {
    return `Keeps ${p(g.forget)} of its old memory, writes ${p(g.input)} of the new information
            and shows ${p(g.output)} of its memory to the next layer.`;
  }
  if (cell === "gru") {
    return `Update gate ${p(g.update)}: how much of the memory is replaced by new information.
            Reset gate ${p(g.reset)}: how much old memory is used to compute it.`;
  }
  return "";
}

export function drawCell(target, cell, gates, ch) {
  const svg = cell === "lstm" ? lstm(gates, ch) : cell === "gru" ? gru(gates, ch) : rnn(ch);
  const text = caption(cell, gates);
  target.innerHTML = svg + (text ? `<p class="muted" style="margin-top:6px">${text}</p>` : "");
}
