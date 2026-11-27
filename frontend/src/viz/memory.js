import { diverging, hideTip, showChar, showTip } from "../utils.js";

const HEIGHT = 256;
let cache = { memory: null, width: 0, image: null, order: null };

function sortedUnits(memory) {
  const units = memory[0].length;
  const mean = (u) => memory.reduce((a, row) => a + row[u], 0) / memory.length;
  const means = Array.from({ length: units }, (_, u) => mean(u));
  return means.map((_, u) => u).sort((a, b) => means[a] - means[b]);
}

function renderBase(memory, order, width, ratio) {
  const steps = memory.length;
  const units = memory[0].length;
  const off = document.createElement("canvas");
  off.width = width * ratio;
  off.height = HEIGHT * ratio;
  const ctx = off.getContext("2d");
  ctx.scale(ratio, ratio);
  const cw = width / steps;
  const ch = HEIGHT / units;
  for (let t = 0; t < steps; t++) {
    for (let row = 0; row < units; row++) {
      ctx.fillStyle = diverging(memory[t][order[row]]);
      ctx.fillRect(t * cw, row * ch, Math.ceil(cw), Math.ceil(ch));
    }
  }
  return off;
}

export function drawMemory(canvas, memory, text, selected, onSelect) {
  const steps = memory.length;
  const units = memory[0].length;
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  if (cache.memory !== memory || cache.width !== width) {
    const order = sortedUnits(memory);
    cache = { memory, width, order, image: renderBase(memory, order, width, ratio) };
  }
  canvas.width = width * ratio;
  canvas.height = HEIGHT * ratio;
  canvas.style.height = `${HEIGHT}px`;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(cache.image, 0, 0);
  if (selected !== null) {
    const cw = width / steps;
    ctx.scale(ratio, ratio);
    ctx.strokeStyle = "#1f1d1a";
    ctx.lineWidth = 2;
    ctx.strokeRect(selected * cw - 1, 1, Math.max(cw, 2) + 2, HEIGHT - 2);
  }

  const position = (e) => {
    const r = canvas.getBoundingClientRect();
    return [Math.floor(((e.clientX - r.left) / r.width) * steps), Math.floor(((e.clientY - r.top) / r.height) * units)];
  };
  canvas.onmousemove = (e) => {
    const [t, row] = position(e);
    if (t < 0 || t >= steps || row < 0 || row >= units) return hideTip();
    const u = cache.order[row];
    showTip(`character ${t + 1} “${showChar(text[t])}” · unit #${u}<br>value <b>${memory[t][u].toFixed(2)}</b>`, e);
  };
  canvas.onmouseleave = hideTip;
  canvas.onclick = (e) => onSelect(Math.max(0, Math.min(steps - 1, position(e)[0])));
}
