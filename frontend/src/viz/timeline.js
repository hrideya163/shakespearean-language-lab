import { hideTip, showChar, showTip } from "../utils.js";

const COLOURS = { input: "#2a78d6", forget: "#eb6834", output: "#1baf7a", update: "#eb6834", reset: "#2a78d6" };
const H = 170;
const PAD = { left: 36, right: 70, top: 10, bottom: 24 };

export function drawTimeline(target, steps, gateNames, text, selected, onSelect) {
  if (!gateNames.length) {
    target.innerHTML = `<p class="empty">The plain RNN has no gates: every step rewrites its whole memory.</p>`;
    return;
  }
  const width = target.clientWidth || 800;
  const n = steps.length;
  const x = (i) => PAD.left + (i / Math.max(n - 1, 1)) * (width - PAD.left - PAD.right);
  const values = gateNames.flatMap((g) => steps.map((s) => s.gates[g]));
  const lo = Math.max(0, Math.floor(Math.min(...values) * 10) / 10);
  const hi = Math.min(1, Math.ceil(Math.max(...values) * 10) / 10);
  const y = (v) => PAD.top + (1 - (v - lo) / Math.max(hi - lo, 1e-6)) * (H - PAD.top - PAD.bottom);

  let svg = `<svg width="${width}" height="${H}" style="display:block">`;
  for (const v of [lo, (lo + hi) / 2, hi]) {
    svg += `<line x1="${PAD.left}" x2="${width - PAD.right}" y1="${y(v)}" y2="${y(v)}" stroke="#ebe7de"/>
            <text x="${PAD.left - 6}" y="${y(v) + 4}" text-anchor="end" font-size="11" fill="#8a857c">${v.toFixed(1)}</text>`;
  }
  svg += `<text x="${PAD.left}" y="${H - 6}" font-size="11" fill="#8a857c">start of text</text>
          <text x="${width - PAD.right}" y="${H - 6}" font-size="11" fill="#8a857c" text-anchor="end">character ${n}</text>`;
  for (const g of gateNames) {
    const pts = steps.map((s, i) => `${x(i).toFixed(1)},${y(s.gates[g]).toFixed(1)}`).join(" ");
    svg += `<polyline points="${pts}" fill="none" stroke="${COLOURS[g]}" stroke-width="1.5" stroke-linejoin="round"/>`;
  }
  const labels = gateNames.map((g) => ({ g, y: y(steps[n - 1].gates[g]) + 4 })).sort((a, b) => a.y - b.y);
  labels.forEach((l, i) => { if (i > 0) l.y = Math.max(l.y, labels[i - 1].y + 14); });
  for (const l of labels) {
    svg += `<text x="${width - PAD.right + 6}" y="${l.y}" font-size="12" fill="#57534c">${l.g}</text>`;
  }
  if (selected !== null) {
    svg += `<line x1="${x(selected)}" x2="${x(selected)}" y1="${PAD.top}" y2="${H - PAD.bottom}" stroke="#1f1d1a" stroke-width="1.5"/>`;
    for (const g of gateNames) {
      svg += `<circle cx="${x(selected)}" cy="${y(steps[selected].gates[g])}" r="4.5" fill="${COLOURS[g]}" stroke="#fffefb" stroke-width="2"/>`;
    }
  }
  svg += `<rect x="${PAD.left}" y="0" width="${width - PAD.left - PAD.right}" height="${H}" fill="transparent" style="cursor:crosshair"/></svg>`;
  const legend = gateNames
    .map((g) => `<span><span class="sw" style="background:${COLOURS[g]}"></span>${g} gate</span>`)
    .join("");
  target.innerHTML = `<div class="legend" style="margin-bottom:6px">${legend}</div>${svg}`;

  const hit = target.querySelector("rect:last-of-type");
  const stepAt = (e) => {
    const r = hit.getBoundingClientRect();
    return Math.max(0, Math.min(n - 1, Math.round(((e.clientX - r.left) / r.width) * (n - 1))));
  };
  hit.onmousemove = (e) => {
    const i = stepAt(e);
    const rows = gateNames.map((g) => `${g}: <b>${steps[i].gates[g].toFixed(2)}</b>`).join("<br>");
    showTip(`character ${i + 1} “${showChar(text[i])}”<br>${rows}`, e);
  };
  hit.onmouseleave = hideTip;
  hit.onclick = (e) => onSelect(stepAt(e));
}
