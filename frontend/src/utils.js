export const $ = (id) => document.getElementById(id);

export function esc(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

export async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `Server error ${res.status}`);
  }
  return res.json();
}

export function segmented(id, onChange) {
  const root = $(id);
  root.addEventListener("click", (e) => {
    const button = e.target.closest("button");
    if (!button) return;
    root.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b === button));
    onChange?.(button.dataset.v);
  });
  return () => root.querySelector("button.active").dataset.v;
}

const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
const rgb = (c) => `rgb(${c[0]}, ${c[1]}, ${c[2]})`;

const BLUE = hex("#2a78d6");
const RED = hex("#e34948");
const MID = hex("#f0efec");
const SEQ_LOW = hex("#f4f8fd");
const SEQ_HIGH = hex("#104281");

export function diverging(v, strength = 1) {
  const t = Math.max(-1, Math.min(1, v));
  return rgb(mix(MID, t < 0 ? BLUE : RED, Math.abs(t) * strength));
}

export function sequential(v, low = SEQ_LOW, high = SEQ_HIGH) {
  return rgb(mix(low, high, Math.max(0, Math.min(1, v))));
}
export const BLUE_TEXT = [hex("#f4f8fd"), hex("#6da7ec")];
export const ORANGE_TEXT = [hex("#fdf1ea"), hex("#f19a6f")];

export function scaleLegend(kind, left, right) {
  const stops =
    kind === "diverging"
      ? `${diverging(-1)}, ${diverging(0)}, ${diverging(1)}`
      : kind === "orange"
        ? `${sequential(0, ...ORANGE_TEXT)}, ${sequential(1, ...ORANGE_TEXT)}`
        : kind === "blue-text"
          ? `${sequential(0, ...BLUE_TEXT)}, ${sequential(1, ...BLUE_TEXT)}`
          : `${sequential(0)}, ${sequential(1)}`;
  return `<span>${left}</span><span class="bar" style="background:linear-gradient(90deg, ${stops})"></span><span>${right}</span>`;
}

export function showTip(html, event) {
  const tip = $("tooltip");
  tip.innerHTML = html;
  tip.style.display = "block";
  const x = Math.min(event.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
  const y = event.clientY + 16 + tip.offsetHeight > window.innerHeight ? event.clientY - tip.offsetHeight - 10 : event.clientY + 16;
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}
export function hideTip() {
  $("tooltip").style.display = "none";
}

export const pct = (v) => `${Math.round(v * 100)}%`;

export function showChar(ch) {
  if (ch === "\n") return "↵";
  if (ch === " ") return "␣";
  return ch;
}
