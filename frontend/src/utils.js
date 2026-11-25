// Small helpers shared by both tabs.

export const $ = (id) => document.getElementById(id);

// Escape text before putting it into HTML (stops user input breaking the page).
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
