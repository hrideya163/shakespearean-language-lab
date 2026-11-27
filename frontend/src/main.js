import "./style.css";
import { $ } from "./utils.js";
import { setupGenerator } from "./pages/generator.js";
import { setupTranslator } from "./pages/translator.js";
import { setupResearch } from "./pages/research.js";

function showPage(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  for (const page of ["generator", "translator", "research"]) $(`page-${page}`).hidden = page !== name;
  window.location.hash = name;
}

function setStatus(text, ok) {
  $("status").classList.toggle("bad", !ok);
  $("status").lastElementChild.textContent = text;
}

async function start() {
  document.querySelector(".tabs").onclick = (e) => e.target.dataset.tab && showPage(e.target.dataset.tab);
  setupTranslator();
  setupResearch();
  const pageFromUrl = () => (["translator", "research"].includes(location.hash.slice(1)) ? location.hash.slice(1) : "generator");
  showPage(pageFromUrl());
  window.addEventListener("hashchange", () => showPage(pageFromUrl()));
  try {
    const status = await fetch("/api/status").then((r) => r.json());
    setupGenerator(status.generators);
    const problems = [];
    if (!status.generators.length) problems.push("no generator trained");
    if (status.translator !== "lstm") problems.push("translator: dictionary baseline");
    setStatus(problems.length ? problems.join(" · ") : `${status.generators.length} generators · LSTM translator`, !problems.length);
  } catch {
    setupGenerator([]);
    setStatus("backend offline: uvicorn server:app --port 8000", false);
  }
}

start();
