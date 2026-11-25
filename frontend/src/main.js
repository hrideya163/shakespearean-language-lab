// Entry point: tab switching + loading which models are available.
import "./style.css";
import { $ } from "./utils.js";
import { setupGenerator } from "./generator.js";
import { setupTranslator } from "./translator.js";

function showTab(name) {
  for (const tab of ["generator", "translator"]) {
    $(`tab-${tab}`).classList.toggle("active", tab === name);
    $(`panel-${tab}`).hidden = tab !== name;
  }
}

async function start() {
  $("tab-generator").addEventListener("click", () => showTab("generator"));
  $("tab-translator").addEventListener("click", () => showTab("translator"));
  setupTranslator();
  try {
    const status = await fetch("/api/status").then((r) => r.json());
    setupGenerator(status.generators);
    const notes = [];
    if (status.generators.length === 0) {
      notes.push("No generator trained yet: run python train_generator.py");
    }
    if (status.translator !== "lstm") {
      notes.push("No translator trained yet (using the dictionary baseline): run python train_translator.py");
    }
    $("model-status").textContent = notes.join(" · ");
  } catch {
    setupGenerator([]);
    $("model-status").textContent = "Backend not reachable. Start it with: uvicorn server:app --port 8000";
  }
}

start();
