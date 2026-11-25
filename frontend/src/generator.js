// Generator tab: calls POST /api/generate and shows the text in a "neuron viewer".
import { $, esc, postJSON } from "./utils.js";

let last = null; // the most recent response, so we can re-colour without generating again

// Colour of one character. Neuron values are -1..1: red = positive, blue = negative.
// Gate values: the darker the orange, the more the model is FORGETTING (gate lower than usual).
// The gate is scaled to this sample's own min..max so small differences are visible.
function colourFor(mode, value, min = 0, max = 1) {
  if (mode === "gate") {
    const alpha = (max - value) / Math.max(max - min, 1e-6);
    return `rgba(230, 120, 0, ${alpha.toFixed(2)})`;
  }
  const v = Math.max(-1, Math.min(1, value));
  return v > 0 ? `rgba(220, 40, 40, ${v.toFixed(2)})` : `rgba(40, 90, 220, ${(-v).toFixed(2)})`;
}

function renderText() {
  if (!last) return;
  const choice = $("neuron-select").value;
  let values = null;
  let mode = "neuron";
  if (choice === "__gate") {
    values = last.gate;
    mode = "gate";
  } else if (choice) {
    values = last.neurons[choice];
  }

  const min = values ? Math.min(...values) : 0;
  const max = values ? Math.max(...values) : 1;
  const chars = [...last.text].map((ch, i) => {
    const shown = ch === "\n" ? "↵<br>" : esc(ch);
    if (!values) return ch === "\n" ? "<br>" : shown;
    return `<span style="background:${colourFor(mode, values[i], min, max)}">${shown}</span>`;
  });
  $("gen-output").innerHTML = `<span class="prompt">${esc(last.prompt)}</span>${chars.join("")}`;
}

function fillNeuronSelect(data) {
  const options = ['<option value="">no colouring</option>'];
  if (data.gate_name) {
    options.push(`<option value="__gate">mean ${data.gate_name} gate</option>`);
  }
  for (const name of Object.keys(data.neurons)) {
    options.push(`<option value="${esc(name)}">${esc(name)}</option>`);
  }
  const previous = $("neuron-select").value;
  $("neuron-select").innerHTML = options.join("");
  if ([...$("neuron-select").options].some((o) => o.value === previous)) {
    $("neuron-select").value = previous;
  }
}

async function generateText() {
  const button = $("generate-btn");
  button.disabled = true;
  button.textContent = "Generating…";
  try {
    last = await postJSON("/api/generate", {
      prompt: $("gen-prompt").value,
      temperature: parseFloat($("gen-temperature").value),
      length: parseInt($("gen-length").value, 10),
      cell: $("gen-cell").value,
    });
    fillNeuronSelect(last);
    renderText();
    $("gen-style").textContent =
      last.style_score === null
        ? ""
        : `Shakespeare-style score: ${Math.round(last.style_score * 100)}% (from the style classifier)`;
  } catch (err) {
    $("gen-output").innerHTML = `<p class="warning">${esc(err.message)}</p>`;
  } finally {
    button.disabled = false;
    button.textContent = "Generate";
  }
}

export function setupGenerator(availableCells) {
  $("gen-cell").innerHTML = availableCells.length
    ? availableCells.map((c) => `<option value="${c}">${c.toUpperCase()}</option>`).join("")
    : '<option value="lstm">LSTM (not trained yet)</option>';
  $("gen-temperature").addEventListener("input", () => {
    $("gen-temperature-value").textContent = $("gen-temperature").value;
  });
  $("generate-btn").addEventListener("click", generateText);
  $("neuron-select").addEventListener("change", renderText);
}
