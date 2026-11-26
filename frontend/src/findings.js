// The study's findings, shown on the Research page. Numbers come from RESULTS.md
// (which is written from the experiment reports in results/). Update both together.

export const HEADLINES = [
  ["0.001%", "of the gradient survives 50 characters back in a plain RNN (vanishing gradient)"],
  ["2.17 bpc", "best LSTM after 9,000 steps: it caught up with the GRU, the RNN stayed stuck"],
  ["14.6% → 0.6%", "of translations stuck in a loop: greedy decoding vs our beam search, same model"],
  ["90.5%", "of negation flips: the translator prefers the correct meaning when shown both"],
];

export const QUESTIONS = [
  {
    tag: "RQ1",
    title: "Why LSTM? RNN vs GRU vs LSTM",
    claim: "We measured the vanishing gradient directly. Gated cells keep learning; the plain RNN gets stuck.",
    points: [
      "50 characters back the plain RNN keeps <b>0.001%</b> of its gradient; the GRU about <b>0.3%</b>.",
      "Same number of weights: the GRU learns fastest (test <b>2.16</b> bits per character after 3k steps).",
      "Trained 3× longer, the LSTM catches up with the GRU (<b>2.166</b> vs <b>2.162</b>); the RNN stops improving at ~2k steps.",
      "Gradient flow is learned: after longer training every model passes back much more gradient.",
    ],
    figures: [
      ["rq1_long_budget.png", "Long training budget"],
      ["rq1_gradient_flow.png", "Gradient flow"],
      ["rq1_learning_curves.png", "Learning curves"],
      ["rq1_memory_curve.png", "Memory curve"],
    ],
    report: "rq1_compare_cells.md",
  },
  {
    tag: "RQ2",
    title: "What do individual LSTM cells learn?",
    claim: "Some single cells have readable, causal jobs, and they keep those jobs as training continues.",
    points: [
      "Cell <b>#53</b> tracks the position in the line. Switching it off hurts line-end predictions 13× more than a random cell.",
      "Cell <b>#77</b> tracks capital letters, and switching it off matters (5× a random cell).",
      "Correlation ≠ causation: the strongest end-of-line cell (r = 0.43) changes nothing when removed.",
      "The forget gate clears memory most after sentence ends (0.58 vs 0.71 on letters).",
    ],
    figures: [
      ["rq2_gate_saturation.png", "Gate saturation"],
      ["rq2_embeddings.png", "Character embeddings"],
    ],
    report: "rq2_inside_lstm.md",
  },
  {
    tag: "RQ3",
    title: "Temperature and generation quality",
    claim: "Temperature trades correctness for diversity; around T = 0.8 the text statistics match real Shakespeare best.",
    points: [
      "T = 0.2: <b>99%</b> real words but <b>26%</b> repeated phrases.",
      "T = 0.8: diversity, repetition and style score all closest to real Shakespeare.",
      "The model uses archaic words at less than half the real rate (<b>2.6%</b> vs <b>6.6%</b>).",
    ],
    figures: [["rq3_temperature.png", "Temperature sweep"]],
    report: "rq3_temperature.md",
  },
  {
    tag: "RQ4",
    title: "Copy gate, extended vocabulary and decoding",
    claim: "The copy gate is the biggest improvement and also a faithful explanation of every output word.",
    points: [
      "Copy gate: <b>+2.2 BLEU</b> and <b>75%</b> vs <b>64%</b> of names, numbers and negations kept.",
      "Copy probability: <b>0.47</b> for kept words vs <b>0.03</b> for swapped words (thou → you).",
      "Attention points at the replaced word <b>68%</b> of the time (chance ≈ 9%).",
      "Decoding matters: greedy decoding loops on <b>14.6%</b> of sentences, the final beam search on <b>0.6%</b>; the coverage penalty adds <b>+0.8 BLEU</b>.",
      "Tried and rejected: the pointer-generator extended vocabulary kept <i>fewer</i> names (61% vs 70%); dictionary hints and clause splitting did not help either.",
    ],
    figures: [
      ["decoding.png", "Decoding study"],
      ["rq4_calibration.png", "Calibration"],
      ["rq4_overfitting.png", "Overfitting & dropout"],
    ],
    report: "rq4_copy_gate.md",
  },
  {
    tag: "RQ5",
    title: "Can we detect meaning drift?",
    claim: "Comparing two versions of a sentence is a strong signal; an absolute score is not.",
    points: [
      "Shown both versions, the model prefers the correct negation <b>90.5%</b> of the time.",
      "As a stand-alone detector its AUROC is only <b>0.61</b>: sentence difficulty dominates.",
      "The rule-based meaning guard catches <b>88%</b> of name swaps, at a 17% false-alarm cost.",
    ],
    figures: [],
    report: "rq5_drift.md",
  },
];
