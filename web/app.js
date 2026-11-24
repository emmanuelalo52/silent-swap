// Participant interface. One question at a time, three steps per question:
//   own       the participant answers alone first (so the assistant cannot anchor them)
//   ai        the assistant's suggestion and confidence appear; the participant commits a final answer
//   feedback  the correct answer is revealed and the running scores update
// In the "notice" condition a one-time update notice appears at the swap, and a build label sits
// beside every assistant answer. The other conditions show neither.

const params = new URLSearchParams(location.search);
const pid = params.get("pid") || "P00";
const condition = params.get("c") || "silent";
const demo = params.get("demo") === "1";
const LETTERS = ["A", "B", "C", "D"];

const state = {
  session: null, idx: 0, stage: "own", initial: null, final: null, feedback: null,
  shownAt: 0, aiShownAt: 0, msToInitial: null, noticeSeen: false,
  you: { right: 0, n: 0 }, ai: { right: 0, n: 0 }, bySubject: {},
};

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const item = () => state.session.items[state.idx];
const swapped = () => state.session.notice && item().n >= state.session.swap_at;

async function post(body) {
  const r = await fetch("/api/answer", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body) });
  if (!r.ok) throw new Error((await r.json()).error);
  return r.json();
}

function tally(it, fb) {
  state.you.n += 1; state.you.right += fb.you_correct ? 1 : 0;
  state.ai.n += 1; state.ai.right += fb.ai_correct ? 1 : 0;
  const s = (state.bySubject[it.subject] ||= { right: 0, n: 0 });
  s.n += 1; s.right += fb.ai_correct ? 1 : 0;
}

function renderHeader() {
  const s = state.session;
  $("progress").textContent = state.stage === "done" ? "Finished" : `Question ${item().n} of ${s.n_items}`;
  const chips = [`<span class="chip">You <b>${state.you.right}/${state.you.n}</b></span>`,
    `<span class="chip">Assistant <b>${state.ai.right}/${state.ai.n}</b></span>`];
  for (const subj of s.subjects) {
    const r = state.bySubject[subj] || { right: 0, n: 0 };
    chips.push(`<span class="chip">Assistant on ${esc(subj)}: <b>${r.right}/${r.n}</b></span>`);
  }
  $("record").innerHTML = chips.join("");
  const build = $("build");
  build.hidden = !(state.session.notice && state.stage !== "done" && state.noticeSeen && swapped());
  if (!build.hidden) build.textContent = `Updated at question ${s.swap_at}: answers now come from the ${item().build}.`;
}

function optionRow(letter, text) {
  const it = item();
  const classes = ["opt"], tags = [];
  if (state.stage === "own" && state.initial === letter) classes.push("selected");
  if (state.stage === "ai") {
    if (state.final === letter) classes.push("selected");
    if (it.ai.letter === letter) { classes.push("ai-pick"); tags.push("assistant's suggestion"); }
    if (state.initial === letter) tags.push("your first answer");
  }
  if (state.stage === "feedback") {
    const fb = state.feedback;
    if (fb.answer === letter) { classes.push("correct"); tags.push("✓ correct answer"); }
    else if (state.final === letter) classes.push("wrong");
    if (state.final === letter) tags.push("your final answer");
    if (fb.ai === letter) tags.push("assistant");
  }
  const disabled = state.stage === "feedback" ? "disabled" : "";
  return `<button class="${classes.join(" ")}" data-letter="${letter}" ${disabled}>
    <span class="letter">${letter}</span><span>${esc(text)}</span>
    <span class="tag">${tags.join(" · ")}</span></button>`;
}

function aiPanel() {
  const it = item();
  const pct = Math.round(it.ai.confidence * 100);
  // provenance beside the answer, notice condition only
  const prov = state.session.notice ? `<div class="prov">Answered by the ${esc(it.build)}</div>` : "";
  return `<div class="ai"><div class="says">Assistant suggests <b>${it.ai.letter}</b></div>
    <div class="conf"><span>Confidence</span><span class="bar"><span style="width:${pct}%"></span></span>
    <span>${pct}%</span></div>${prov}</div>`;
}

function render() {
  renderHeader();
  if (state.stage === "done") return renderDone();
  const it = item();
  const steps = {
    own: `<p class="step"><b>Step 1 of 2.</b> Pick your own answer first.</p>`,
    ai: `<p class="step"><b>Step 2 of 2.</b> Here is the assistant's suggestion. Choose your final answer; you can keep yours or change it.</p>`,
    feedback: "",
  };
  let html = `<div class="card"><div class="subject">${esc(it.subject)}</div>
    <div class="question">${esc(it.question)}</div>${steps[state.stage]}
    <div class="options">${it.choices.map((c, i) => optionRow(LETTERS[i], c)).join("")}</div>`;
  if (state.stage !== "own") html += aiPanel();
  if (state.stage === "feedback") {
    const fb = state.feedback;
    html += `<div class="result">You were <span class="${fb.you_correct ? "ok" : "no"}">${fb.you_correct ? "right" : "wrong"}</span>.
      The assistant was <span class="${fb.ai_correct ? "ok" : "no"}">${fb.ai_correct ? "right" : "wrong"}</span>.
      The correct answer is <b>${fb.answer}</b>.</div>`;
  }
  const label = { own: "Lock in and see the assistant", ai: "Submit final answer",
    feedback: state.idx + 1 < state.session.n_items ? "Next question" : "Finish" }[state.stage];
  const ready = state.stage === "own" ? state.initial : state.stage === "ai" ? state.final : true;
  html += `<div class="actions"><button class="btn" id="go" ${ready ? "" : "disabled"}>${label}</button></div></div>`;
  $("main").innerHTML = html;

  document.querySelectorAll(".opt:not(:disabled)").forEach((b) => b.addEventListener("click", () => {
    if (state.stage === "own") state.initial = b.dataset.letter;
    else if (state.stage === "ai") state.final = b.dataset.letter;
    render();
  }));
  $("go").addEventListener("click", advance);
}

async function advance() {
  const it = item();
  if (state.stage === "own") {
    state.msToInitial = Date.now() - state.shownAt;
    state.final = state.initial;
    state.stage = "ai"; state.aiShownAt = Date.now();
  } else if (state.stage === "ai") {
    state.feedback = await post({ pid, condition, demo, n: it.n, initial: state.initial, final: state.final,
      ms_to_initial: state.msToInitial, ms_to_final: Date.now() - state.aiShownAt });
    tally(it, state.feedback);
    state.stage = "feedback";
  } else {
    state.idx += 1;
    if (state.idx >= state.session.n_items) { state.stage = "done"; return render(); }
    Object.assign(state, { stage: "own", initial: null, final: null, feedback: null, shownAt: Date.now() });
    if (state.session.notice && item().n === state.session.swap_at && !state.noticeSeen) return showNotice();
  }
  render();
}

function showNotice() {
  const n = state.session.notice;
  const ov = $("overlay");
  render(); // draw the question behind the dialog
  ov.innerHTML = `<div class="notice" role="dialog" aria-modal="true" aria-labelledby="nt">
    <h2 id="nt">${esc(n.title)}</h2><dl>
    <dt>What changed</dt><dd>${esc(n.changed)}</dd>
    <dt>What stayed the same</dt><dd>${esc(n.unchanged)}</dd>
    <dt>What to watch</dt><dd>${esc(n.watch)}</dd></dl>
    <div class="actions"><button class="btn" id="ack">Got it, continue</button></div></div>`;
  ov.hidden = false;
  renderHeader();
  $("ack").addEventListener("click", () => {
    ov.hidden = true; state.noticeSeen = true; state.shownAt = Date.now(); render();
  });
}

function renderDone() {
  const s = state.session;
  const debrief = {
    control: "The assistant was the same model and build for all questions.",
    silent: `From question ${s.swap_at} the assistant was switched to a cheaper, compressed build of the same model. We did not tell you at the time because the study looks at whether people notice such changes. Its overall accuracy was about the same, but some individual answers changed.`,
    notice: `From question ${s.swap_at} the assistant ran a cheaper, compressed build of the same model, as the update notice said.`,
  }[s.condition];
  $("main").innerHTML = `<div class="card done"><h2>Thank you, you have finished.</h2>
    <p>You got <b>${state.you.right} of ${state.you.n}</b> right. The assistant's suggestions were right
    <b>${state.ai.right} of ${state.ai.n}</b> times.</p>
    <div class="debrief"><b>About this study.</b> ${debrief}</div></div>`;
}

// Demo jump (storyboards): replay earlier items as if the participant followed the assistant,
// then open item `jump` at `stage`, with `own`/`final` answers preselected. Never logged.
async function demoJump() {
  const jump = parseInt(params.get("jump"), 10);
  for (let i = 0; i < jump - 1; i++) {
    state.idx = i;
    const it = item();
    const fb = await post({ pid, condition, demo: true, n: it.n, initial: it.ai.letter, final: it.ai.letter });
    tally(it, fb);
  }
  state.idx = jump - 1;
  state.noticeSeen = item().n > state.session.swap_at || params.get("stage") !== "notice";
  const stage = params.get("stage") || "own";
  if (stage === "notice") { state.noticeSeen = false; return showNotice(); }
  state.initial = params.get("own");
  if (stage === "ai" || stage === "feedback") { state.stage = "ai"; state.final = params.get("final") || state.initial; }
  if (stage === "feedback") {
    const it = item();
    state.feedback = await post({ pid, condition, demo: true, n: it.n, initial: state.initial, final: state.final });
    tally(it, state.feedback);
    state.stage = "feedback";
  }
  render();
}

(async function start() {
  const r = await fetch(`/api/session?c=${encodeURIComponent(condition)}`);
  state.session = await r.json();
  if (state.session.error) { $("main").textContent = state.session.error; return; }
  state.shownAt = Date.now();
  if (demo && params.get("jump")) return demoJump();
  render();
})();
