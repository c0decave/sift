"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const results = $("#results");
const tpl = $("#card-tpl");
const clearBtn = $("#clear-btn");

const ICONS = {
  csv: "📊", tsv: "📊", xlsx: "📊", email: "✉️", json: "🔧", jsonl: "🔧",
  yaml: "🔧", toml: "🔧", ini: "🔧", xml: "🔖", html: "🌐", pdf: "📕",
  docx: "📝", pptx: "📽️", odt: "📝", markdown: "📓", log: "📜", batch: "⚙️",
  shell: "⚙️", powershell: "⚙️", "source-code": "💻", image: "🖼️",
  archive: "🗜️", binary: "🔩", text: "📄", empty: "⬜", error: "⛔",
};

// ---- backend status ---------------------------------------------------
async function refreshHealth() {
  const dot = $("#status-dot");
  const text = $("#status-text");
  try {
    const res = await fetch("/api/health");
    // A refused request is not an answer. The Host allow-list that closes
    // DNS rebinding answers 403 with a body, and a 404 or a 503 answers with
    // none — every one of them parsed or threw, left llm_enabled undefined,
    // and the widget then asserted "heuristic mode (no LLM)": a working
    // backend deliberately without a model, when in fact the request never
    // reached one.
    if (!res.ok) {
      dot.className = "dot err";
      text.textContent = `backend refused the status request (HTTP ${res.status})`;
      return;
    }
    const h = await res.json();
    if (!h.llm_enabled) {
      dot.className = "dot warn";
      text.textContent = "heuristic mode (no LLM)";
    } else if (h.available) {
      dot.className = "dot ok";
      const model = (h.models && h.models[0]) || h.model;
      text.textContent = `${h.provider} · ${model}`;
    } else {
      dot.className = "dot off";
      text.textContent = `${h.provider || "LLM"} unreachable`;
    }
  } catch (e) {
    dot.className = "dot off";
    text.textContent = "backend offline";
  }
}

// ---- summarization ----------------------------------------------------
function pendingCard(name) {
  const node = tpl.content.firstElementChild.cloneNode(true);
  node.classList.add("pending");
  $(".card-icon", node).textContent = "⏳";
  $(".fname", node).textContent = name;
  $(".meta-line", node).innerHTML = '<span class="spinner"></span> analyzing…';
  $(".badge", node).textContent = "…";
  $(".card-body", node).style.display = "none";
  $(".card-foot", node).textContent = "";
  results.prepend(node);
  clearBtn.hidden = false;
  return node;
}

function httpReason(res) {
  if (res.status === 503) return "the server is busy — try again in a moment";
  if (res.status === 413) return "this file is larger than the server accepts";
  if (res.status === 403) return "the server refused this request";
  if (res.status === 400) return "the server could not read this upload";
  return `the server answered HTTP ${res.status}`;
}

function renderResult(node, d) {
  node.classList.remove("pending");
  // The common failure fills in a heuristic summary *and* sets error, so
  // `d.error && !d.summary` never fired and the reason the model was skipped
  // was nowhere: the footer said "heuristic (LLM error)" — the parenthetical,
  // not the cause. The CLI prints it in red and the markdown emits it.
  if (d.error && !d.summary) {
    node.classList.add("error");
    $(".card-icon", node).textContent = "⛔";
    $(".badge", node).textContent = "error";
    $(".meta-line", node).textContent = d.error;
    $(".card-body", node).style.display = "none";
    return;
  }
  $(".card-body", node).style.display = "";
  $(".card-icon", node).textContent = ICONS[d.kind] || "📄";
  $(".fname", node).textContent = d.filename || "(file)";
  const conf = Math.round((d.detection_confidence || 0) * 100);
  $(".meta-line", node).textContent =
    `${d.mime || d.kind} · ${d.size_human || ""} · ${conf}% confidence`;
  $(".badge", node).textContent = d.kind;

  setField($(".classification", node), d.classification);
  setField($(".summary", node), d.summary);

  const kp = $(".key-points", node);
  const ul = $("ul", kp);
  ul.innerHTML = "";
  if (d.key_points && d.key_points.length) {
    d.key_points.forEach(p => {
      const li = document.createElement("li");
      li.textContent = p;
      ul.appendChild(li);
    });
    kp.style.display = "";
  } else {
    kp.style.display = "none";
  }

  const ent = $(".entities", node);
  if (d.entities && d.entities.length) {
    ent.innerHTML = "<b>Entities:</b> " + d.entities.map(escapeHtml).join(", ");
    ent.style.display = "";
  } else {
    ent.style.display = "none";
  }

  const warn = $(".warnings", node);
  warn.innerHTML = "";
  const flags = (d.warnings || []).slice();
  if (d.error && d.summary) {
    // The common failure fills in a heuristic summary *and* sets error, so
    // `d.error && !d.summary` never fired and the reason the model was
    // skipped appeared nowhere: the footer said "heuristic (LLM error)" —
    // the parenthetical, not the cause. The CLI prints it in red and the
    // markdown emits it; the browser was the one surface that hid it.
    flags.unshift("the model was not used: " + d.error);
  }
  if (d.truncated) {
    // The one fact a reader must not miss: the counts above describe the part
    // that was read, not the file. It is easy to scroll past in the footer.
    flags.unshift("only part of this file was read — the figures above "
                  + "describe that part, not the whole file");
  }
  // Notes carry findings, not only housekeeping: a phrase addressed to an
  // assistant in a position that reads as a clause, a mail whose HTML states
  // a different amount than its text, an attachment that is not what its
  // name says. Run into the footer beside "context window 262144 tokens"
  // they were invisible — the same defect the terminal output had.
  const noted = (d.notes || []).filter(isAFinding);
  const housekeeping = (d.notes || []).filter(n => !isAFinding(n));

  if (flags.length || noted.length) {
    flags.forEach(w => {
      const div = document.createElement("div");
      div.className = "warn-item";
      div.textContent = "⚠ " + w;
      warn.appendChild(div);
    });
    noted.forEach(n => {
      const div = document.createElement("div");
      div.className = "note-item";
      div.textContent = "· " + n;
      warn.appendChild(div);
    });
    warn.classList.add("show");
  } else {
    warn.classList.remove("show");
  }

  $(".metadata pre", node).textContent = JSON.stringify(d.metadata || {}, null, 2);

  const engine = d.llm_used ? `${d.provider}:${d.model}`
    : (d.error ? "heuristic (LLM error)" : "heuristic");
  const notes = housekeeping.length ? " · " + housekeeping.join(" · ") : "";
  $(".card-foot", node).textContent = `⟳ ${engine} · ${d.elapsed_seconds}s${notes}`;
}

// The same test report.py uses: a note that says something about the file,
// as against one that says something about the run. Kept in step with
// _IS_A_FINDING there — a test asserts the two lists match.
const FINDING_MARKERS = [
  "noted, not flagged", "addresses an AI assistant", "is not what its name",
  "HTML version of this message", "own rendering", "not summarized",
  "marked missing", "the summary says", "stopped after", "treats as secret",
];

function isAFinding(note) {
  const lower = String(note).toLowerCase();
  return FINDING_MARKERS.some(marker => lower.includes(marker.toLowerCase()));
}

function setField(el, value) {
  if (value) { el.textContent = value; el.style.display = ""; }
  else { el.style.display = "none"; }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

async function summarizeFile(file) {
  const node = pendingCard(file.name);
  const fd = new FormData();
  fd.append("file", file, file.name);
  const vision = $("#vision-toggle").checked ? "1" : "0";
  try {
    const res = await fetch(`/api/summarize?vision=${vision}`, {
      method: "POST", body: fd,
    });
    // res.json() on a body-less refusal throws SyntaxError, which the catch
    // below rendered as the card's whole explanation: dropping more files at
    // once than the server has connection slots produced "SyntaxError:
    // Unexpected end of JSON input" instead of "the server is busy".
    let data;
    try {
      data = await res.json();
    } catch (parseError) {
      data = { error: httpReason(res), filename: file.name, kind: "error" };
    }
    if (!res.ok && !data.error) data.error = httpReason(res);
    renderResult(node, data);
  } catch (e) {
    renderResult(node, { error: String(e), filename: file.name, kind: "error" });
  }
}

function handleFiles(files) {
  [...files].forEach(summarizeFile);
}

// ---- wiring -----------------------------------------------------------
function init() {
  const dz = $("#dropzone");
  const input = $("#file-input");

  dz.addEventListener("click", () => input.click());
  dz.addEventListener("keydown", e => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
  });
  input.addEventListener("change", () => { handleFiles(input.files); input.value = ""; });

  ["dragenter", "dragover"].forEach(ev =>
    dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach(ev =>
    dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", e => {
    if (e.dataTransfer && e.dataTransfer.files) handleFiles(e.dataTransfer.files);
  });

  clearBtn.addEventListener("click", () => {
    results.innerHTML = "";
    clearBtn.hidden = true;
  });

  refreshHealth();
  setInterval(refreshHealth, 15000);
}

document.addEventListener("DOMContentLoaded", init);
