"use strict";
/* The page. Rules this file keeps:
 * - Text from the server (model answers, order notes, tool messages) is untrusted: it only ever goes
 *   into the DOM as text nodes, never parsed as markup.
 * - The browser does no scheduling arithmetic. Every number shown (KPIs, deltas, times) was computed
 *   by the server from the stored schedules.
 */

const CSRF = document.querySelector('meta[name="csrf-token"]').content;
const SVG_NS = "http://www.w3.org/2000/svg";
const EXAMPLES = [
  "M2 is down from 11:00 to 14:00 today. What happens to the plan?",
  "Make O-103 urgent.",
  "Which orders are late in the current plan?",
];

let state = null;
let ganttData = { live: null, draft: null };
let pollTimer = null;

// ---------------------------------------------------------------- small DOM helpers

function make(ns, tag, props, kids) {
  const node = ns ? document.createElementNS(ns, tag) : document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  node.append(...(kids || []).flat().filter((k) => k !== null && k !== undefined && k !== false));
  return node;
}
const h = (tag, props, ...kids) => make(null, tag, props, kids);
const s = (tag, props, ...kids) => make(SVG_NS, tag, props, kids);
const $ = (id) => document.getElementById(id);
const clear = (node) => node.replaceChildren();

async function api(path, options = {}) {
  const init = { ...options, headers: { ...(options.headers || {}) } };
  if (options.body !== undefined) {
    init.method = "POST";
    init.headers["Content-Type"] = "application/json";
    init.headers["X-CSRF-Token"] = CSRF;
    init.body = JSON.stringify(options.body);
  }
  const response = await fetch(path, init);
  let data = null;
  try { data = await response.json(); } catch { /* not JSON */ }
  if (!response.ok) {
    const detail = data && data.detail;
    const text = Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : detail;
    throw new Error(text || `Request failed (${response.status})`);
  }
  return data;
}

// ---------------------------------------------------------------- chat

function renderTranscript() {
  const box = $("transcript");
  const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 60;
  clear(box);
  if (!state.transcript.length) {
    box.append(h("div", { class: "msg system" }, "Describe a disruption or ask about the plan. Proposals appear on the right for you to approve."));
  }
  for (const m of state.transcript) {
    if (m.role === "system") { box.append(h("div", { class: "msg system" }, m.text)); continue; }
    const who = m.role === "user" ? "You" : m.kind === "clarify" ? "Assistant asks" : "Assistant";
    const node = h("div", { class: `msg ${m.role}` }, h("span", { class: "who" }, who), m.text);
    for (const w of m.warnings || []) node.append(h("span", { class: "warn" }, `! ${w}`));
    if (m.usage) {
      const cost = m.usage.cost_usd == null ? "" : `, $${m.usage.cost_usd.toFixed(4)}`;
      node.append(h("span", { class: "use" }, `${m.usage.steps} model calls, ${m.usage.tokens} tokens${cost}`));
    }
    box.append(node);
  }
  if (nearBottom) box.scrollTop = box.scrollHeight;
}

function setBusy(busy, progress) {
  $("send").disabled = busy || !state.model_configured;
  $("message").disabled = busy;
  const steps = progress && progress.length ? `: ${progress.join(" → ")}` : "";
  $("activity").textContent = busy ? `Working${steps}…` : "";
  $("approve").disabled = busy;
  $("reject").disabled = busy;
}

async function sendMessage(text) {
  const message = text.trim();
  if (!message) return;
  $("message").value = "";
  setBusy(true, []);
  state.transcript.push({ role: "user", text: message });
  renderTranscript();
  try {
    const { turn_id } = await api("/api/chat", { body: { message } });
    await followTurn(turn_id);
  } catch (e) {
    await refresh();
    $("activity").textContent = e.message;            // after the refresh, which would clear it
  }
}

async function followTurn(turnId) {
  for (;;) {
    const turn = await api(`/api/chat/${turnId}`);
    setBusy(turn.status === "running", turn.progress);
    if (turn.status === "done") break;
    await new Promise((resolve) => setTimeout(resolve, 600));
  }
  await refresh();
}

// ---------------------------------------------------------------- key figures

const KPIS = [
  { key: "late_orders", label: "Late orders", delta: "delta_late_orders", better: "lower" },
  { key: "total_tardiness_min", label: "Total tardiness", unit: "min", delta: "delta_total_tardiness_min", better: "lower" },
  { key: "weighted_tardiness", label: "Weighted tardiness", delta: "delta_weighted_tardiness", better: "lower" },
  { key: "all_orders_done_at", label: "All orders done", delta: "delta_makespan_min", unit_delta: "min", better: "lower", text: true },
  { key: "mean_utilization_pct", label: "Mean utilization", unit: "%", delta: "delta_mean_utilization_pct", better: null },
];

function deltaChip(value, better, unit) {
  const sign = value > 0 ? "+" : value < 0 ? "−" : "";
  const arrow = value > 0 ? "▲" : value < 0 ? "▼" : "–";
  let verdict = "same";
  if (value !== 0 && better) verdict = (value < 0) === (better === "lower") ? "better" : "worse";
  const words = { better: "better", worse: "worse", same: value === 0 ? "no change" : "change" }[verdict];
  const label = `${words}: ${sign}${Math.abs(value)}${unit ? " " + unit : ""}`;
  return h("span", { class: `delta ${verdict}`, "aria-label": label, title: label }, `${arrow} ${sign}${Math.abs(value)}${unit ? " " + unit : ""}`);
}

// A timestamp on the plant's own date reads better as just the time.
function shortStamp(value) {
  return value.slice(0, 10) === state.plant_time.slice(0, 10) ? value.slice(11) : value;
}

function renderKpis() {
  const box = $("kpis");
  clear(box);
  const live = state.kpi_live;
  const diff = state.proposal ? state.proposal.comparison.diff : null;
  if (!live) return;
  for (const k of KPIS) {
    const tile = h("div", { class: "tile" }, h("p", { class: "label" }, k.label),
      h("p", { class: k.text ? "value text" : "value" }, k.text ? shortStamp(live[k.key]) : String(live[k.key]),
        k.unit && !k.text ? h("span", { class: "unit" }, k.unit) : null));
    if (diff) {
      const after = k.text ? shortStamp(diff.kpi_after[k.key]) : diff.kpi_after[k.key];
      tile.append(h("p", { class: "draft" }, `Draft: ${after}${k.unit && !k.text ? " " + k.unit : ""}`,
        deltaChip(diff[k.delta], k.better, k.text ? k.unit_delta : k.unit)));
    }
    box.append(tile);
  }
}

// ---------------------------------------------------------------- proposal

function renderProposal() {
  const p = state.proposal;
  $("proposal").hidden = !p;
  $("proposal-error").hidden = true;
  if (!p) return;
  const diff = p.comparison.diff;
  $("proposal-title").textContent = `Proposal: draft ${p.draft_id}`;
  const changes = $("proposal-changes");
  clear(changes);
  for (const c of p.changes) changes.append(h("li", {}, c));

  const facts = $("proposal-facts");
  clear(facts);
  facts.append(h("div", {}, `${diff.moved_operation_count} operations move (${diff.machine_change_count} change machine).`));
  facts.append(h("div", {}, `Newly late: ${diff.newly_late_orders.join(", ") || "none"}. No longer late: ${diff.no_longer_late_orders.join(", ") || "none"}.`));
  const after = p.comparison.solve_after;
  facts.append(h("div", {}, `Solver: ${after.status}; tardiness ${after.tardiness_proven_optimal ? "proven optimal" : "not proven optimal"}; ` +
    `fewest moves ${after.stability_proven_optimal ? "proven" : "not proven"}.`));
  if (p.comparison.confidence_note) facts.append(h("div", {}, p.comparison.confidence_note));

  $("approve").onclick = () => decide("/api/approve", { draft_id: p.draft_id, digest: p.digest });
  $("reject").onclick = () => decide("/api/reject", { draft_id: p.draft_id });
}

async function decide(path, body) {
  setBusy(true, []);
  try {
    await api(path, { body });
  } catch (e) {
    const box = $("proposal-error");
    box.textContent = e.message;
    box.hidden = false;
  }
  await refresh(true);
}

// ---------------------------------------------------------------- Gantt chart

const hm = (stamp) => stamp.slice(11);
const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

function tickLabel(g, minute) {
  const [datePart, timePart] = g.t0.split(" ");
  const [y, mo, d] = datePart.split("-").map(Number);
  const [hh, mm] = timePart.split(":").map(Number);
  const total = hh * 60 + mm + minute;
  const day = Math.floor(total / 1440);
  const hour = Math.floor((total % 1440) / 60);
  const name = WEEKDAYS[new Date(Date.UTC(y, mo - 1, d + day)).getUTCDay()];
  return { text: `${String(hour).padStart(2, "0")}:00`, day: name, atMidnight: hour === 0 };
}

function showTip(lines, x, y) {
  const tip = $("tip");
  clear(tip);
  tip.append(h("strong", {}, lines[0]), ...lines.slice(1).map((l) => h("div", {}, l)));
  tip.hidden = false;
  const box = tip.getBoundingClientRect();
  tip.style.left = `${Math.max(8, Math.min(x + 12, window.innerWidth - box.width - 8))}px`;
  tip.style.top = `${Math.max(8, Math.min(y + 12, window.innerHeight - box.height - 8))}px`;
}
const hideTip = () => { $("tip").hidden = true; };

function opLines(op) {
  const lines = [`${op.order_id} · ${op.op_id}`, `${op.machine_id}, ${hm(op.start_at)}–${hm(op.end_at)}`, `Priority ${op.priority}, ${op.family}`];
  if (op.changed === "moved") lines.push(`Moved from ${op.was}`);
  if (op.changed === "new") lines.push("New in this draft");
  if (op.late_min > 0) lines.push(`Order ${op.order_id} finishes ${op.late_min} min late`);
  return lines;
}

function drawGantt(g, draft) {
  const left = 44, right = 14, top = 40, rowH = 34, barH = 20;
  const width = Math.max(560, ($(draft ? "draft-chart" : "live-chart").clientWidth || 800) - 26);
  const height = top + g.machines.length * rowH + 6;
  const span = g.axis_end - g.axis_start;
  const x = (m) => left + ((m - g.axis_start) / span) * (width - left - right);
  const svg = s("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "group", "aria-label": `${draft ? "Proposed" : "Live"} plan by machine` });

  svg.append(s("defs", {}, s("pattern", { id: "hatch", width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" },
    s("rect", { width: 6, height: 6, fill: "Canvas" }), s("line", { x1: 0, y1: 0, x2: 0, y2: 6, stroke: "CanvasText", "stroke-width": 2 }))));

  // hour grid and labels
  const firstHour = Math.ceil(g.axis_start / 60) * 60;
  const every = (width - left - right) / (span / 60) < 38 ? 2 : 1;
  for (let m = firstHour, n = 0; m <= g.axis_end; m += 60, n += 1) {
    if (n % every) continue;
    const t = tickLabel(g, m);
    svg.append(s("line", { class: "gridline", x1: x(m), x2: x(m), y1: top, y2: height - 6 }));
    svg.append(s("text", { class: "ticklabel", x: x(m), y: top - 8, "text-anchor": "middle" }, t.text));
    if (n === 0 || t.atMidnight) svg.append(s("text", { class: "ticklabel day", x: x(m), y: top - 21, "text-anchor": "middle" }, t.day));
  }
  svg.append(s("line", { class: "axisline", x1: left, x2: width - right, y1: top, y2: top }));

  g.machines.forEach((row, i) => {
    const y0 = top + i * rowH;
    svg.append(s("line", { class: "gridline", x1: left, x2: width - right, y1: y0 + rowH, y2: y0 + rowH }));
    svg.append(s("text", { class: "rowlabel", x: left - 8, y: y0 + rowH / 2 + 4, "text-anchor": "end" }, row.machine_id));

    let cursor = g.axis_start;                                            // shaded gaps between shifts
    for (const w of [...row.open].sort((a, b) => a.start - b.start)) {
      if (w.start > cursor) svg.append(s("rect", { class: "closed", x: x(cursor), y: y0, width: x(w.start) - x(cursor), height: rowH }));
      cursor = Math.max(cursor, w.end);
    }
    if (cursor < g.axis_end) svg.append(s("rect", { class: "closed", x: x(cursor), y: y0, width: x(g.axis_end) - x(cursor), height: rowH }));

    for (const w of row.downtime) {                                       // outages
      const wx = x(w.start), ww = x(w.end) - x(w.start);
      svg.append(s("rect", { class: "down", x: wx, y: y0 + 3, width: ww, height: rowH - 6 }));
      if (ww >= 34) svg.append(s("text", { class: "downlabel", x: wx + 4, y: y0 + 12 }, "down"));
    }

    for (const op of row.ops) {
      const bx = x(op.start) + 1, bw = Math.max(2, x(op.end) - x(op.start) - 2), by = y0 + (rowH - barH) / 2;
      const kind = op.changed ? "changed" : "base";
      const group = s("g", { class: "op", tabindex: 0, role: "img", "aria-label": opLines(op).join(". ") },
        s("rect", { class: "hit", x: bx - 1, y: y0, width: bw + 2, height: rowH }),
        s("rect", { class: `bar ${kind}`, x: bx, y: by, width: bw, height: barH, rx: 4 }));
      if (bw >= op.order_id.length * 6.6 + 10) {                            // label only if it fits
        group.append(s("text", { class: `barlabel ${kind}`, x: bx + bw / 2, y: by + barH / 2 + 4, "text-anchor": "middle" }, op.order_id));
      }
      if (op.last_of_order && op.late_min > 0) {                            // late order: marker + (in the caption) the name
        group.append(s("rect", { class: "latemark", x: bx + bw - 9, y: by - 6, width: 9, height: 9, transform: `rotate(45 ${bx + bw - 4.5} ${by - 1.5})` }));
      }
      group.addEventListener("mousemove", (e) => showTip(opLines(op), e.clientX, e.clientY));
      group.addEventListener("mouseleave", hideTip);
      group.addEventListener("focus", () => { const r = group.getBoundingClientRect(); showTip(opLines(op), r.left, r.bottom); });
      group.addEventListener("blur", hideTip);
      svg.append(group);
    }
  });

  if (g.now >= g.axis_start && g.now <= g.axis_end) {                    // the plant clock
    svg.append(s("line", { class: "nowline", x1: x(g.now), x2: x(g.now), y1: top - 4, y2: height - 6 }));
    svg.append(s("text", { class: "nowlabel", x: x(g.now) + 4, y: height - 10 }, "now"));
  }
  return svg;
}

function tableTwin(g) {
  const rows = [];
  for (const m of g.machines) for (const op of m.ops) {
    const notes = [op.changed === "moved" ? `moved from ${op.was}` : op.changed === "new" ? "new" : null,
      op.last_of_order && op.late_min > 0 ? `order ${op.late_min} min late` : null].filter(Boolean).join("; ");
    rows.push(h("tr", {}, h("td", {}, m.machine_id), h("td", {}, op.order_id), h("td", {}, op.op_id), h("td", {}, op.start_at), h("td", {}, op.end_at), h("td", {}, notes)));
  }
  const head = h("tr", {}, ...["Machine", "Order", "Operation", "Start", "End", "Note"].map((t) => h("th", { scope: "col" }, t)));
  return h("details", { class: "table" }, h("summary", {}, "Table view"),
    h("div", { class: "scroll" }, h("table", {}, h("thead", {}, head), h("tbody", {}, rows))));
}

function renderChart(container, g, draft) {
  clear(container);
  if (!g) return;
  const lateText = g.late_orders.length ? `Late: ${g.late_orders.join(", ")}` : "No late orders";
  const legend = h("ul", { class: "legend" },
    h("li", {}, h("span", { class: "swatch base" }), "Operation"),
    draft ? h("li", {}, h("span", { class: "swatch changed" }), "Moved or new vs live plan") : null,
    h("li", {}, h("span", { class: "swatch down" }), "Machine outage"),
    h("li", {}, h("span", { class: "swatch latemark" }), "Last operation of a late order"));
  const title = draft ? `Proposed plan — draft ${g.source}` : "Live plan";
  const sub = draft ? `${g.moved_count} operations moved. ${lateText}.` : `${lateText}. Plant time ${g.now_at}.`;
  container.append(h("h2", {}, title), h("p", { class: "sub" }, sub), legend, h("div", { class: "plot" }, drawGantt(g, draft)), tableTwin(g));
}

async function loadGantt(source) {
  try { return await api(`/api/gantt?source=${encodeURIComponent(source)}`); } catch { return null; }
}

async function renderCharts() {
  const board = [$("live-chart"), $("draft-chart")];
  board.forEach((c) => c.querySelector(".plot")?.classList.add("loading"));   // hold the old render, dimmed
  const [live, draft] = await Promise.all([loadGantt("committed"), state.proposal ? loadGantt(state.proposal.draft_id) : null]);
  ganttData = { live, draft };
  $("draft-chart").hidden = !draft;
  renderChart($("live-chart"), live, false);
  renderChart($("draft-chart"), draft, true);
}

// ---------------------------------------------------------------- page

function renderHeader() {
  $("meta").textContent = state.busy ? "" : `Plant time ${state.plant_time} · live plan v${state.live_version}` + (state.model ? ` · model ${state.model}` : "");
  const banner = $("banner");
  banner.hidden = state.model_configured;
  banner.textContent = state.model_configured ? "" : "Chat is disabled: set ANTHROPIC_API_KEY and ANTHROPIC_MODEL, then restart the server. You can still look at the plan.";
}

async function refresh(afterDecision = false) {
  state = await api("/api/state");
  renderTranscript();
  if (state.busy) {                                    // the agent owns the store; show progress and wait
    setBusy(true, state.progress);
    clearTimeout(pollTimer);
    pollTimer = setTimeout(() => refresh(), 800);
    return;
  }
  setBusy(false);
  renderHeader();
  renderKpis();
  renderProposal();
  await renderCharts();
  if (afterDecision && !state.proposal) $("message").focus();
}

function init() {
  const chips = $("chips");
  for (const text of EXAMPLES) chips.append(h("button", { type: "button", onclick: () => { $("message").value = text; $("message").focus(); } }, text));
  $("composer").addEventListener("submit", (e) => { e.preventDefault(); sendMessage($("message").value); });
  $("message").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage($("message").value); }
  });
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (!ganttData.live) return;
      renderChart($("live-chart"), ganttData.live, false);
      renderChart($("draft-chart"), ganttData.draft, true);
    }, 150);
  });
  refresh().catch((e) => { $("banner").hidden = false; $("banner").textContent = `Could not load: ${e.message}`; });
}

init();
