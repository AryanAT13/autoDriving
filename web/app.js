/* Dashboard: configure a run, stream its metrics, replay its policy, or drive yourself. */
(() => {
  const $ = (id) => document.getElementById(id);
  const css = (name) => getComputedStyle(document.documentElement)
    .getPropertyValue(name).trim();

  const ACTION = { MAINTAIN: 0, ACCELERATE: 1, BRAKE: 2, LANE_LEFT: 3, LANE_RIGHT: 4 };
  /* Only these three clear the all-pairs colour-separation floor on this dark surface,
     so at most three runs are compared at once and each keeps its slot while shown. */
  const SLOTS = ["--series-1", "--series-2", "--series-3"];
  const DRIVE_HZ = 8;

  const state = {
    algorithms: [], runId: null, socket: null, player: null,
    charts: {}, slots: [null, null, null], order: [],
    mode: "idle", drive: null, lastDraw: null,
  };

  const setStatus = (text, kind) => {
    $("status").textContent = text;
    $("status").dataset.state = kind ?? "";
  };

  /* ---------- controls ---------- */

  async function loadAlgorithms() {
    const data = await (await fetch("/api/algorithms")).json();
    state.algorithms = data.learners;
    $("algo").innerHTML = data.learners
      .map((a) => `<option value="${a.name}">${a.name}</option>`).join("");
    renderHyper();
  }

  const selected = () => state.algorithms.find((a) => a.name === $("algo").value);

  /* Only the settings that actually apply to the chosen algorithm are shown. */
  function relevantKeys(algo) {
    if (algo.name === "actor_critic") {
      return ["gamma", "alpha_policy", "alpha_value", "entropy_beta", "hidden_units"];
    }
    const shared = ["alpha", "gamma", "epsilon_end", "epsilon_decay_episodes"];
    return algo.name.startsWith("dyna_q")
      ? [...shared, "planning_steps", "model_capacity"] : shared;
  }

  function renderHyper() {
    const algo = selected();
    if (!algo) return;
    $("hyper").innerHTML = relevantKeys(algo).map((key) => `
      <div class="field">
        <label for="hp-${key}">${key}</label>
        <input id="hp-${key}" data-key="${key}" type="number" step="any"
               value="${algo.defaults[key]}">
      </div>`).join("");
    syncDecay();
  }

  /* An exploration schedule longer than the run never finishes decaying. */
  function syncDecay() {
    const field = $("hp-epsilon_decay_episodes");
    if (!field) return;
    field.value = Math.max(1, Math.round(Number($("episodes").value) * 0.6));
  }

  function overrides() {
    const out = {};
    for (const input of $("hyper").querySelectorAll("input[data-key]")) {
      const value = Number(input.value);
      if (Number.isFinite(value)) out[input.dataset.key] = value;
    }
    return out;
  }

  /* ---------- charts ---------- */

  function makeChart(canvasId, yLabel) {
    return new Chart($(canvasId), {
      type: "line",
      data: { datasets: [] },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false, parsing: false,
        interaction: { mode: "nearest", intersect: false },
        scales: {
          x: { type: "linear", ticks: { color: css("--faint"), maxTicksLimit: 6 },
               grid: { color: css("--border-soft") },
               title: { display: true, text: "episode", color: css("--faint") } },
          y: { ticks: { color: css("--faint"), maxTicksLimit: 6 },
               grid: { color: css("--border-soft") },
               title: { display: true, text: yLabel, color: css("--faint") } },
        },
        plugins: {
          legend: { labels: { color: css("--muted"), boxWidth: 10, font: { size: 10 } } },
        },
      },
    });
  }

  function initCharts() {
    state.charts.reward = makeChart("returnChart", "greedy return");
    state.charts.collision = makeChart("collisionChart", "collisions / episode");
  }

  /* A run keeps its colour for as long as it is on the chart. */
  function claimSlot(runId, label) {
    let index = state.slots.indexOf(null);
    if (index === -1) {
      // dropSlot needs the run still in `order` to find its dataset, so evict through it
      dropSlot(state.slots.indexOf(state.order[0]));
      index = state.slots.indexOf(null);
    }
    state.slots[index] = runId;
    state.order.push(runId);
    const colour = css(SLOTS[index]);
    for (const chart of [state.charts.reward, state.charts.collision]) {
      chart.data.datasets.push({
        label, data: [], borderColor: colour, backgroundColor: colour,
        borderWidth: 2, pointRadius: 0, tension: 0.2,
      });
      chart.update("none");
    }
    return index;
  }

  function dropSlot(index) {
    const runId = state.slots[index];
    if (runId === null) return;
    const position = datasetIndex(runId);
    for (const chart of [state.charts.reward, state.charts.collision]) {
      chart.data.datasets.splice(position, 1);
      chart.update("none");
    }
    state.slots[index] = null;
    state.order = state.order.filter((id) => id !== runId);
  }

  /* Datasets sit in the order their runs were added, which is `order`. */
  const datasetIndex = (runId) => state.order.indexOf(runId);

  function clearCharts() {
    for (const chart of [state.charts.reward, state.charts.collision]) {
      chart.data.datasets = [];
      chart.update("none");
    }
    state.slots = [null, null, null];
    state.order = [];
  }

  function pushBlocks(runId, blocks) {
    const index = datasetIndex(runId);
    if (index < 0) return;
    const series = state.charts.collision.data.datasets[index].data;
    for (const block of blocks) series.push({ x: block.episode, y: block.collision_rate });
    state.charts.collision.update("none");
  }

  function pushEvals(runId, evals) {
    const index = datasetIndex(runId);
    if (index < 0 || !evals.length) return;
    const series = state.charts.reward.data.datasets[index].data;
    for (const point of evals) series.push({ x: point.episode, y: point.reward });
    state.charts.reward.update("none");
    renderSummary(evals[evals.length - 1]);
  }

  const PERCENT = new Set(["collision_rate", "goal_rate"]);
  function renderSummary(metrics) {
    const keys = ["reward", "collision_rate", "goal_rate", "mean_speed", "steps", "red_lights"];
    $("summary").querySelector("tbody").innerHTML = keys.map((key) => {
      const value = metrics[key];
      if (value === undefined) return "";
      const shown = PERCENT.has(key) ? `${(value * 100).toFixed(1)}%` : value.toFixed(2);
      return `<tr><td>${key.replace(/_/g, " ")}</td><td>${shown}</td></tr>`;
    }).join("");
  }

  /* ---------- training ---------- */

  async function startTraining() {
    const body = {
      algo: $("algo").value,
      episodes: Number($("episodes").value),
      seed: Number($("seed").value),
      eval_every: Number($("evalEvery").value),
      eval_episodes: 20,
      overrides: overrides(),
    };
    const response = await fetch("/api/runs", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      setStatus(`error: ${detail.detail ?? response.status}`, "error");
      return;
    }
    const run = await response.json();
    state.runId = run.id;
    claimSlot(run.id, `${run.algo} · s${run.seed} · ${run.id.slice(0, 4)}`);
    $("train").disabled = true;
    $("stop").disabled = false;
    $("watchAgent").disabled = false;
    $("showPolicy").disabled = false;
    $("runLabel").textContent = `run ${run.id}`;
    setStatus(`training ${run.algo}`, "running");
    openSocket(run.id);
  }

  function openSocket(runId) {
    state.socket?.close();
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${protocol}://${location.host}/ws/runs/${runId}`);
    state.socket = socket;
    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      pushBlocks(runId, payload.blocks);
      pushEvals(runId, payload.evals);
      const { status, episode, episodes, error } = payload.summary;
      if (status === "running") {
        setStatus(`training ${episode.toLocaleString()} / ${episodes.toLocaleString()}`,
                  "running");
      } else {
        setStatus(error ? `error: ${error}`
                        : `${status} at ${episode.toLocaleString()} episodes`,
                  status === "error" ? "error" : "done");
        finishTraining();
      }
    };
    socket.onerror = () => setStatus("connection lost", "error");
  }

  function finishTraining() {
    $("train").disabled = false;
    $("stop").disabled = true;
    state.socket?.close();
    state.socket = null;
  }

  async function stopTraining() {
    if (!state.runId) return;
    $("stop").disabled = true;
    await fetch(`/api/runs/${state.runId}/stop`, { method: "POST" });
  }

  /* ---------- the road ---------- */

  const overlay = (show) => { $("stageOverlay").hidden = !show; };

  function stopPlayback() {
    clearInterval(state.player);
    state.player = null;
  }

  function outcomeOf(stats) {
    return stats.collision ? "collision" : stats.goal ? "goal" : null;
  }

  function describe(stats) {
    if (stats.collision) return "collided";
    return stats.goal ? "reached the goal" : "ran out of time";
  }

  function renderStats(stats, note) {
    const tiles = [
      ["return", stats.reward.toFixed(1), stats.reward > 15 ? "good" : stats.reward < -20 ? "bad" : ""],
      ["steps", stats.steps, ""],
      ["mean speed", stats.mean_speed.toFixed(2), ""],
      ["lane changes", stats.lane_changes, ""],
      ["red lights", stats.red_lights, stats.red_lights ? "bad" : ""],
      ["outcome", stats.collision ? "crashed" : stats.goal ? "goal" : "driving",
        stats.goal ? "good" : stats.collision ? "bad" : ""],
    ];
    $("episodeStats").innerHTML = tiles.map(([label, value, tone]) =>
      `<div class="stat"><span>${label}</span><strong class="${tone}">${value}</strong></div>`)
      .join("");
    $("episodeInfo").textContent = note ?? "";
  }

  function play(frames, stats, label) {
    stopPlayback();
    overlay(false);
    let index = 0;
    const tick = () => {
      const last = index === frames.length - 1;
      RENDER.draw($("road"), frames[index], frames[index - 1], label,
                  last ? outcomeOf(stats) : null);
      state.lastDraw = { frame: frames[index], previous: frames[index - 1], label,
                         outcome: last ? outcomeOf(stats) : null };
      index += 1;
      if (index >= frames.length) {
        stopPlayback();
        renderStats(stats, `${label} ${describe(stats)} after ${stats.steps} steps.`);
      }
    };
    tick();
    state.player = setInterval(tick, 1000 / Number($("speed").value));
  }

  async function watch(url, label) {
    if (state.mode === "drive") endDrive();
    $("episodeInfo").textContent = "running an episode…";
    const seed = Math.floor(Math.random() * 10000);
    const response = await fetch(`${url}${url.includes("?") ? "&" : "?"}seed=${seed}`);
    if (!response.ok) {
      $("episodeInfo").textContent = "could not run an episode";
      return;
    }
    const payload = await response.json();
    play(payload.frames, payload.stats, label);
  }

  /* ---------- drive it yourself ---------- */

  const KEYS = { ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right",
                 w: "up", s: "down", a: "left", d: "right" };

  async function startDrive() {
    stopPlayback();
    const seed = Math.floor(Math.random() * 10000);
    const session = await (await fetch(`/api/drive?seed=${seed}`, { method: "POST" })).json();
    state.drive = { id: session.id, previous: null,
                    held: new Set(), pressed: new Set(), busy: false };
    state.mode = "drive";
    overlay(false);
    $("driveHint").hidden = false;
    $("drive").textContent = "Stop driving";
    RENDER.draw($("road"), session.frame, null, "you");
    renderStats(session.stats, "Your turn — the agents score this exact road.");
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    state.player = setInterval(driveTick, 1000 / DRIVE_HZ);
  }

  function endDrive(note) {
    stopPlayback();
    window.removeEventListener("keydown", onKeyDown);
    window.removeEventListener("keyup", onKeyUp);
    state.mode = "idle";
    state.drive = null;
    $("driveHint").hidden = true;
    $("drive").textContent = "Drive it yourself";
    if (note) $("episodeInfo").textContent = note;
  }

  function onKeyDown(event) {
    if (state.mode !== "drive") return;
    if (event.key === "Escape") { endDrive("Stopped driving."); return; }
    const key = KEYS[event.key];
    if (!key) return;
    event.preventDefault();
    state.drive.pressed.add(key);          // a tap still registers between ticks
    if (key === "up" || key === "down") state.drive.held.add(key);
  }

  function onKeyUp(event) {
    const key = KEYS[event.key];
    if (key && state.drive) state.drive.held.delete(key);
  }

  async function driveTick() {
    const drive = state.drive;
    if (!drive || drive.busy) return;

    const pressed = drive.pressed;
    drive.pressed = new Set();
    let action = ACTION.MAINTAIN;
    if (pressed.has("left")) action = ACTION.LANE_LEFT;
    else if (pressed.has("right")) action = ACTION.LANE_RIGHT;
    else if (drive.held.has("up") || pressed.has("up")) action = ACTION.ACCELERATE;
    else if (drive.held.has("down") || pressed.has("down")) action = ACTION.BRAKE;

    drive.busy = true;
    const response = await fetch(`/api/drive/${drive.id}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }),
    }).catch(() => null);
    drive.busy = false;
    if (!response || !response.ok) { endDrive("Lost the drive session."); return; }

    const data = await response.json();
    const outcome = data.done ? outcomeOf(data.stats) : null;
    RENDER.draw($("road"), data.frame, drive.previous, "you", outcome);
    state.lastDraw = { frame: data.frame, previous: drive.previous, label: "you", outcome };
    drive.previous = data.frame;
    renderStats(data.stats, data.done ? "" : "Your turn — the agents score this exact road.");
    if (data.done) {
      endDrive(`You ${describe(data.stats)} after ${data.stats.steps} steps `
             + `for a return of ${data.stats.reward.toFixed(1)}. `
             + `A trained agent scores about 40 here, the hand-written driver 43.5.`);
    }
  }

  /* ---------- learned policy ---------- */

  async function showPolicy() {
    const panel = $("policy");
    const response = await fetch(`/api/runs/${state.runId}/policy`);
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      panel.innerHTML = `<p class="policy-note">${detail.detail ?? "unavailable"}</p>`;
      panel.hidden = false;
      return;
    }
    const table = await response.json();
    const head = `<tr><th>speed</th>${table.gaps.map((g) => `<th>${g}</th>`).join("")}</tr>`;
    const body = table.rows.map((row) => `<tr><td>${row.speed}</td>${
      row.actions.map((a) => `<td class="a-${a}">${a.replace("LANE_", "")}</td>`).join("")
    }</tr>`).join("");
    panel.innerHTML = `<table>${head}${body}</table>
      <p class="policy-note">Middle lane, both neighbours free, light ${table.light}.
      ${table.states_learned} of ${table.states_reachable} reachable states learned.</p>`;
    panel.hidden = false;
  }

  /* ---------- wiring ---------- */

  document.addEventListener("DOMContentLoaded", () => {
    initCharts();
    loadAlgorithms();
    RENDER.clear($("road"));

    $("algo").addEventListener("change", renderHyper);
    $("episodes").addEventListener("change", syncDecay);
    $("train").addEventListener("click", startTraining);
    $("stop").addEventListener("click", stopTraining);
    $("clear").addEventListener("click", clearCharts);
    $("showPolicy").addEventListener("click", showPolicy);
    $("watchAgent").addEventListener("click",
      () => watch(`/api/runs/${state.runId}/episode`, `${$("algo").value} (greedy)`));
    $("watchScripted").addEventListener("click",
      () => watch("/api/episode?algo=scripted", "scripted driver"));
    $("drive").addEventListener("click",
      () => (state.mode === "drive" ? endDrive("Stopped driving.") : startDrive()));

    window.addEventListener("resize", () => {
      if (!state.lastDraw) { RENDER.clear($("road")); return; }
      const { frame, previous, label, outcome } = state.lastDraw;
      RENDER.draw($("road"), frame, previous, label, outcome);
    });
  });
})();
