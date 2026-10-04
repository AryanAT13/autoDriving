/* Dashboard: configure a run, stream its metrics, and replay its policy on the canvas. */
(() => {
  const $ = (id) => document.getElementById(id);
  const css = (name) => getComputedStyle(document.documentElement)
    .getPropertyValue(name).trim();

  const state = { algorithms: [], runId: null, socket: null, timer: null, charts: {} };

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

  function selected() {
    return state.algorithms.find((a) => a.name === $("algo").value);
  }

  /* Only the settings that matter for the chosen algorithm are shown. */
  function relevantKeys(algo) {
    const shared = ["alpha", "gamma", "epsilon_end"];
    if (algo.name === "actor_critic") {
      return ["gamma", "alpha_policy", "alpha_value", "entropy_beta", "hidden_units"];
    }
    if (algo.name.startsWith("dyna_q")) return [...shared, "planning_steps", "model_capacity"];
    return shared;
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
    const algo = selected();
    if (!algo || algo.name === "actor_critic") return;
    const episodes = Number($("episodes").value);
    const suggested = Math.max(1, Math.round(episodes * 0.6));
    let note = $("decayNote");
    if (!note) {
      note = document.createElement("div");
      note.id = "decayNote";
      note.className = "field";
      note.innerHTML = `<label for="hp-epsilon_decay_episodes">epsilon_decay_episodes</label>
        <input id="hp-epsilon_decay_episodes" data-key="epsilon_decay_episodes" type="number">`;
      $("hyper").appendChild(note);
    }
    $("hp-epsilon_decay_episodes").value = suggested;
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

  function makeChart(canvasId, label, colour, yLabel) {
    return new Chart($(canvasId), {
      type: "line",
      data: { datasets: [{ label, data: [], borderColor: colour, backgroundColor: colour,
                           borderWidth: 2, pointRadius: 0, tension: 0.15 }] },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        parsing: false, interaction: { mode: "nearest", intersect: false },
        scales: {
          x: { type: "linear", title: { display: true, text: "episode", color: css("--muted") },
               ticks: { color: css("--muted") }, grid: { color: css("--border") } },
          y: { title: { display: true, text: yLabel, color: css("--muted") },
               ticks: { color: css("--muted") }, grid: { color: css("--border") } },
        },
        plugins: { legend: { labels: { color: css("--muted"), boxWidth: 12 } } },
      },
    });
  }

  function initCharts() {
    state.charts.reward = makeChart("returnChart", "greedy evaluation return",
                                    css("--series-1"), "return");
    state.charts.collision = makeChart("collisionChart", "training collision rate",
                                       css("--series-2"), "collisions / episode");
  }

  function resetCharts() {
    for (const chart of Object.values(state.charts)) {
      chart.data.datasets[0].data = [];
      chart.update("none");
    }
  }

  /* Blocks arrive already averaged by the server, so the chart just plots them. */
  function pushBlocks(blocks) {
    const series = state.charts.collision.data.datasets[0].data;
    for (const block of blocks) series.push({ x: block.episode, y: block.collision_rate });
    state.charts.collision.update("none");
  }

  function pushEvals(evals) {
    const series = state.charts.reward.data.datasets[0].data;
    for (const point of evals) series.push({ x: point.episode, y: point.reward });
    state.charts.reward.update("none");
    if (evals.length) renderSummary(evals[evals.length - 1]);
  }

  const PERCENT = new Set(["collision_rate", "goal_rate"]);
  function renderSummary(metrics) {
    const rows = ["reward", "collision_rate", "goal_rate", "mean_speed", "steps", "red_lights"];
    $("summary").querySelector("tbody").innerHTML = rows.map((key) => {
      const value = metrics[key];
      if (value === undefined) return "";
      const shown = PERCENT.has(key) ? `${(value * 100).toFixed(1)}%` : value.toFixed(2);
      return `<tr><td>${key.replace(/_/g, " ")}</td><td>${shown}</td></tr>`;
    }).join("");
  }

  /* ---------- training ---------- */

  async function startTraining() {
    resetCharts();
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
    $("train").disabled = true;
    $("stop").disabled = false;
    $("watchAgent").disabled = false;
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
      pushBlocks(payload.blocks);
      pushEvals(payload.evals);
      const { status, episode, episodes, error } = payload.summary;
      if (status === "running") {
        setStatus(`training ${episode}/${episodes}`, "running");
      } else {
        setStatus(error ? `error: ${error}` : `${status} at ${episode} episodes`,
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
    await fetch(`/api/runs/${state.runId}/stop`, { method: "POST" });
    $("stop").disabled = true;
  }

  /* ---------- playback ---------- */

  function animate(payload, label) {
    clearInterval(state.timer);
    const { frames, stats } = payload;
    let index = 0;
    const tick = () => {
      RENDER.draw($("road"), frames[index], label);
      index += 1;
      if (index >= frames.length) {
        clearInterval(state.timer);
        const outcome = stats.collision ? "collision" : stats.goal ? "reached the goal" : "ran out of time";
        $("episodeInfo").textContent =
          `${label}: ${outcome} after ${stats.steps} steps, return ${stats.reward.toFixed(1)}, ` +
          `mean speed ${stats.mean_speed.toFixed(2)}, ${stats.lane_changes} lane changes, ` +
          `${stats.red_lights} red-light violations.`;
      }
    };
    state.timer = setInterval(tick, 1000 / Number($("speed").value));
  }

  async function watch(url, label) {
    $("episodeInfo").textContent = "running an episode...";
    const seed = Math.floor(Math.random() * 10000);
    const response = await fetch(`${url}${url.includes("?") ? "&" : "?"}seed=${seed}`);
    if (!response.ok) {
      $("episodeInfo").textContent = "could not run an episode";
      return;
    }
    animate(await response.json(), label);
  }

  /* ---------- wiring ---------- */

  document.addEventListener("DOMContentLoaded", () => {
    initCharts();
    loadAlgorithms();
    $("algo").addEventListener("change", renderHyper);
    $("episodes").addEventListener("change", syncDecay);
    $("train").addEventListener("click", startTraining);
    $("stop").addEventListener("click", stopTraining);
    $("watchAgent").addEventListener("click",
      () => watch(`/api/runs/${state.runId}/episode`, `${$("algo").value} (greedy)`));
    $("watchScripted").addEventListener("click",
      () => watch("/api/episode?algo=scripted", "scripted driver"));
  });
})();
