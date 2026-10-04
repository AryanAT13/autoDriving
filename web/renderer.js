/* Top-down canvas view of one simulation frame. The viewport follows the ego vehicle. */
const RENDER = (() => {
  const WINDOW = 420;        // world units visible across the canvas
  const EGO_FRACTION = 0.32; // where the ego sits horizontally
  const CAR_LENGTH = 10;

  const css = (name) => getComputedStyle(document.documentElement)
    .getPropertyValue(name).trim();

  function draw(canvas, frame, label) {
    const ctx = canvas.getContext("2d");
    const { width: w, height: h } = canvas;
    ctx.clearRect(0, 0, w, h);
    if (!frame) return;

    const laneHeight = (h - 40) / frame.lanes;
    const originX = frame.ego.x - WINDOW * EGO_FRACTION;
    const toScreen = (x) => ((x - originX) / WINDOW) * w;
    const laneY = (lane) => 24 + lane * laneHeight;

    ctx.fillStyle = css("--road");
    ctx.fillRect(0, 24, w, h - 40);

    ctx.strokeStyle = css("--lane");
    ctx.lineWidth = 2;
    for (let lane = 1; lane < frame.lanes; lane += 1) {
      ctx.setLineDash([12, 12]);
      ctx.beginPath();
      ctx.moveTo(0, laneY(lane));
      ctx.lineTo(w, laneY(lane));
      ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(0, 24); ctx.lineTo(w, 24);
    ctx.moveTo(0, h - 16); ctx.lineTo(w, h - 16);
    ctx.stroke();

    const signal = { green: css("--good"), yellow: "#eda100", red: css("--bad") };
    for (const light of frame.lights) {
      const x = toScreen(light.x);
      if (x < -10 || x > w + 10) continue;
      ctx.strokeStyle = signal[light.phase];
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(x, 24); ctx.lineTo(x, h - 16);
      ctx.stroke();
      ctx.fillStyle = signal[light.phase];
      ctx.beginPath();
      ctx.arc(x, 14, 6, 0, Math.PI * 2);
      ctx.fill();
    }

    const goal = toScreen(frame.road_length);
    if (goal > -10 && goal < w + 10) {
      ctx.strokeStyle = css("--ink");
      ctx.setLineDash([5, 5]);
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(goal, 24); ctx.lineTo(goal, h - 16);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    const carW = Math.max(14, (CAR_LENGTH / WINDOW) * w * 1.6);
    const carH = laneHeight * 0.52;
    const car = (x, lane, fill) => {
      const sx = toScreen(x) - carW / 2;
      const sy = laneY(lane) + (laneHeight - carH) / 2;
      ctx.fillStyle = fill;
      ctx.beginPath();
      ctx.roundRect(sx, sy, carW, carH, 4);
      ctx.fill();
    };

    ctx.fillStyle = css("--muted");
    for (const vehicle of frame.traffic) {
      const x = toScreen(vehicle.x);
      if (x < -carW || x > w + carW) continue;
      car(vehicle.x, vehicle.lane, css("--muted"));
    }
    car(frame.ego.x, frame.ego.lane, css("--series-1"));

    ctx.fillStyle = css("--muted");
    ctx.font = "12px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(label ?? "", 8, h - 3);
    ctx.textAlign = "right";
    ctx.fillText(
      `step ${frame.step}   speed ${frame.ego.speed}   ${Math.round(frame.ego.x)}/${frame.road_length}`,
      w - 8, h - 3);
  }

  return { draw };
})();
