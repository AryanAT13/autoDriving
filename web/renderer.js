/* Top-down view of one simulation frame. The viewport follows the ego vehicle, and the
   lane markings are anchored in world coordinates so they scroll with the road. */
const RENDER = (() => {
  const WINDOW = 340;        // world units visible across the canvas
  const EGO_AT = 0.27;       // ego's horizontal position, as a fraction of the width
  const DASH = 16, DASH_GAP = 16;
  const HUD_H = 44;

  const SIGNAL = { green: "#2fd98a", yellow: "#f0b429", red: "#ef4d4d" };
  const TRAFFIC = ["#8d94a3", "#a8846a", "#7d8f8a", "#9a7f92", "#879bb0", "#9c9079"];

  let noise = null;

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.roundRect(x, y, Math.max(1, w), Math.max(1, h), r);
  }

  function shade(hex, factor) {
    const n = parseInt(hex.slice(1), 16);
    const clamp = (v) => Math.max(0, Math.min(255, Math.round(v)));
    return `rgb(${clamp((n >> 16) * factor)},${clamp(((n >> 8) & 255) * factor)},`
         + `${clamp((n & 255) * factor)})`;
  }

  /* A tiny translucent noise tile, built once, tiled over the asphalt. */
  function grain(ctx) {
    if (noise) return noise;
    const tile = document.createElement("canvas");
    tile.width = tile.height = 72;
    const tctx = tile.getContext("2d");
    const image = tctx.createImageData(72, 72);
    for (let i = 0; i < image.data.length; i += 4) {
      image.data[i] = image.data[i + 1] = image.data[i + 2] = 255;
      image.data[i + 3] = Math.random() * 16;
    }
    tctx.putImageData(image, 0, 0);
    noise = ctx.createPattern(tile, "repeat");
    return noise;
  }

  function fit(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

  function backdrop(ctx, w, h) {
    const sky = ctx.createLinearGradient(0, 0, 0, h);
    sky.addColorStop(0, "#131922");
    sky.addColorStop(1, "#0a0d11");
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, w, h);
  }

  function asphalt(ctx, w, top, bottom) {
    const height = bottom - top;
    const surface = ctx.createLinearGradient(0, top, 0, bottom);
    surface.addColorStop(0, "#32373f");
    surface.addColorStop(0.5, "#3c424b");
    surface.addColorStop(1, "#2d323a");
    ctx.fillStyle = surface;
    ctx.fillRect(0, top, w, height);

    ctx.save();
    ctx.globalAlpha = 0.55;
    ctx.fillStyle = grain(ctx);
    ctx.fillRect(0, top, w, height);
    ctx.restore();

    const vignette = ctx.createLinearGradient(0, top, 0, bottom);
    vignette.addColorStop(0, "rgba(0,0,0,.42)");
    vignette.addColorStop(0.14, "rgba(0,0,0,0)");
    vignette.addColorStop(0.86, "rgba(0,0,0,0)");
    vignette.addColorStop(1, "rgba(0,0,0,.42)");
    ctx.fillStyle = vignette;
    ctx.fillRect(0, top, w, height);
  }

  function markings(ctx, w, frame, top, laneH, originX, sx) {
    ctx.fillStyle = "rgba(242,240,230,.88)";
    ctx.fillRect(0, top, w, 3);
    ctx.fillRect(0, top + laneH * frame.lanes - 3, w, 3);

    const period = DASH + DASH_GAP;
    const start = Math.floor(originX / period) * period;
    ctx.fillStyle = "rgba(238,235,222,.6)";
    for (let lane = 1; lane < frame.lanes; lane += 1) {
      const y = top + lane * laneH - 1.5;
      for (let x = start; x < originX + WINDOW + period; x += period) {
        const a = sx(x);
        const b = sx(x + DASH);
        if (b < -12 || a > w + 12) continue;
        ctx.fillRect(a, y, b - a, 3);
      }
    }
  }

  function goalLine(ctx, w, frame, top, bottom, sx) {
    const x = sx(frame.road_length);
    if (x < -40 || x > w + 40) return;
    const size = 9;
    for (let y = top, row = 0; y < bottom; y += size, row += 1) {
      for (let column = 0; column < 2; column += 1) {
        ctx.fillStyle = (row + column) % 2 ? "#f4f4f1" : "#15171b";
        ctx.fillRect(x + column * size, y, size, Math.min(size, bottom - y));
      }
    }
  }

  function signals(ctx, w, frame, top, bottom, sx) {
    for (const light of frame.lights) {
      const x = sx(light.x);
      if (x < -70 || x > w + 70) continue;
      const colour = SIGNAL[light.phase];

      ctx.save();
      ctx.globalAlpha = 0.18;
      const wash = ctx.createLinearGradient(x - 16, 0, x + 16, 0);
      wash.addColorStop(0, "transparent");
      wash.addColorStop(0.5, colour);
      wash.addColorStop(1, "transparent");
      ctx.fillStyle = wash;
      ctx.fillRect(x - 16, top, 32, bottom - top);
      ctx.restore();

      ctx.fillStyle = "rgba(242,240,230,.8)";
      ctx.fillRect(x - 1.5, top, 3, bottom - top);

      ctx.fillStyle = "#232932";
      ctx.fillRect(x - 2, top - 20, 4, 20);
      roundRect(ctx, x - 8.5, top - 33, 17, 15, 5);
      ctx.fill();

      ctx.save();
      ctx.shadowColor = colour;
      ctx.shadowBlur = 18;
      ctx.fillStyle = colour;
      ctx.beginPath();
      ctx.arc(x, top - 25.5, 4.6, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 30;
      ctx.fill();
      ctx.restore();
    }
  }

  function vehicle(ctx, cx, cy, len, wid, colour, braking, isEgo) {
    const x = cx - len / 2;
    const y = cy - wid / 2;

    const shadow = ctx.createRadialGradient(cx, cy + wid * 0.45, 1, cx, cy + wid * 0.45, len * 0.62);
    shadow.addColorStop(0, "rgba(0,0,0,.5)");
    shadow.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = shadow;
    ctx.fillRect(cx - len, cy - wid * 0.4, len * 2, wid * 1.9);

    const body = ctx.createLinearGradient(0, y, 0, y + wid);
    body.addColorStop(0, shade(colour, 1.38));
    body.addColorStop(0.42, colour);
    body.addColorStop(1, shade(colour, 0.58));
    ctx.fillStyle = body;
    roundRect(ctx, x, y, len, wid, Math.min(wid * 0.36, 7));
    ctx.fill();

    ctx.fillStyle = "rgba(255,255,255,.14)";
    roundRect(ctx, x + len * 0.1, y + 1.5, len * 0.8, wid * 0.16, 2);
    ctx.fill();

    ctx.fillStyle = "rgba(11,15,21,.8)";
    roundRect(ctx, x + len * 0.3, y + wid * 0.2, len * 0.36, wid * 0.6, 3);
    ctx.fill();

    const lampH = Math.max(2, wid * 0.19);
    ctx.save();
    ctx.shadowColor = "rgba(255,236,180,.9)";
    ctx.shadowBlur = 9;
    ctx.fillStyle = "#ffeec4";
    roundRect(ctx, x + len - 3.6, y + wid * 0.14, 3.2, lampH, 1.5); ctx.fill();
    roundRect(ctx, x + len - 3.6, y + wid * 0.67, 3.2, lampH, 1.5); ctx.fill();
    ctx.restore();

    ctx.save();
    ctx.shadowColor = braking ? "rgba(255,70,70,.95)" : "rgba(180,45,45,.5)";
    ctx.shadowBlur = braking ? 15 : 4;
    ctx.fillStyle = braking ? "#ff5f5f" : "#8e2f2f";
    roundRect(ctx, x + 0.6, y + wid * 0.14, 3, lampH, 1.5); ctx.fill();
    roundRect(ctx, x + 0.6, y + wid * 0.67, 3, lampH, 1.5); ctx.fill();
    ctx.restore();

    if (isEgo) {
      ctx.strokeStyle = "rgba(255,255,255,.4)";
      ctx.lineWidth = 1.2;
      roundRect(ctx, x, y, len, wid, Math.min(wid * 0.36, 7));
      ctx.stroke();
    }
  }

  function hud(ctx, w, h, frame, label) {
    ctx.fillStyle = "rgba(9,11,15,.72)";
    ctx.fillRect(0, h - HUD_H, w, HUD_H);
    ctx.fillStyle = "rgba(255,255,255,.06)";
    ctx.fillRect(0, h - HUD_H, w, 1);

    const pad = 14;
    const baseline = h - 22;
    ctx.font = "600 12.5px Inter, system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.fillStyle = "#e9edf3";
    ctx.fillText(label ?? "", pad, baseline);

    ctx.textAlign = "right";
    ctx.fillStyle = "#98a2b0";
    ctx.font = "500 12px 'JetBrains Mono', ui-monospace, monospace";
    ctx.fillText(`step ${frame.step}    speed ${frame.ego.speed}/${frame.ego.max_speed}`
               + `    ${Math.round(frame.ego.x)} / ${frame.road_length}`, w - pad, baseline);

    const barY = h - 12;
    const barW = w - pad * 2;
    ctx.fillStyle = "rgba(255,255,255,.1)";
    roundRect(ctx, pad, barY, barW, 5, 2.5); ctx.fill();
    const progress = Math.max(0, Math.min(1, frame.ego.x / frame.road_length));
    const fill = ctx.createLinearGradient(pad, 0, pad + barW, 0);
    fill.addColorStop(0, "#2f6fd0");
    fill.addColorStop(1, "#74b6ff");
    ctx.fillStyle = fill;
    roundRect(ctx, pad, barY, Math.max(4, barW * progress), 5, 2.5); ctx.fill();
  }

  /* An edge vignette rather than a wash, so the final frame stays readable. */
  function flash(ctx, w, h, colour) {
    const glow = ctx.createRadialGradient(w * EGO_AT, h / 2, Math.min(w, h) * 0.2,
                                          w * EGO_AT, h / 2, w * 0.78);
    glow.addColorStop(0, colour.replace("ALPHA", "0"));
    glow.addColorStop(0.55, colour.replace("ALPHA", ".12"));
    glow.addColorStop(1, colour.replace("ALPHA", ".5"));
    ctx.fillStyle = glow;
    ctx.fillRect(0, 0, w, h);
  }

  function draw(canvas, frame, previous, label, outcome) {
    const { ctx, w, h } = fit(canvas);
    backdrop(ctx, w, h);
    if (!frame) return;

    const top = 36;
    const bottom = h - HUD_H - 8;
    const laneH = (bottom - top) / frame.lanes;
    const originX = frame.ego.x - WINDOW * EGO_AT;
    const sx = (x) => ((x - originX) / WINDOW) * w;
    const laneCentre = (lane) => top + (lane + 0.5) * laneH;

    asphalt(ctx, w, top, bottom);
    markings(ctx, w, frame, top, laneH, originX, sx);
    goalLine(ctx, w, frame, top, bottom, sx);
    signals(ctx, w, frame, top, bottom, sx);

    const carLen = Math.max(22, (10 / WINDOW) * w * 1.9);
    const carWid = Math.min(laneH * 0.56, carLen * 0.56);

    frame.traffic.forEach((car, index) => {
      const cx = sx(car.x);
      if (cx < -carLen || cx > w + carLen) return;
      const before = previous && previous.traffic[index];
      vehicle(ctx, cx, laneCentre(car.lane), carLen, carWid,
              TRAFFIC[index % TRAFFIC.length],
              Boolean(before && car.speed < before.speed - 0.01), false);
    });

    const slowing = Boolean(previous && frame.ego.speed < previous.ego.speed);
    vehicle(ctx, sx(frame.ego.x), laneCentre(frame.ego.lane), carLen, carWid,
            "#3987e5", slowing, true);

    if (outcome === "collision") flash(ctx, w, h, "rgba(239,77,77,ALPHA)");
    else if (outcome === "goal") flash(ctx, w, h, "rgba(47,217,138,ALPHA)");

    hud(ctx, w, h, frame, label);
  }

  function clear(canvas) {
    const { ctx, w, h } = fit(canvas);
    backdrop(ctx, w, h);
  }

  return { draw, clear };
})();
