// Frontend WC2026 Predictor
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

const api = {
  matches: (q = "") => fetch(`/api/matches${q}`).then(r => r.json()),
  match: (id) => fetch(`/api/match/${id}`).then(r => r.json()),
  predict: (id) => fetch(`/api/predict/${id}`, { method: "POST" }).then(r => r.json()),
  predictAll: () => fetch(`/api/predict_all`, { method: "POST" }).then(r => r.json()),
  refresh: () => fetch(`/api/refresh`, { method: "POST" }).then(r => r.json()),
  teams: () => fetch(`/api/teams`).then(r => r.json()),
  groups: () => fetch(`/api/groups`).then(r => r.json()),
  accuracy: () => fetch(`/api/accuracy`).then(r => r.json()),
  modelInfo: () => fetch(`/api/model_info`).then(r => r.json()),
  trainNow: () => fetch(`/api/train_now`, { method: "POST" }).then(r => r.json()),
  trainStatus: () => fetch(`/api/train_status`).then(r => r.json()),
  live: () => fetch(`/api/live`).then(r => r.json()),
  liveOne: (id) => fetch(`/api/live/${id}`).then(r => r.json()),
  liveTick: () => fetch(`/api/live/tick`, { method: "POST" }).then(r => r.json()),
  liveStatus: () => fetch(`/api/live/status`).then(r => r.json()),
};

const flag = (code) => `<span class="flag">${code ?? "?"}</span>`;
const pct = (x) => (x * 100).toFixed(1) + "%";
const fmtDate = (iso) => {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString("es-AR", {
    day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit",
  });
};

let STATE = { tab: "quiniela" };

// Devuelve {label, color, emoji, recommend} según probabilidad máxima del modelo
function confidenceTier(pred) {
  const top = Math.max(pred.p_home, pred.p_draw, pred.p_away);
  // Cuál es el pick (home/draw/away)
  const pick = pred.p_home >= pred.p_draw && pred.p_home >= pred.p_away
    ? "home"
    : (pred.p_away >= pred.p_draw ? "away" : "draw");
  if (top >= 0.72) return { tier: "SEGURO", color: "emerald", emoji: "🔒", recommend: true, top, pick };
  if (top >= 0.55) return { tier: "FAVORITO", color: "blue", emoji: "✅", recommend: true, top, pick };
  if (top >= 0.42) return { tier: "RIESGO", color: "amber", emoji: "⚠️", recommend: false, top, pick };
  return { tier: "NO APOSTAR", color: "rose", emoji: "🚫", recommend: false, top, pick };
}

// ===== KPIs =====
async function loadKpis() {
  const [acc, info, matches] = await Promise.all([
    api.accuracy(), api.modelInfo(), api.matches("?status=upcoming"),
  ]);
  $("#kpi-model").textContent = info.model_loaded
    ? (info.meta?.version || "ML")
    : "Elo+Poisson";
  if (acc.played_with_prediction > 0) {
    $("#kpi-acc").textContent = (acc.accuracy_1x2 * 100).toFixed(1) + "%";
    $("#kpi-brier").textContent = acc.brier_score.toFixed(3);
  } else {
    $("#kpi-acc").textContent = "—";
    $("#kpi-brier").textContent = "—";
  }
  $("#kpi-played").textContent = acc.played_with_prediction ?? 0;
  const now = new Date();
  const in24h = new Date(now.getTime() + 24 * 3600 * 1000);
  const next = matches.filter(m => m.datetime && new Date(m.datetime) >= now && new Date(m.datetime) <= in24h);
  $("#kpi-next").textContent = next.length;
}

// ===== Match card =====
function matchCard(m) {
  const pred = m.prediction;
  const probBar = pred
    ? `<div class="prob-bar mt-2">
         <div class="prob-home" style="width:${pred.p_home*100}%"></div>
         <div class="prob-draw" style="width:${pred.p_draw*100}%"></div>
         <div class="prob-away" style="width:${pred.p_away*100}%"></div>
       </div>
       <div class="flex justify-between text-xs mt-1 text-slate-600">
         <span>${m.home}: ${pct(pred.p_home)}</span>
         <span>X: ${pct(pred.p_draw)}</span>
         <span>${m.away}: ${pct(pred.p_away)}</span>
       </div>`
    : `<div class="text-xs text-slate-500 italic mt-2">Sin predicción aún. Click → predecir.</div>`;
  const result = m.finished
    ? `<div class="text-2xl font-bold tabular-nums">${m.home_goals} - ${m.away_goals}</div>`
    : (pred
      ? `<div class="text-sm text-slate-600">Predicho: <span class="font-semibold tabular-nums text-base">${Math.round(pred.pred_home_goals)} - ${Math.round(pred.pred_away_goals)}</span></div>`
      : `<div class="text-sm text-slate-400">vs</div>`);
  const stage = m.stage === "GROUP"
    ? `Grupo ${m.group} · J${m.matchday}`
    : m.stage;
  return `
    <div class="match-card bg-white rounded-xl shadow-sm p-4 cursor-pointer" data-id="${m.id}">
      <div class="flex justify-between items-center text-xs text-slate-500 mb-2">
        <span class="font-semibold uppercase tracking-wide">${stage}</span>
        <span>${fmtDate(m.datetime)}</span>
      </div>
      <div class="flex items-center justify-between gap-3">
        <div class="flex items-center gap-2 flex-1">
          ${flag(m.home)} <span class="font-semibold">${m.home_name}</span>
        </div>
        ${result}
        <div class="flex items-center gap-2 flex-1 justify-end">
          <span class="font-semibold">${m.away_name}</span> ${flag(m.away)}
        </div>
      </div>
      ${probBar}
      ${pred ? `<div class="flex gap-3 mt-2 text-xs text-slate-500">
        <span>O2.5: <b>${pct(pred.over_25)}</b></span>
        <span>BTTS: <b>${pct(pred.btts)}</b></span>
        <span>conf: <b>${(pred.confidence*100).toFixed(0)}%</b></span>
      </div>`:""}
    </div>`;
}

// ===== Tab EN VIVO =====
async function renderLive() {
  const [data, status] = await Promise.all([api.live(), api.liveStatus()]);
  // Banner explicativo
  const banner = `
    <div class="bg-gradient-to-r from-rose-50 to-orange-50 border border-rose-200 rounded-xl p-4 mb-4 flex items-center justify-between gap-4">
      <div>
        <div class="flex items-center gap-2 mb-1">
          <span class="inline-block w-2.5 h-2.5 bg-rose-500 rounded-full animate-pulse"></span>
          <h3 class="font-bold">Predicción in-play (Dixon-Robinson 1998)</h3>
        </div>
        <p class="text-sm text-slate-700">El modelo re-predice cada 60s con minuto + marcador + tiros + expulsiones + xG live.</p>
      </div>
      <div class="text-right text-xs text-slate-500">
        <div>API: ${status.api_enabled ? '<span class="text-emerald-600 font-semibold">conectada</span>' : '<span class="text-rose-600">sin token</span>'}</div>
        <div>Último tick: ${status.last_tick ? fmtDate(status.last_tick) : 'pendiente'}</div>
        <div>Partidos en vivo: <b>${status.live_count ?? 0}</b></div>
        ${status.last_error ? `<div class="text-rose-600 mt-1">⚠ ${status.last_error}</div>` : ''}
      </div>
    </div>`;

  if (!data.length) {
    $("#view").innerHTML = banner + `
      <div class="bg-white rounded-xl shadow-sm p-8 text-center">
        <div class="text-5xl mb-2">⚽</div>
        <p class="font-semibold text-slate-700 mb-1">No hay partidos en este momento</p>
        <p class="text-sm text-slate-500">Cuando empiece un partido del Mundial, aparecerá aquí con stats live y probabilidad ajustada minuto a minuto.</p>
        <button onclick="api.liveTick().then(()=>renderLive())" class="mt-4 bg-rose-600 hover:bg-rose-700 text-white text-sm font-semibold px-3 py-2 rounded-lg">↻ Consultar API ahora</button>
      </div>`;
    return;
  }

  $("#view").innerHTML = banner + `<div class="grid md:grid-cols-2 gap-4">${data.map(liveCard).join("")}</div>`;
}

function liveCard(m) {
  const s = m.stats; const p = m.prediction_live;
  const fmtPoss = (v) => v == null ? '—' : `${Math.round(v)}%`;
  const fmtXg = (v) => v == null ? '—' : v.toFixed(2);
  const top = Math.max(p.p_home, p.p_draw, p.p_away);
  const expFinal = `${Math.round(p.expected_final_home)} - ${Math.round(p.expected_final_away)}`;
  return `
    <div class="bg-white rounded-xl shadow-md overflow-hidden border-2 border-rose-200">
      <div class="bg-gradient-to-r from-rose-600 to-orange-500 text-white px-4 py-3 flex justify-between items-center">
        <div class="flex items-center gap-2">
          <span class="inline-block w-2 h-2 bg-white rounded-full animate-pulse"></span>
          <span class="font-bold uppercase text-xs tracking-wide">${m.status}</span>
          <span class="bg-white/20 px-2 py-0.5 rounded-full text-xs font-bold">${m.minute}'</span>
        </div>
        <span class="text-xs opacity-90">${fmtDate(m.updated_at)}</span>
      </div>

      <div class="p-4">
        <div class="flex items-center justify-between gap-3 mb-3">
          <div class="flex items-center gap-2 flex-1">
            ${flag(m.home)} <span class="font-bold text-lg">${m.home_name}</span>
          </div>
          <div class="text-3xl font-bold tabular-nums">${m.home_goals_live} - ${m.away_goals_live}</div>
          <div class="flex items-center gap-2 flex-1 justify-end">
            <span class="font-bold text-lg">${m.away_name}</span> ${flag(m.away)}
          </div>
        </div>

        <div class="bg-slate-50 rounded-lg p-3 mb-3">
          <div class="flex justify-between items-center mb-1">
            <span class="text-xs uppercase text-slate-500 font-semibold">Predicción in-play</span>
            <span class="text-xs text-slate-500">Final esperado: <b>${expFinal}</b></span>
          </div>
          <div class="prob-bar mt-1">
            <div class="prob-home" style="width:${p.p_home*100}%"></div>
            <div class="prob-draw" style="width:${p.p_draw*100}%"></div>
            <div class="prob-away" style="width:${p.p_away*100}%"></div>
          </div>
          <div class="flex justify-between text-xs mt-1 font-semibold">
            <span class="text-emerald-700">${m.home}: ${pct(p.p_home)}</span>
            <span class="text-slate-600">X: ${pct(p.p_draw)}</span>
            <span class="text-brand-600">${m.away}: ${pct(p.p_away)}</span>
          </div>
        </div>

        <table class="w-full text-sm">
          <thead><tr class="text-xs text-slate-400 uppercase">
            <th class="text-right">${m.home}</th><th>Stat</th><th class="text-left">${m.away}</th>
          </tr></thead>
          <tbody>
            <tr><td class="text-right font-bold">${fmtPoss(s.possession_h)}</td><td class="text-center text-xs text-slate-500">Posesión</td><td class="font-bold">${fmtPoss(s.possession_a)}</td></tr>
            <tr><td class="text-right font-bold">${s.shots_h ?? 0}</td><td class="text-center text-xs text-slate-500">Tiros</td><td class="font-bold">${s.shots_a ?? 0}</td></tr>
            <tr><td class="text-right font-bold">${s.shots_on_target_h ?? 0}</td><td class="text-center text-xs text-slate-500">Al arco</td><td class="font-bold">${s.shots_on_target_a ?? 0}</td></tr>
            <tr><td class="text-right font-bold">${fmtXg(s.xg_live_h)}</td><td class="text-center text-xs text-slate-500">xG live</td><td class="font-bold">${fmtXg(s.xg_live_a)}</td></tr>
            <tr><td class="text-right font-bold">${s.corners_h ?? 0}</td><td class="text-center text-xs text-slate-500">Tiros esquina</td><td class="font-bold">${s.corners_a ?? 0}</td></tr>
            <tr><td class="text-right font-bold">${s.fouls_h ?? 0}</td><td class="text-center text-xs text-slate-500">Faltas</td><td class="font-bold">${s.fouls_a ?? 0}</td></tr>
            <tr><td class="text-right"><span class="bg-yellow-300 px-1.5 rounded text-xs font-bold">${s.yellow_h ?? 0}</span></td><td class="text-center text-xs text-slate-500">Amarillas</td><td><span class="bg-yellow-300 px-1.5 rounded text-xs font-bold">${s.yellow_a ?? 0}</span></td></tr>
            <tr><td class="text-right"><span class="bg-rose-500 text-white px-1.5 rounded text-xs font-bold">${s.red_h ?? 0}</span></td><td class="text-center text-xs text-slate-500">Rojas</td><td><span class="bg-rose-500 text-white px-1.5 rounded text-xs font-bold">${s.red_a ?? 0}</span></td></tr>
          </tbody>
        </table>
      </div>
    </div>`;
}

// ===== Modo Quiniela =====
async function renderQuiniela() {
  const data = await api.matches("?status=upcoming");
  const withPred = data.filter(m => m.prediction && m.home && m.away);
  if (!withPred.length) {
    $("#view").innerHTML = `<div class="bg-amber-50 border border-amber-200 rounded-xl p-6 text-amber-900">
      <p class="font-semibold mb-1">Sin predicciones todavía.</p>
      <p class="text-sm">Click en <b>⚡ Predecir todos</b> arriba para generar predicciones de los próximos partidos.</p>
    </div>`;
    return;
  }
  // Ordenar por probabilidad máxima desc
  withPred.sort((a, b) => {
    const ta = confidenceTier(a.prediction).top;
    const tb = confidenceTier(b.prediction).top;
    return tb - ta;
  });
  const safe = withPred.filter(m => confidenceTier(m.prediction).tier === "SEGURO");
  const fav = withPred.filter(m => confidenceTier(m.prediction).tier === "FAVORITO");
  const risk = withPred.filter(m => confidenceTier(m.prediction).tier === "RIESGO");
  const nope = withPred.filter(m => confidenceTier(m.prediction).tier === "NO APOSTAR");

  const explainBanner = `
    <div class="bg-gradient-to-r from-emerald-50 to-amber-50 border border-emerald-200 rounded-xl p-4 mb-4">
      <div class="flex items-center gap-2 mb-1"><span class="text-lg">🎯</span><h3 class="font-bold">Estrategia para ganar quinielas</h3></div>
      <p class="text-sm text-slate-700">No apostás todos los partidos: apostás los <b>de alta confianza</b>. Los modelos top del mundo aciertan ~58-65% global, pero <b>~80-90% en sus picks 🔒 SEGURO</b>. Confianza calibrada → cuando dice 80% acierta 80% real.</p>
    </div>`;

  const section = (title, items, badge, hint) => items.length ? `
    <div class="mb-5">
      <div class="flex items-center gap-2 mb-2">
        <h3 class="font-bold text-lg">${badge} ${title}</h3>
        <span class="text-xs text-slate-500">(${items.length} ${items.length===1?'partido':'partidos'})</span>
      </div>
      <p class="text-xs text-slate-500 mb-2">${hint}</p>
      <div class="grid md:grid-cols-2 gap-3">${items.map(quinielaCard).join("")}</div>
    </div>` : "";

  $("#view").innerHTML = explainBanner +
    section("Picks SEGUROS", safe, "🔒", "Confianza ≥ 72%. Estos son tus mejores tickets — apostá fuerte acá.") +
    section("FAVORITOS claros", fav, "✅", "Confianza 55-72%. Apuesta sólida pero con margen de error real.") +
    section("Partidos de RIESGO", risk, "⚠️", "Confianza 42-55%. Solo si necesitás llenar la quiniela.") +
    section("Mejor NO APOSTAR", nope, "🚫", "Confianza < 42%. Es 50/50 — guardá la plata.");
  bindMatchClicks();
}

function quinielaCard(m) {
  const p = m.prediction;
  const t = confidenceTier(p);
  const pickName = t.pick === "home" ? m.home_name : (t.pick === "away" ? m.away_name : "Empate");
  const colorClasses = {
    emerald: "border-emerald-300 bg-emerald-50",
    blue: "border-blue-300 bg-blue-50",
    amber: "border-amber-300 bg-amber-50",
    rose: "border-rose-300 bg-rose-50",
  }[t.color];
  return `
    <div class="match-card rounded-xl shadow-sm p-4 cursor-pointer border-2 ${colorClasses}" data-id="${m.id}">
      <div class="flex justify-between items-center text-xs text-slate-500 mb-2">
        <span class="font-semibold uppercase tracking-wide">${m.stage === "GROUP" ? `Grupo ${m.group} · J${m.matchday}` : m.stage}</span>
        <span>${fmtDate(m.datetime)}</span>
      </div>
      <div class="flex items-center justify-between gap-3 mb-2">
        <div class="flex items-center gap-2 flex-1">
          ${flag(m.home)} <span class="font-semibold">${m.home_name}</span>
        </div>
        <span class="text-xs text-slate-400">vs</span>
        <div class="flex items-center gap-2 flex-1 justify-end">
          <span class="font-semibold">${m.away_name}</span> ${flag(m.away)}
        </div>
      </div>
      <div class="bg-white rounded-lg p-3 border border-slate-200">
        <div class="flex justify-between items-center mb-1">
          <span class="text-xs uppercase text-slate-500 font-semibold">Pick recomendado</span>
          <span class="text-xs font-bold">${t.emoji} ${t.tier}</span>
        </div>
        <div class="flex justify-between items-end">
          <span class="text-lg font-bold">${pickName}</span>
          <span class="text-2xl font-bold tabular-nums text-${t.color}-700">${(t.top*100).toFixed(0)}%</span>
        </div>
        <div class="prob-bar mt-2">
          <div class="prob-home" style="width:${p.p_home*100}%"></div>
          <div class="prob-draw" style="width:${p.p_draw*100}%"></div>
          <div class="prob-away" style="width:${p.p_away*100}%"></div>
        </div>
        <div class="flex justify-between text-xs mt-1 text-slate-600">
          <span>${pct(p.p_home)}</span><span>${pct(p.p_draw)}</span><span>${pct(p.p_away)}</span>
        </div>
      </div>
      <div class="flex gap-3 mt-2 text-xs text-slate-500">
        <span>Marcador predicho: <b class="tabular-nums">${Math.round(p.pred_home_goals)}-${Math.round(p.pred_away_goals)}</b></span>
        <span>O2.5: <b>${pct(p.over_25)}</b></span>
        <span>BTTS: <b>${pct(p.btts)}</b></span>
      </div>
    </div>`;
}

// ===== Tabs =====
async function renderUpcoming() {
  const data = await api.matches("?status=upcoming");
  if (!data.length) {
    $("#view").innerHTML = `<p class="text-slate-500 italic">No hay partidos pendientes.</p>`;
    return;
  }
  $("#view").innerHTML = `<div class="grid md:grid-cols-2 gap-3">${data.map(matchCard).join("")}</div>`;
  bindMatchClicks();
}

async function renderPlayed() {
  const data = await api.matches("?status=played");
  if (!data.length) {
    $("#view").innerHTML = `<p class="text-slate-500 italic">Aún no se cargaron partidos jugados.</p>`;
    return;
  }
  // Conteo de aciertos para el banner
  let hits = 0, misses = 0, pending = 0;
  data.forEach(m => {
    if (!m.finished || m.home_goals == null) { pending++; return; }
    const r = realOutcome(m);
    const p = predOutcome(m);
    if (!p) return;
    if (r === p) hits++; else misses++;
  });
  const total = hits + misses;
  const acc = total ? (hits/total*100).toFixed(1) : "—";
  const banner = `
    <div class="bg-gradient-to-r from-emerald-50 to-amber-50 border border-emerald-200 rounded-xl p-4 mb-4 flex justify-between items-center">
      <div>
        <h3 class="font-bold flex items-center gap-2"><span>📊</span> Aciertos del modelo en partidos jugados</h3>
        <p class="text-sm text-slate-700 mt-1">
          <span class="text-emerald-700 font-bold">${hits} aciertos ✓</span> ·
          <span class="text-amber-700 font-bold">${misses} fallas ✗</span> ·
          ${pending} en curso/sin resultado · Accuracy: <b>${acc}${total?'%':''}</b>
        </p>
      </div>
      <div class="text-xs text-slate-500 text-right">
        <div>🟢 verde = el modelo acertó</div>
        <div>🟡 amarillo = el modelo falló</div>
      </div>
    </div>`;
  $("#view").innerHTML = banner + `<div class="grid md:grid-cols-2 gap-3">${data.map(playedCard).join("")}</div>`;
  bindMatchClicks();
}

async function renderGroups() {
  const data = await api.groups();
  if (!Object.keys(data).length) {
    $("#view").innerHTML = `<p class="text-slate-500 italic">Sin grupos disponibles.</p>`;
    return;
  }
  const html = Object.entries(data).sort().map(([g, table]) => `
    <div class="bg-white rounded-xl shadow-sm overflow-hidden">
      <div class="bg-brand-600 text-white px-3 py-2 font-bold">Grupo ${g}</div>
      <table class="w-full text-sm">
        <thead class="bg-slate-100 text-slate-600 text-xs uppercase">
          <tr><th class="text-left px-2 py-1">Equipo</th><th>PJ</th><th>G</th><th>E</th><th>P</th><th>GF</th><th>GC</th><th>DG</th><th>Pts</th></tr>
        </thead>
        <tbody>
          ${table.map((r,i)=>`
            <tr class="${i<2?'bg-emerald-50':''}">
              <td class="text-left px-2 py-1 font-medium flex items-center gap-2">${flag(r.code)} ${r.name}</td>
              <td class="text-center">${r.pj}</td><td class="text-center">${r.g}</td><td class="text-center">${r.e}</td>
              <td class="text-center">${r.p}</td><td class="text-center">${r.gf}</td><td class="text-center">${r.gc}</td>
              <td class="text-center">${r.dg>=0?'+':''}${r.dg}</td>
              <td class="text-center font-bold">${r.pts}</td>
            </tr>`).join("")}
        </tbody>
      </table>
    </div>`).join("");
  $("#view").innerHTML = `<div class="grid md:grid-cols-2 xl:grid-cols-3 gap-3">${html}</div>`;
}

async function renderTeams() {
  const data = await api.teams();
  $("#view").innerHTML = `
    <div class="bg-white rounded-xl shadow-sm overflow-hidden">
      <table class="w-full text-sm">
        <thead class="bg-slate-100 text-slate-600 text-xs uppercase">
          <tr><th class="text-left px-3 py-2">#</th><th class="text-left">Selección</th><th>Conf.</th><th>Elo</th><th>xG/p</th><th>xGA/p</th></tr>
        </thead>
        <tbody>
          ${data.map((t,i)=>`
            <tr class="border-t border-slate-100">
              <td class="px-3 py-1.5 text-slate-500">${i+1}</td>
              <td class="text-left font-medium flex items-center gap-2">${flag(t.code)} ${t.name}</td>
              <td class="text-center text-xs text-slate-600">${t.confederation}</td>
              <td class="text-center font-semibold tabular-nums">${t.elo}</td>
              <td class="text-center tabular-nums">${t.xg_for.toFixed(2)}</td>
              <td class="text-center tabular-nums">${t.xg_against.toFixed(2)}</td>
            </tr>`).join("")}
        </tbody>
      </table>
    </div>`;
}

async function renderAccuracy() {
  const a = await api.accuracy();
  if (!a.played_with_prediction) {
    $("#view").innerHTML = `<p class="text-slate-500 italic">Aún no hay partidos jugados con predicción guardada. Primero corré "Predecir todos".</p>`;
    return;
  }
  $("#view").innerHTML = `
    <div class="grid grid-cols-3 gap-3 mb-4">
      <div class="bg-white rounded-xl shadow-sm p-4"><div class="text-xs text-slate-500">Aciertos 1X2</div><div class="text-2xl font-bold mt-1">${pct(a.accuracy_1x2)}</div></div>
      <div class="bg-white rounded-xl shadow-sm p-4"><div class="text-xs text-slate-500">Brier (↓ mejor)</div><div class="text-2xl font-bold mt-1">${a.brier_score.toFixed(3)}</div></div>
      <div class="bg-white rounded-xl shadow-sm p-4"><div class="text-xs text-slate-500">N</div><div class="text-2xl font-bold mt-1">${a.played_with_prediction}</div></div>
    </div>
    <div class="bg-white rounded-xl shadow-sm overflow-hidden">
      <table class="w-full text-sm">
        <thead class="bg-slate-100 text-slate-600 text-xs uppercase">
          <tr><th class="text-left px-3 py-2">Partido</th><th>Real</th><th>Predicho</th><th>Score esperado</th><th>H/X/A</th><th></th></tr>
        </thead>
        <tbody>
          ${a.details.map(d=>`
            <tr class="border-t border-slate-100 ${d.hit?'bg-emerald-50/40':'bg-rose-50/40'}">
              <td class="px-3 py-1.5 flex items-center gap-2">${flag(d.home)} ${d.home} vs ${d.away} ${flag(d.away)}</td>
              <td class="text-center font-mono">${d.real}</td>
              <td class="text-center text-xs uppercase">${d.pred_outcome}</td>
              <td class="text-center font-mono">${d.pred_score}</td>
              <td class="text-center text-xs">${(d.p_home*100).toFixed(0)}/${(d.p_draw*100).toFixed(0)}/${(d.p_away*100).toFixed(0)}</td>
              <td class="text-center">${d.hit?'<span class="text-emerald-600 font-bold">✓</span>':'<span class="text-rose-600">✗</span>'}</td>
            </tr>`).join("")}
        </tbody>
      </table>
    </div>`;
}

function componentRow(name, comp, weight) {
  if (!comp) {
    return `<tr class="text-slate-400 italic">
      <td class="py-1">${name}</td>
      <td class="text-center" colspan="4">no disponible</td>
    </tr>`;
  }
  return `<tr class="border-t border-slate-100">
    <td class="py-1">${name}</td>
    <td class="text-center text-slate-500">${weight!=null?(weight*100).toFixed(0)+'%':'—'}</td>
    <td class="text-center">${pct(comp.p_home)}</td>
    <td class="text-center">${pct(comp.p_draw)}</td>
    <td class="text-center">${pct(comp.p_away)}</td>
  </tr>`;
}

// ===== Modal =====
function bindMatchClicks() {
  $$(".match-card").forEach(el => {
    el.addEventListener("click", () => openMatch(el.dataset.id));
  });
}

async function openMatch(id) {
  $("#modal").classList.remove("hidden");
  $("#modal-body").innerHTML = `<div class="p-8 text-center text-slate-500">Cargando…</div>`;
  let m = await api.match(id);
  let liveDetail = null;
  if (m.home && m.away) {
    // Pedimos siempre la predicción "fresca" para tener el detalle por modelo
    liveDetail = await api.predict(id);
    m = await api.match(id);
  }
  const p = m.prediction;
  $("#modal-body").innerHTML = `
    <div class="p-6 border-b border-slate-200 flex justify-between items-start">
      <div>
        <div class="text-xs uppercase text-slate-500 font-semibold">${m.stage === "GROUP" ? `Grupo ${m.group} · Jornada ${m.matchday}` : m.stage}</div>
        <div class="text-2xl font-bold mt-1">${flag(m.home)} ${m.home_name} vs ${m.away_name} ${flag(m.away)}</div>
        <div class="text-sm text-slate-500 mt-1">${fmtDate(m.datetime)}</div>
      </div>
      <button onclick="document.getElementById('modal').classList.add('hidden')" class="text-slate-400 hover:text-slate-600 text-2xl">×</button>
    </div>
    ${m.finished ? `<div class="bg-slate-50 px-6 py-4 text-center">
      <div class="text-xs uppercase text-slate-500 font-semibold">Resultado</div>
      <div class="text-4xl font-bold tabular-nums mt-1">${m.home_goals} - ${m.away_goals}</div>
    </div>`:''}
    ${p ? `
      <div class="p-6 space-y-5">
        <div>
          <div class="flex justify-between text-sm font-semibold mb-1">
            <span class="text-emerald-700">${m.home_name} ${pct(p.p_home)}</span>
            <span class="text-slate-500">Empate ${pct(p.p_draw)}</span>
            <span class="text-brand-600">${m.away_name} ${pct(p.p_away)}</span>
          </div>
          <div class="prob-bar">
            <div class="prob-home" style="width:${p.p_home*100}%"></div>
            <div class="prob-draw" style="width:${p.p_draw*100}%"></div>
            <div class="prob-away" style="width:${p.p_away*100}%"></div>
          </div>
        </div>

        <div class="grid grid-cols-2 gap-3">
          <div class="bg-slate-50 rounded-lg p-3">
            <div class="text-xs text-slate-500">Marcador más probable</div>
            <div class="text-2xl font-bold tabular-nums mt-1">${Math.round(p.pred_home_goals)} - ${Math.round(p.pred_away_goals)}</div>
          </div>
          <div class="bg-slate-50 rounded-lg p-3">
            <div class="text-xs text-slate-500">Confianza modelo</div>
            <div class="text-xl font-bold mt-1">${(p.confidence*100).toFixed(0)}%</div>
          </div>
          <div class="bg-slate-50 rounded-lg p-3">
            <div class="text-xs text-slate-500">Over 2.5 goles</div>
            <div class="text-xl font-bold mt-1">${pct(p.over_25)}</div>
          </div>
          <div class="bg-slate-50 rounded-lg p-3">
            <div class="text-xs text-slate-500">BTTS (ambos marcan)</div>
            <div class="text-xl font-bold mt-1">${pct(p.btts)}</div>
          </div>
        </div>

        ${liveDetail?.components ? `
        <div class="border-t border-slate-200 pt-4">
          <h4 class="font-semibold text-sm mb-2">📐 Voto de cada modelo matemático</h4>
          <table class="w-full text-xs">
            <thead><tr class="text-slate-500">
              <th class="text-left py-1">Modelo</th><th>Peso</th><th>Local</th><th>Empate</th><th>Visit.</th>
            </tr></thead>
            <tbody>
              ${componentRow("HGB+RF+LR (ML)", liveDetail.components.ml, liveDetail.weights.ml)}
              ${componentRow("Dixon-Coles (1997)", liveDetail.components.dixon_coles, liveDetail.weights.dixon_coles)}
              ${componentRow("Pi-ratings (2013)", liveDetail.components.pi_ratings, liveDetail.weights.pi_ratings)}
              ${componentRow("Elo + Poisson", liveDetail.components.elo_poisson, liveDetail.weights.elo_poisson)}
              <tr class="border-t-2 border-slate-300 font-bold bg-amber-50">
                <td class="py-1">META-ENSEMBLE</td><td class="text-center">—</td>
                <td class="text-center">${pct(p.p_home)}</td>
                <td class="text-center">${pct(p.p_draw)}</td>
                <td class="text-center">${pct(p.p_away)}</td>
              </tr>
            </tbody>
          </table>
          <p class="text-xs text-slate-500 mt-2">Marcador entero predicho usando <b>${liveDetail.score_source}</b>.</p>
        </div>
        `:""}

        <div class="text-xs text-slate-500 italic">
          Modelo: <span class="font-mono">${p.model_version}</span> · Generado: ${fmtDate(p.created_at)}
        </div>
      </div>
    `:`<div class="p-6 text-slate-500 italic">No se pudo generar predicción para este partido (probablemente knockout aún sin equipos).</div>`}
    <div class="px-6 py-3 bg-slate-50 border-t border-slate-200 flex justify-end gap-2">
      <button id="btn-repredict" class="bg-brand-600 hover:bg-brand-700 text-white text-sm font-semibold px-3 py-2 rounded-lg">⟳ Re-predecir</button>
    </div>`;
  $("#btn-repredict")?.addEventListener("click", async () => {
    await api.predict(id);
    openMatch(id);
  });
}

// ===== Tab nav =====
$$(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    $$(".tab").forEach(t => t.classList.remove("active","border-brand-600","text-brand-700"));
    $$(".tab").forEach(t => t.classList.add("border-transparent","text-slate-500"));
    btn.classList.add("active","border-brand-600","text-brand-700");
    btn.classList.remove("border-transparent","text-slate-500");
    STATE.tab = btn.dataset.tab;
    renderTab();
  });
});

function renderTab() {
  $("#view").innerHTML = `<div class="text-slate-400 italic">Cargando…</div>`;
  if (STATE.tab === "quiniela") return renderQuiniela();
  if (STATE.tab === "live") return renderLive();
  if (STATE.tab === "upcoming") return renderUpcoming();
  if (STATE.tab === "played") return renderPlayed();
  if (STATE.tab === "groups") return renderGroups();
  if (STATE.tab === "teams") return renderTeams();
  if (STATE.tab === "accuracy") return renderAccuracy();
}

// Auto-refresh del tab Live cada 30s
setInterval(() => {
  if (STATE.tab === "live") renderLive();
  updateLiveBadge();
}, 30000);

async function updateLiveBadge() {
  try {
    const s = await api.liveStatus();
    const badge = $("#live-badge");
    if (s.live_count > 0) {
      badge.textContent = s.live_count;
      badge.classList.remove("hidden");
    } else {
      badge.classList.add("hidden");
    }
  } catch {}
}
updateLiveBadge();

// ===== Header buttons =====
$("#btn-refresh").addEventListener("click", async () => {
  const btn = $("#btn-refresh"); btn.disabled = true;
  const oldTxt = btn.textContent;
  btn.textContent = "Scrapeando…";
  try {
    const r = await api.refresh();
    btn.textContent = r.ok ? `✓ ${r.updated} actualizados` : "✗ Sin red";
  } catch { btn.textContent = "✗ Error"; }
  setTimeout(() => { btn.textContent = oldTxt; btn.disabled = false; loadKpis(); renderTab(); }, 1800);
});

$("#btn-train").addEventListener("click", async () => {
  const btn = $("#btn-train"); const oldTxt = btn.textContent;
  btn.disabled = true; btn.textContent = "Entrenando…";
  try {
    await api.trainNow();
  } catch { btn.textContent = "✗ Error"; }
  // Polling al status
  const poll = setInterval(async () => {
    const s = await api.trainStatus();
    if (!s.running) {
      clearInterval(poll);
      btn.disabled = false; btn.textContent = oldTxt;
      loadKpis(); renderTab(); updateTrainStatus();
    }
  }, 1500);
});

async function updateTrainStatus() {
  try {
    const s = await api.trainStatus();
    const dot = $("#train-dot"); const txt = $("#train-text");
    if (s.running) {
      dot.className = "w-2 h-2 rounded-full bg-amber-300 animate-pulse";
      txt.textContent = `entrenando… (${s.last_trigger || "auto"})`;
    } else if (s.last_error) {
      dot.className = "w-2 h-2 rounded-full bg-rose-400";
      txt.textContent = `error: ${s.last_error.slice(0,30)}`;
    } else if (s.last_finished) {
      dot.className = "w-2 h-2 rounded-full bg-emerald-400";
      const acc = s.last_metrics?.accuracy_1x2;
      txt.textContent = acc
        ? `acc ${(acc*100).toFixed(1)}% · ${s.total_runs} runs`
        : `listo (${s.total_runs} runs)`;
    } else {
      dot.className = "w-2 h-2 rounded-full bg-slate-300";
      txt.textContent = "auto-train: pendiente";
    }
    $("#train-status").title = `Próximo: cada ${Math.round(s.scheduled_every_s/3600)}h · Último: ${s.last_finished || '—'} (${s.last_duration_s||'?'}s)`;
  } catch {}
}

setInterval(updateTrainStatus, 5000);
updateTrainStatus();

$("#btn-predict-all").addEventListener("click", async () => {
  const btn = $("#btn-predict-all"); btn.disabled = true;
  const oldTxt = btn.textContent; btn.textContent = "Calculando…";
  try {
    const r = await api.predictAll();
    btn.textContent = `✓ ${r.predicted} listos`;
  } catch { btn.textContent = "✗ Error"; }
  setTimeout(() => { btn.textContent = oldTxt; btn.disabled = false; loadKpis(); renderTab(); }, 1600);
});

// Init
loadKpis();
renderTab();
