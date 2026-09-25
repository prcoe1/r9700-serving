/* r9700 dashboard — live stats + sweep history.
 * Single file (no bundler): sections are factories, not copy-paste.
 * Backend route map: POST /api/{bench,depth,conc}, GET/POST …/status,
 * …/cancel, …/history, …/download/{ts}, …/clear, …/delete, plus the legacy
 * GET/POST /api/history* compat shim for bench. */
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmtT = ts => new Date((ts || 0) * 1000).toLocaleString();
const fmtClock = ts => new Date(ts).toLocaleTimeString([], {hour12:false});

/* ---------------------------------------------------------------- charts */

let kvChart, reqChart, hitChart, thrChart, histChart, depthChart, concChart;

function chartDefaults(){
  if(typeof Chart === 'undefined') return false;
  // One place for the dark theme; every chart below inherits it.
  Chart.defaults.color = '#9aa0b8';
  Chart.defaults.borderColor = 'rgba(42,46,62,0.6)';
  Chart.defaults.font.family = 'Inter,system-ui,sans-serif';
  Chart.defaults.plugins.tooltip.callbacks = Chart.defaults.plugins.tooltip.callbacks || {};
  return true;
}

function baseLineOptions(extraScales){
  return {
    animation:false, responsive:true, maintainAspectRatio:false,
    // 1s polls × 600 pts: skip overlap instead of drawing every point.
    parsing:false, normalized:true,
    interaction:{mode:'index', intersect:false},
    plugins:{
      legend:{display:false},
      decimation:{enabled:true, algorithm:'min-max'},
      tooltip:{callbacks:{title: items => items.length ? fmtClock(items[0].parsed.x) : ''}},
    },
    scales:Object.assign({x:{type:'linear', display:false}}, extraScales),
  };
}

function makeLineChart(canvas, label, color, yOpts){
  return new Chart(canvas, {
    type:'line',
    data:{datasets:[{label, data:[], borderColor:color, backgroundColor:color+'22', tension:0.25, pointRadius:0, borderWidth:1.5, fill:true}]},
    options: baseLineOptions({y:Object.assign({beginAtZero:true, ticks:{font:{size:10}}}, yOpts)}),
  });
}

/* Dual-axis pp/tg chart shared by bench history, depth and concurrency. */
function makePpTgChart(canvas, xTitle){
  return new Chart(canvas, {
    type:'line',
    data:{datasets:[
      {label:'pp', data:[], borderColor:'#6c7bff', backgroundColor:'transparent', tension:0.2, pointRadius:3, yAxisID:'y'},
      {label:'tg', data:[], borderColor:'#2ecc71', backgroundColor:'transparent', tension:0.2, pointRadius:3, yAxisID:'y1'},
    ]},
    options:{
      animation:false, responsive:true, maintainAspectRatio:false,
      parsing:false, normalized:true,
      interaction:{mode:'index', intersect:false},
      plugins:{
        legend:{labels:{color:'#e6e8ef'}},
        decimation:{enabled:true, algorithm:'min-max'},
        tooltip:{callbacks:{title: items => items.length ? String(items[0].parsed.x) : ''}},
      },
      scales:{
        x:{type:'linear', ticks:{}, title:{display:!!xTitle, text:xTitle || '', color:'#9aa0b8'}},
        y:{type:'linear', position:'left', beginAtZero:true, title:{display:true, text:'pp t/s', color:'#9aa0b8'}},
        y1:{type:'linear', position:'right', beginAtZero:true, grid:{drawOnChartArea:false}, title:{display:true, text:'tg t/s', color:'#9aa0b8'}},
      },
    },
  });
}

function initCharts(){
  if(!chartDefaults()){
    console.error('Chart.js not loaded');
    $('benchStatus').textContent = 'Chart.js failed to load — tables still work, check /static/chart.umd.min.js';
    return;
  }
  try{
    kvChart = makeLineChart($('kvChart'), 'kv %', '#6c7bff', {min:0, max:100});
    reqChart = makeLineChart($('reqChart'), 'running', '#2ecc71', {ticks:{precision:0}});
    hitChart = makeLineChart($('hitChart'), 'hit %', '#f1c40f', {min:0, max:100});
    thrChart = new Chart($('thrChart'), {
      type:'line',
      data:{datasets:[
        {label:'prompt t/s (EMA)', data:[], borderColor:'#6c7bff', backgroundColor:'rgba(108,123,255,0.12)', tension:0.25, pointRadius:0, borderWidth:1.5, fill:true, yAxisID:'y'},
        {label:'gen t/s (EMA)', data:[], borderColor:'#2ecc71', backgroundColor:'transparent', tension:0.25, pointRadius:0, borderWidth:1.5, yAxisID:'y1'},
      ]},
      options: baseLineOptions({
        y:{beginAtZero:true, position:'left'},
        y1:{beginAtZero:true, position:'right', grid:{drawOnChartArea:false}},
      }),
    });
    // Bench history x is wall-clock: reuse the time tooltip from baseLineOptions.
    histChart = new Chart($('histChart'), {
      type:'line',
      data:{datasets:[
        {label:'pp2048', data:[], borderColor:'#6c7bff', backgroundColor:'transparent', tension:0.2, pointRadius:3, yAxisID:'y'},
        {label:'tg32', data:[], borderColor:'#2ecc71', backgroundColor:'transparent', tension:0.2, pointRadius:3, yAxisID:'y1'},
        {label:'tg128', data:[], borderColor:'#e74c3c', backgroundColor:'transparent', tension:0.2, pointRadius:3, yAxisID:'y1'},
      ]},
      options:{
        animation:false, responsive:true, maintainAspectRatio:false,
        parsing:false, normalized:true,
        interaction:{mode:'index', intersect:false},
        plugins:{
          legend:{labels:{color:'#e6e8ef'}},
          decimation:{enabled:true, algorithm:'min-max'},
          tooltip:{callbacks:{title: items => items.length ? fmtClock(items[0].parsed.x) : ''}},
        },
        scales:{
          x:{type:'linear', ticks:{callback: v => fmtClock(v)}},
          y:{type:'linear', position:'left', beginAtZero:true, title:{display:true, text:'pp t/s', color:'#9aa0b8'}},
          y1:{type:'linear', position:'right', beginAtZero:true, grid:{drawOnChartArea:false}, title:{display:true, text:'tg t/s', color:'#9aa0b8'}},
        },
      },
    });
    depthChart = makePpTgChart($('depthChart'), 'depth tokens');
    concChart = makePpTgChart($('concChart'), 'depth tokens @ max conc');
  }catch(e){
    console.error('chart init failed', e);
  }
}

/* ------------------------------------------------------------------ store */

/* Timestamped ring — seeded from GET /api/metrics/history so charts survive
 * reload/tab switches (previously client-memory only). {x,y} points so the
 * shared time axis tooltips work. */
const RING_CAP = 900; // 15 min @ 1s poll (matches server maxlen)
const ring = {t:[], kv:[], running:[], hit:[], pp:[], tg:[]};
let failStreak = 0;

function ringPush(tsMs, kv, running, hit, pp, tg){
  ring.t.push(tsMs);
  ring.kv.push(kv != null ? {x:tsMs, y:kv} : null);
  ring.running.push(running != null ? {x:tsMs, y:running} : null);
  ring.hit.push(hit != null ? {x:tsMs, y:hit} : null);
  ring.pp.push(pp != null ? {x:tsMs, y:pp} : null);
  ring.tg.push(tg != null ? {x:tsMs, y:tg} : null);
  while(ring.t.length > RING_CAP){
    ring.t.shift(); ring.kv.shift(); ring.running.shift();
    ring.hit.shift(); ring.pp.shift(); ring.tg.shift();
  }
}

async function seedRing(){
  try{
    const r = await fetch('/api/metrics/history?limit=600');
    if(!r.ok) return;
    for(const p of await r.json()){
      if(p.ts == null) continue;
      const tsMs = p.ts > 1e12 ? p.ts : p.ts * 1000;
      ringPush(tsMs, p.kv, p.running, p.hit, p.pp, p.tg);
    }
    updateRing();
  }catch(e){ console.warn('ring seed failed', e); }
}

function updateRing(){
  if(kvChart){kvChart.data.datasets[0].data = ring.kv.filter(Boolean); kvChart.update();}
  if(reqChart){reqChart.data.datasets[0].data = ring.running.filter(Boolean); reqChart.update();}
  if(hitChart){hitChart.data.datasets[0].data = ring.hit.filter(Boolean); hitChart.update();}
  if(thrChart){thrChart.data.datasets[0].data = ring.pp.filter(Boolean); thrChart.data.datasets[1].data = ring.tg.filter(Boolean); thrChart.update();}
}

const fmtRate = v => v == null || !isFinite(v) ? '—' : v >= 1e6 ? (v/1e6).toFixed(2)+'M' : v >= 1e4 ? (v/1e3).toFixed(1)+'k' : v >= 100 ? v.toFixed(0) : v.toFixed(1);
const fmtCount = n => n == null ? '—' : n >= 1e6 ? (n/1e6).toFixed(2)+'M' : n >= 1e3 ? (n/1e3).toFixed(1)+'k' : Math.round(n).toLocaleString();

function showError(msg){
  const el = $('errBanner');
  if(!msg){ el.hidden = true; el.textContent = ''; return; }
  el.hidden = false; el.textContent = msg;
}

/* --------------------------------------------------------------- live tick */

// Client-side deltas are a fallback only: the server now ships EMA/windowed
// keys (pp_ema, tg_ema, hit_window_pct). Prefer those;
// the KV-growth prompt estimate is gone (it conflated decode KV growth).
let prevQueries = null, prevHits = null;

async function tick(){
  if(document.hidden) return; // visibility pause — no stacking, no gaps
  try{
    const r = await fetch('/api/metrics');
    const healthEl = $('health');
    if(!r.ok){ healthEl.textContent = 'vLLM down'; healthEl.className = 'badge bad'; throw new Error('metrics '+r.status); }
    healthEl.textContent = 'vLLM up'; healthEl.className = 'badge ok';
    showError(null);
    failStreak = 0;
    const j = await r.json();
    const kv = j['vllm:kv_cache_usage_perc'];
    const kvMaxTok = j['kv_cache_size_tokens'];
    const kvBlocks = j['num_gpu_blocks'];
    const kvBlockSize = j['block_size'];
    const kvDtype = j['cache_dtype'];
    if(kv != null){
      $('kv').textContent = (kv*100).toFixed(1);
      $('kvbar').style.width = (kv*100).toFixed(1)+'%';
      const usedTok = kvMaxTok != null ? Math.round(kv * kvMaxTok) : null;
      $('kvdetail').textContent = usedTok != null ? `${usedTok.toLocaleString()} / ${kvMaxTok.toLocaleString()} tokens used` : `gpu_cache ${j['vllm:gpu_cache_usage_perc'] != null ? (j['vllm:gpu_cache_usage_perc']*100).toFixed(1)+'%' : '—'}`;
      if(kvMaxTok != null) $('kvMax').textContent = `max ${kvMaxTok.toLocaleString()} tokens`;
      if(kvBlocks != null && kvBlockSize != null){
        $('kvCap').textContent = `${kvBlocks} blocks × ${kvBlockSize} · ${kvDtype || ''} · util ${j['gpu_memory_utilization'] ?? '—'}`.trim();
      } else if(kvMaxTok != null){
        $('kvCap').textContent = `capacity ${kvMaxTok.toLocaleString()} tokens · ${kvDtype || ''}`.trim();
      }
    }
    const asInt = v => v != null ? String(Math.round(v)) : null;
    $('running').textContent = asInt(j['vllm:num_requests_running']) ?? '—';
    $('waiting').textContent = asInt(j['vllm:num_requests_waiting']) ?? '—';
    $('swapped').textContent = asInt(j['vllm:num_requests_swapped']) ?? '0';
    // Throughput: server EMA first, raw single-poll delta as fallback.
    const ppRate = j['pp_ema'] ?? j['pp_raw'] ?? null;
    const tgRate = j['tg_ema'] ?? j['tg_raw'] ?? null;
    $('ppRate').textContent = fmtRate(ppRate);
    $('tgRate').textContent = fmtRate(tgRate);
    const pt = j['vllm:prompt_tokens_total'], gt = j['vllm:generation_tokens_total'];
    $('thrDetail').textContent = (pt != null && gt != null) ? `cumulative ${fmtCount(pt)} / ${fmtCount(gt)} tokens` : '';
    const spec = j['spec_accept_pct'];
    $('specDetail').textContent = spec != null ? `spec-decode acceptance ${spec.toFixed(1)}% (cumulative)` : '';
    // Prefix hits: server window first, else client 1s delta, else cumulative.
    const hit = j['prefix_hit_pct'];
    const queries = j['vllm:prefix_cache_queries_total'];
    const hits = j['vllm:prefix_cache_hits_total'];
    let deltaPct = null, deltaQ = null, deltaH = null;
    if(prevQueries != null && prevHits != null && queries != null && hits != null){
      deltaQ = queries - prevQueries;
      deltaH = hits - prevHits;
      if(deltaQ > 0 && deltaH >= 0) deltaPct = deltaH / deltaQ * 100;
    }
    const winPct = j['hit_window_pct'];
    $('hitpct').textContent = (winPct ?? deltaPct ?? hit ?? NaN).toFixed?.(1) ?? '—';
    if($('hitpct').textContent === 'NaN') $('hitpct').textContent = '—';
    $('hitCumulative').textContent = hit != null ? `(cum ${hit.toFixed(1)}%)` : '';
    $('cumPct').textContent = hit != null ? hit.toFixed(1) : '—';
    $('hitdetail').textContent = (queries != null && hits != null) ? `${fmtCount(hits)} hits / ${fmtCount(queries)} queries (${hit != null ? hit.toFixed(1)+'% hitrate' : '—'})` : '';
    $('hitDelta').textContent = deltaQ != null ? `Δ ${fmtCount(deltaH)} hits / ${fmtCount(deltaQ)} queries (${deltaPct != null ? deltaPct.toFixed(1)+'% hitrate' : '—'}) per poll` : 'Δ — (waiting for next poll)';
    $('blocksize').textContent = `block_size ${j['block_size'] ?? '—'} · mamba ${j['mamba_cache_mode'] ?? '—'}`;
    $('lastTs').textContent = `last update ${new Date(j.ts*1000).toLocaleTimeString()} · vllm_up ${j.vllm_up}`;
    ringPush(Date.now(), kv != null ? kv*100 : null, j['vllm:num_requests_running'] ?? null, winPct ?? deltaPct ?? hit ?? null, ppRate, tgRate);
    updateRing();
    prevQueries = queries;
    prevHits = hits;
  }catch(e){
    failStreak++;
    $('lastTs').textContent = 'metrics fetch failed: '+e;
    if(failStreak >= 3) showError('vLLM metrics unreachable ('+e+') — retrying with backoff.');
  }
}

/* setTimeout chain (not setInterval): a slow /metrics can never stack
 * overlapping ticks, and failures back off 1s → 10s instead of hammering. */
function scheduleTick(){
  tick().finally(() => {
    const delay = failStreak === 0 ? 1000 : Math.min(1000 * Math.pow(2, failStreak - 1), 10000);
    setTimeout(scheduleTick, delay);
  });
}

/* -------------------------------------------------------- sweep framework */

/* One definition per sweep kind drives status polling, run/cancel, history
 * table+chart, download/delete/clear — previously ×3 copy-pasted blocks. */
const SWEEPS = {
  bench: {
    runBtn:'runBench', cancelBtn:'cancelBench', refreshBtn:'refreshHist', clearBtn:'clearBench',
    statusEl:'benchStatus', logEl:'benchLog', tableId:'histTable', cardId:'benchCard',
    chartWrap:'benchChartWrap', tableWrap:'benchTableWrap',
    base:'/api/bench', histUrl:'/api/bench/history',
    runLabel:'▶ Run bench (pp2048 + tg32/128 ×3)', activeLabel:'⏳ bench running…',
    emptyCols:8, emptyText:'no runs yet',
  },
  depth: {
    runBtn:'runDepth', cancelBtn:'cancelDepth', refreshBtn:'refreshDepth', clearBtn:'clearDepth',
    statusEl:'depthStatus', logEl:'depthLog', tableId:'depthTable', cardId:'depthCard',
    chartWrap:'depthChartWrap', tableWrap:'depthTableWrap',
    base:'/api/depth', histUrl:'/api/depth/history',
    runLabel:'▶ Run depth sweep (0–200K) — est. 20min', activeLabel:'⏳ depth running…',
    confirmRun:'Run full depth sweep 0–200K? est. 20min, hits vLLM heavily. LAN-exposed.',
    emptyCols:9, emptyText:'no sweeps yet',
  },
  conc: {
    runBtn:'runConc', cancelBtn:'cancelConc', refreshBtn:'refreshConc', clearBtn:'clearConc',
    statusEl:'concStatus', logEl:'concLog', tableId:'concTable', cardId:'concCard',
    chartWrap:'concChartWrap', tableWrap:'concTableWrap',
    base:'/api/conc', histUrl:'/api/conc/history',
    runLabel:'▶ Run concurrency sweep — est. 40min', activeLabel:'⏳ conc running…',
    confirmRun:'Run concurrency sweep up to max_num_seqs? est. 40min, hits vLLM with corpus.',
    emptyCols:10, emptyText:'no sweeps yet',
  },
};
const sweepStatus = {bench:{running:false}, depth:{running:false}, conc:{running:false}};

function setLog(kind, text){
  const el = $(SWEEPS[kind].logEl);
  el.style.display = 'block';
  el.textContent = text;
  el.scrollTop = el.scrollHeight; // follow the live tail
}

function anySweepRunning(){
  return Object.values(sweepStatus).some(s => s.running);
}

/* Backend 409s a second concurrent sweep; reflect that in ALL run buttons so
 * the conflict is visible before clicking, not after. */
function syncRunButtons(){
  const any = anySweepRunning();
  for(const [kind, d] of Object.entries(SWEEPS)){
    const run = $(d.runBtn), cancel = $(d.cancelBtn);
    if(sweepStatus[kind].running){ run.disabled = true; cancel.disabled = false; run.textContent = d.activeLabel; }
    else { run.disabled = any; cancel.disabled = true; run.textContent = d.runLabel; }
  }
}

function setEmptyState(kind, isEmpty, statusText){
  const d = SWEEPS[kind];
  $(d.statusEl).textContent = statusText;
  $(d.chartWrap).style.display = isEmpty ? 'none' : 'block';
  $(d.tableWrap).style.display = isEmpty ? 'none' : 'block';
  $(d.cardId).classList.toggle('minimised', isEmpty);
}

function dlCell(base, ts){
  return ts ? `<a class="btn download small" href="${base}/download/${ts}" download>download</a>` : '';
}
function delCell(kind, ts){
  return ts ? `<button class="btn danger small" data-del-kind="${kind}" data-del-ts="${ts}">delete</button>` : '';
}

function extractThroughput(rec){
  let pp = rec.pp2048 ?? null, tg32 = rec.tg32 ?? null, tg128 = rec.tg128 ?? null;
  const bms = rec.benchmarks || rec.results || [];
  if(Array.isArray(bms)){
    for(const b of bms){
      const ps = b.prompt_size ?? b.pp ?? b.prompt_tokens;
      const rs = b.response_size ?? b.tg ?? b.response_tokens ?? b.num_tokens;
      const ppM = b.pp_throughput?.mean ?? b.pp?.mean ?? b.pp_tokens_per_second ?? b.throughput;
      const tgM = b.tg_throughput?.mean ?? b.tg?.mean ?? b.tg_tokens_per_second;
      if(ps === 2048 && rs === 32){ if(ppM != null) pp = ppM; if(tgM != null) tg32 = tgM; }
      else if(ps === 2048 && rs === 128){ if(pp == null && ppM != null) pp = ppM; if(tgM != null) tg128 = tgM; }
    }
  }
  return {pp, tg32, tg128};
}

function depthPoints(rec, key){
  let pts = rec[key] || [];
  if(pts.length === 0 && rec.benchmarks){
    pts = rec.benchmarks.map(b => ({depth:b.depth ?? b.context_size ?? 0, concurrency:b.concurrency, pp:b.pp_throughput?.mean, tg:b.tg_throughput?.mean, ttft:b.e2e_ttft?.mean}));
  }
  return pts;
}

/* --- per-kind renderers (table rows + chart series only) --- */

function renderBench(items){
  const tbody = $('histTable').querySelector('tbody');
  tbody.innerHTML = '';
  const labels = [], ppD = [], t32 = [], t128 = [];
  items.slice(-20).forEach(rec => {
    const {pp, tg32, tg128} = extractThroughput(rec);
    const tsMs = (rec.ts || 0) * 1000;
    labels.push(tsMs); ppD.push(pp != null ? {x:tsMs, y:pp} : null);
    t32.push(tg32 != null ? {x:tsMs, y:tg32} : null); t128.push(tg128 != null ? {x:tsMs, y:tg128} : null);
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${fmtT(rec.ts)}</td><td>${esc(rec.model) || ''}</td>` +
      `<td class="num">${pp != null ? pp.toFixed(0) : '—'}</td><td class="num">${tg32 != null ? tg32.toFixed(1) : '—'}</td><td class="num">${tg128 != null ? tg128.toFixed(1) : '—'}</td>` +
      `<td class="num">${rec.elapsed ? rec.elapsed.toFixed(0)+'s' : ''}</td><td>${dlCell('/api/bench', rec.ts)}</td><td>${delCell('bench', rec.ts)}</td>`;
    tbody.appendChild(tr);
  });
  if(histChart){
    histChart.data.datasets[0].data = ppD.filter(Boolean);
    histChart.data.datasets[1].data = t32.filter(Boolean);
    histChart.data.datasets[2].data = t128.filter(Boolean);
    histChart.update();
  }
  window._hist = items;
}

function renderDepth(items){
  const tbody = $('depthTable').querySelector('tbody');
  tbody.innerHTML = '';
  const latest = items[items.length - 1];
  const points = depthPoints(latest, 'depth_results');
  items.slice(-20).forEach(rec => {
    const dr = rec.depth_results || [];
    const lastPt = dr[dr.length - 1] || {};
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${fmtT(rec.ts)}</td><td>${esc(rec.model) || ''}</td>` +
      `<td class="num">${dr.length ? dr.length : rec.depths?.length || '—'} depths</td>` +
      `<td class="num">${lastPt.pp != null ? lastPt.pp.toFixed(0) : '—'}</td><td class="num">${lastPt.tg != null ? lastPt.tg.toFixed(1) : '—'}</td>` +
      `<td class="num">${lastPt.ttft != null ? lastPt.ttft.toFixed(1) : '—'}</td>` +
      `<td class="num">${rec.elapsed ? rec.elapsed.toFixed(0)+'s' : ''}</td><td>${dlCell('/api/depth', rec.ts)}</td><td>${delCell('depth', rec.ts)}</td>`;
    tbody.appendChild(tr);
  });
  if(points.length > 0 && points.length <= 20){
    const hdr = document.createElement('tr');
    hdr.innerHTML = '<td colspan="9" class="muted">latest sweep depths (TTFT in table only)</td>';
    tbody.appendChild(hdr);
    points.forEach(p => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td></td><td></td><td class="num">${p.depth}</td><td class="num">${p.pp != null ? p.pp.toFixed(0) : '—'}</td>` +
        `<td class="num">${p.tg != null ? p.tg.toFixed(1) : '—'}</td><td class="num">${p.ttft != null ? p.ttft.toFixed(1) : '—'}</td><td></td><td></td><td></td>`;
      tbody.appendChild(tr);
    });
  }
  if(depthChart){
    depthChart.data.datasets[0].data = points.map(p => p.pp != null ? {x:p.depth, y:p.pp} : null).filter(Boolean);
    depthChart.data.datasets[1].data = points.map(p => p.tg != null ? {x:p.depth, y:p.tg} : null).filter(Boolean);
    depthChart.update();
  }
  window._depthHist = items;
}

function renderConc(items){
  const tbody = $('concTable').querySelector('tbody');
  tbody.innerHTML = '';
  const latest = items[items.length - 1];
  const points = depthPoints(latest, 'conc_results');
  items.slice(-20).forEach(rec => {
    const cr = rec.conc_results || [];
    const conc = rec.concurrency || cr[0]?.concurrency || rec.max_conc || '—';
    const lastPt = cr[cr.length - 1] || {};
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${fmtT(rec.ts)}</td><td>${esc(rec.model) || ''}</td>` +
      `<td class="num">${cr.length ? cr[0].depth+'…'+cr[cr.length-1].depth : rec.depths ? rec.depths[0]+'…'+rec.depths[rec.depths.length-1] : '—'}</td>` +
      `<td class="num">${conc}</td><td class="num">${lastPt.pp != null ? lastPt.pp.toFixed(0) : '—'}</td>` +
      `<td class="num">${lastPt.tg != null ? lastPt.tg.toFixed(1) : '—'}</td><td class="num">${lastPt.ttft != null ? lastPt.ttft.toFixed(1) : '—'}</td>` +
      `<td class="num">${rec.elapsed ? rec.elapsed.toFixed(0)+'s' : ''}</td><td>${dlCell('/api/conc', rec.ts)}</td><td>${delCell('conc', rec.ts)}</td>`;
    tbody.appendChild(tr);
  });
  if(points.length > 0){
    const hdr = document.createElement('tr');
    hdr.innerHTML = `<td colspan="10" class="muted">latest sweep @ conc ${(latest.concurrency || latest.max_conc || '')} (TTFT in table only)</td>`;
    tbody.appendChild(hdr);
    points.forEach(p => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td></td><td></td><td class="num">${p.depth}</td><td class="num">${p.concurrency ?? ''}</td>` +
        `<td class="num">${p.pp != null ? p.pp.toFixed(0) : '—'}</td><td class="num">${p.tg != null ? p.tg.toFixed(1) : '—'}</td>` +
        `<td class="num">${p.ttft != null ? p.ttft.toFixed(1) : '—'}</td><td></td><td></td><td></td>`;
      tbody.appendChild(tr);
    });
  }
  if(concChart){
    concChart.data.datasets[0].data = points.map(p => p.pp != null ? {x:p.depth, y:p.pp} : null).filter(Boolean);
    concChart.data.datasets[1].data = points.map(p => p.tg != null ? {x:p.depth, y:p.tg} : null).filter(Boolean);
    concChart.update();
  }
  window._concHist = items;
}

const RENDERERS = {bench:renderBench, depth:renderDepth, conc:renderConc};
const COUNT_NOUN = {bench:['bench run', 'bench runs'], depth:['sweep', 'sweeps'], conc:['sweep', 'sweeps']};

async function refreshSweep(kind){
  const d = SWEEPS[kind];
  try{
    const r = await fetch(d.histUrl);
    if(!r.ok) throw new Error('history fetch '+r.status);
    const items = await r.json();
    const [one, many] = COUNT_NOUN[kind];
    const tbody = $(d.tableId).querySelector('tbody');
    if(items.length === 0){
      tbody.innerHTML = `<tr><td colspan="${d.emptyCols}" class="muted">${d.emptyText}</td></tr>`;
      setEmptyState(kind, true, kind === 'bench' ? 'no history yet — run a bench or `just bench-json`' : 'no sweeps yet');
      if(histChart && kind === 'bench'){ histChart.data.datasets.forEach(ds => ds.data = []); histChart.update(); }
    } else {
      setEmptyState(kind, false, `${items.length} ${items.length === 1 ? one : many}`);
      RENDERERS[kind](items);
    }
  }catch(e){
    console.error('refresh '+kind+' failed', e);
    $(d.statusEl).textContent = 'history load failed: '+e;
  }
}

async function pollSweepStatus(kind){
  const d = SWEEPS[kind];
  try{
    const r = await fetch(d.base+'/status').then(r => r.json());
    sweepStatus[kind].running = !!r.running;
    syncRunButtons();
    if(r.running){
      $(d.statusEl).textContent = kind === 'bench' ? 'bench running — ~2-4 min for 3 runs' : kind === 'depth' ? (r.progress || 'depth sweep running — est. 20min') : 'concurrency sweep running — est. 40min';
      if(r.log) setLog(kind, r.log.slice(-10000));
      setTimeout(() => pollSweepStatus(kind), kind === 'bench' ? 2000 : 3000);
    } else {
      if(r.last) refreshSweep(kind);
      else syncRunButtons();
      if(r.log) setLog(kind, r.log.slice(-10000));
    }
  }catch(e){ console.warn('status poll '+kind+' failed', e); }
}

function pollAllSweeps(){
  for(const kind of Object.keys(SWEEPS)) pollSweepStatus(kind);
}

function wireSweep(kind, confirmRun){
  const d = SWEEPS[kind];
  if(confirmRun) d.confirmRun = confirmRun; // boot-time default; refreshSweepLabels() rewrites it live
  $(d.runBtn).addEventListener('click', async () => {
    if(!confirm(d.confirmRun)) return;
    const r = await fetch(d.base, {method:'POST'});
    if(r.status === 409){ alert('Another bench is already running'); pollAllSweeps(); return; }
    if(!r.ok){ alert('Start failed: '+ await r.text()); return; }
    pollSweepStatus(kind);
  });
  $(d.cancelBtn).addEventListener('click', async () => {
    if(!confirm('Cancel the running '+kind+' sweep?')) return;
    await fetch(d.base+'/cancel', {method:'POST'});
    pollSweepStatus(kind);
  });
  $(d.refreshBtn).addEventListener('click', () => refreshSweep(kind));
  $(d.clearBtn).addEventListener('click', async () => {
    const n = $(d.tableId).querySelectorAll('tbody tr').length;
    if(!confirm('Clear all '+kind+' history? This deletes all '+n+' rows.')) return;
    const r = await fetch(d.base+'/clear', {method:'POST'});
    if(!r.ok){ alert('Clear failed: '+await r.text()); return; }
    $(d.logEl).style.display = 'none'; $(d.logEl).textContent = '';
    refreshSweep(kind);
  });
}

/* Delegated delete buttons (rows are re-rendered, so no inline onclick). */
document.addEventListener('click', async e => {
  const btn = e.target.closest('[data-del-kind]');
  if(!btn) return;
  const kind = btn.dataset.delKind, ts = btn.dataset.delTs;
  if(!confirm('Delete this '+kind+' run?')) return;
  const r = await fetch(SWEEPS[kind].base+'/delete', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ts:Number(ts)})});
  if(!r.ok){ alert('Delete failed: '+await r.text()); return; }
  refreshSweep(kind);
});

/* Collapse toggles per sweep card (persisted). */
function initCollapse(){
  document.querySelectorAll('[data-collapse]').forEach(btn => {
    const card = $(btn.dataset.collapse);
    let saved = null;
    try{ saved = localStorage.getItem('collapse-'+btn.dataset.collapse); }catch{}
    if(saved === '1'){ card.classList.add('collapsed'); btn.setAttribute('aria-expanded', 'false'); btn.textContent = '▸'; }
    btn.addEventListener('click', () => {
      const collapsed = card.classList.toggle('collapsed');
      btn.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
      btn.textContent = collapsed ? '▸' : '▾';
      try{ localStorage.setItem('collapse-'+btn.dataset.collapse, collapsed ? '1' : '0'); }catch{}
    });
  });
}

async function refreshInfo(){
  try{
    const r = await fetch('/api/info');
    if(!r.ok) throw new Error(r.status);
    const j = await r.json();
    if(j.model) $('model').textContent = `model: ${j.model}`;
    $('kvmeta').textContent = `kv: ${j.kv_dtype || '—'}`;
    // Pin the requests chart scale to the configured concurrency cap so a
    // full engine (2/2 running) reads as full-scale, not autoscaled noise.
    if(j.max_conc != null && reqChart){
      reqChart.options.scales.y.max = j.max_conc;
      reqChart.update();
      $('reqCap').textContent = `cap ${j.max_conc} concurrent (max-num-seqs)`;
    }
  }catch(e){
    console.warn('info fetch failed', e);
  }
}

/* Benchmarks-tab history refreshes pause when the tab is hidden — the
 * status pollers still run (they drive the run/cancel buttons). */
function benchmarksVisible(){
  return $('tab-benchmarks').classList.contains('active') && !document.hidden;
}

/* Rewrite sweep section titles, run-button labels, confirm prompts, and
 * corpus notes from the live sweep geometry (GET /api/sweep-config).
 * Hardcoded HTML/JS text stays as the fallback when the fetch fails. */
async function refreshSweepLabels(){
  try{
    const r = await fetch('/api/sweep-config');
    if(!r.ok) return;
    const c = await r.json();
    const depths = c.depths || [];
    if(!depths.length) return;
    const top = depths[depths.length - 1];
    const k = v => v >= 1000 ? Math.round(v / 1000) + 'K' : String(v);
    const kl = v => v >= 1000 ? Math.round(v / 1000) + 'k' : String(v);
    const range = `0–${k(top)}`;
    const list = depths.map(kl).join(' ');
    const conc = c.max_conc || '?';
    const ds = $('depthSubtitle');
    if(ds) ds.textContent = `full ${range} corpus (pp2048/tg1024, TTFT in table) — est. 20min`;
    const cs = $('concSubtitle');
    if(cs) cs.textContent = `same depths ${range} at max concurrency (×${conc} parallel, pp2048/tg1024) — est. 40min`;
    SWEEPS.depth.runLabel = `▶ Run depth sweep (${range}) — est. 20min`;
    SWEEPS.depth.confirmRun = `Run full depth sweep ${range}? est. 20min, hits vLLM heavily. LAN-exposed.`;
    SWEEPS.conc.runLabel = `▶ Run concurrency sweep (${range} ×${conc}) — est. 40min`;
    SWEEPS.conc.confirmRun = `Run concurrency sweep ${range} up to conc ${conc}? est. 40min, hits vLLM with corpus.`;
    const dn = $('depthNote');
    if(dn) dn.innerHTML = `History file: <code>observability/depth_history.jsonl</code> (limit 20, book corpus <code>--depth ${list} --tg 1024 --no-cache</code>).`;
    const cn = $('concNote');
    if(cn) cn.innerHTML = `History file: <code>observability/conc_history.jsonl</code> (limit 20, <code>--depth ${list} --tg 1024 --concurrency ${conc} --no-cache</code>). Same book corpus as depth, but with <code>x = max_num_seqs</code> parallel runs.`;
    syncRunButtons(); // re-paint idle run buttons with the new labels
  }catch(e){ console.warn('sweep-config fetch failed', e); }
}

function initTabs(){
  const btns = Array.from(document.querySelectorAll('.tab-btn'));
  const panels = document.querySelectorAll('.tab-panel');
  const names = btns.map(b => b.dataset.tab);
  function resizeCharts(){
    // double rAF: the panel must be laid out (non-zero width) first, or the
    // charts resize to 0 (notably iOS Safari).
    requestAnimationFrame(() => requestAnimationFrame(() => {
      [kvChart, reqChart, hitChart, thrChart, histChart, depthChart, concChart].forEach(c => {
        try{ if(c) c.resize(); }catch{}
      });
      updateRing();
      if(histChart) histChart.update();
      if(depthChart) depthChart.update();
      if(concChart) concChart.update();
    }));
  }
  function activate(name, focus=false){
    btns.forEach(b => {
      const isActive = b.dataset.tab === name;
      b.classList.toggle('active', isActive);
      b.setAttribute('aria-selected', isActive ? 'true' : 'false');
      b.setAttribute('tabindex', isActive ? '0' : '-1');
      if(isActive && focus) b.focus();
    });
    panels.forEach(p => {
      p.classList.toggle('active', p.id === 'tab-'+name);
    });
    resizeCharts();
    // entering Benchmarks with stale tables → refresh once
    if(name === 'benchmarks'){ refreshSweep('bench'); refreshSweep('depth'); refreshSweep('conc'); }
    try{ localStorage.setItem('dashboard-tab', name); }catch{}
    history.replaceState(null, '', '#'+name);
  }
  btns.forEach(b => b.addEventListener('click', () => activate(b.dataset.tab)));
  // Keyboard nav per the ARIA tabs pattern
  btns.forEach((b, i) => b.addEventListener('keydown', e => {
    let j = null;
    if(e.key === 'ArrowRight') j = (i+1) % btns.length;
    else if(e.key === 'ArrowLeft') j = (i-1+btns.length) % btns.length;
    else if(e.key === 'Home') j = 0;
    else if(e.key === 'End') j = btns.length-1;
    if(j != null){ e.preventDefault(); activate(names[j], true); }
  }));
  // restore from hash or storage, default stats
  let initial = 'stats';
  if(location.hash && names.includes(location.hash.slice(1))) initial = location.hash.slice(1);
  else try{ const s = localStorage.getItem('dashboard-tab'); if(s && names.includes(s)) initial = s; }catch{}
  activate(initial);
  window.addEventListener('hashchange', () => {
    const h = location.hash.slice(1);
    if(names.includes(h)) activate(h);
  });
}

(async () => {
  initCharts();
  initTabs();
  initCollapse();
  wireSweep('bench', 'Run bench now? This hits vLLM with pp2048 + tg32/128 ×3 (~2-4 min). LAN-exposed — anyone can trigger.');
  wireSweep('depth');
  wireSweep('conc');
  $('vllmurl').textContent = location.hostname+':8180';
  await seedRing();
  await refreshSweepLabels(); // live ladder before first paint of labels
  scheduleTick();
  refreshSweep('bench'); refreshSweep('depth'); refreshSweep('conc'); refreshInfo();
  setInterval(() => { if(benchmarksVisible()) refreshSweep('bench'); }, 10000);
  setInterval(() => { if(benchmarksVisible()) refreshSweep('depth'); }, 15000);
  setInterval(() => { if(benchmarksVisible()) refreshSweep('conc'); }, 15000);
  setInterval(() => { refreshInfo(); refreshSweepLabels(); }, 15000);
  document.addEventListener('visibilitychange', () => { if(!document.hidden){ tick(); pollAllSweeps(); } });
  pollAllSweeps();
})();
