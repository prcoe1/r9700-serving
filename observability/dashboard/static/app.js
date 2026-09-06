const $ = id => document.getElementById(id);
let kvChart, reqChart, hitChart, histChart, depthChart, concChart;
const ring = {kv:[], running:[], waiting:[], hit:[]};
let prevQueries = null, prevHits = null;

function makeChart(canvas, label, color){
  return new Chart(canvas, {
    type:'line',
    data:{labels:[], datasets:[{label, data:[], borderColor:color, backgroundColor:color+'22', tension:0.25, pointRadius:0, borderWidth:1.5, fill:true}]},
    options:{animation:false, responsive:true, maintainAspectRatio:false, plugins:{legend:{display:false}}, scales:{x:{display:false}, y:{beginAtZero:true, ticks:{color:'#9aa0b8', font:{size:10}}}} }
  });
}

function initCharts(){
  if(typeof Chart==='undefined'){
    console.error('Chart.js not loaded');
    $('benchStatus').textContent='Chart.js failed to load — table still works, check /static/chart.umd.min.js';
    return;
  }
  try{
    kvChart = makeChart($('kvChart'), 'kv %', '#6c7bff');
    kvChart.options.scales.y.min = 0;
    kvChart.options.scales.y.max = 100;
    kvChart.options.scales.y.ticks = {color:'#9aa0b8', font:{size:10}, stepSize:25};
    kvChart.update();
    reqChart = makeChart($('reqChart'), 'running', '#2ecc71');
    hitChart = makeChart($('hitChart'), 'hit %', '#f1c40f');
    hitChart.options.scales.y.min = 0;
    hitChart.options.scales.y.max = 100;
    hitChart.options.scales.y.ticks = {color:'#9aa0b8', font:{size:10}, stepSize:25};
    hitChart.update();
    histChart = new Chart($('histChart'), {
      type:'line',
      data:{labels:[], datasets:[
        {label:'pp2048', data:[], borderColor:'#6c7bff', backgroundColor:'transparent', tension:0.2, pointRadius:3, yAxisID:'y'},
        {label:'tg32', data:[], borderColor:'#2ecc71', backgroundColor:'transparent', tension:0.2, pointRadius:3, yAxisID:'y1'},
        {label:'tg128', data:[], borderColor:'#e74c3c', backgroundColor:'transparent', tension:0.2, pointRadius:3, yAxisID:'y1'},
      ]},
      options:{
        animation:false, responsive:true,
        plugins:{legend:{labels:{color:'#e6e8ef'}}},
        scales:{
          x:{ticks:{color:'#9aa0b8', maxRotation:45}},
          y:{type:'linear', position:'left', ticks:{color:'#9aa0b8'}, title:{display:true, text:'pp t/s', color:'#9aa0b8'}},
          y1:{type:'linear', position:'right', grid:{drawOnChartArea:false}, ticks:{color:'#9aa0b8'}, title:{display:true, text:'tg t/s', color:'#9aa0b8'}}
        }
      }
    });
    depthChart = new Chart($('depthChart'), {
      type:'line',
      data:{labels:[], datasets:[
        {label:'pp', data:[], borderColor:'#6c7bff', backgroundColor:'transparent', tension:0.2, pointRadius:3, yAxisID:'y'},
        {label:'tg', data:[], borderColor:'#2ecc71', backgroundColor:'transparent', tension:0.2, pointRadius:3, yAxisID:'y1'},
      ]},
      options:{
        animation:false, responsive:true,
        plugins:{legend:{labels:{color:'#e6e8ef'}}},
        scales:{
          x:{ticks:{color:'#9aa0b8'}, title:{display:true, text:'depth tokens', color:'#9aa0b8'}},
          y:{type:'linear', position:'left', ticks:{color:'#9aa0b8'}, title:{display:true, text:'pp t/s', color:'#9aa0b8'}},
          y1:{type:'linear', position:'right', grid:{drawOnChartArea:false}, ticks:{color:'#9aa0b8'}, title:{display:true, text:'tg t/s', color:'#9aa0b8'}}
        }
      }
    });
    concChart = new Chart($('concChart'), {
      type:'line',
      data:{labels:[], datasets:[
        {label:'pp', data:[], borderColor:'#6c7bff', backgroundColor:'transparent', tension:0.2, pointRadius:3, yAxisID:'y'},
        {label:'tg', data:[], borderColor:'#2ecc71', backgroundColor:'transparent', tension:0.2, pointRadius:3, yAxisID:'y1'},
      ]},
      options:{
        animation:false, responsive:true,
        plugins:{legend:{labels:{color:'#e6e8ef'}}},
        scales:{
          x:{ticks:{color:'#9aa0b8'}, title:{display:true, text:'depth tokens @ max conc', color:'#9aa0b8'}},
          y:{type:'linear', position:'left', ticks:{color:'#9aa0b8'}, title:{display:true, text:'pp t/s', color:'#9aa0b8'}},
          y1:{type:'linear', position:'right', grid:{drawOnChartArea:false}, ticks:{color:'#9aa0b8'}, title:{display:true, text:'tg t/s', color:'#9aa0b8'}}
        }
      }
    });
  }catch(e){
    console.error('chart init failed', e);
  }
}

async function tick(){
  try{
    const r = await fetch('/api/metrics');
    if(!r.ok) throw new Error(r.statusText);
    const j = await r.json();
    const kv = j['vllm:kv_cache_usage_perc'];
    const kvMaxTok = j['kv_cache_size_tokens'];
    const kvBlocks = j['num_gpu_blocks'];
    const kvBlockSize = j['block_size'];
    const kvDtype = j['cache_dtype'];
    if(kv!=null){
      $('kv').textContent = (kv*100).toFixed(1);
      $('kvbar').style.width = (kv*100).toFixed(1)+'%';
      const usedTok = kvMaxTok!=null? Math.round(kv * kvMaxTok) : null;
      $('kvdetail').textContent = usedTok!=null? `${usedTok.toLocaleString()} / ${kvMaxTok.toLocaleString()} tokens used` : `gpu_cache ${j['vllm:gpu_cache_usage_perc']!=null?(j['vllm:gpu_cache_usage_perc']*100).toFixed(1)+'%':'—'}`;
      if(kvMaxTok!=null){
        $('kvMax').textContent = `max ${kvMaxTok.toLocaleString()} tokens`;
      }
      if(kvBlocks!=null && kvBlockSize!=null){
        $('kvCap').textContent = `${kvBlocks} blocks × ${kvBlockSize} · ${kvDtype||''} · util ${j['gpu_memory_utilization']??'—'}`.trim();
      } else if(kvMaxTok!=null){
        $('kvCap').textContent = `capacity ${kvMaxTok.toLocaleString()} tokens · ${kvDtype||''}`.trim();
      }
    }
    $('running').textContent = j['vllm:num_requests_running']!=null? j['vllm:num_requests_running']: '—';
    $('waiting').textContent = j['vllm:num_requests_waiting']!=null? j['vllm:num_requests_waiting']: '—';
    $('swapped').textContent = j['vllm:num_requests_swapped']!=null? j['vllm:num_requests_swapped']: '0';
    const hit = j['prefix_hit_pct'];
    const queries = j['vllm:prefix_cache_queries_total'];
    const hits = j['vllm:prefix_cache_hits_total'];
    let deltaPct = null, deltaQ = null, deltaH = null;
    if(prevQueries!=null && prevHits!=null && queries!=null && hits!=null){
      deltaQ = queries - prevQueries;
      deltaH = hits - prevHits;
      if(deltaQ>0) deltaPct = deltaH / deltaQ * 100;
    }
    if(deltaPct!=null){
      $('hitpct').textContent = deltaPct.toFixed(1);
    } else if(hit!=null){
      $('hitpct').textContent = hit.toFixed(1);
    }
    const fmt = n => n==null? '—' : n>=1e6? (n/1e6).toFixed(2)+'M' : n>=1e3? (n/1e3).toFixed(1)+'k' : n.toLocaleString();
    $('hitCumulative').textContent = hit!=null? `(cum ${hit.toFixed(1)}%)` : '';
    $('cumPct').textContent = hit!=null? hit.toFixed(1) : '—';
    $('hitdetail').textContent = (queries!=null && hits!=null)? `${fmt(hits)} hits / ${fmt(queries)} queries (${hit!=null?hit.toFixed(1)+'% hitrate':'—'})` : `cumulative queries ${queries??'—'} · hits ${hits??'—'}`;
    $('hitDelta').textContent = deltaQ!=null? `Δ ${fmt(deltaH)} hits / ${fmt(deltaQ)} queries (${deltaPct!=null?deltaPct.toFixed(1)+'% hitrate':'—'}) per poll` : `Δ — (waiting for next poll)`;
    $('blocksize').textContent = `block_size ${j['block_size']??'—'} · mamba ${j['mamba_cache_mode']??'—'}`;
    $('lastTs').textContent = `last update ${new Date(j.ts*1000).toLocaleTimeString()} · vllm_up ${j.vllm_up}`;
    const chartVal = deltaPct!=null? deltaPct : (hit??0);
    ring.kv.push(kv!=null? kv*100 : null);
    ring.running.push(j['vllm:num_requests_running']??0);
    ring.hit.push(chartVal);
    if(ring.kv.length>300) {ring.kv.shift(); ring.running.shift(); ring.hit.shift();}
    updateRing();
    prevQueries = queries;
    prevHits = hits;
  }catch(e){
    $('lastTs').textContent = 'metrics fetch failed: '+e;
  }
  try{
    const h = await fetch('/api/health').then(r=>r.json());
    const el=$('health');
    if(h.vllm_up){ el.textContent='vLLM up'; el.className='badge ok'; } else { el.textContent='vLLM down'; el.className='badge bad'; }
  }catch{}
}

function updateRing(){
  const labels = ring.kv.map((_,i)=>i);
  if(kvChart){kvChart.data.labels = labels; kvChart.data.datasets[0].data = ring.kv; kvChart.update();}
  if(reqChart){reqChart.data.labels = labels; reqChart.data.datasets[0].data = ring.running; reqChart.update();}
  if(hitChart){hitChart.data.labels = labels; hitChart.data.datasets[0].data = ring.hit; hitChart.update();}
}

function extractThroughput(rec){
  let pp=null, tg32=null, tg128=null;
  if(rec.pp2048!=null) pp=rec.pp2048;
  if(rec.tg32!=null) tg32=rec.tg32;
  if(rec.tg128!=null) tg128=rec.tg128;
  const bms = rec.benchmarks || rec.results || [];
  if(Array.isArray(bms)){
    for(const b of bms){
      const ps = b.prompt_size ?? b.pp ?? b.prompt_tokens;
      const rs = b.response_size ?? b.tg ?? b.response_tokens ?? b.num_tokens;
      const ppM = b.pp_throughput?.mean ?? b.pp?.mean ?? b.pp_tokens_per_second ?? b.throughput;
      const tgM = b.tg_throughput?.mean ?? b.tg?.mean ?? b.tg_tokens_per_second;
      if(ps===2048 && rs===32){
        if(ppM!=null) pp=ppM;
        if(tgM!=null) tg32=tgM;
      } else if(ps===2048 && rs===128){
        if(pp==null && ppM!=null) pp=ppM;
        if(tgM!=null) tg128=tgM;
      }
    }
  }
  if(pp==null && typeof rec.raw==='string'){
    const m = rec.raw.match(/pp\s*2048[^0-9]*([0-9.]+)/i);
    if(m) pp=parseFloat(m[1]);
  }
  return {pp, tg32, tg128};
}

async function refreshHistory(){
  try{
    const r = await fetch('/api/history');
    if(!r.ok) throw new Error('history fetch '+r.status);
    const items = await r.json();
    $('benchStatus').textContent = items.length ? `${items.length} bench runs` : 'no history yet — run a bench or `just bench-json`';
    const tbody = $('histTable').querySelector('tbody');
    tbody.innerHTML='';
    const labels=[], ppData=[], tg32Data=[], tg128Data=[];
    const chartWrap=$('benchChartWrap'), tableWrap=$('benchTableWrap');
    if(items.length===0){
      tbody.innerHTML='<tr><td colspan="8" class="muted">no runs yet</td></tr>';
      if(chartWrap) chartWrap.style.display='none';
      if(tableWrap) tableWrap.style.display='none';
      // minimise card when empty (keep header + buttons)
      $('benchCard').classList.add('minimised');
    } else {
      if(chartWrap) chartWrap.style.display='block';
      if(tableWrap) tableWrap.style.display='block';
      $('benchCard').classList.remove('minimised');
    }
    items.slice(-20).forEach(rec=>{
      const d = new Date((rec.ts||0)*1000);
      const label = d.toLocaleString();
      const {pp, tg32, tg128} = extractThroughput(rec);
      labels.push(label);
      ppData.push(pp); tg32Data.push(tg32); tg128Data.push(tg128);
      const tr=document.createElement('tr');
      const dlBtn = rec.ts? `<a href="/api/history/download/${rec.ts}" download>download</a>` : '';
      const delBtn = rec.ts? `<button class="btn danger small" onclick="deleteBench(${rec.ts})">delete</button>` : '';
      tr.innerHTML = `<td>${label}</td><td>${rec.model||''}</td><td>${pp!=null?pp.toFixed(0):'—'}</td><td>${tg32!=null?tg32.toFixed(1):'—'}</td><td>${tg128!=null?tg128.toFixed(1):'—'}</td><td>${rec.elapsed? rec.elapsed.toFixed(0)+'s':''}</td><td>${dlBtn}</td><td>${delBtn}</td>`;
      tbody.appendChild(tr);
    });
    if(histChart){
      histChart.data.labels = labels;
      histChart.data.datasets[0].data = ppData;
      histChart.data.datasets[1].data = tg32Data;
      histChart.data.datasets[2].data = tg128Data;
      histChart.update();
    }
    window._hist = items;
    if(items.length){
      const last = items[items.length-1];
      if(last.model) $('model').textContent = `model: ${last.model}`;
    }
  }catch(e){
    console.error('refreshHistory failed', e);
    $('benchStatus').textContent='history load failed: '+e;
  }
}

window.deleteBench = async (ts)=>{
  if(!confirm('Delete this bench run?')) return;
  const r=await fetch('/api/history/delete',{method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ts})});
  if(!r.ok){ alert('Delete failed: '+await r.text()); return; }
  refreshHistory();
};

async function pollBenchStatus(){
  const r = await fetch('/api/bench/status').then(r=>r.json());
  const btn=$('runBench');
  if(r.running){
    btn.disabled=true; btn.textContent='⏳ bench running…';
    $('benchStatus').textContent='bench running — ~2-4 min for 3 runs';
    if(r.log){ $('benchLog').style.display='block'; $('benchLog').textContent = r.log.slice(-8000); }
    setTimeout(pollBenchStatus, 2000);
  } else {
    btn.disabled=false; btn.textContent='▶ Run bench (pp2048 + tg32/128 ×3)';
    if(r.last) refreshHistory();
    if(r.log && r.running===false && r.log){ $('benchLog').style.display='block'; $('benchLog').textContent = r.log.slice(-8000); }
  }
}

$('runBench').addEventListener('click', async ()=>{
  if(!confirm('Run bench now? This hits vLLM with pp2048 + tg32/128 ×3 (~2-4 min). LAN-exposed — anyone can trigger.')) return;
  const r = await fetch('/api/bench', {method:'POST'});
  if(r.status===409){ alert('Bench already running'); return; }
  if(!r.ok){ alert('Bench start failed: '+ await r.text()); return; }
  pollBenchStatus();
});
$('refreshHist').addEventListener('click', refreshHistory);
$('clearBench').addEventListener('click', async ()=>{
  if(!confirm('Clear all bench history? This deletes all '+$('histTable').querySelectorAll('tbody tr').length+' rows.')) return;
  const r=await fetch('/api/history/clear',{method:'POST'});
  if(!r.ok){ alert('Clear failed: '+await r.text()); return; }
  $('benchLog').style.display='none'; $('benchLog').textContent='';
  refreshHistory();
});

// Depth sweep
async function refreshDepth(){
  try{
    const r = await fetch('/api/depth/history');
    if(!r.ok) throw new Error('depth history '+r.status);
    const items = await r.json();
    $('depthStatus').textContent = items.length ? `${items.length} sweeps` : 'no sweeps yet';
    const tbody=$('depthTable').querySelector('tbody');
    tbody.innerHTML='';
    const labels=[], ppData=[], tgData=[];
    const chartWrap=$('depthChartWrap'), tableWrap=$('depthTableWrap');
    if(items.length===0){
      tbody.innerHTML='<tr><td colspan="9" class="muted">no sweeps yet</td></tr>';
      if(chartWrap) chartWrap.style.display='none';
      if(tableWrap) tableWrap.style.display='none';
      $('depthCard').classList.add('minimised');
    } else {
      if(chartWrap) chartWrap.style.display='block';
      if(tableWrap) tableWrap.style.display='block';
      $('depthCard').classList.remove('minimised');
      // Show latest sweep in chart, table shows all depths of latest sweep + history
      const latest = items[items.length-1];
      const results = latest.depth_results || [];
      // If depth_results not present, try to parse benchmarks
      let points = results;
      if(points.length===0 && latest.benchmarks){
        points = latest.benchmarks.map(b=>({depth:b.depth||b.context_size||0, pp:b.pp_throughput?.mean, tg:b.tg_throughput?.mean, ttft:b.e2e_ttft?.mean}));
      }
      points.forEach(p=>{
        labels.push(String(p.depth));
        ppData.push(p.pp); tgData.push(p.tg);
      });
      // Table: last 20 sweeps expanded? Show only latest sweep depths for brevity, plus history rows
      // For history, show one row per sweep with summary
      if(items.length>1){
        // history summary rows
        items.slice(-20).forEach(rec=>{
          const d=new Date((rec.ts||0)*1000).toLocaleString();
          const dr=rec.depth_results || [];
          const lastPt=dr[dr.length-1]||{};
          const tr=document.createElement('tr');
          tr.innerHTML=`<td>${d}</td><td>${rec.model||''}</td><td>${dr.length?dr.length:rec.depths?.length||'—'} depths</td><td>${lastPt.pp!=null?lastPt.pp.toFixed(0):'—'}</td><td>${lastPt.tg!=null?lastPt.tg.toFixed(1):'—'}</td><td>${lastPt.ttft!=null?lastPt.ttft.toFixed(1):'—'}</td><td>${rec.elapsed?rec.elapsed.toFixed(0)+'s':''}</td><td><a href="/api/depth/download/${rec.ts}" download>download</a></td><td><button class="btn danger small" onclick="deleteDepth(${rec.ts})">delete</button></td>`;
          tbody.appendChild(tr);
        });
      } else if(items.length===1){
        // single sweep: show delete for the only row
        const rec=items[0];
        const d=new Date((rec.ts||0)*1000).toLocaleString();
        const dr=rec.depth_results || [];
        const lastPt=dr[dr.length-1]||{};
        const tr=document.createElement('tr');
        tr.innerHTML=`<td>${d}</td><td>${rec.model||''}</td><td>${dr.length?dr.length:rec.depths?.length||'—'} depths</td><td>${lastPt.pp!=null?lastPt.pp.toFixed(0):'—'}</td><td>${lastPt.tg!=null?lastPt.tg.toFixed(1):'—'}</td><td>${lastPt.ttft!=null?lastPt.ttft.toFixed(1):'—'}</td><td>${rec.elapsed?rec.elapsed.toFixed(0)+'s':''}</td><td><a href="/api/depth/download/${rec.ts}" download>download</a></td><td><button class="btn danger small" onclick="deleteDepth(${rec.ts})">delete</button></td>`;
        tbody.appendChild(tr);
      }
      // Also expand latest depths as sub-rows if few sweeps
      if(points.length>0 && points.length<=20){
        const hdr=document.createElement('tr'); hdr.innerHTML='<td colspan="9" class="muted">latest sweep depths (TTFT in table only)</td>'; tbody.appendChild(hdr);
        points.forEach(p=>{
          const tr=document.createElement('tr');
          tr.innerHTML=`<td></td><td></td><td>${p.depth}</td><td>${p.pp!=null?p.pp.toFixed(0):'—'}</td><td>${p.tg!=null?p.tg.toFixed(1):'—'}</td><td>${p.ttft!=null?p.ttft.toFixed(1):'—'}</td><td></td><td></td><td></td>`;
          tbody.appendChild(tr);
        });
      }
    }
    if(depthChart){
      depthChart.data.labels = labels;
      depthChart.data.datasets[0].data = ppData;
      depthChart.data.datasets[1].data = tgData;
      depthChart.update();
    }
    window._depthHist = items;
  }catch(e){
    console.error('refreshDepth failed', e);
    $('depthStatus').textContent='depth load failed: '+e;
  }
}
window.deleteDepth = async (ts)=>{
  if(!confirm('Delete this depth sweep?')) return;
  const r=await fetch('/api/depth/delete',{method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ts})});
  if(!r.ok){ alert('Delete failed: '+await r.text()); return; }
  refreshDepth();
};
async function pollDepthStatus(){
  const r=await fetch('/api/depth/status').then(r=>r.json());
  const btn=$('runDepth'), cancel=$('cancelDepth');
  if(r.running){
    btn.disabled=true; cancel.disabled=false; btn.textContent='⏳ depth running…';
    $('depthStatus').textContent='depth sweep running — est. 20min';
    if(r.log){$('depthLog').style.display='block'; $('depthLog').textContent=r.log.slice(-10000);}
    if(r.progress) $('depthStatus').textContent=r.progress;
    setTimeout(pollDepthStatus, 3000);
  } else {
    btn.disabled=false; cancel.disabled=true; btn.textContent='▶ Run depth sweep (0–200K) — est. 20min';
    if(r.last) refreshDepth();
    if(r.log){$('depthLog').style.display='block'; $('depthLog').textContent=r.log.slice(-10000);}
  }
}
$('runDepth').addEventListener('click', async()=>{
  if(!confirm('Run full depth sweep 0–200K? est. 20min, hits vLLM heavily. LAN-exposed.')) return;
  const r=await fetch('/api/depth',{method:'POST'});
  if(r.status===409){alert('Another bench running'); return;}
  if(!r.ok){alert('Depth start failed: '+await r.text()); return;}
  pollDepthStatus();
});
$('cancelDepth').addEventListener('click', async()=>{
  if(!confirm('Cancel depth sweep?')) return;
  await fetch('/api/depth/cancel',{method:'POST'});
});
$('refreshDepth').addEventListener('click', refreshDepth);
$('clearDepth').addEventListener('click', async()=>{
  if(!confirm('Clear all depth history?')) return;
  const r=await fetch('/api/depth/clear',{method:'POST'});
  if(!r.ok){ alert('Clear failed: '+await r.text()); return; }
  $('depthLog').style.display='none'; $('depthLog').textContent='';
  refreshDepth();
});

// Concurrency sweep (same depths 0-200K at max concurrency)
async function refreshConc(){
  try{
    const r=await fetch('/api/conc/history');
    if(!r.ok) throw new Error('conc history '+r.status);
    const items=await r.json();
    $('concStatus').textContent = items.length ? `${items.length} sweeps` : 'no sweeps yet';
    const tbody=$('concTable').querySelector('tbody');
    tbody.innerHTML='';
    const labels=[], ppData=[], tgData=[];
    const chartWrap=$('concChartWrap'), tableWrap=$('concTableWrap');
    if(items.length===0){
      tbody.innerHTML='<tr><td colspan="10" class="muted">no sweeps yet</td></tr>';
      if(chartWrap) chartWrap.style.display='none';
      if(tableWrap) tableWrap.style.display='none';
      $('concCard').classList.add('minimised');
    } else {
      if(chartWrap) chartWrap.style.display='block';
      if(tableWrap) tableWrap.style.display='block';
      $('concCard').classList.remove('minimised');
      const latest=items[items.length-1];
      const results=latest.conc_results || [];
      let points=results;
      if(points.length===0 && latest.benchmarks){
        points=latest.benchmarks.map(b=>({depth:b.depth||b.context_size||0, concurrency:b.concurrency, pp:b.pp_throughput?.mean, tg:b.tg_throughput?.mean, ttft:b.e2e_ttft?.mean}));
      }
      points.forEach(p=>{
        labels.push(String(p.depth));
        ppData.push(p.pp); tgData.push(p.tg);
      });
      items.slice(-20).forEach(rec=>{
        const d=new Date((rec.ts||0)*1000).toLocaleString();
        const cr=rec.conc_results||[];
        const conc=rec.concurrency || (cr[0]?.concurrency) || rec.max_conc || '—';
        const lastPt=cr[cr.length-1]||{};
        const tr=document.createElement('tr');
        tr.innerHTML=`<td>${d}</td><td>${rec.model||''}</td><td>${cr.length?cr[0].depth+'…'+cr[cr.length-1].depth : rec.depths?rec.depths[0]+'…'+rec.depths[rec.depths.length-1] : '—'}</td><td>${conc}</td><td>${lastPt.pp!=null?lastPt.pp.toFixed(0):'—'}</td><td>${lastPt.tg!=null?lastPt.tg.toFixed(1):'—'}</td><td>${lastPt.ttft!=null?lastPt.ttft.toFixed(1):'—'}</td><td>${rec.elapsed?rec.elapsed.toFixed(0)+'s':''}</td><td><a href="/api/conc/download/${rec.ts}" download>download</a></td><td><button class="btn danger small" onclick="deleteConc(${rec.ts})">delete</button></td>`;
        tbody.appendChild(tr);
      });
      if(points.length>0){
        const hdr=document.createElement('tr'); hdr.innerHTML='<td colspan="10" class="muted">latest sweep @ conc '+ (latest.concurrency||latest.max_conc||'') +' (TTFT in table only)</td>'; tbody.appendChild(hdr);
        points.forEach(p=>{
          const tr=document.createElement('tr');
          tr.innerHTML=`<td></td><td></td><td>${p.depth}</td><td>${p.concurrency}</td><td>${p.pp!=null?p.pp.toFixed(0):'—'}</td><td>${p.tg!=null?p.tg.toFixed(1):'—'}</td><td>${p.ttft!=null?p.ttft.toFixed(1):'—'}</td><td></td><td></td><td></td>`;
          tbody.appendChild(tr);
        });
      }
    }
    if(concChart){
      concChart.data.labels = labels;
      concChart.data.datasets[0].data = ppData;
      concChart.data.datasets[1].data = tgData;
      concChart.update();
    }
    window._concHist = items;
  }catch(e){
    console.error('refreshConc failed', e);
    $('concStatus').textContent='conc load failed: '+e;
  }
}
window.deleteConc = async (ts)=>{
  if(!confirm('Delete this concurrency sweep?')) return;
  const r=await fetch('/api/conc/delete',{method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ts})});
  if(!r.ok){ alert('Delete failed: '+await r.text()); return; }
  refreshConc();
};
async function pollConcStatus(){
  const r=await fetch('/api/conc/status').then(r=>r.json());
  const btn=$('runConc'), cancel=$('cancelConc');
  if(r.running){
    btn.disabled=true; cancel.disabled=false; btn.textContent='⏳ conc running…';
    $('concStatus').textContent='concurrency sweep running — est. 40min';
    if(r.log){$('concLog').style.display='block'; $('concLog').textContent=r.log.slice(-10000);}
    setTimeout(pollConcStatus, 3000);
  } else {
    btn.disabled=false; cancel.disabled=true; btn.textContent='▶ Run concurrency sweep — est. 40min';
    if(r.last) refreshConc();
    if(r.log){$('concLog').style.display='block'; $('concLog').textContent=r.log.slice(-10000);}
  }
}
$('runConc').addEventListener('click', async()=>{
  if(!confirm('Run concurrency sweep up to max_num_seqs? est. 40min, hits vLLM with corpus.')) return;
  const r=await fetch('/api/conc',{method:'POST'});
  if(r.status===409){alert('Another bench running'); return;}
  if(!r.ok){alert('Conc start failed: '+await r.text()); return;}
  pollConcStatus();
});
$('cancelConc').addEventListener('click', async()=>{
  if(!confirm('Cancel concurrency sweep?')) return;
  await fetch('/api/conc/cancel',{method:'POST'});
});
$('refreshConc').addEventListener('click', refreshConc);
$('clearConc').addEventListener('click', async()=>{
  if(!confirm('Clear all concurrency history?')) return;
  const r=await fetch('/api/conc/clear',{method:'POST'});
  if(!r.ok){ alert('Clear failed: '+await r.text()); return; }
  $('concLog').style.display='none'; $('concLog').textContent='';
  refreshConc();
});

async function refreshInfo(){
  try{
    const r = await fetch('/api/info');
    if(!r.ok) throw new Error(r.status);
    const j = await r.json();
    if(j.model) $('model').textContent = `model: ${j.model}`;
    if(j.kv_dtype) $('kvmeta').textContent = `kv: ${j.kv_dtype}`;
    else $('kvmeta').textContent = `kv: —`;
  }catch(e){
    console.warn('info fetch failed', e);
  }
}

function initTabs(){
  const btns = document.querySelectorAll('.tab-btn');
  const panels = document.querySelectorAll('.tab-panel');
  function activate(name){
    btns.forEach(b=>{
      const isActive = b.dataset.tab===name;
      b.classList.toggle('active', isActive);
      b.setAttribute('aria-selected', isActive?'true':'false');
    });
    panels.forEach(p=>{
      p.classList.toggle('active', p.id==='tab-'+name);
    });
    // resize charts that may have been hidden
    setTimeout(()=>{
      [kvChart, reqChart, hitChart, histChart, depthChart, concChart].forEach(c=>{
        try{ if(c) c.resize();}
        catch{}
      });
      updateRing();
      if(histChart) histChart.update();
      if(depthChart) depthChart.update();
      if(concChart) concChart.update();
    }, 50);
    try{ localStorage.setItem('dashboard-tab', name); }catch{}
    history.replaceState(null,'','#'+name);
  }
  btns.forEach(b=> b.addEventListener('click', ()=> activate(b.dataset.tab)));
  // restore from hash or storage, default stats
  let initial='stats';
  if(location.hash && ['stats','benchmarks'].includes(location.hash.slice(1))) initial=location.hash.slice(1);
  else try{ const s=localStorage.getItem('dashboard-tab'); if(s && ['stats','benchmarks'].includes(s)) initial=s; }catch{}
  activate(initial);
  window.addEventListener('hashchange', ()=>{
    const h=location.hash.slice(1);
    if(['stats','benchmarks'].includes(h)) activate(h);
  });
}

(async()=>{
  initCharts();
  initTabs();
  fetch('/api/metrics').then(r=>r.json()).catch(()=>null);
  $('vllmurl').textContent = location.hostname+':8180';
  tick(); refreshHistory(); refreshDepth(); refreshConc(); refreshInfo();
  setInterval(tick, 1000);
  setInterval(refreshHistory, 10000);
  setInterval(refreshDepth, 15000);
  setInterval(refreshConc, 15000);
  setInterval(refreshInfo, 15000);
  pollBenchStatus(); pollDepthStatus(); pollConcStatus();
})();
