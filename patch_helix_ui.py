"""
patch_helix_ui.py — 在 index.html 加入 HELIX 模式
用法：python3 ~/UFC/patch_helix_ui.py ~/UFC/index.html
"""

import sys, shutil, re
from pathlib import Path
from datetime import datetime

if len(sys.argv) < 2:
    print("用法: python3 patch_helix_ui.py ~/UFC/index.html")
    sys.exit(1)

target = Path(sys.argv[1])
if not target.exists():
    print(f"ERROR: 找不到 {target}")
    sys.exit(1)

backup = target.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
shutil.copy2(target, backup)
print(f"備份: {backup}")

content = target.read_text(encoding="utf-8")

# ── 1. 加 D3 CDN（在 Chart.js 之後）──────────────────────────
D3_SCRIPT = '  <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>'
OLD_CHARTJS = '  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>'
if D3_SCRIPT not in content:
    content = content.replace(OLD_CHARTJS, OLD_CHARTJS + "\n" + D3_SCRIPT)
    print("✅ D3 CDN 加入")
else:
    print("⏭  D3 CDN 已存在")

# ── 2. 加 HELIX CSS（在 </style> 之前）──────────────────────
HELIX_CSS = """
/* ─── HELIX MODE ─── */
#helix-mode {
  display: none; position: absolute; inset: 0;
  background: var(--obsidian); z-index: 5;
  flex-direction: row; overflow: hidden;
}
#helix-mode.active { display: flex; }

/* Graph 主畫布區 */
#helix-graph-wrap {
  flex: 1; min-width: 0; position: relative;
  background: var(--obsidian);
}
#helix-graph-wrap svg { width: 100%; height: 100%; }

/* 右側 Panel */
#helix-panel {
  width: 280px; flex-shrink: 0;
  border-left: 1px solid var(--border);
  background: var(--surface);
  display: flex; flex-direction: column;
  overflow: hidden;
}

/* Panel header（搜尋 + legend）*/
#helix-panel-head {
  padding: 14px 16px 10px;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}
#helix-search-wrap { position: relative; margin-bottom: 10px; }
#helix-search {
  width: 100%;
  background: var(--surface-2); border: 1px solid var(--border-2);
  color: var(--text-hi); padding: 6px 12px; border-radius: 3px;
  font-family: 'Barlow', sans-serif; font-size: 11px; outline: none;
  transition: border-color 0.15s;
}
#helix-search::placeholder { color: var(--text-lo); }
#helix-search:focus { border-color: var(--champagne-dim); }
#helix-ac {
  position: absolute; top: calc(100% + 2px); left: 0; right: 0;
  background: var(--surface-2); border: 1px solid var(--border-2);
  border-radius: 0 0 3px 3px; max-height: 180px; overflow-y: auto;
  z-index: 200; display: none;
  box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}
.helix-ac-item {
  padding: 7px 12px; font-size: 11px; cursor: pointer;
  color: var(--text-md); font-family: 'Barlow', sans-serif;
  display: flex; justify-content: space-between; align-items: center;
  transition: background 0.1s;
}
.helix-ac-item:hover { background: var(--surface-3); color: var(--text-hi); }
.helix-ac-pr {
  font-family: 'DM Mono', monospace; font-size: 8px; color: var(--text-lo);
}

/* Legend */
#helix-legend {
  display: flex; gap: 10px; flex-wrap: wrap;
}
.helix-leg-item {
  display: flex; align-items: center; gap: 5px;
  font-size: 9px; color: var(--text-lo); font-family: 'Barlow', sans-serif;
}
.helix-leg-dot {
  width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0;
}

/* Panel body（選手詳情 + beat chain）*/
#helix-panel-body {
  flex: 1; overflow-y: auto; padding: 14px 16px;
}

/* Empty state */
#helix-empty {
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; gap: 10px; height: 100%;
  padding: 20px;
}
#helix-empty .helix-empty-icon { font-size: 28px; opacity: 0.08; }
#helix-empty p {
  font-family: 'DM Mono', monospace; font-size: 9px;
  color: var(--text-lo); text-align: center; line-height: 1.8;
  letter-spacing: 0.04em;
}

/* Fighter detail */
#helix-fighter-detail { display: none; }
#helix-fighter-name {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 20px; font-weight: 700; color: var(--text-hi);
  letter-spacing: 0.04em; text-transform: uppercase; line-height: 1.1;
  margin-bottom: 2px;
}
#helix-fighter-meta {
  font-family: 'DM Mono', monospace; font-size: 9px;
  color: var(--text-lo); margin-bottom: 12px; letter-spacing: 0.04em;
}
.helix-stat-row {
  display: flex; justify-content: space-between; align-items: baseline;
  margin-bottom: 6px; padding-bottom: 6px;
  border-bottom: 1px solid var(--border);
}
.helix-stat-label {
  font-size: 9px; color: var(--text-lo); font-family: 'Barlow', sans-serif;
  letter-spacing: 0.02em;
}
.helix-stat-val {
  font-family: 'DM Mono', monospace; font-size: 11px; color: var(--text-md);
}
.helix-stat-val.gold { color: var(--champagne); }

/* Beat chain */
.helix-chain-title {
  font-family: 'DM Mono', monospace; font-size: 8px; font-weight: 500;
  color: var(--text-lo); letter-spacing: 0.18em; text-transform: uppercase;
  margin: 12px 0 6px; padding-bottom: 5px; border-bottom: 1px solid var(--border);
}
.helix-chain-item {
  display: flex; justify-content: space-between; align-items: center;
  padding: 5px 0; border-bottom: 1px solid var(--border);
  cursor: pointer; transition: background 0.1s;
}
.helix-chain-item:hover { background: var(--surface-2); }
.helix-chain-name {
  font-size: 11px; color: var(--text-md); font-family: 'Barlow', sans-serif;
  transition: color 0.12s;
}
.helix-chain-item:hover .helix-chain-name { color: var(--text-hi); }
.helix-chain-method {
  font-family: 'DM Mono', monospace; font-size: 8px;
}
.helix-chain-method.ko  { color: var(--crimson-2); }
.helix-chain-method.sub { color: var(--cobalt-2); }
.helix-chain-method.dec { color: var(--text-lo); }

/* Overlay controls */
#helix-controls {
  position: absolute; top: 12px; left: 14px;
  display: flex; gap: 6px; z-index: 10;
}
.helix-ctrl-btn {
  font-family: 'DM Mono', monospace; font-size: 8px; font-weight: 500;
  letter-spacing: 0.12em; text-transform: uppercase;
  padding: 4px 10px; border-radius: 2px; cursor: pointer;
  border: 1px solid var(--border-2); background: rgba(9,9,13,0.85);
  color: var(--text-lo); transition: all 0.18s;
}
.helix-ctrl-btn:hover { color: var(--text-md); border-color: var(--border-2); }
.helix-ctrl-btn.active { color: var(--champagne); border-color: var(--champagne-dim); }

/* Loading */
#helix-loading {
  position: absolute; inset: 0; display: flex;
  align-items: center; justify-content: center;
  background: var(--obsidian); z-index: 20;
  font-family: 'DM Mono', monospace; font-size: 10px;
  color: var(--text-lo); letter-spacing: 0.12em;
  flex-direction: column; gap: 10px;
}
.helix-loading-bar {
  width: 120px; height: 1px; background: var(--border-2);
  position: relative; overflow: hidden;
}
.helix-loading-fill {
  position: absolute; top: 0; left: -100%; width: 100%;
  height: 100%; background: var(--champagne-dim);
  animation: helixLoad 1.2s ease-in-out infinite;
}
@keyframes helixLoad {
  0%   { left: -100%; }
  100% { left: 100%; }
}
"""

STYLE_CLOSE = "</style>"
count_style = content.count(STYLE_CLOSE)
if count_style != 1:
    print(f"ERROR: </style> 出現 {count_style} 次，無法安全插入")
    sys.exit(1)

if "HELIX MODE" not in content:
    # Insert before last </style>
    content = content.replace(STYLE_CLOSE, HELIX_CSS + "\n" + STYLE_CLOSE, 1)
    print("✅ HELIX CSS 加入")
else:
    print("⏭  HELIX CSS 已存在")

# ── 3. 加 nav button ──────────────────────────────────────────
OLD_DNA_BTN = '<button class="mode-btn" id="btn-dna" onclick="setMode(\'dna\')">META</button>'
NEW_DNA_BTN = OLD_DNA_BTN + '\n    <button class="mode-btn" id="btn-helix" onclick="setMode(\'helix\')">Helix</button>'
if 'btn-helix' not in content:
    content = content.replace(OLD_DNA_BTN, NEW_DNA_BTN)
    print("✅ Nav button 加入")
else:
    print("⏭  Nav button 已存在")

# ── 4. 加 HTML #helix-mode div（在 #dna-mode 之後）────────────
HELIX_HTML = """
  <!-- HELIX MODE -->
  <div id="helix-mode">

    <!-- Loading overlay -->
    <div id="helix-loading">
      <span>INITIALISING HELIX</span>
      <div class="helix-loading-bar"><div class="helix-loading-fill"></div></div>
    </div>

    <!-- Graph area -->
    <div id="helix-graph-wrap">
      <!-- Overlay controls -->
      <div id="helix-controls">
        <div class="helix-ctrl-btn active" id="helix-btn-ranked" onclick="helixToggleRanked()">Ranked</div>
        <div class="helix-ctrl-btn" id="helix-btn-all" onclick="helixToggleAll()">+ Unranked</div>
        <div class="helix-ctrl-btn" id="helix-btn-reset" onclick="helixResetZoom()">Reset</div>
      </div>
      <svg id="helix-svg"></svg>
    </div>

    <!-- Right panel -->
    <div id="helix-panel">
      <div id="helix-panel-head">
        <div id="helix-search-wrap">
          <input id="helix-search" type="text" placeholder="Search fighter…" autocomplete="off">
          <div id="helix-ac"></div>
        </div>
        <div id="helix-legend">
          <div class="helix-leg-item">
            <div class="helix-leg-dot" style="background:#FFD700"></div>Champion
          </div>
          <div class="helix-leg-item">
            <div class="helix-leg-dot" style="background:#C0A060"></div>Former
          </div>
          <div class="helix-leg-item">
            <div class="helix-leg-dot" style="background:#9a9a9a"></div>Ranked
          </div>
          <div class="helix-leg-item">
            <div class="helix-leg-dot" style="background:#2a2a3a;border:1px solid #3a3a4a"></div>Unranked
          </div>
        </div>
      </div>
      <div id="helix-panel-body">
        <div id="helix-empty">
          <div class="helix-empty-icon">⬡</div>
          <p>Click any node to explore<br>their beat chain and<br>network influence</p>
        </div>
        <div id="helix-fighter-detail"></div>
      </div>
    </div>

  </div><!-- end #helix-mode -->
"""

DNA_MODE_CLOSE = "  </div><!-- end #dna-mode -->"
if "helix-mode" not in content:
    content = content.replace(DNA_MODE_CLOSE, DNA_MODE_CLOSE + "\n" + HELIX_HTML)
    print("✅ HELIX HTML 加入")
else:
    print("⏭  HELIX HTML 已存在")

# ── 5. 更新 setMode 加入 helix 分支 ──────────────────────────
OLD_SETMODE_CLOSE = """  } else if(mode==='dna'){
    univEl.style.display='none'; sideEl.style.display='none';
    vsEl.classList.remove('active'); bEl.classList.remove('active'); dnaEl.classList.add('active');
    searchWrap.style.display='none';
    document.getElementById('header-sub').textContent='META · Striking vs Grappling';
    dnaInit();
  }
}"""

NEW_SETMODE_CLOSE = """  } else if(mode==='dna'){
    univEl.style.display='none'; sideEl.style.display='none';
    vsEl.classList.remove('active'); bEl.classList.remove('active'); dnaEl.classList.add('active');
    if(helixEl) helixEl.classList.remove('active');
    searchWrap.style.display='none';
    document.getElementById('header-sub').textContent='META · Striking vs Grappling';
    dnaInit();
  } else if(mode==='helix'){
    univEl.style.display='none'; sideEl.style.display='none';
    vsEl.classList.remove('active'); bEl.classList.remove('active'); dnaEl.classList.remove('active');
    if(helixEl) helixEl.classList.add('active');
    searchWrap.style.display='none';
    document.getElementById('header-sub').textContent='HELIX · Style Network & PageRank';
    document.getElementById('btn-helix').classList.add('active');
    helixInit();
  }
}"""

if "mode==='helix'" not in content:
    content = content.replace(OLD_SETMODE_CLOSE, NEW_SETMODE_CLOSE)
    print("✅ setMode helix 分支加入")
else:
    print("⏭  setMode helix 已存在")

# 也要在 setMode 開頭加入 helixEl 變數
OLD_SETMODE_VARS = """  const univEl=document.getElementById('plot');
  const sideEl=document.getElementById('sidebar');
  const vsEl=document.getElementById('vs-mode');
  const bEl=document.getElementById('builder-mode');
  const dnaEl=document.getElementById('dna-mode');
  const searchWrap=document.getElementById('search-wrap');
  document.getElementById('btn-universe').classList.toggle('active',mode==='universe');
  document.getElementById('btn-vs').classList.toggle('active',mode==='vs');
  document.getElementById('btn-builder').classList.toggle('active',mode==='builder');
  document.getElementById('btn-dna').classList.toggle('active',mode==='dna');"""

NEW_SETMODE_VARS = """  const univEl=document.getElementById('plot');
  const sideEl=document.getElementById('sidebar');
  const vsEl=document.getElementById('vs-mode');
  const bEl=document.getElementById('builder-mode');
  const dnaEl=document.getElementById('dna-mode');
  const helixEl=document.getElementById('helix-mode');
  const searchWrap=document.getElementById('search-wrap');
  document.getElementById('btn-universe').classList.toggle('active',mode==='universe');
  document.getElementById('btn-vs').classList.toggle('active',mode==='vs');
  document.getElementById('btn-builder').classList.toggle('active',mode==='builder');
  document.getElementById('btn-dna').classList.toggle('active',mode==='dna');
  document.getElementById('btn-helix').classList.toggle('active',mode==='helix');"""

if "helixEl" not in content:
    content = content.replace(OLD_SETMODE_VARS, NEW_SETMODE_VARS)
    print("✅ setMode vars 更新")
else:
    print("⏭  setMode vars 已更新")

# ── 6. 加 helixInit JS（在 window.setMode=setMode 之前）──────
HELIX_JS = """
// ══════════════════════════════════════════════════════
//  HELIX MODE — D3 Force Graph
// ══════════════════════════════════════════════════════

let _helixInited   = false;
let _helixData     = null;   // full graph JSON
let _helixShowAll  = false;  // ranked only vs all
let _helixSim      = null;   // D3 simulation
let _helixSelected = null;   // selected fighter name

// Tier → color（完全沿用 Universe Classic 系統）
const HELIX_TIER_COLOR = {
  champion:    '#FFD700',
  ex_champion: '#C0A060',
  ranked:      '#9a9a9a',
  unranked:    '#2a2a3a',
};

function helixNodeColor(d){
  if(d.is_ranked){
    // Use pr_rank to approximate tier：這裡簡單分三層
    // 實際 tier 資料在 fighters.json 不在 helix_graph，用 pr_rank 近似
    if(d.pr >= 80) return '#FFD700';        // champion level
    if(d.pr >= 40) return '#C0A060';        // ex-champ / top ranked
    return '#9a9a9a';                        // ranked
  }
  return '#2a2a3a';
}
function helixNodeStroke(d){
  if(d.id === _helixSelected) return '#c9a96e';
  if(d.pr >= 80) return 'rgba(255,215,0,0.5)';
  if(d.pr >= 40) return 'rgba(192,160,96,0.3)';
  if(d.is_ranked) return 'rgba(154,154,154,0.2)';
  return 'rgba(42,42,58,0.0)';
}
function helixNodeR(d){
  // Size = PageRank 0-100 → radius 3-14
  const base = 3 + (d.pr / 100) * 11;
  if(d.id === _helixSelected) return base + 2;
  return base;
}

async function helixInit(){
  if(_helixInited && _helixData) { helixRender(); return; }
  if(_helixInited) return;
  _helixInited = true;

  try {
    const res = await fetch(API_BASE + '/api/helix/top?n=300');
    if(!res.ok) throw new Error('HTTP ' + res.status);
    _helixData = await res.json();
  } catch(e){
    console.error('HELIX load failed:', e);
    document.getElementById('helix-loading').innerHTML =
      '<span style="color:var(--crimson-2)">Failed to load graph data</span>';
    return;
  }

  helixBuildSearch();
  helixRender();
}

function helixBuildSearch(){
  if(!_helixData) return;
  const nodes = _helixData.nodes || [];
  const inp = document.getElementById('helix-search');
  const ac  = document.getElementById('helix-ac');

  inp.addEventListener('input', ()=>{
    const q = inp.value.toLowerCase().trim();
    ac.innerHTML = '';
    if(!q){ ac.style.display='none'; return; }
    const matches = nodes
      .filter(n => n.id.toLowerCase().includes(q))
      .sort((a,b) => b.pr - a.pr)
      .slice(0,8);
    if(!matches.length){ ac.style.display='none'; return; }
    matches.forEach(n=>{
      const div = document.createElement('div');
      div.className = 'helix-ac-item';
      div.innerHTML = `<span>${n.id}</span><span class="helix-ac-pr">PR #${n.pr_rank}</span>`;
      div.onclick = ()=>{
        inp.value = n.id;
        ac.style.display = 'none';
        helixSelectFighter(n.id);
      };
      ac.appendChild(div);
    });
    ac.style.display = 'block';
  });

  document.addEventListener('click', e=>{
    if(!inp.contains(e.target)) ac.style.display='none';
  });
}

function helixRender(){
  const loading = document.getElementById('helix-loading');
  const svg     = document.getElementById('helix-svg');
  if(!_helixData){ return; }

  let nodes = (_helixData.nodes || []).filter(n => _helixShowAll ? true : n.is_ranked);
  const nodeIds = new Set(nodes.map(n=>n.id));
  let edges = (_helixData.edges || []).filter(e => nodeIds.has(e.winner) && nodeIds.has(e.loser));

  // D3 needs mutable objects
  nodes = nodes.map(n => ({...n}));
  const nodeMap = Object.fromEntries(nodes.map(n=>[n.id,n]));
  const links   = edges.map(e => ({
    source: nodeMap[e.winner],
    target: nodeMap[e.loser],
    weight: e.weight,
    count:  e.count,
    method: e.bouts && e.bouts[0] ? e.bouts[0].method : '',
  })).filter(l => l.source && l.target);

  const wrap   = document.getElementById('helix-graph-wrap');
  const W      = wrap.clientWidth  || 900;
  const H      = wrap.clientHeight || 700;

  const svgEl = d3.select('#helix-svg')
    .attr('viewBox', `0 0 ${W} ${H}`)
    .style('background', '#09090d');
  svgEl.selectAll('*').remove();

  // Arrow marker
  svgEl.append('defs').append('marker')
    .attr('id','helix-arrow')
    .attr('viewBox','0 -4 8 8')
    .attr('refX', 14).attr('refY', 0)
    .attr('markerWidth', 4).attr('markerHeight', 4)
    .attr('orient', 'auto')
    .append('path')
    .attr('d','M0,-4L8,0L0,4')
    .attr('fill','rgba(201,169,110,0.25)');

  const g = svgEl.append('g').attr('class','helix-g');

  // Zoom
  const zoom = d3.zoom()
    .scaleExtent([0.2, 8])
    .on('zoom', e => g.attr('transform', e.transform));
  svgEl.call(zoom);
  window._helixZoom = zoom;
  window._helixSvgEl = svgEl;

  // Simulation
  const sim = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d=>d.id)
      .distance(d => 60 - d.weight * 8)
      .strength(0.6))
    .force('charge', d3.forceManyBody().strength(d => -80 - d.pr * 0.8))
    .force('center', d3.forceCenter(W/2, H/2))
    .force('collision', d3.forceCollide().radius(d => helixNodeR(d) + 3));
  _helixSim = sim;

  // Edges
  const link = g.append('g').attr('class','links')
    .selectAll('line')
    .data(links)
    .join('line')
    .attr('stroke', d => {
      if(d.method === 'KO/TKO')     return 'rgba(122,31,40,0.25)';
      if(d.method === 'Submission') return 'rgba(26,48,80,0.30)';
      return 'rgba(31,31,44,0.35)';
    })
    .attr('stroke-width', d => Math.min(2, 0.5 + d.weight * 0.3))
    .attr('marker-end', 'url(#helix-arrow)');

  // Nodes
  const node = g.append('g').attr('class','nodes')
    .selectAll('circle')
    .data(nodes)
    .join('circle')
    .attr('r', d => helixNodeR(d))
    .attr('fill', d => helixNodeColor(d))
    .attr('stroke', d => helixNodeStroke(d))
    .attr('stroke-width', 1)
    .attr('opacity', d => d.is_ranked ? 0.9 : 0.45)
    .style('cursor','pointer')
    .on('click', (event, d) => {
      event.stopPropagation();
      helixSelectFighter(d.id);
    })
    .call(d3.drag()
      .on('start', (e,d) => { if(!e.active) sim.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; })
      .on('drag',  (e,d) => { d.fx=e.x; d.fy=e.y; })
      .on('end',   (e,d) => { if(!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; }));

  // Tooltip on hover
  const tooltip = d3.select('body').select('#helix-tooltip-d3').node()
    ? d3.select('#helix-tooltip-d3')
    : d3.select('body').append('div')
        .attr('id','helix-tooltip-d3')
        .style('position','fixed').style('pointer-events','none')
        .style('background','rgba(14,14,20,0.95)').style('border','1px solid #1f1f2c')
        .style('border-radius','3px').style('padding','7px 10px')
        .style('font-family','"DM Mono",monospace').style('font-size','10px')
        .style('color','#b8b4c0').style('display','none').style('z-index','9999')
        .style('max-width','180px').style('line-height','1.6');

  node.on('mouseover', (e,d) => {
    tooltip.style('display','block')
      .html(`<strong style="color:#e8e4dc;font-family:\'Barlow Condensed\',sans-serif;font-size:13px;text-transform:uppercase">${d.id}</strong><br>` +
            `PR Rank &nbsp;<span style="color:#c9a96e">#${d.pr_rank}</span><br>` +
            `Style &nbsp;<span style="color:#e8e4dc">${d.style||'—'}</span><br>` +
            `W/L (graph) &nbsp;${d.wins_graph}/${d.losses_graph}`);
  })
  .on('mousemove', e => {
    tooltip.style('left', (e.clientX+14)+'px').style('top', (e.clientY-8)+'px');
  })
  .on('mouseout', () => tooltip.style('display','none'));

  // Labels for top-ranked nodes
  const label = g.append('g').attr('class','labels')
    .selectAll('text')
    .data(nodes.filter(n => n.pr >= 55 || n.id === _helixSelected))
    .join('text')
    .attr('font-family','"Barlow Condensed",sans-serif')
    .attr('font-size', d => d.pr >= 75 ? 10 : 8)
    .attr('fill', d => d.pr >= 80 ? '#c9a96e' : '#6e6a7c')
    .attr('text-anchor','middle')
    .attr('dy','2em')
    .attr('pointer-events','none')
    .text(d => d.id.split(' ').pop());

  sim.on('tick', ()=>{
    link
      .attr('x1', d=>d.source.x).attr('y1', d=>d.source.y)
      .attr('x2', d=>d.target.x).attr('y2', d=>d.target.y);
    node.attr('cx', d=>d.x).attr('cy', d=>d.y);
    label.attr('x', d=>d.x).attr('y', d=>d.y);
  });

  // Hide loading after simulation settles
  sim.on('end', ()=>{
    loading.style.display = 'none';
  });
  setTimeout(() => { loading.style.display='none'; }, 2000);

  // Click background to deselect
  svgEl.on('click', () => {
    _helixSelected = null;
    helixClearPanel();
    node.attr('stroke', d => helixNodeStroke(d))
        .attr('r', d => helixNodeR(d));
  });
}

function helixSelectFighter(name){
  _helixSelected = name;

  // Update node highlight
  d3.selectAll('#helix-svg circle')
    .attr('stroke', d => helixNodeStroke(d))
    .attr('stroke-width', d => d.id===name ? 2 : 1)
    .attr('r', d => helixNodeR(d))
    .attr('opacity', d => {
      if(d.id === name) return 1;
      if(d.is_ranked) return 0.9;
      return 0.45;
    });

  helixShowPanel(name);
}

async function helixShowPanel(name){
  const emptyEl  = document.getElementById('helix-empty');
  const detailEl = document.getElementById('helix-fighter-detail');
  emptyEl.style.display  = 'none';
  detailEl.style.display = 'block';
  detailEl.innerHTML = '<div style="color:var(--text-lo);font-family:\'DM Mono\',monospace;font-size:9px;letter-spacing:0.1em">LOADING...</div>';

  try {
    const res  = await fetch(API_BASE + '/api/helix/fighter/' + encodeURIComponent(name));
    const data = await res.json();
    const f    = data.fighter;
    const beats    = data.beat_chain  || [];
    const lostTo   = data.loss_chain  || [];

    const styleMethodColor = m => {
      if(!m) return 'dec';
      if(m.includes('KO')) return 'ko';
      if(m.includes('Sub')) return 'sub';
      return 'dec';
    };
    const methodShort = m => {
      if(!m) return 'DEC';
      if(m.includes('KO')) return 'KO';
      if(m.includes('Sub')) return 'SUB';
      return 'DEC';
    };

    const prColor = f.pr >= 80 ? 'gold' : '';

    detailEl.innerHTML = `
      <div id="helix-fighter-name">${f.id}</div>
      <div id="helix-fighter-meta">
        ${f.style || '—'} · ${f.stance || '—'} · ${f.is_ranked ? '★ Ranked' : 'Unranked'}
      </div>
      <div class="helix-stat-row">
        <span class="helix-stat-label">PageRank</span>
        <span class="helix-stat-val ${prColor}">#${f.pr_rank} &nbsp;<span style="color:var(--text-lo);font-size:9px">(${f.pr.toFixed(1)}/100)</span></span>
      </div>
      <div class="helix-stat-row">
        <span class="helix-stat-label">Wins / Losses (graph)</span>
        <span class="helix-stat-val">${f.wins_graph} / ${f.losses_graph}</span>
      </div>
      <div class="helix-stat-row">
        <span class="helix-stat-label">Community</span>
        <span class="helix-stat-val">#${f.community >= 0 ? f.community : '—'}</span>
      </div>

      <div class="helix-chain-title">Beat Chain — victories</div>
      ${beats.slice(0,10).map(b=>`
        <div class="helix-chain-item" onclick="helixSelectFighter('${b.opponent.replace(/'/g,"\\'")}')">
          <span class="helix-chain-name">${b.opponent}</span>
          <span class="helix-chain-method ${styleMethodColor(b.bouts[0]&&b.bouts[0].method)}">${methodShort(b.bouts[0]&&b.bouts[0].method)}</span>
        </div>`).join('')}
      ${beats.length === 0 ? '<div style="font-size:9px;color:var(--text-lo);padding:4px 0">No recorded victories</div>' : ''}

      <div class="helix-chain-title">Lost to</div>
      ${lostTo.slice(0,5).map(b=>`
        <div class="helix-chain-item" onclick="helixSelectFighter('${b.opponent.replace(/'/g,"\\'")}')">
          <span class="helix-chain-name">${b.opponent}</span>
          <span class="helix-chain-method ${styleMethodColor(b.bouts[0]&&b.bouts[0].method)}">${methodShort(b.bouts[0]&&b.bouts[0].method)}</span>
        </div>`).join('')}
      ${lostTo.length === 0 ? '<div style="font-size:9px;color:var(--text-lo);padding:4px 0">No recorded losses</div>' : ''}
    `;
  } catch(e){
    console.error(e);
    detailEl.innerHTML = '<div style="color:var(--crimson-2);font-size:9px">Failed to load fighter data</div>';
  }
}

function helixClearPanel(){
  document.getElementById('helix-empty').style.display='flex';
  document.getElementById('helix-fighter-detail').style.display='none';
  document.getElementById('helix-fighter-detail').innerHTML='';
}

function helixToggleRanked(){
  _helixShowAll = false;
  document.getElementById('helix-btn-ranked').classList.add('active');
  document.getElementById('helix-btn-all').classList.remove('active');
  document.getElementById('helix-loading').style.display='flex';
  _helixInited = false;  // force re-render
  helixInit();
}

function helixToggleAll(){
  _helixShowAll = true;
  document.getElementById('helix-btn-ranked').classList.remove('active');
  document.getElementById('helix-btn-all').classList.add('active');
  document.getElementById('helix-loading').style.display='flex';
  // fetch all graph
  fetch(API_BASE + '/api/helix/graph?ranked_only=false')
    .then(r=>r.json())
    .then(data=>{
      _helixData = data;
      _helixInited = true;
      helixRender();
    });
}

function helixResetZoom(){
  if(window._helixSvgEl && window._helixZoom){
    window._helixSvgEl.transition().duration(400)
      .call(window._helixZoom.transform, d3.zoomIdentity);
  }
}
"""

WINDOW_SETMODE = "  window.setMode=setMode;"
if "helixInit" not in content:
    content = content.replace(WINDOW_SETMODE, HELIX_JS + "\n  " + WINDOW_SETMODE)
    print("✅ helixInit JS 加入")
else:
    print("⏭  helixInit JS 已存在")

# ── 7. expose helixSelectFighter to window ───────────────────
OLD_WINDOW_EXPOSE = "  window.bSubmit=bSubmit;"
NEW_WINDOW_EXPOSE = "  window.bSubmit=bSubmit;\n  window.helixSelectFighter=helixSelectFighter;"
if "window.helixSelectFighter" not in content:
    content = content.replace(OLD_WINDOW_EXPOSE, NEW_WINDOW_EXPOSE)
    print("✅ window.helixSelectFighter 加入")
else:
    print("⏭  已存在")

# ── 驗證 </style> 數量 ────────────────────────────────────────
style_count = content.count("</style>")
print(f"\n驗證: </style> 出現 {style_count} 次（應為 1）")
if style_count != 1:
    print("ERROR: </style> 數量異常，請檢查")
    sys.exit(1)

target.write_text(content, encoding="utf-8")
print(f"\n✅ 完成！輸出: {target}")
print("\n下一步：")
print("  1. 本地確認外觀：python3 -m http.server 8080 (在 ~/UFC/)")
print("  2. git add -A && git commit -m 'feat: HELIX frontend D3 graph' && git push")
