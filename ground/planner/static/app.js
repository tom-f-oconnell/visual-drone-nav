// NAIP flight planner front end: Leaflet map, waypoint editing, cache + export calls, simulations.
const $ = (id) => document.getElementById(id);
const FIELDS = ["default_alt", "takeoff_alt", "speed", "corridor_m"];

const map = L.map("map").setView([39.74, -104.99], 13);
const osm = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19, attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);
const esri = L.tileLayer(
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  { maxZoom: 19, attribution: "Esri World Imagery (display only)" });
const naip = L.tileLayer("/api/tiles/{z}/{x}/{y}.png", {
  minZoom: 12, maxZoom: 20, maxNativeZoom: 18, attribution: "USDA NAIP (cached)",
}).addTo(map);
const cellsLayer = L.geoJSON(null, { style: { color: "#888", weight: 1, fill: false, dashArray: "2 3" } });
L.control.layers({ "OpenStreetMap": osm, "Esri imagery": esri },
  { "Cached NAIP": naip, "Cache cells": cellsLayer }).addTo(map);

const route = L.polyline([], { color: "#1f6feb", weight: 3 }).addTo(map);
const corridor = L.geoJSON(null, { style: { color: "#1f6feb", weight: 1, fillOpacity: 0.06 } }).addTo(map);
let markers = [];
let wps = [];  // {lat, lon, alt|null, hold_s}

function mission() {
  const m = {
    name: $("name").value, waypoints: wps, end_action: $("end_action").value,
    camera: { hfov_deg: +$("hfov_deg").value },
  };
  FIELDS.forEach((f) => (m[f] = +$(f).value));
  return m;
}

function loadMission(m) {
  $("name").value = m.name;
  FIELDS.forEach((f) => ($(f).value = m[f]));
  $("end_action").value = m.end_action;
  $("hfov_deg").value = m.camera.hfov_deg;
  wps = m.waypoints;
  redraw();
  if (wps.length) map.fitBounds(route.getBounds().pad(0.3));
}

function icon(i) { return L.divIcon({ className: "wp-label", html: String(i + 1) }); }

function redraw() {
  markers.forEach((mk) => mk.remove());
  markers = wps.map((wp, i) => {
    const mk = L.marker([wp.lat, wp.lon], { draggable: true, icon: icon(i) }).addTo(map);
    mk.on("drag", (e) => { wp.lat = e.latlng.lat; wp.lon = e.latlng.lng; route.setLatLngs(wps.map((w) => [w.lat, w.lon])); });
    mk.on("dragend", () => { renderList(); check(); });
    return mk;
  });
  route.setLatLngs(wps.map((w) => [w.lat, w.lon]));
  renderList();
  check();
}

function renderList() {
  const ol = $("wps");
  ol.innerHTML = "";
  wps.forEach((wp, i) => {
    const li = document.createElement("li");
    li.innerHTML = `<div class="wp"><span>${wp.lat.toFixed(5)}, ${wp.lon.toFixed(5)}</span>
      <input type="number" placeholder="alt" title="Altitude of this waypoint in metres above the first waypoint. Blank uses the cruise altitude." value="${wp.alt ?? ""}">
      <button title="move up">↑</button><button title="delete">✕</button></div>`;
    const [alt, up, del] = li.querySelectorAll("input, button");
    alt.onchange = () => { wp.alt = alt.value === "" ? null : +alt.value; check(); };
    up.onclick = () => { if (i > 0) { [wps[i - 1], wps[i]] = [wps[i], wps[i - 1]]; redraw(); } };
    del.onclick = () => { wps.splice(i, 1); redraw(); };
    ol.appendChild(li);
  });
}

map.on("click", (e) => { wps.push({ lat: e.latlng.lat, lon: e.latlng.lng, alt: null, hold_s: 0 }); redraw(); });
$("clear").onclick = () => { wps = []; redraw(); };
[...FIELDS, "hfov_deg", "end_action"].forEach((f) => ($(f).onchange = check));

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r;
}

let checkTimer, routeCached = false, routeCells = "";
function check() {
  clearTimeout(checkTimer);
  checkTimer = setTimeout(async () => {
    const res = await (await api("/api/plan/check", { method: "POST", body: JSON.stringify(mission()) })).json();
    corridor.clearLayers();
    if (res.corridor) corridor.addData(res.corridor);
    $("stats").innerHTML = res.length_m === undefined ? "" :
      `Route ${(res.length_m / 1000).toFixed(2)} km, about ${Math.round(res.est_time_s / 60)} min at cruise.<br>
       Camera footprint at cruise: ${res.footprint_m} m wide.<br>
       NAIP cells: ${res.cached}/${res.cells} cached` +
      (res.cells > res.cached ? `, about ${res.est_download_mb} MB to fetch.` : ".");
    $("warnings").innerHTML = res.warnings.map((w) => `<li>${w}</li>`).join("");
    routeCached = Boolean(res.cells && res.cached === res.cells);
    routeCells = (res.cell_keys || []).join();
    $("export").disabled = !routeCached;
    updateSimButton();
    refreshImagery(res.cached);
  }, 250);
}

// Which NAIP years the route uses. The server may have to ask the imagery source, so only when the route's cells change.
let imageryKey = null;
async function refreshImagery(cached) {
  const key = `${routeCells}/${cached}`;
  if (key === imageryKey) return;
  imageryKey = key;
  try {
    const { label } = await (await api("/api/plan/imagery", { method: "POST", body: JSON.stringify(mission()) })).json();
    if (key === imageryKey) $("naip_years").textContent = label;
  } catch (e) { imageryKey = null; console.warn(e); }
}

$("cache").onclick = async () => {
  $("cache").disabled = true;
  try {
    let job = await (await api("/api/cache/jobs", { method: "POST", body: JSON.stringify(mission()) })).json();
    $("progress").hidden = false;
    while (job.state === "running") {
      const shown = job.done;
      $("bar").style.width = `${job.total ? (100 * job.done) / job.total : 100}%`;
      $("ptext").textContent = `${job.done} / ${job.total} new cells`;
      await new Promise((r) => setTimeout(r, 1000));
      job = await (await api(`/api/cache/jobs/${job.id}`)).json();
      if (job.done !== shown) naip.setUrl(`/api/tiles/{z}/{x}/{y}.png?v=${job.done}`);
    }
    $("bar").style.width = "100%";
    if (job.total) naip.setUrl(`/api/tiles/{z}/{x}/{y}.png?v=${Date.now()}`);
    const notes = [];
    if (job.no_coverage.length) notes.push(`${job.no_coverage.length} cells have no NAIP coverage`);
    if (job.errors.length) notes.push(`${job.errors.length} errors (see server log)`);
    const kept = job.route_cells - job.total;
    $("ptext").textContent = notes.length ? notes.join("; ")
      : job.total ? `Downloaded ${job.total} new, ${kept} already cached` : `All ${kept} cells already cached`;
    refreshCells();
    check();
  } catch (e) { alert(e.message); }
  $("cache").disabled = false;
};

async function download(resp, filename) {
  const url = URL.createObjectURL(await resp.blob());
  Object.assign(document.createElement("a"), { href: url, download: filename }).click();
  URL.revokeObjectURL(url);
}

$("export").onclick = async () => {
  $("export").disabled = true;
  try {
    const m = mission();
    await download(await api("/api/export", { method: "POST", body: JSON.stringify(m) }), `${m.name}.zip`);
  } catch (e) { alert(e.message); }
  $("export").disabled = false;
};

$("wpl").onclick = async () => {
  const m = mission();
  await api(`/api/missions/${encodeURIComponent(m.name)}`, { method: "PUT", body: JSON.stringify(m) });
  await download(await api(`/api/missions/${encodeURIComponent(m.name)}/mission.waypoints`), `${m.name}.waypoints`);
};

$("save").onclick = async () => {
  const m = mission();
  await api(`/api/missions/${encodeURIComponent(m.name)}`, { method: "PUT", body: JSON.stringify(m) });
  refreshSaved();
};

$("saved").onchange = async () => {
  if (!$("saved").value) return;
  loadMission(await (await api(`/api/missions/${encodeURIComponent($("saved").value)}`)).json());
};

async function refreshSaved() {
  const names = await (await api("/api/missions")).json();
  $("saved").innerHTML = `<option value="">Load saved…</option>` + names.map((n) => `<option>${n}</option>`).join("");
}

async function refreshCells() {
  cellsLayer.clearLayers();
  cellsLayer.addData(await (await api("/api/cache/cells")).json());
}

// Simulation: runs execute on the server, the 3D viewer of each is shown in an iframe.
const ACTIVE = ["preparing", "fetching", "running"];
let simulator = false, openRun = null, simTimer;

function showView(name) {
  $("map").hidden = name !== "plan";
  $("simview").hidden = name !== "sim";
  $("compareview").hidden = name !== "compare";
  $("tab_plan").setAttribute("aria-pressed", name === "plan");
  $("tab_sim").setAttribute("aria-pressed", name === "sim");
  $("tab_compare").setAttribute("aria-pressed", name === "compare");
  if (name === "plan") map.invalidateSize();
}
$("tab_plan").onclick = () => showView("plan");
$("tab_sim").onclick = () => showView("sim");
$("tab_compare").onclick = () => showView("compare");

function updateSimButton() {
  const ready = simulator && routeCached && wps.length >= 2;
  $("sim_run").disabled = $("sim_compare").disabled = !ready;
  $("sim_run").title = ready ? "" : "Add at least two waypoints and cache NAIP for the route first";
  if (ready) refreshYears();
}

// Other NAIP years over the route, looked up once it is cached so the list is ready before it is opened.
let yearsKey = null;
async function refreshYears() {
  const key = routeCells;
  if (key === yearsKey) return;
  yearsKey = key;
  try {
    const { years } = await (await api("/api/sim/years", { method: "POST", body: JSON.stringify(mission()) })).json();
    if (key !== yearsKey) return;
    const sel = $("sim_world"), keep = sel.value;
    while (sel.options.length > 1) sel.remove(1);
    years.forEach((y) => sel.add(new Option(`NAIP ${y} (realistic, downloads it if needed)`, y)));
    sel.value = keep;
    if (sel.selectedIndex < 0) sel.selectedIndex = 0;
  } catch (e) { yearsKey = null; console.warn(e); }
}

// Camera appearance: each field's value in the page is the one that changes nothing.
const lookInputs = [...document.querySelectorAll("[data-look]")];

function lookValues() {
  const out = {};
  lookInputs.forEach((el) => { if (+el.value !== +el.defaultValue) out[el.dataset.look] = +el.value; });
  return out;
}

function showLookState() {
  const changed = Object.keys(lookValues()).some((name) => name !== "drift_tau_s");
  $("look_state").textContent = changed ? "changed" : "as the imagery";
}

lookInputs.forEach((el) => { el.oninput = showLookState; });

$("look_reset").onclick = () => {
  lookInputs.forEach((el) => { el.value = el.defaultValue; });
  showLookState();
};

$("look_random").onclick = async () => {
  try {
    const values = await (await api("/api/sim/appearance/random")).json();
    lookInputs.forEach((el) => { el.value = values[el.dataset.look]; });
    showLookState();
  } catch (e) { alert(e.message); }
};

function simRequest() {
  return JSON.stringify({
    mission: mission(),
    options: {
      seed: +$("sim_seed").value, wind_speed: +$("sim_wind").value, wind_from_deg: +$("sim_wind_from").value,
      gust: +$("sim_gust").value, realtime: +$("sim_pace").value,
      world_year: $("sim_world").value ? +$("sim_world").value : null,
      appearance: lookValues(), navigation: $("sim_nav").value,
      compass_bias_deg: +$("sim_compass").value, height_bias_pct: +$("sim_height").value,
    },
  });
}

$("sim_run").onclick = async () => {
  $("sim_run").disabled = $("sim_compare").disabled = true;
  try {
    const run = await (await api("/api/sim/runs", { method: "POST", body: simRequest() })).json();
    await refreshRuns();
    followRun(run.id);
  } catch (e) { alert(e.message); }
  updateSimButton();
};

$("sim_compare").onclick = async () => {
  $("sim_run").disabled = $("sim_compare").disabled = true;
  try {
    const [first] = await (await api("/api/sim/compare", { method: "POST", body: simRequest() })).json();
    await refreshRuns();
    followRun(first.id, "compare");
  } catch (e) { alert(e.message); }
  updateSimButton();
};

let openPartner = null, wantedView = "sim";
function followRun(id, wanted = "sim") {
  clearTimeout(simTimer);
  openRun = id;
  openPartner = null;
  wantedView = wanted;
  const view = $("simview");
  delete view.dataset.run;
  view.removeAttribute("src");
  showView("plan");
  $("tabs").hidden = true;
  $("tab_compare").hidden = true;
  $("sim_runs").value = id || "";
  $("sim_result").textContent = "";
  $("sim_progress").hidden = $("sim_stop").hidden = $("sim_delete").hidden = true;
  if (id) pollRun();
}

async function pollRun() {
  const id = openRun;
  let run;
  try {
    run = await (await api(`/api/sim/runs/${id}`)).json();
  } catch (e) {
    if (id === openRun) followRun(null);
    return;
  }
  if (id !== openRun) return;
  renderRun(run);
  if (ACTIVE.includes(run.state)) simTimer = setTimeout(pollRun, 1000);
  else refreshRuns();
}

function clock(t) { return `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`; }

let shownState = null;
function renderRun(run) {
  // The panel is long: keep the progress or the result in sight.
  if (run.state !== shownState) requestAnimationFrame(() => $("simsec").scrollIntoView({ block: "end" }));
  shownState = run.state;
  openPartner = run.partner || null;
  if (openPartner && $("tab_compare").hidden) {
    const page = `/static/compare.html?a=${run.id}&b=${openPartner}`, frame = $("compareview");
    if (frame.getAttribute("src") !== page) frame.src = page;
    $("tab_compare").hidden = false;
    $("tabs").hidden = false;
    if (wantedView === "compare") showView("compare");
  }
  const view = $("simview");
  if (run.viewable && view.dataset.run !== run.id) {
    view.dataset.run = run.id;
    view.src = `/sim/${run.id}/`;
    $("tabs").hidden = false;
    if (wantedView === "sim") showView("sim");
  }
  const active = ACTIVE.includes(run.state);
  $("sim_progress").hidden = !active;
  $("sim_stop").hidden = !active;
  $("sim_delete").hidden = active;
  if (active) {
    let frac = 0, text = "Building the map…";
    if (run.state === "fetching" && run.cells) {
      frac = run.cells.total ? run.cells.done / run.cells.total : 1;
      text = `Downloading NAIP ${run.options.world_year}: ${run.cells.done} / ${run.cells.total} cells`;
    } else if (run.state === "running") {
      text = "Starting the simulator…";
      if (run.progress) {
        frac = run.progress.leg / run.progress.legs;
        text = `Leg ${Math.min(run.progress.leg + 1, run.progress.legs)} of ${run.progress.legs}, ${clock(run.progress.t)} flown`;
      }
    }
    $("sim_bar").style.width = `${100 * frac}%`;
    $("sim_ptext").textContent = text;
  }
  $("sim_result").textContent = active ? "" : resultText(run);
}

function resultText(run) {
  if (run.state === "failed") return `Failed: ${run.error}`;
  if (run.state === "stopped") return "Stopped before the end of the flight.";
  const s = run.summary;
  const end = s.landed ? `Landed ${s.landing_error_m} m from the target.`
    : s.mission_complete ? `Reached the last waypoint, ${s.landing_error_m} m off.` : "Ran out of time before finishing.";
  const alone = run.options.navigation === "odometry";
  const note = (run.options.world_year ? `Camera saw NAIP ${run.options.world_year}.`
    : alone ? "" : "The camera saw the map's own imagery, so this is optimistic.") +
    (s.compass_bias_deg ? ` Compass bias ${s.compass_bias_deg}°.` : "") + (s.height_bias_pct ? ` Height bias ${s.height_bias_pct}%.` : "");
  const look = Object.entries(s.appearance || {}).map(([name, v]) => `${name.replace(/_/g, " ")} ${v}`).join(", ");
  return `${end} Position error: median ${s.ekf_err_median_m} m, 95% ${s.ekf_err_p95_m} m, ` +
    `max ${s.ekf_err_max_m} m. ${alone ? "Visual odometry only, no map fixes." : `Map fixes on ${Math.round(100 * s.fix_rate)}% of frames.`} ${note}` +
    (look ? ` Camera appearance: ${look}.` : "");
}

$("sim_stop").onclick = () => {
  [openRun, openPartner].filter(Boolean).forEach((id) => api(`/api/sim/runs/${id}/stop`, { method: "POST" }).catch((e) => alert(e.message)));
};

$("sim_delete").onclick = async () => {
  if (!confirm(openPartner ? "Delete this simulation and the one it was compared with?" : "Delete this simulation?")) return;
  try {
    await Promise.all([openRun, openPartner].filter(Boolean).map((id) => api(`/api/sim/runs/${id}`, { method: "DELETE" })));
    $("compareview").removeAttribute("src");
    followRun(null);
    await refreshRuns();
  } catch (e) { alert(e.message); }
};

$("sim_runs").onchange = () => followRun($("sim_runs").value || null);

async function refreshRuns() {
  const runs = await (await api("/api/sim/runs")).json();
  const sel = $("sim_runs");
  while (sel.options.length > 1) sel.remove(1);
  runs.forEach((r) => {
    const outcome = r.state === "finished" ? `${r.summary.ekf_err_median_m} m median` : r.state;
    const nav = r.options.navigation === "odometry" ? "odometry only, " : "";
    sel.add(new Option(`${r.name}, ${r.created.slice(5, 16).replace("T", " ")}: ${nav}${outcome}`, r.id));
  });
  sel.value = openRun || "";
  return runs;
}

(async () => {
  const cfg = await (await api("/api/config")).json();
  $("cfg").textContent = `Imagery source: ${cfg.source}, cache GSD ${cfg.gsd} m, data in ${cfg.data_dir}`;
  simulator = cfg.simulator;
  $("simsec").hidden = !simulator;
  $("sim_missing").hidden = simulator;
  refreshSaved();
  refreshCells();
  check();
  if (simulator) {
    const running = (await refreshRuns()).find((r) => ACTIVE.includes(r.state));
    if (running) followRun(running.id);
  }
})();
