// Flight planner front end: Leaflet map, waypoint editing, cache + export calls, simulations.
const $ = (id) => document.getElementById(id);
const FIELDS = ["default_alt", "takeoff_alt", "speed", "corridor_m"];

const map = L.map("map");  // placed once the cached regions are known
const osm = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19, attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);
const esri = L.tileLayer(
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  { maxZoom: 19, attribution: "Esri World Imagery (display only)" });
const IMAGERY_MIN_ZOOM = 12;
const IMAGERY = {
  naip: { label: "NAIP", maxNativeZoom: 18, attribution: "USDA NAIP (cached)" },
  sentinel2: { label: "Sentinel-2", maxNativeZoom: 16, attribution: "Copernicus Sentinel-2 (cached)" },
};
const planDate = () => ($("imagery").value === "sentinel2" && $("imagery_date").value) || null;
const dateQuery = (name) => (name === "sentinel2" && planDate() ? `&date=${planDate()}` : "");
const imageryUrl = (name, v) => `/api/tiles/{z}/{x}/{y}.png?imagery=${name}${dateQuery(name)}&v=${v}`;
const imageryLayers = Object.fromEntries(Object.entries(IMAGERY).map(([name, i]) => [name,
  L.tileLayer(imageryUrl(name, 0), { minZoom: IMAGERY_MIN_ZOOM, maxZoom: 20, maxNativeZoom: i.maxNativeZoom, attribution: i.attribution })]));
const cellsLayer = L.geoJSON(null, { style: { color: "#888", weight: 1, fill: false, dashArray: "2 3" } });
L.control.layers({ "OpenStreetMap": osm, "Esri imagery": esri }, { "Cache cells": cellsLayer }).addTo(map);

// Which cached imagery the map shows, separately from which one the plan uses.
let shownImagery = null;
const ImageryControl = L.Control.extend({
  onAdd() {
    const box = L.DomUtil.create("div", "imagery-view");
    box.title = "Cached imagery to show on the map. This does not change what the plan uses (Map imagery in the panel).";
    [["naip", "NAIP"], ["sentinel2", "Sentinel-2"], ["", "Off"]].forEach(([name, label]) => {
      const b = L.DomUtil.create("button", "", box);
      b.textContent = label;
      b.dataset.view = name;
      b.onclick = () => showImagery(name);
    });
    L.DomEvent.disableClickPropagation(box);  // a click here must not add a waypoint
    return box;
  },
});
map.addControl(new ImageryControl({ position: "bottomleft" }));

function showImagery(name) {
  Object.entries(imageryLayers).forEach(([k, layer]) => { if (k === name) layer.addTo(map); else layer.remove(); });
  shownImagery = name;
  document.querySelectorAll(".imagery-view button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.view === name)));
  refreshCells();
}

const route = L.polyline([], { color: "#1f6feb", weight: 3 }).addTo(map);
const corridor = L.geoJSON(null, { style: { color: "#1f6feb", weight: 1, fillOpacity: 0.06 } }).addTo(map);
let markers = [];
let wps = [];  // {lat, lon, alt|null, hold_s}

function mission() {
  const m = {
    name: $("name").value, waypoints: wps, end_action: $("end_action").value,
    camera: { hfov_deg: +$("hfov_deg").value }, imagery: $("imagery").value, imagery_date: planDate(),
  };
  FIELDS.forEach((f) => (m[f] = +$(f).value));
  return m;
}

function loadMission(m) {
  $("name").value = m.name;
  FIELDS.forEach((f) => ($(f).value = m[f]));
  $("end_action").value = m.end_action;
  $("hfov_deg").value = m.camera.hfov_deg;
  $("imagery").value = m.imagery || "naip";
  $("imagery_date").value = m.imagery_date || "";
  $("imagery_date_box").hidden = $("imagery").value !== "sentinel2";
  imageryLayers.sentinel2.setUrl(imageryUrl("sentinel2", 0));
  showImagery($("imagery").value);
  refreshRegions();
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
function planImageryChanged() {
  $("imagery_date_box").hidden = $("imagery").value !== "sentinel2";
  imageryLayers.sentinel2.setUrl(imageryUrl("sentinel2", 0));
  showImagery($("imagery").value);
  refreshRegions();
  check();
}
$("imagery").onchange = planImageryChanged;
$("imagery_date").onchange = planImageryChanged;

// A request that fails validation gets a list of problems, each located by "body" and then the path to the field.
function errorText(detail) {
  if (!Array.isArray(detail)) return detail;
  return detail.map((d) => [d.loc.slice(1).join("."), d.msg].filter(Boolean).join(": ")).join("\n");
}

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!r.ok) throw new Error(errorText((await r.json().catch(() => ({}))).detail) || r.statusText);
  return r;
}

let checkTimer, routeCached = false, routeCells = "";
function check() {
  showCloudState();
  clearTimeout(checkTimer);
  checkTimer = setTimeout(async () => {
    const res = await (await api("/api/plan/check", { method: "POST", body: JSON.stringify(mission()) })).json();
    corridor.clearLayers();
    if (res.corridor) corridor.addData(res.corridor);
    $("stats").innerHTML = res.length_m === undefined ? "" :
      `Route ${(res.length_m / 1000).toFixed(2)} km, about ${Math.round(res.est_time_s / 60)} min at cruise.<br>
       Camera footprint at cruise: ${res.footprint_m} m wide.<br>
       ${IMAGERY[$("imagery").value].label} cells: ${res.cached}/${res.cells} cached` +
      (res.cells > res.cached ? `, about ${res.est_download_mb} MB to fetch.` : ".");
    $("warnings").innerHTML = res.warnings.map((w) => `<li>${w}</li>`).join("");
    routeCached = Boolean(res.cells && res.cached === res.cells);
    routeCells = (res.cell_keys || []).join();
    $("export").disabled = !routeCached;
    updateSimButton();
    refreshImagery(res.cached);
  }, 250);
}

// Whether simulated flights meet clouds by default follows the plan's altitude, and the control beside this can override it.
let cloudBase = 750, cloudDefault = null;
const flightMeetsClouds = () => Math.max(+$("default_alt").value, +$("takeoff_alt").value,
  ...wps.map((w) => (w.alt === null || w.alt === undefined ? +$("default_alt").value : w.alt))) >= cloudBase;

function cloudsInFlight() {
  const choice = $("sim_clouds").value;
  return choice === "auto" ? (flightMeetsClouds() ? "real" : "none") : choice;
}

function showCloudState() {
  const high = flightMeetsClouds(), state = $("clouds_state");
  const byDefault = high ? `included above ${cloudBase} m` : `not included below ${cloudBase} m`;
  const chosen = { real: "included at every height", shadows: "only their shadows", none: "not included" }[$("sim_clouds").value];
  state.textContent = chosen ? `Clouds in simulation: ${chosen} (default here: ${byDefault})` : `Clouds in simulation: ${byDefault}`;
  if (cloudDefault !== null && cloudDefault !== high) {
    state.classList.remove("changed");
    void state.offsetWidth;  // restarts the animation
    state.classList.add("changed");
  }
  cloudDefault = high;
  $("sim_pass_box").hidden = cloudsInFlight() === "none" && worldChoice().world_imagery === "naip";
}
$("default_alt").oninput = showCloudState;
$("takeoff_alt").oninput = showCloudState;

// What imagery the route uses, and whether NAIP covers it. The server may have to ask the imagery
// source, so only when the route's cells change.
let imageryKey = null;
async function refreshImagery(cached) {
  const plan = $("imagery");
  $("cache_name").textContent = `Cache ${IMAGERY[plan.value].label} for route`;
  const key = `${routeCells}/${cached}/${plan.value}/${planDate()}`;
  if (key === imageryKey) return;
  imageryKey = key;
  try {
    const { label, naip_here } = await (await api("/api/plan/imagery", { method: "POST", body: JSON.stringify(mission()) })).json();
    if (key !== imageryKey) return;
    $("naip_years").textContent = label;
    plan.options[0].disabled = naip_here === false;
    plan.options[0].textContent = naip_here === false ? "NAIP (none here)" : "NAIP (0.6 m aerial)";
    $("imagery_note").hidden = naip_here !== false;
    $("imagery_note").textContent = "There is no NAIP here, so this plan uses Sentinel-2.";
    if (naip_here === false && plan.value === "naip") {
      plan.value = "sentinel2";
      planImageryChanged();
    }
  } catch (e) { imageryKey = null; console.warn(e); }
}

$("cache").onclick = async () => {
  $("cache").disabled = true;
  const name = $("imagery").value, layer = imageryLayers[name];
  try {
    let job = await (await api("/api/cache/jobs", { method: "POST", body: JSON.stringify(mission()) })).json();
    $("progress").hidden = false;
    while (job.state === "running") {
      const shown = job.done;
      $("bar").style.width = `${job.total ? (100 * job.done) / job.total : 100}%`;
      $("ptext").textContent = `${job.done} / ${job.total} new cells`;
      await new Promise((r) => setTimeout(r, 1000));
      job = await (await api(`/api/cache/jobs/${job.id}`)).json();
      if (job.done !== shown) layer.setUrl(imageryUrl(name, job.done));
    }
    $("bar").style.width = "100%";
    if (job.total) layer.setUrl(imageryUrl(name, Date.now()));
    const notes = [];
    if (job.no_coverage.length) {
      notes.push(`${job.no_coverage.length} cells have no ${IMAGERY[name].label} coverage` + (name === "naip" ? "; Sentinel-2 may" : ""));
    }
    if (job.errors.length) notes.push(`${job.errors.length} errors (see server log)`);
    const kept = job.route_cells - job.total;
    $("ptext").textContent = notes.length ? notes.join("; ")
      : job.total ? `Downloaded ${job.total} new, ${kept} already cached` : `All ${kept} cells already cached`;
    refreshCells();
    refreshRegions();
    check();
    refreshPasses(true);
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
  if (shownImagery) cellsLayer.addData(await (await api(`/api/cache/cells?imagery=${shownImagery}${dateQuery(shownImagery)}`)).json());
}

function regionTag(region, name) {
  const { cells, on_map } = region[name], { label } = IMAGERY[name];
  const count = cells < region.cells ? ` ${cells}/${region.cells}` : "";
  if (on_map) return `<span class="tag">${label}${count}</span>`;
  const why = `Downloaded for simulated flights or for another date, so the map's ${label} button shows nothing here.`;
  return `<span class="tag off" title="${why}">${label}${count}, not on map</span>`;
}

// Lists the regions with cached imagery and returns them, or none if that fails.
async function refreshRegions() {
  const list = $("regions");
  let regions;
  try {
    regions = await (await api(`/api/cache/regions?${dateQuery("sentinel2")}`)).json();
  } catch (e) {
    list.innerHTML = `<li class="hint">Could not list cached imagery: ${e.message}</li>`;
    return [];
  }
  list.innerHTML = regions.length ? "" : `<li class="hint">Nothing downloaded yet.</li>`;
  const km = (a, b) => Math.max(1, Math.round(L.latLng(a).distanceTo(b) / 1000));
  regions.forEach((region) => {
    const [w, s, e, n] = region.bbox;
    const li = list.appendChild(document.createElement("li"));
    const b = li.appendChild(document.createElement("button"));
    b.className = "region";
    b.innerHTML = `<span>${region.center[0].toFixed(3)}, ${region.center[1].toFixed(3)}</span>
      <small>${region.cells} cell${region.cells === 1 ? "" : "s"}, ${km([s, w], [s, e])} × ${km([s, w], [n, w])} km,
      ${new Date(region.downloaded * 1000).toLocaleDateString()}</small>
      <div>${Object.keys(IMAGERY).filter((name) => region[name].cells).map((name) => regionTag(region, name)).join(" ")}</div>`;
    b.onclick = () => goToRegion(region);
  });
  return regions;
}

function goToRegion(region) {
  showView("plan");
  const [w, s, e, n] = region.bbox, bounds = L.latLngBounds([s, w], [n, e]).pad(0.1);
  // Cached imagery is not drawn when zoomed out further, so a very long region only shows around its middle.
  if (map.getBoundsZoom(bounds) < IMAGERY_MIN_ZOOM) map.setView(region.center, IMAGERY_MIN_ZOOM);
  else map.fitBounds(bounds);
  if (shownImagery && !region[shownImagery].on_map) {
    const other = Object.keys(IMAGERY).find((name) => region[name].on_map);
    if (other) showImagery(other);
  }
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
  const satellite = $("imagery").value === "sentinel2";
  const key = `${routeCells}/${satellite}`;
  if (key === yearsKey) return;
  yearsKey = key;
  try {
    const body = JSON.stringify(mission());
    const [{ years }, s2] = await Promise.all([
      api("/api/sim/years", { method: "POST", body }).then((r) => r.json()),
      api("/api/sentinel2/passes", { method: "POST", body }).then((r) => r.json()),
    ]);
    if (key !== yearsKey) return;
    const sel = $("sim_world"), keep = sel.value;
    while (sel.options.length > 1) sel.remove(1);
    // A camera seeing 10 m imagery from 120 m would be far blurrier than a real one, so NAIP is the default there.
    if (satellite && years.length) sel.add(new Option("Newest NAIP (realistic, downloads it if needed)", "newest"));
    years.forEach((y) => sel.add(new Option(`NAIP ${y} (realistic, downloads it if needed)`, y)));
    if (s2.years.length) sel.add(new Option("Sentinel-2, last pass (blurry for a camera)", "s2:newest"));
    s2.years.forEach((y) => sel.add(new Option(`Sentinel-2 ${y} (blurry for a camera)`, `s2:${y}`)));
    sel.value = keep || (satellite && years.length ? "newest" : "");
    if (sel.selectedIndex < 0) sel.value = satellite && years.length ? "newest" : "";
    refreshPasses(true);
  } catch (e) { yearsKey = null; console.warn(e); }
}

function worldChoice() {
  const value = $("sim_world").value, satellite = value.startsWith("s2:");
  const year = value.replace("s2:", "");
  return { world_imagery: satellite ? "sentinel2" : "naip", world_year: year === "newest" ? "newest" : year ? +year : null };
}

// The passes a flight could use: a chosen year's, or the last 12 months'.
let passesKey = null;
async function refreshPasses(force) {
  const { world_imagery, world_year } = worldChoice();
  const year = Number.isInteger(world_year) ? world_year : null;
  const key = `${routeCells}/${year}`;
  $("sim_pass_box").hidden = cloudsInFlight() === "none" && world_imagery === "naip";
  if (key === passesKey && !force) return;
  passesKey = key;
  try {
    const url = `/api/sentinel2/passes${year ? `?year=${year}` : ""}`;
    const { passes, stored } = await (await api(url, { method: "POST", body: JSON.stringify(mission()) })).json();
    if (key !== passesKey) return;
    const sel = $("sim_pass"), keep = sel.value;
    while (sel.options.length) sel.remove(0);
    const newest = passes[0];
    const auto = year ? (passes.length ? `Picked by the seed, from ${passes.length} that year` : "None that year, so no clouds")
      : newest ? `Last pass (${newest.date})` : "None found, so no clouds";
    sel.add(new Option(auto, ""));
    passes.forEach((p) => sel.add(new Option(`${p.date}${p.cloud_pct === null ? "" : `, ${p.cloud_pct}% cloud`}${stored.passes.includes(p.date) ? ", stored" : ""}`, p.date)));
    sel.value = keep;
    if (sel.selectedIndex < 0) sel.value = "";
    const parts = [];
    if (stored.clear.length) parts.push(`clear ${stored.clear.join(", ")}`);
    if (stored.passes.length) parts.push(`with clouds ${stored.passes.join(", ")}`);
    $("s2_stored").hidden = !parts.length;
    $("s2_stored").textContent = `Sentinel-2 stored for this route: ${parts.join("; ")}.`;
  } catch (e) { passesKey = null; console.warn(e); }
}
$("sim_world").onchange = () => refreshPasses();
$("sim_clouds").onchange = () => { showCloudState(); refreshPasses(); };

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

// Fog: each choice fills in the values, and changing a value makes it Custom.
const FOG_PRESETS = {
  haze: { visibility: 5000, top: 1500, patchiness: 0.2, patch: 2000, brightness: 0.75 },
  mist: { visibility: 1500, top: 400, patchiness: 0.3, patch: 800, brightness: 0.8 },
  fog: { visibility: 400, top: 200, patchiness: 0.4, patch: 300, brightness: 0.8 },
  dense: { visibility: 100, top: 150, patchiness: 0.5, patch: 200, brightness: 0.85 },
};
const fogInputs = [...document.querySelectorAll("[data-fog]")];

function fogValues() {
  if (!$("sim_fog").value) return {};
  return Object.fromEntries(fogInputs.map((el) => [el.dataset.fog, +el.value]));
}

$("sim_fog").onchange = () => {
  const preset = FOG_PRESETS[$("sim_fog").value];
  if (preset) fogInputs.forEach((el) => { el.value = preset[el.dataset.fog]; });
  $("fog_box").hidden = !$("sim_fog").value;
};
fogInputs.forEach((el) => { el.oninput = () => { $("sim_fog").value = "custom"; }; });

function simRequest() {
  return JSON.stringify({
    mission: mission(),
    options: {
      seed: +$("sim_seed").value, wind_speed: +$("sim_wind").value, wind_from_deg: +$("sim_wind_from").value,
      gust: +$("sim_gust").value, realtime: +$("sim_pace").value,
      ...worldChoice(), clouds: $("sim_clouds").value, pass_date: $("sim_pass").value || null,
      appearance: lookValues(), fog: fogValues(), navigation: $("sim_nav").value,
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
  else {
    refreshRuns();
    refreshRegions();
  }
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
      text = `Downloading ${run.fetching}: ${run.cells.done} / ${run.cells.total} cells`;
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

function worldName({ world_imagery: imagery, world_year: year }) {
  if (imagery === "sentinel2") return year === "newest" ? "the last Sentinel-2 pass" : `Sentinel-2 from ${year}`;
  return year === "newest" ? "the newest NAIP" : `NAIP ${year}`;
}

function skyText(run) {
  const w = run.world;
  if (!w) return "";
  if (w.note) return ` ${w.note}`;
  if (w.clouds === "none") return w.choice === "auto" ? ` No clouds, as the flight stays below ${cloudBase} m.` : " No clouds.";
  const where = w.base_m ? `, seen above ${w.base_m} m` : "";
  const what = w.clouds === "real" ? `${w.cloud_pct}% of the route under cloud${where}, ${w.shadow_pct}% in shadow`
    : `${w.shadow_pct}% of the route in shadow, as seen from below the clouds`;
  return ` Sky from the Sentinel-2 pass of ${w.pass_date}: ${what}.`;
}

function resultText(run) {
  if (run.state === "failed") return `Failed: ${run.error}`;
  if (run.state === "stopped") return "Stopped before the end of the flight.";
  const s = run.summary;
  const end = s.landed ? `Landed ${s.landing_error_m} m from the target.`
    : s.mission_complete ? `Reached the last waypoint, ${s.landing_error_m} m off.` : "Ran out of time before finishing.";
  const alone = run.options.navigation === "odometry";
  const note = (run.imagery === "sentinel2" ? "Map from Sentinel-2. " : "") +
    (run.options.world_year ? `Camera saw ${worldName(run.options)}.`
    : alone ? "" : "The camera saw the map's own imagery, so this is optimistic.") + skyText(run) +
    (s.fog ? ` Fog: ${s.fog.visibility} m visibility, up to ${s.fog.top} m.` : "") +
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
    const nav = (r.imagery === "sentinel2" ? "Sentinel-2, " : "") + (r.options.navigation === "odometry" ? "odometry only, " : "");
    sel.add(new Option(`${r.name}, ${r.created.slice(5, 16).replace("T", " ")}: ${nav}${outcome}`, r.id));
  });
  sel.value = openRun || "";
  return runs;
}

(async () => {
  const cfg = await (await api("/api/config")).json();
  $("cfg").textContent = `NAIP: ${cfg.source}, ${cfg.gsd} m. Sentinel-2: ${cfg.s2_source}, ${cfg.s2_gsd} m. Data in ${cfg.data_dir}`;
  simulator = cfg.simulator;
  cloudBase = cfg.cloud_base_m;
  document.querySelector(".clouds-row").hidden = !simulator;
  $("simsec").hidden = !simulator;
  $("sim_missing").hidden = simulator;
  refreshSaved();
  showImagery($("imagery").value);
  const [last] = await refreshRegions();
  if (last) goToRegion(last);
  else map.setView([39.74, -104.99], 13);
  check();
  if (simulator) {
    const running = (await refreshRuns()).find((r) => ACTIVE.includes(r.state));
    if (running) followRun(running.id);
  }
})();
