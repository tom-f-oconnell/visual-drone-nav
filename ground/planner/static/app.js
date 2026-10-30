// NAIP flight planner front end: Leaflet map, waypoint editing, cache + export calls.
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
      <input type="number" placeholder="alt" title="altitude m (blank = cruise)" value="${wp.alt ?? ""}">
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

let checkTimer;
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
    $("export").disabled = !(res.cells && res.cached === res.cells);
  }, 250);
}

$("cache").onclick = async () => {
  $("cache").disabled = true;
  try {
    let job = await (await api("/api/cache/jobs", { method: "POST", body: JSON.stringify(mission()) })).json();
    $("progress").hidden = false;
    while (job.state === "running") {
      $("bar").style.width = `${job.total ? (100 * job.done) / job.total : 100}%`;
      $("ptext").textContent = `${job.done} / ${job.total} cells`;
      await new Promise((r) => setTimeout(r, 1000));
      job = await (await api(`/api/cache/jobs/${job.id}`)).json();
      naip.setUrl(`/api/tiles/{z}/{x}/{y}.png?v=${job.done}`);
    }
    $("bar").style.width = "100%";
    naip.setUrl(`/api/tiles/{z}/{x}/{y}.png?v=${Date.now()}`);
    const notes = [];
    if (job.no_coverage.length) notes.push(`${job.no_coverage.length} cells have no NAIP coverage`);
    if (job.errors.length) notes.push(`${job.errors.length} errors (see server log)`);
    $("ptext").textContent = notes.length ? notes.join("; ") : `Cached ${job.total} cells`;
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

(async () => {
  const cfg = await (await api("/api/config")).json();
  $("cfg").textContent = `Imagery source: ${cfg.source}, cache GSD ${cfg.gsd} m, data in ${cfg.data_dir}`;
  refreshSaved();
  refreshCells();
  check();
})();
