/**
 * FLICK Wind Webapp – Three.js scene + pipeline orchestration.
 *
 * Coordinate convention:
 *   STL files are Z-up (X-right, Y-forward, Z-up).
 *   A worldGroup rotated -π/2 around X converts to Three.js Y-up for rendering.
 *   Heatmap plane lives in the worldGroup in the original Z-up frame
 *   (horizontal, in the XY plane at Z = 3 m), rotated by -wind_angle around Z
 *   to un-rotate the CFD grid back into the building frame.
 */

import * as THREE from 'three';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

// ---------------------------------------------------------------------------
// Scene setup
// ---------------------------------------------------------------------------
const canvasWrap = document.getElementById('canvas-wrap');

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x1a1d23);
canvasWrap.appendChild(renderer.domElement);

const scene = new THREE.Scene();

// worldGroup holds all geometry in Z-up coords; the group itself is tilted
// so Three.js Y-up camera works naturally.
const worldGroup = new THREE.Group();
worldGroup.rotation.x = -Math.PI / 2;
scene.add(worldGroup);

// Lights
scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
dirLight.position.set(200, 400, 300);
scene.add(dirLight);

// Camera – starts from above, looking down
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1e6);
camera.position.set(0, 600, 200);
camera.lookAt(0, 0, 0);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

// Resize handler
function onResize() {
  const w = canvasWrap.clientWidth;
  const h = canvasWrap.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', onResize);
onResize();

// Render loop
(function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
})();

// ---------------------------------------------------------------------------
// Scene objects (replaced on each result)
// ---------------------------------------------------------------------------
let stlMesh = null;
let heatmapMesh = null;
let arrowGroup = null;
let streamlineGroup = null;
let windData = null;       // cached {u, v, n_points, step_size}
let currentResults = null; // cached results.json

function clearScene() {
  if (stlMesh)         { worldGroup.remove(stlMesh);         stlMesh = null; }
  if (heatmapMesh)     { worldGroup.remove(heatmapMesh);     heatmapMesh = null; }
  if (arrowGroup)      { worldGroup.remove(arrowGroup);      arrowGroup = null; }
  if (streamlineGroup) { worldGroup.remove(streamlineGroup);  streamlineGroup = null; }
  windData = null;
  currentResults = null;
}

function loadSTL(url) {
  return new Promise((resolve, reject) => {
    new STLLoader().load(
      url,
      (geometry) => {
        geometry.computeVertexNormals();
        const mat = new THREE.MeshPhongMaterial({
          color: 0x8899aa,
          specular: 0x334455,
          shininess: 30,
          side: THREE.DoubleSide,
        });
        const mesh = new THREE.Mesh(geometry, mat);
        resolve(mesh);
      },
      undefined,
      reject,
    );
  });
}

function placeHeatmap(results, heatmapUrl) {
  return new Promise((resolve) => {
    const { step_size, stl_bbox: bbox, wind_angle, ref_height } = results;
    const size = 2 * step_size;
    const zPos = ref_height ?? 3;

    const geometry = new THREE.PlaneGeometry(size, size);
    new THREE.TextureLoader().load(heatmapUrl + '&t=' + Date.now(), (tex) => {
      // PlaneGeometry lies in the XY plane by default (normal = +Z) → perfect
      // for our Z-up worldGroup frame.
      const mat = new THREE.MeshBasicMaterial({
        map: tex,
        transparent: true,
        opacity: 0.88,
        depthWrite: false,
        side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(geometry, mat);
      mesh.position.set(bbox.centre_x, bbox.centre_y, zPos);
      // Un-rotate CFD grid back into original building frame
      mesh.rotation.z = -(wind_angle * Math.PI) / 180;
      resolve(mesh);
    });
  });
}

// CSS gradient approximating matplotlib Spectral_r (shared by all fields)
const SPECTRAL_R_GRADIENT = 'linear-gradient(to right,#9e0142,#d53e4f,#f46d43,#fdae61,#fee08b,#ffffbf,#e6f598,#abdda4,#66c2a5,#3288bd,#5e4fa2)';
const FIELD_META = {
  UMAG: { label: 'Wind speed magnitude (m/s)', gradient: SPECTRAL_R_GRADIENT },
  UGT:  { label: 'U component \u2013 X velocity (m/s)', gradient: SPECTRAL_R_GRADIENT },
  VGT:  { label: 'V component \u2013 Y velocity (m/s)', gradient: SPECTRAL_R_GRADIENT },
};

/**
 * Update colorbar: gradient, title, and tick marks.
 */
function buildColorbar(field, vmin, vmax) {
  const meta = FIELD_META[field] || FIELD_META.UMAG;
  legendRamp.style.background  = meta.gradient;
  legendTitle.textContent = meta.label;

  legendTicks.innerHTML = '';
  const N_TICKS = 5;
  for (let i = 0; i < N_TICKS; i++) {
    const frac = i / (N_TICKS - 1);
    const val  = vmin + frac * (vmax - vmin);
    const tick = document.createElement('span');
    tick.className = 'tick';
    tick.style.left = (frac * 100) + '%';
    tick.textContent = val.toFixed(2);
    legendTicks.appendChild(tick);
  }
}

// ---------------------------------------------------------------------------
// Wind data fetch
// ---------------------------------------------------------------------------
async function fetchWindData() {
  if (windData) return windData;
  const res = await fetch(`/api/jobs/${jobId}/wind_data`);
  if (!res.ok) throw new Error('Failed to fetch wind data');
  windData = await res.json();
  return windData;
}

// ---------------------------------------------------------------------------
// Arrow field (quiver) rendering
// ---------------------------------------------------------------------------
const BASE_STEP = 8; // default grid spacing; scaled by arrowDensity slider

function buildArrowGeometry() {
  // Shaft (thin cylinder along +Y)
  const shaft = new THREE.CylinderGeometry(0.15, 0.15, 0.7, 6);
  shaft.translate(0, 0.35, 0);
  // Head (cone at the tip)
  const head = new THREE.ConeGeometry(0.35, 0.3, 6);
  head.translate(0, 0.85, 0);
  // Merge and rotate so the arrow points along +X (default direction)
  const merged = mergeGeometries([shaft, head]);
  merged.rotateZ(-Math.PI / 2); // now points along +X
  return merged;
}

// velScale: multiplier applied only to colour (size never changes)
function buildArrows(results, wd, velScale = 1.0) {
  const { step_size, stl_bbox: bbox, wind_angle, ref_height } = results;
  const { u, v, n_points } = wd;
  const cellSize = 2 * step_size / n_points;

  // Color scale: use time-series range when active, else pipeline field range
  const vmin = timeSeries.length > 0 ? animVmin : (windFields[activeField]?.min ?? 0);
  const vmax = timeSeries.length > 0 ? animVmax : (windFields[activeField]?.max ?? 1);
  const vRange = vmax - vmin || 1;

  const arrowGeo = buildArrowGeometry();
  const arrowMat = new THREE.MeshPhongMaterial({
    vertexColors: false,
    transparent: true,
    opacity: 0.9,
  });

  const instances = [];
  const colours = [];
  // Arrow size fixed relative to domain — never changes with wind speed
  const baseScale = step_size * 0.055;
  const dummy = new THREE.Object3D();
  const col = new THREE.Color();

  // Density slider: lower value = smaller step = more arrows
  const step = Math.max(1, Math.round(BASE_STEP * arrowDensity));

  for (let r = step; r < n_points - step; r += step) {
    for (let c = step; c < n_points - step; c += step) {
      const ux = u[r][c];
      // V centred: 0.5 = no lateral flow, <0.5 = −y, >0.5 = +y
      const vy = v[r][c] - 0.5;
      if (ux < 0.01) continue; // solid / stagnant cell

      const scale = baseScale;

      // Grid coords: row 0 = +y (origin='upper'), col 0 = −x
      const x = -step_size + (c + 0.5) * cellSize;
      const y =  step_size - (r + 0.5) * cellSize;

      dummy.position.set(x, y, 0);
      dummy.rotation.set(0, 0, Math.atan2(vy, ux));
      dummy.scale.set(scale, scale, scale);
      dummy.updateMatrix();
      instances.push(dummy.matrix.clone());

      // Colour via Spectral_r, matching the heatmap colour scale
      const rawMag = wd.umag ? wd.umag[r][c] : Math.sqrt(ux * ux + vy * vy);
      const t = Math.max(0, Math.min(1, (rawMag * velScale - vmin) / vRange));
      const [cr, cg, cb] = spectralRColor(t);
      col.setRGB(cr, cg, cb);
      colours.push(col.clone());
    }
  }

  const mesh = new THREE.InstancedMesh(arrowGeo, arrowMat, instances.length);
  instances.forEach((m, i) => {
    mesh.setMatrixAt(i, m);
    mesh.setColorAt(i, colours[i]);
  });
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;

  const group = new THREE.Group();
  group.add(mesh);
  const zPos = ref_height ?? 3;
  group.position.set(bbox.centre_x, bbox.centre_y, zPos + 0.5);
  group.rotation.z = -(wind_angle * Math.PI) / 180;
  return group;
}

// ---------------------------------------------------------------------------
// Streamline rendering (RK2 integration)
// ---------------------------------------------------------------------------
function buildStreamlines(results, wd) {
  const { step_size, stl_bbox: bbox, wind_angle, ref_height } = results;
  const { u, v, n_points } = wd;
  const cellSize = 2 * step_size / n_points;

  // Bilinear interpolation returning signed (ux, vy) in CFD frame
  // V is centred around 0.5: subtract to get signed lateral component
  function sampleUV(gx, gy) {
    const c0 = Math.floor(gx), r0 = Math.floor(gy);
    if (c0 < 0 || r0 < 0 || c0 >= n_points - 1 || r0 >= n_points - 1) return null;
    const c1 = c0 + 1, r1 = r0 + 1;
    const fc = gx - c0, fr = gy - r0;
    const lerp = (a, b, t) => a + (b - a) * t;
    const ux = lerp(lerp(u[r0][c0], u[r0][c1], fc), lerp(u[r1][c0], u[r1][c1], fc), fr);
    const vyRaw = lerp(lerp(v[r0][c0], v[r0][c1], fc), lerp(v[r1][c0], v[r1][c1], fc), fr);
    return [ux, vyRaw - 0.5]; // centre V so 0 = no lateral
  }

  // Max speed for stopping threshold (sample on sub-grid)
  let maxMag = 0;
  for (let r = 0; r < n_points; r += 4) {
    for (let c = 0; c < n_points; c += 4) {
      const vy = v[r][c] - 0.5;
      const m = Math.sqrt(u[r][c] ** 2 + vy ** 2);
      if (m > maxMag) maxMag = m;
    }
  }
  const minMag = maxMag * 0.02;
  const maxSteps = 400;
  const dt = 0.5; // step size in grid units

  function integrate(gx0, gy0) {
    const pts = [];
    let gx = gx0, gy = gy0;
    for (let s = 0; s < maxSteps; s++) {
      const uv1 = sampleUV(gx, gy);
      if (!uv1) break;
      const [ux1, vy1] = uv1;
      if (Math.sqrt(ux1 * ux1 + vy1 * vy1) < minMag) break;

      // Convert world velocity → grid velocity (V negated: row↑ = world-y↑)
      const dg1x = ux1 / cellSize, dg1y = -vy1 / cellSize;
      const mx = gx + 0.5 * dt * dg1x, my = gy + 0.5 * dt * dg1y;
      const uv2 = sampleUV(mx, my);
      if (!uv2) break;
      const dg2x = uv2[0] / cellSize, dg2y = -uv2[1] / cellSize;

      gx += dt * dg2x;
      gy += dt * dg2y;
      if (gx < 0 || gx >= n_points || gy < 0 || gy >= n_points) break;

      pts.push(new THREE.Vector3(
        -step_size + gx * cellSize,
         step_size - gy * cellSize,
        0,
      ));
    }
    return pts;
  }

  // Seed points on a regular grid; skip solid cells (u ≈ 0)
  const nSeeds = 7;
  const margin = n_points * 0.08;
  const spacing = (n_points - 2 * margin) / (nSeeds - 1);
  const lines = [];

  for (let si = 0; si < nSeeds; si++) {
    for (let sj = 0; sj < nSeeds; sj++) {
      const gx0 = margin + sj * spacing;
      const gy0 = margin + si * spacing;
      // Skip seed if in solid region
      const r0 = Math.round(gy0), c0 = Math.round(gx0);
      if (r0 >= 0 && r0 < n_points && c0 >= 0 && c0 < n_points && u[r0][c0] < 0.02) continue;

      const pts = integrate(gx0, gy0);
      if (pts.length < 3) continue;

      const geom = new THREE.BufferGeometry().setFromPoints(pts);
      // Colour: cyan at start fading to white at end
      const colors = new Float32Array(pts.length * 3);
      for (let i = 0; i < pts.length; i++) {
        const t = i / pts.length;
        colors[i * 3]     = 0.2 + 0.8 * t;
        colors[i * 3 + 1] = 0.85;
        colors[i * 3 + 2] = 1.0;
      }
      geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      const mat = new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.75,
        linewidth: 1,
      });
      lines.push(new THREE.Line(geom, mat));
    }
  }

  const group = new THREE.Group();
  lines.forEach(l => group.add(l));
  const zPos = ref_height ?? 3;
  group.position.set(bbox.centre_x, bbox.centre_y, zPos + 0.5);
  group.rotation.z = -(wind_angle * Math.PI) / 180;
  return group;
}

// ---------------------------------------------------------------------------
// UI state
// ---------------------------------------------------------------------------
let jobId = null;
let pollTimer = null;
let windAngle = 0;
let pxResolution = null; // null = auto
let refHeight = 10.0;
let refVelocity = 1.0;
let activeField = 'UMAG';
let windFields  = {};    // populated from results.json after pipeline

const dropZone    = document.getElementById('drop-zone');
const fileInput   = document.getElementById('file-input');
const fileNameEl  = document.getElementById('file-name');
const windSlider  = document.getElementById('wind-slider');
const angleLbl    = document.getElementById('angle-label');
const heightInput = document.getElementById('height-input');
const heightLbl   = document.getElementById('height-label');
const velInput    = document.getElementById('vel-input');
const velLbl      = document.getElementById('vel-label');
const resSelect   = document.getElementById('res-select');
const resLabel    = document.getElementById('res-label');
const resInfo     = document.getElementById('res-info');
const fieldSection= document.getElementById('field-section');
const fieldSelect = document.getElementById('field-select');
const runBtn      = document.getElementById('run-btn');
const statusMsg   = document.getElementById('status-msg');
const progressBar = document.getElementById('progress-bar');
const legendEl    = document.getElementById('legend');
const legendTitle = document.getElementById('legend-title');
const legendRamp  = document.getElementById('legend-ramp');
const legendTicks = document.getElementById('legend-ticks');
const logSection     = document.getElementById('log-section');
const logOutput      = document.getElementById('log-output');
const overlaySection    = document.getElementById('overlay-section');
const arrowsToggle      = document.getElementById('arrows-toggle');
const arrowDensitySlider= document.getElementById('arrow-density');
const arrowDensityLabel = document.getElementById('arrow-density-label');
const streamToggle      = document.getElementById('streamlines-toggle');
const downloadSection= document.getElementById('download-section');
const dlCsvBtn       = document.getElementById('dl-csv-btn');
const dlGeotiffBtn   = document.getElementById('dl-geotiff-btn');
const tsSection      = document.getElementById('ts-section');
const tsUploadBtn    = document.getElementById('ts-upload-btn');
const tsManualBtn    = document.getElementById('ts-manual-btn');
const tsFileInput    = document.getElementById('ts-file-input');
const tsTableWrap    = document.getElementById('ts-table-wrap');
const tsTbody        = document.getElementById('ts-tbody');
const tsAddRow       = document.getElementById('ts-add-row');
const tsSaveBtn      = document.getElementById('ts-save-btn');
const tsInfo         = document.getElementById('ts-info');
const animSection    = document.getElementById('anim-section');
const animSlider     = document.getElementById('anim-slider');
const animTimeLabel  = document.getElementById('anim-time-label');
const animSpeedLabel = document.getElementById('anim-speed-label');
const animPlayBtn    = document.getElementById('anim-play-btn');
const animStopBtn    = document.getElementById('anim-stop-btn');
const animResetBtn   = document.getElementById('anim-reset-btn');
const animRateSlider = document.getElementById('anim-rate');
const animRateLabel  = document.getElementById('anim-rate-label');

// Time series + animation state
let arrowDensity = 1.0; // slider multiplier: <1 = denser, >1 = sparser
let timeSeries  = [];   // [{t (minutes), v (m/s)}, …] sorted by t
let animTimer   = null;
let animT       = 0;    // current time in minutes
let animRate    = 15;   // sim-minutes per real second (15 min/s → 24h in ~96s)
let animVmin    = 0;    // color scale lower bound (derived from time series)
let animVmax    = 1;    // color scale upper bound (derived from time series)

// progress bar animation states
const PROGRESS_STAGES = {
  pending:  5,
  running:  60,   // advanced by progress_msg keywords below
  done:     100,
  error:    100,
};

function setProgress(status, msg) {
  let pct = PROGRESS_STAGES[status] ?? 0;
  if (status === 'running') {
    if (msg.includes('inference'))  pct = 80;
    if (msg.includes('heatmap'))    pct = 92;
  }
  progressBar.style.width = pct + '%';
  statusMsg.textContent = msg || status;
}

// ---------------------------------------------------------------------------
// Drag-and-drop + file upload
// ---------------------------------------------------------------------------
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', (e) => { if (e.key === 'Enter') fileInput.click(); });
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

async function handleFile(file) {
  fileNameEl.textContent = file.name;
  setProgress('pending', 'Uploading STL…');
  progressBar.style.width = '10%';

  const fd = new FormData();
  fd.append('file', file);
  try {
    const res = await fetch('/api/upload', { method: 'POST', body: fd });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    jobId = data.job_id;
    runBtn.disabled = false;
    setProgress('pending', 'STL uploaded. Set wind direction and click Run.');
    progressBar.style.width = '5%';
  } catch (err) {
    setProgress('error', 'Upload failed: ' + err.message);
  }
}

// ---------------------------------------------------------------------------
// Wind slider
// ---------------------------------------------------------------------------
windSlider.addEventListener('input', () => {
  windAngle = parseFloat(windSlider.value);
  angleLbl.textContent = windAngle + '°';
});

// ---------------------------------------------------------------------------
// Reference height & velocity inputs
// ---------------------------------------------------------------------------
heightInput.addEventListener('input', () => {
  refHeight = parseFloat(heightInput.value) || 10;
  heightLbl.textContent = refHeight + ' m';
});
velInput.addEventListener('input', () => {
  refVelocity = parseFloat(velInput.value) || 1;
  velLbl.textContent = refVelocity.toFixed(1) + ' m/s';
});

// ---------------------------------------------------------------------------
// Pixel resolution select
// ---------------------------------------------------------------------------
resSelect.addEventListener('change', () => {
  const val = resSelect.value;
  if (val === 'auto') {
    pxResolution = null;
    resLabel.textContent = 'auto';
  } else {
    pxResolution = parseFloat(val);
    resLabel.textContent = val + ' m/px';
  }
  // Hide stale info on change
  resInfo.style.display = 'none';
});

// ---------------------------------------------------------------------------
// Field selector (live swap of heatmap texture)
// ---------------------------------------------------------------------------
fieldSelect.addEventListener('change', async () => {
  activeField = fieldSelect.value;
  const fieldData = windFields[activeField];
  if (!fieldData || !jobId || !heatmapMesh) return;

  if (timeSeries.length > 0) {
    // Re-derive color scale for the new field using the existing time series
    applyTimeSeries(timeSeries);
    return;
  }

  buildColorbar(activeField, fieldData.min, fieldData.max);

  const url = `/api/jobs/${jobId}/heatmap?field=${activeField}&t=${Date.now()}`;
  new THREE.TextureLoader().load(url, (tex) => {
    heatmapMesh.material.map = tex;
    heatmapMesh.material.needsUpdate = true;
  });
});

// ---------------------------------------------------------------------------
// Overlay toggles
// ---------------------------------------------------------------------------
arrowsToggle.addEventListener('change', async () => {
  if (arrowsToggle.checked) {
    try {
      const wd = await fetchWindData();
      if (arrowGroup) worldGroup.remove(arrowGroup);
      arrowGroup = buildArrows(currentResults, wd);
      worldGroup.add(arrowGroup);
    } catch (e) { console.warn('Arrow load failed:', e); }
  } else if (arrowGroup) {
    worldGroup.remove(arrowGroup);
  }
});

arrowDensitySlider.addEventListener('input', () => {
  arrowDensity = parseFloat(arrowDensitySlider.value);
  arrowDensityLabel.textContent = arrowDensity.toFixed(2).replace(/\.?0+$/, '') + '×';
  if (!arrowsToggle.checked || !currentResults || !windData) return;
  if (arrowGroup) worldGroup.remove(arrowGroup);
  arrowGroup = buildArrows(currentResults, windData);
  worldGroup.add(arrowGroup);
});

streamToggle.addEventListener('change', async () => {
  if (streamToggle.checked) {
    try {
      const wd = await fetchWindData();
      if (!streamlineGroup) streamlineGroup = buildStreamlines(currentResults, wd);
      worldGroup.add(streamlineGroup);
    } catch (e) { console.warn('Streamline load failed:', e); }
  } else if (streamlineGroup) {
    worldGroup.remove(streamlineGroup);
  }
});

// ---------------------------------------------------------------------------
// Download buttons
// ---------------------------------------------------------------------------
dlCsvBtn.addEventListener('click', () => {
  if (jobId) window.location.href = `/api/jobs/${jobId}/download/csv`;
});
dlGeotiffBtn.addEventListener('click', () => {
  if (jobId) window.location.href = `/api/jobs/${jobId}/download/geotiff?field=${activeField}`;
});

// ---------------------------------------------------------------------------
// Jet colormap + client-side heatmap rendering
// ---------------------------------------------------------------------------
// Spectral_r colormap (matches matplotlib Spectral_r via piecewise RGB)
function spectralRColor(t) {
  // Control points sampled from Spectral_r (t=0 → dark red, t=1 → dark purple)
  const stops = [
    [0.000, [0.620, 0.004, 0.259]],
    [0.100, [0.835, 0.243, 0.310]],
    [0.200, [0.957, 0.427, 0.263]],
    [0.300, [0.992, 0.682, 0.380]],
    [0.400, [0.996, 0.878, 0.545]],
    [0.500, [1.000, 1.000, 0.749]],
    [0.600, [0.902, 0.961, 0.596]],
    [0.700, [0.671, 0.867, 0.643]],
    [0.800, [0.400, 0.761, 0.647]],
    [0.900, [0.196, 0.533, 0.741]],
    [1.000, [0.369, 0.310, 0.635]],
  ];
  t = Math.max(0, Math.min(1, t));
  let i = 0;
  while (i < stops.length - 2 && stops[i + 1][0] <= t) i++;
  const [t0, c0] = stops[i], [t1, c1] = stops[i + 1];
  const f = (t - t0) / (t1 - t0);
  return [c0[0] + f * (c1[0] - c0[0]), c0[1] + f * (c1[1] - c0[1]), c0[2] + f * (c1[2] - c0[2])];
}

function renderHeatmapCanvas(umag, velScale, vmin, vmax) {
  const n = umag.length;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = n;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(n, n);
  for (let row = 0; row < n; row++) {
    for (let col = 0; col < n; col++) {
      const idx = (row * n + col) * 4;
      const raw = umag[row][col];
      if (raw === 0) { img.data[idx + 3] = 0; continue; } // solid → transparent
      const val = raw * velScale;
      const t = Math.max(0, Math.min(1, (val - vmin) / (vmax - vmin)));
      const [r, g, b] = spectralRColor(t);
      img.data[idx]     = r * 255;
      img.data[idx + 1] = g * 255;
      img.data[idx + 2] = b * 255;
      img.data[idx + 3] = 224;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

// ---------------------------------------------------------------------------
// Time series: interpolation + animation
// ---------------------------------------------------------------------------
function interpolateWindSpeed(t) {
  if (timeSeries.length === 0) return currentResults ? currentResults.ref_velocity : 1;
  if (t <= timeSeries[0].t) return timeSeries[0].v;
  if (t >= timeSeries.at(-1).t) return timeSeries.at(-1).v;
  const i = timeSeries.findIndex(p => p.t > t) - 1;
  const p0 = timeSeries[i], p1 = timeSeries[i + 1];
  return p0.v + (p1.v - p0.v) * (t - p0.t) / (p1.t - p0.t);
}

function applyAnimFrame(t) {
  if (!currentResults || !windData || !windData.umag || !heatmapMesh) return;
  const ws = interpolateWindSpeed(t);
  const scale = ws / (currentResults.ref_velocity || 1);

  // Use the time-series-derived color scale so all frames are comparable
  const canvas = renderHeatmapCanvas(windData.umag, scale, animVmin, animVmax);
  const tex = new THREE.CanvasTexture(canvas);
  tex.flipY = false; // canvas already in correct orientation
  heatmapMesh.material.map = tex;
  heatmapMesh.material.needsUpdate = true;

  // Rebuild arrows with updated scale for colour
  if (arrowsToggle.checked) {
    if (arrowGroup) worldGroup.remove(arrowGroup);
    arrowGroup = buildArrows(currentResults, windData, scale);
    worldGroup.add(arrowGroup);
  }

  // Update UI
  const tRange = timeSeries.at(-1).t - timeSeries[0].t;
  animSlider.value = tRange > 0 ? (t - timeSeries[0].t) / tRange : 0;
  animTimeLabel.textContent = `t = ${t.toFixed(1)} min`;
  animSpeedLabel.textContent = `Wind: ${ws.toFixed(2)} m/s`;
}

function stopAnimation() {
  if (animTimer) { clearInterval(animTimer); animTimer = null; }
}

function startAnimation() {
  stopAnimation();
  const TICK_MS = 100; // update every 100 ms
  animTimer = setInterval(() => {
    animT += (TICK_MS / 1000) * animRate;
    if (timeSeries.length > 0 && animT > timeSeries.at(-1).t) {
      animT = timeSeries[0].t; // loop
    }
    applyAnimFrame(animT);
  }, TICK_MS);
}

animPlayBtn.addEventListener('click', () => {
  if (timeSeries.length < 2) return;
  startAnimation();
});
animStopBtn.addEventListener('click', stopAnimation);
animResetBtn.addEventListener('click', () => {
  stopAnimation();
  animT = timeSeries.length > 0 ? timeSeries[0].t : 0;
  applyAnimFrame(animT);
});
animSlider.addEventListener('input', () => {
  stopAnimation();
  if (timeSeries.length < 2) return;
  const tRange = timeSeries.at(-1).t - timeSeries[0].t;
  animT = timeSeries[0].t + parseFloat(animSlider.value) * tRange;
  applyAnimFrame(animT);
});
animRateSlider.addEventListener('input', () => {
  animRate = parseFloat(animRateSlider.value);
  animRateLabel.textContent = animRate >= 60
    ? `${(animRate / 60).toFixed(0)}h/s`
    : `${animRate.toFixed(1)} min/s`;
});

// ---------------------------------------------------------------------------
// Time series UI (upload + manual table)
// ---------------------------------------------------------------------------
function applyTimeSeries(data) {
  timeSeries = data.slice().sort((a, b) => a.t - b.t);

  const tsMinWS = Math.min(...timeSeries.map(p => p.v));
  const tsMaxWS = Math.max(...timeSeries.map(p => p.v));
  const ref = currentResults?.ref_velocity || 1;
  const fieldData = windFields[activeField] || {};

  // Derive color scale from the full range of wind speeds in the time series
  animVmin = (fieldData.min ?? 0) * (tsMinWS / ref);
  animVmax = (fieldData.max ?? 1) * (tsMaxWS / ref);
  if (animVmax <= animVmin) animVmax = animVmin + 0.01;

  // Update colorbar to reflect the animation's effective range
  buildColorbar(activeField, animVmin, animVmax);

  tsInfo.textContent = `${timeSeries.length} entries · `
    + `${timeSeries[0].t.toFixed(1)}–${timeSeries.at(-1).t.toFixed(1)} min · `
    + `${tsMinWS.toFixed(2)}–${tsMaxWS.toFixed(2)} m/s`;
  animSection.style.display = 'block';
  animT = timeSeries[0].t;
  applyAnimFrame(animT);
}

function parseCSVTimeSeries(text) {
  const rows = [];
  for (const line of text.split('\n')) {
    const parts = line.trim().split(',');
    if (parts.length < 2) continue;
    const t = parseFloat(parts[0]), v = parseFloat(parts[1]);
    if (isNaN(t) || isNaN(v)) continue;
    rows.push({ t, v });
  }
  return rows;
}

tsUploadBtn.addEventListener('click', () => tsFileInput.click());
tsFileInput.addEventListener('change', async () => {
  const file = tsFileInput.files[0];
  if (!file) return;
  const text = await file.text();
  const rows = parseCSVTimeSeries(text);
  if (!rows.length) { tsInfo.textContent = 'No valid rows found.'; return; }

  // Persist to backend
  const fd = new FormData();
  fd.append('file', file);
  await fetch(`/api/jobs/${jobId}/timeseries`, { method: 'POST', body: fd });

  // Populate manual table too (for review)
  rebuildTable(rows);
  tsTableWrap.style.display = 'block';
  applyTimeSeries(rows);
});

tsManualBtn.addEventListener('click', () => {
  tsTableWrap.style.display = tsTableWrap.style.display === 'none' ? 'block' : 'none';
  if (!tsTbody.children.length) addTableRow(0, 1.0);
});

function addTableRow(t = '', v = '') {
  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td><input type="number" class="ts-t" value="${t}" min="0" step="1"/></td>
    <td><input type="number" class="ts-v" value="${v}" min="0" step="0.01"/></td>
    <td><button class="ts-del-btn" title="Remove">&times;</button></td>`;
  tr.querySelector('.ts-del-btn').addEventListener('click', () => tr.remove());
  tsTbody.appendChild(tr);
}

function rebuildTable(rows) {
  tsTbody.innerHTML = '';
  rows.forEach(r => addTableRow(r.t, r.v));
}

tsAddRow.addEventListener('click', () => addTableRow());

tsSaveBtn.addEventListener('click', async () => {
  const rows = [];
  for (const tr of tsTbody.querySelectorAll('tr')) {
    const t = parseFloat(tr.querySelector('.ts-t')?.value);
    const v = parseFloat(tr.querySelector('.ts-v')?.value);
    if (!isNaN(t) && !isNaN(v)) rows.push({ t, v });
  }
  if (rows.length < 2) { tsInfo.textContent = 'Need at least 2 entries.'; return; }

  await fetch(`/api/jobs/${jobId}/timeseries`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(rows),
  });
  applyTimeSeries(rows);
});

// ---------------------------------------------------------------------------
// Run button
// ---------------------------------------------------------------------------
runBtn.addEventListener('click', async () => {
  if (!jobId) return;
  stopPolling();
  stopAnimation();
  animVmin = 0; animVmax = 1; // reset until time series loaded
  clearScene();
  legendEl.style.display = 'none';
  fieldSection.style.display = 'none';
  overlaySection.style.display = 'none';
  downloadSection.style.display = 'none';
  tsSection.style.display = 'none';
  animSection.style.display = 'none';
  timeSeries = [];
  tsTableWrap.style.display = 'none';
  tsTbody.innerHTML = '';
  tsInfo.textContent = '';
  arrowsToggle.checked = false;
  streamToggle.checked = false;

  runBtn.disabled = true;
  logOutput.textContent = '';
  logSection.style.display = 'block';
  setProgress('running', 'Starting pipeline…');
  progressBar.style.width = '5%';

  try {
    const params = new URLSearchParams({
      wind_angle: windAngle,
      ref_height: refHeight,
      ref_velocity: refVelocity,
    });
    if (pxResolution !== null) params.append('px_resolution', pxResolution);
    const res = await fetch(
      `/api/jobs/${jobId}/process?${params}`,
      { method: 'POST' },
    );
    if (!res.ok) throw new Error(await res.text());
    startPolling();
  } catch (err) {
    setProgress('error', 'Failed to start: ' + err.message);
    runBtn.disabled = false;
  }
});

// ---------------------------------------------------------------------------
// Status polling
// ---------------------------------------------------------------------------
function startPolling() {
  pollTimer = setInterval(async () => {
    try {
      const [statusRes, logRes] = await Promise.all([
        fetch(`/api/jobs/${jobId}/status`),
        fetch(`/api/jobs/${jobId}/log`),
      ]);
      const { status, progress_msg } = await statusRes.json();
      const { log } = await logRes.json();

      setProgress(status, progress_msg);

      if (log) {
        const atBottom = logOutput.scrollHeight - logOutput.scrollTop <= logOutput.clientHeight + 20;
        logOutput.textContent = log;
        if (atBottom) logOutput.scrollTop = logOutput.scrollHeight;
      }

      if (status === 'done') {
        stopPolling();
        await onPipelineDone();
        runBtn.disabled = false;
      } else if (status === 'error') {
        stopPolling();
        runBtn.disabled = false;
      }
    } catch (_) { /* network hiccup – keep polling */ }
  }, 2000);
}

function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

// ---------------------------------------------------------------------------
// Load results into Three.js scene
// ---------------------------------------------------------------------------
async function onPipelineDone() {
  const [resRes] = await Promise.all([
    fetch(`/api/jobs/${jobId}/results`),
  ]);
  const results = await resRes.json();

  clearScene();
  currentResults = results;

  // Load STL mesh
  try {
    stlMesh = await loadSTL(`/api/jobs/${jobId}/stl`);
    worldGroup.add(stlMesh);
  } catch (e) {
    console.warn('STL load failed:', e);
  }

  // Store wind field stats and reset to UMAG
  windFields   = results.wind_fields || {};
  activeField  = 'UMAG';
  fieldSelect.value = 'UMAG';
  fieldSection.style.display = Object.keys(windFields).length > 1 ? 'block' : 'none';

  // Load heatmap plane (default: UMAG)
  const heatmapUrl = `/api/jobs/${jobId}/heatmap?field=${activeField}`;
  try {
    heatmapMesh = await placeHeatmap(results, heatmapUrl);
    worldGroup.add(heatmapMesh);
  } catch (e) {
    console.warn('Heatmap load failed:', e);
  }

  // Update colorbar
  const initField = windFields[activeField] || {};
  buildColorbar(activeField, initField.min ?? 0, initField.max ?? 1);
  legendEl.style.display = 'flex';

  // Show effective resolution, grid size, and reference parameters
  if (results.px_resolution != null) {
    resLabel.textContent = results.px_resolution.toFixed(2) + ' m/px';
  }
  if (results.n_points != null) {
    const parts = [`${results.n_points}x${results.n_points} grid`,
                   `${results.step_size * 2} m domain`];
    if (results.ref_height != null)
      parts.push(`z = ${results.ref_height} m`);
    if (results.ref_velocity != null && results.ref_velocity !== 1)
      parts.push(`Uref = ${results.ref_velocity} m/s`);
    resInfo.textContent = parts.join(' · ');
    resInfo.style.display = 'block';
  }

  // Show overlay toggles, download buttons, and time series section
  overlaySection.style.display = 'block';
  downloadSection.style.display = 'block';
  tsSection.style.display = 'block';

  // Frame camera on geometry
  if (stlMesh) {
    const box = new THREE.Box3().setFromObject(worldGroup);
    const ctr = new THREE.Vector3();
    box.getCenter(ctr);
    const sz = box.getSize(new THREE.Vector3()).length();
    controls.target.copy(ctr);
    camera.position.set(ctr.x, ctr.y + sz * 1.2, ctr.z + sz * 0.5);
    camera.lookAt(ctr);
    controls.update();
  }
}
