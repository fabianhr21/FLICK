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

function clearScene() {
  if (stlMesh)    { worldGroup.remove(stlMesh);    stlMesh = null; }
  if (heatmapMesh){ worldGroup.remove(heatmapMesh); heatmapMesh = null; }
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
    const { step_size, stl_bbox: bbox, wind_angle } = results;
    const size = 2 * step_size;

    const geometry = new THREE.PlaneGeometry(size, size);
    new THREE.TextureLoader().load(heatmapUrl + '?t=' + Date.now(), (tex) => {
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
      mesh.position.set(bbox.centre_x, bbox.centre_y, 3);
      // Un-rotate CFD grid back into original building frame
      mesh.rotation.z = -(wind_angle * Math.PI) / 180;
      resolve(mesh);
    });
  });
}

// ---------------------------------------------------------------------------
// UI state
// ---------------------------------------------------------------------------
let jobId = null;
let pollTimer = null;
let windAngle = 0;

const dropZone   = document.getElementById('drop-zone');
const fileInput  = document.getElementById('file-input');
const fileNameEl = document.getElementById('file-name');
const windSlider = document.getElementById('wind-slider');
const angleLbl   = document.getElementById('angle-label');
const runBtn     = document.getElementById('run-btn');
const statusMsg  = document.getElementById('status-msg');
const progressBar= document.getElementById('progress-bar');
const legendEl   = document.getElementById('legend');
const legendMin  = document.getElementById('legend-min');
const legendMax  = document.getElementById('legend-max');
const logSection = document.getElementById('log-section');
const logOutput  = document.getElementById('log-output');

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
// Run button
// ---------------------------------------------------------------------------
runBtn.addEventListener('click', async () => {
  if (!jobId) return;
  stopPolling();
  clearScene();
  legendEl.style.display = 'none';

  runBtn.disabled = true;
  logOutput.textContent = '';
  logSection.style.display = 'block';
  setProgress('running', 'Starting pipeline…');
  progressBar.style.width = '5%';

  try {
    const res = await fetch(
      `/api/jobs/${jobId}/process?wind_angle=${windAngle}`,
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

  // Load STL mesh
  try {
    stlMesh = await loadSTL(`/api/jobs/${jobId}/stl`);
    worldGroup.add(stlMesh);
  } catch (e) {
    console.warn('STL load failed:', e);
  }

  // Load heatmap plane
  try {
    heatmapMesh = await placeHeatmap(results, `/api/jobs/${jobId}/heatmap`);
    worldGroup.add(heatmapMesh);
  } catch (e) {
    console.warn('Heatmap load failed:', e);
  }

  // Update legend
  const { min, max } = results.wind_speed;
  legendMin.textContent = min.toFixed(2);
  legendMax.textContent = max.toFixed(2);
  legendEl.style.display = 'flex';

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
