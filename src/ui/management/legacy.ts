// @ts-nocheck

export function startManagementLegacy() {
// ═══════════════════════════════════════════════════════════════════════════════
// Inlined playlist_repo.js (adapted — no ES module imports, no repo URL)
// ═══════════════════════════════════════════════════════════════════════════════

const MIN_DISPLAY_DURATION_SECONDS = 1;
const MAX_DISPLAY_DURATION_SECONDS = 300;
const TRANSITION_OPTIONS = ['crossfade', 'wipe_left', 'wipe_down', 'dissolve', 'cut'];

function isPlainObject(v) { return v != null && typeof v === 'object' && !Array.isArray(v); }

function normalizeOptionalDuration(value, label) {
  if (value == null) return undefined;
  const s = Number(value);
  if (!Number.isInteger(s) || s < MIN_DISPLAY_DURATION_SECONDS || s > MAX_DISPLAY_DURATION_SECONDS)
    throw new Error(`${label} must be an integer from ${MIN_DISPLAY_DURATION_SECONDS} to ${MAX_DISPLAY_DURATION_SECONDS}`);
  return s;
}

function normalizeOptionalTransition(value, label) {
  if (value == null) return undefined;
  const t = String(value).trim();
  if (!t) throw new Error(`${label} must be a non-empty string`);
  if (!TRANSITION_OPTIONS.includes(t)) throw new Error(`${label} must be one of: ${TRANSITION_OPTIONS.join(', ')}`);
  return t;
}

function validateBundlePath(path, label = 'path') {
  if (typeof path !== 'string' || !path.trim()) throw new Error(`${label} must be a non-empty string`);
  const t = path.trim();
  if (t.startsWith('/')) throw new Error(`${label} must be relative`);
  if (t.includes('\\')) throw new Error(`${label} must use forward slashes`);
  if (t.split('/').some(s => s === '.' || s === '..')) throw new Error(`${label} must not contain . or .. segments`);
  if (!t.endsWith('.vzglyd')) throw new Error(`${label} must point to a .vzglyd bundle`);
  return t;
}

function normalizeDefaults(d) {
  if (!d) return {};
  const out = {};
  const dur = normalizeOptionalDuration(d.duration_seconds, 'defaults.duration_seconds');
  const ti = normalizeOptionalTransition(d.transition_in, 'defaults.transition_in');
  const to = normalizeOptionalTransition(d.transition_out, 'defaults.transition_out');
  if (dur != null) out.duration_seconds = dur;
  if (ti != null) out.transition_in = ti;
  if (to != null) out.transition_out = to;
  return out;
}

function normalizeEntry(e, i) {
  if (!isPlainObject(e)) throw new Error(`slides[${i}] must be an object`);
  const out = { path: validateBundlePath(e.path, `slides[${i}].path`) };
  if (e.enabled === false) out.enabled = false;
  const dur = normalizeOptionalDuration(e.duration_seconds, `slides[${i}].duration_seconds`);
  if (dur != null) out.duration_seconds = dur;
  const ti = normalizeOptionalTransition(e.transition_in, `slides[${i}].transition_in`);
  if (ti != null) out.transition_in = ti;
  const to = normalizeOptionalTransition(e.transition_out, `slides[${i}].transition_out`);
  if (to != null) out.transition_out = to;
  if (e.params !== undefined) out.params = e.params;
  if (e.sidecar_params !== undefined) out.sidecar_params = e.sidecar_params;
  return out;
}

function validatePlaylist(p) {
  if (!isPlainObject(p)) throw new Error('playlist.json must be an object');
  if (!Array.isArray(p.slides)) throw new Error('playlist.json must have a slides array');
  return { ...p, defaults: normalizeDefaults(p.defaults), slides: p.slides.map(normalizeEntry) };
}

function parseParamsText(text) {
  if (!text || !text.trim()) return undefined;
  return JSON.parse(text);
}

function toEditablePlaylist(playlist) {
  const v = validatePlaylist(playlist);
  return {
    defaults: {
      duration_seconds: v.defaults.duration_seconds != null ? String(v.defaults.duration_seconds) : '',
      transition_in: v.defaults.transition_in ?? '',
      transition_out: v.defaults.transition_out ?? '',
    },
    display_scale: String(v.display_scale ?? '1.0'),
    slides: v.slides.map(e => ({
      path: e.path,
      enabled: e.enabled !== false,
      duration_seconds: e.duration_seconds != null ? String(e.duration_seconds) : '',
      transition_in: e.transition_in ?? '',
      transition_out: e.transition_out ?? '',
      params_text: e.params !== undefined ? JSON.stringify(e.params, null, 2) : '',
      params_editor_mode: 'raw',
      params_form_values: {},
      params_schema: null,
      params_editor_message: '',
      sidecar_params_text: e.sidecar_params !== undefined ? JSON.stringify(e.sidecar_params, null, 2) : '',
      bundle_manifest: null,
      bundle_manifest_status: 'idle',
      bundle_error: '',
    })),
  };
}

function emptyEditableEntry() {
  return {
    path: '', enabled: true, duration_seconds: '', transition_in: '', transition_out: '',
    params_text: '', params_editor_mode: 'raw', params_form_values: {}, params_schema: null,
    params_editor_message: '', sidecar_params_text: '',
    bundle_manifest: null, bundle_manifest_status: 'idle', bundle_error: '',
  };
}

function serializeEditablePlaylist(ep) {
  const defaults = {};
  const dur = normalizeOptionalDuration(ep.defaults.duration_seconds || null, 'defaults.duration_seconds');
  if (dur != null) defaults.duration_seconds = dur;
  const ti = normalizeOptionalTransition(ep.defaults.transition_in || null, 'defaults.transition_in');
  if (ti != null) defaults.transition_in = ti;
  const to = normalizeOptionalTransition(ep.defaults.transition_out || null, 'defaults.transition_out');
  if (to != null) defaults.transition_out = to;

  const slides = ep.slides.map((e, i) => {
    const item = { path: validateBundlePath(e.path, `slides[${i}].path`) };
    if (e.enabled === false) item.enabled = false;
    const d = normalizeOptionalDuration(e.duration_seconds || null, `slides[${i}].duration_seconds`);
    if (d != null) item.duration_seconds = d;
    const eti = normalizeOptionalTransition(e.transition_in || null, `slides[${i}].transition_in`);
    if (eti != null) item.transition_in = eti;
    const eto = normalizeOptionalTransition(e.transition_out || null, `slides[${i}].transition_out`);
    if (eto != null) item.transition_out = eto;
    const params = parseParamsText(e.params_text);
    if (params !== undefined) item.params = params;
    const sidecarParams = parseParamsText(e.sidecar_params_text);
    if (sidecarParams !== undefined) item.sidecar_params = sidecarParams;
    return item;
  });

  const display_scale = parseFloat(ep.display_scale || '1.0');
  return validatePlaylist({ defaults, display_scale: isFinite(display_scale) ? display_scale : 1.0, slides });
}

// ═══════════════════════════════════════════════════════════════════════════════
// param_schema.js (inlined, essential parts)
// ═══════════════════════════════════════════════════════════════════════════════

function describeParamField(field) {
  const defaultText = field.default === undefined ? '' :
    (typeof field.default === 'string' ? field.default : JSON.stringify(field.default));
  return { label: field.label ?? field.key, help: field.help ?? '', defaultText };
}

// ═══════════════════════════════════════════════════════════════════════════════
// API helpers
// ═══════════════════════════════════════════════════════════════════════════════

async function apiGet(path) {
  const r = await fetch(path);
  if (!r.ok) {
    const j = await r.json().catch(() => ({}));
    throw new Error(j.error || `${r.status} ${r.statusText}`);
  }
  return r.json();
}

async function apiPost(path, body) {
  const r = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const j = await r.json().catch(() => ({}));
    throw new Error(j.error || `${r.status} ${r.statusText}`);
  }
  return r.json();
}

async function apiPostRaw(path, body) {
  const r = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body,
  });
  if (!r.ok) {
    const j = await r.json().catch(() => ({}));
    throw new Error(j.error || `${r.status} ${r.statusText}`);
  }
  return r.json();
}

// ═══════════════════════════════════════════════════════════════════════════════
// App state
// ═══════════════════════════════════════════════════════════════════════════════

const state = {
  editablePlaylist: null,
  savedJson: '',
  secretKeys: [],
  library: [],
  selectedIndex: null,
};

// ═══════════════════════════════════════════════════════════════════════════════
// DOM refs
// ═══════════════════════════════════════════════════════════════════════════════

const statusBadge = document.getElementById('status-badge');
const saveBtn = document.getElementById('save-btn');
const defaultDuration = document.getElementById('default-duration');
const displayScale = document.getElementById('display-scale');
const defaultTransitionIn = document.getElementById('default-transition-in');
const defaultTransitionOut = document.getElementById('default-transition-out');
const entryList = document.getElementById('entry-list');
const addEntryBtn = document.getElementById('add-entry-btn');
const libraryList = document.getElementById('library-list');
const refreshLibraryBtn = document.getElementById('refresh-library-btn');
const uploadInput = document.getElementById('upload-input');
const dropZone = document.getElementById('drop-zone');
const secretsList = document.getElementById('secrets-list');
const newSecretKey = document.getElementById('new-secret-key');
const newSecretValue = document.getElementById('new-secret-value');
const addSecretBtn = document.getElementById('add-secret-btn');
const exportSecretsBtn = document.getElementById('export-secrets-btn');
const previewLabel = document.getElementById('preview-label');
const previewStatus = document.getElementById('preview-status');
const toast = document.getElementById('toast');

// ═══════════════════════════════════════════════════════════════════════════════
// Toast
// ═══════════════════════════════════════════════════════════════════════════════

let toastTimer = null;
function showToast(msg, kind = '') {
  toast.textContent = msg;
  toast.className = kind ? `is-${kind}` : '';
  toast.style.display = 'block';
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.style.display = 'none'; }, 3500);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Transition selects
// ═══════════════════════════════════════════════════════════════════════════════

function fillTransitionSelect(sel, selected = '') {
  sel.replaceChildren();
  for (const val of ['', ...TRANSITION_OPTIONS]) {
    const opt = document.createElement('option');
    opt.value = val;
    opt.textContent = val || 'Inherit / none';
    if (val === selected) opt.selected = true;
    sel.append(opt);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Status polling
// ═══════════════════════════════════════════════════════════════════════════════

async function pollStatus() {
  try {
    const s = await apiGet('/api/status');
    const slide = s.current_slide ?? '—';
    const fps = s.fps > 0 ? `${s.fps.toFixed(1)} fps` : '';
    statusBadge.textContent = fps ? `${slide} • ${fps}` : slide;
  } catch {
    statusBadge.textContent = 'display offline';
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Playlist load / render
// ═══════════════════════════════════════════════════════════════════════════════

async function loadPlaylist() {
  try {
    const raw = await apiGet('/api/playlist');
    state.editablePlaylist = toEditablePlaylist(raw);
    state.savedJson = JSON.stringify(serializeEditablePlaylist(state.editablePlaylist));
    renderEditor();
    await hydrateAllManifests();
  } catch (e) {
    showToast(`Failed to load playlist: ${e.message}`, 'error');
  }
}

function renderEditor() {
  if (!state.editablePlaylist) return;
  const ep = state.editablePlaylist;
  defaultDuration.value = ep.defaults.duration_seconds;
  displayScale.value = ep.display_scale;
  fillTransitionSelect(defaultTransitionIn, ep.defaults.transition_in);
  fillTransitionSelect(defaultTransitionOut, ep.defaults.transition_out);
  renderEntries();
  updateSaveButton();
}

function updateSaveButton() {
  if (!state.editablePlaylist) { saveBtn.disabled = true; return; }
  try {
    const current = JSON.stringify(serializeEditablePlaylist(state.editablePlaylist));
    saveBtn.disabled = false;
    saveBtn.textContent = current !== state.savedJson ? 'Save playlist *' : 'Save playlist';
  } catch {
    saveBtn.disabled = false;
    saveBtn.textContent = 'Save playlist';
  }
}

function escHtml(v) {
  return String(v ?? '')
    .replace(/&/g, '&amp;').replace(/"/g, '&quot;')
    .replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function createNote(text, tone = '') {
  const p = document.createElement('p');
  p.className = `field-note${tone ? ` is-${tone}` : ''}`;
  p.textContent = text;
  return p;
}

function cassetteArtUrl(bundlePath, kind) {
  return `/api/slides/${encodeURIComponent(bundlePath)}/art/${kind}`;
}

function hasCassetteArt(manifest) {
  return Boolean(
    manifest?.assets?.art?.j_card?.path &&
    manifest?.assets?.art?.side_a_label?.path &&
    manifest?.assets?.art?.side_b_label?.path,
  );
}

function createCassetteImage(bundlePath, kind, label) {
  const img = document.createElement('img');
  img.src = cassetteArtUrl(bundlePath, kind);
  img.alt = label;
  img.loading = 'lazy';
  img.addEventListener('error', () => {
    img.replaceWith(createNote(`${label} unavailable`, 'error'));
  }, { once: true });
  return img;
}

function createCassetteCaption(text, className) {
  const caption = document.createElement('span');
  caption.className = className;
  caption.textContent = text;
  return caption;
}

function buildCassetteArt(bundlePath, manifest, { compact = false } = {}) {
  if (!bundlePath || !hasCassetteArt(manifest)) return null;

  const shell = document.createElement('div');
  shell.className = `cassette-art${compact ? ' is-compact' : ''}`;

  const cover = document.createElement('div');
  cover.className = 'cassette-j-card';
  cover.append(
    createCassetteImage(bundlePath, 'j-card', `${manifest.name ?? bundlePath} J-card`),
    createCassetteCaption('J-card', 'cassette-caption'),
  );

  const tape = document.createElement('div');
  tape.className = 'cassette-tape';
  const sideA = document.createElement('div');
  sideA.className = 'cassette-label';
  sideA.append(
    createCassetteImage(bundlePath, 'side-a', `${manifest.name ?? bundlePath} side A label`),
    createCassetteCaption('Side A', 'cassette-label-title'),
  );
  const sideB = document.createElement('div');
  sideB.className = 'cassette-label';
  sideB.append(
    createCassetteImage(bundlePath, 'side-b', `${manifest.name ?? bundlePath} side B label`),
    createCassetteCaption('Side B', 'cassette-label-title'),
  );
  tape.append(sideA, sideB);

  shell.append(cover, tape);
  return shell;
}

function describeInherited(entry, key) {
  const own = entry[key];
  const def = state.editablePlaylist?.defaults?.[key] ?? '';
  const bun = entry.bundle_manifest?.display?.[key] ?? null;
  if (own) return `Override. Playlist default: ${def || 'none'}.`;
  if (def) return `Inherited from defaults: ${def}.`;
  if (bun) return `Bundle default: ${bun}.`;
  return 'No default set.';
}

function buildManifestBadges(entry) {
  const m = entry.bundle_manifest;
  const badges = [];
  if (entry.bundle_manifest_status === 'loading') badges.push('loading…');
  if (entry.bundle_manifest_status === 'error') badges.push('metadata unavailable');
  if (!m) return badges;
  if (m.author) badges.push(`by ${m.author}`);
  if (m.scene_space) badges.push(m.scene_space);
  if (m.display?.duration_seconds != null) badges.push(`${m.display.duration_seconds}s`);
  if (hasCassetteArt(m)) badges.push('cassette art');
  if (m.params?.fields?.length) badges.push(`${m.params.fields.length} params`);
  return badges;
}

function renderEntries() {
  entryList.replaceChildren();
  if (!state.editablePlaylist) return;

  for (const [idx, entry] of state.editablePlaylist.slides.entries()) {
    const card = document.createElement('section');
    card.className = `editor-entry${state.selectedIndex === idx ? ' is-selected' : ''}`;
    card.dataset.index = String(idx);

    // Top line
    const topLine = document.createElement('div');
    topLine.className = 'entry-topline';
    const indexLabel = document.createElement('span');
    indexLabel.className = 'entry-index';
    indexLabel.textContent = `Slide ${idx + 1}`;
    const actions = document.createElement('div');
    actions.className = 'entry-actions';

    const previewBtn = document.createElement('button');
    previewBtn.className = 'secondary-btn';
    previewBtn.dataset.action = 'preview';
    previewBtn.textContent = 'Preview';

    const upBtn = document.createElement('button');
    upBtn.className = 'icon-btn'; upBtn.dataset.action = 'move-up';
    upBtn.textContent = '↑'; upBtn.disabled = idx === 0;

    const downBtn = document.createElement('button');
    downBtn.className = 'icon-btn'; downBtn.dataset.action = 'move-down';
    downBtn.textContent = '↓';
    downBtn.disabled = idx === state.editablePlaylist.slides.length - 1;

    const removeBtn = document.createElement('button');
    removeBtn.className = 'icon-btn danger-btn'; removeBtn.dataset.action = 'remove';
    removeBtn.textContent = '✕';

    actions.append(previewBtn, upBtn, downBtn, removeBtn);
    topLine.append(indexLabel, actions);

    // Summary
    const summary = document.createElement('div');
    summary.className = 'entry-summary';
    const title = document.createElement('div');
    title.className = 'entry-summary-title';
    title.textContent = (entry.bundle_manifest?.name ?? entry.path) || 'New slide';
    summary.append(title);
    const badges = buildManifestBadges(entry);
    if (badges.length) {
      const row = document.createElement('div');
      row.className = 'badge-row';
      for (const b of badges) {
        const pill = document.createElement('span');
        pill.className = 'badge-pill'; pill.textContent = b;
        row.append(pill);
      }
      summary.append(row);
    }
    if (entry.bundle_manifest_status === 'error')
      summary.append(createNote(`Metadata unavailable: ${entry.bundle_error}`, 'error'));
    const artPanel = buildCassetteArt(entry.path, entry.bundle_manifest);
    if (artPanel) summary.append(artPanel);

    // Fields grid
    const grid = document.createElement('div');
    grid.className = 'form-grid';

    const pathField = document.createElement('label');
    pathField.className = 'field is-wide';
    pathField.innerHTML = `<span>Bundle path</span>
      <input data-field="path" value="${escHtml(entry.path)}" placeholder="clock.vzglyd" />`;
    pathField.append(createNote('Path relative to the slides directory.'));

    const enabledField = document.createElement('label');
    enabledField.className = 'field';
    enabledField.innerHTML = `<span>Enabled</span>
      <select data-field="enabled">
        <option value="true"${entry.enabled !== false ? ' selected' : ''}>true</option>
        <option value="false"${entry.enabled === false ? ' selected' : ''}>false</option>
      </select>`;

    const durField = document.createElement('label');
    durField.className = 'field';
    durField.innerHTML = `<span>Duration (s)</span>
      <input data-field="duration_seconds" type="number" min="1" max="300"
        value="${escHtml(entry.duration_seconds)}" placeholder="inherit" />`;
    durField.append(createNote(describeInherited(entry, 'duration_seconds')));

    const tiField = document.createElement('label');
    tiField.className = 'field';
    tiField.innerHTML = '<span>Transition in</span>';
    const tiSel = document.createElement('select');
    tiSel.dataset.field = 'transition_in';
    fillTransitionSelect(tiSel, entry.transition_in);
    tiField.append(tiSel, createNote(describeInherited(entry, 'transition_in')));

    const toField = document.createElement('label');
    toField.className = 'field';
    toField.innerHTML = '<span>Transition out</span>';
    const toSel = document.createElement('select');
    toSel.dataset.field = 'transition_out';
    fillTransitionSelect(toSel, entry.transition_out);
    toField.append(toSel, createNote(describeInherited(entry, 'transition_out')));

    grid.append(pathField, enabledField, durField, tiField, toField);

    // Params
    const paramsShell = document.createElement('div');
    paramsShell.className = 'entry-params-shell';
    const paramsLabel = document.createElement('label');
    paramsLabel.className = 'field is-wide';
    paramsLabel.innerHTML = '<span>Params JSON</span>';
    const paramsArea = document.createElement('textarea');
    paramsArea.dataset.field = 'params_text';
    paramsArea.placeholder = '{\n  "key": "value"\n}';
    paramsArea.value = entry.params_text;
    paramsLabel.append(paramsArea);
    paramsShell.append(paramsLabel);

    const sidecarParamsLabel = document.createElement('label');
    sidecarParamsLabel.className = 'field is-wide';
    sidecarParamsLabel.innerHTML = '<span>Sidecar params JSON</span>';
    const sidecarParamsArea = document.createElement('textarea');
    sidecarParamsArea.dataset.field = 'sidecar_params_text';
    sidecarParamsArea.placeholder = '{\n  "api_key": "..." \n}';
    sidecarParamsArea.value = entry.sidecar_params_text ?? '';
    sidecarParamsLabel.append(sidecarParamsArea);
    paramsShell.append(sidecarParamsLabel);

    card.append(topLine, summary, grid, paramsShell);
    entryList.append(card);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Manifest hydration
// ═══════════════════════════════════════════════════════════════════════════════

async function hydrateManifest(entry) {
  const path = entry.path?.trim();
  if (!path) return;
  entry.bundle_manifest_status = 'loading';
  try {
    const m = await apiGet(`/api/slides/${encodeURIComponent(path)}/manifest`);
    entry.bundle_manifest = m;
    entry.bundle_manifest_status = 'ready';
    entry.bundle_error = '';
  } catch (e) {
    entry.bundle_manifest = null;
    entry.bundle_manifest_status = 'error';
    entry.bundle_error = e.message;
  }
}

async function hydrateAllManifests() {
  if (!state.editablePlaylist) return;
  const entries = state.editablePlaylist.slides.filter(e => e.path?.trim());
  await Promise.all(entries.map(e => hydrateManifest(e)));
  renderEditor();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Save playlist
// ═══════════════════════════════════════════════════════════════════════════════

async function savePlaylist() {
  try {
    const serialized = serializeEditablePlaylist(state.editablePlaylist);
    const json = JSON.stringify(serialized, null, 2) + '\n';
    await apiPostRaw('/api/playlist', json);
    state.savedJson = JSON.stringify(serialized);
    updateSaveButton();
    showToast('Playlist saved and applied to display.', 'success');
  } catch (e) {
    showToast(`Save failed: ${e.message}`, 'error');
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Slide library
// ═══════════════════════════════════════════════════════════════════════════════

async function loadLibrary() {
  try {
    state.library = await apiGet('/api/slides');
    renderLibrary();
  } catch (e) {
    showToast(`Failed to load slide library: ${e.message}`, 'error');
  }
}

function renderLibrary() {
  libraryList.replaceChildren();
  if (!state.library.length) {
    libraryList.append(createNote('No .vzglyd files found in slides directory.'));
    return;
  }
  for (const bundle of state.library) {
    const row = document.createElement('div');
    row.className = 'library-row';

    const info = document.createElement('div');
    info.className = 'library-info';
    const name = document.createElement('span');
    name.className = 'library-name';
    name.textContent = bundle.manifest?.name ?? bundle.path;
    const meta = document.createElement('span');
    meta.className = 'library-path';
    meta.textContent = bundle.path;
    info.append(name, meta);
    const artPanel = buildCassetteArt(bundle.path, bundle.manifest, { compact: true });
    if (artPanel) info.append(artPanel);
    else if (!bundle.manifest) info.append(createNote('Cassette artwork unavailable.', 'error'));

    const addBtn = document.createElement('button');
    addBtn.className = 'secondary-btn';
    addBtn.textContent = '+ Add';
    addBtn.style.flexShrink = '0';
    addBtn.addEventListener('click', () => {
      if (!state.editablePlaylist) return;
      const entry = emptyEditableEntry();
      entry.path = bundle.path;
      entry.bundle_manifest = bundle.manifest;
      entry.bundle_manifest_status = bundle.manifest ? 'ready' : 'idle';
      state.editablePlaylist.slides.push(entry);
      renderEditor();
      showToast(`Added ${bundle.path} to playlist.`);
    });

    row.append(info, addBtn);
    libraryList.append(row);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Upload
// ═══════════════════════════════════════════════════════════════════════════════

async function uploadFiles(files) {
  for (const file of files) {
    if (!file.name.endsWith('.vzglyd')) {
      showToast(`${file.name}: not a .vzglyd file`, 'error');
      continue;
    }
    const formData = new FormData();
    formData.append('file', file, file.name);
    try {
      const r = await fetch('/api/slides/upload', { method: 'POST', body: formData });
      if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        throw new Error(j.error || r.statusText);
      }
      showToast(`Uploaded ${file.name}`, 'success');
    } catch (e) {
      showToast(`Upload failed: ${e.message}`, 'error');
    }
  }
  await loadLibrary();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Secrets
// ═══════════════════════════════════════════════════════════════════════════════

async function loadSecrets() {
  try {
    const s = await apiGet('/api/secrets');
    state.secretKeys = s.keys ?? [];
    renderSecrets();
  } catch (e) {
    showToast(`Failed to load secrets: ${e.message}`, 'error');
  }
}

function renderSecrets() {
  secretsList.replaceChildren();
  if (!state.secretKeys.length) {
    secretsList.append(createNote('No secrets configured.'));
    return;
  }
  const row = document.createElement('div');
  for (const key of state.secretKeys) {
    const badge = document.createElement('span');
    badge.className = 'secret-key-badge';
    badge.textContent = key;
    row.append(badge);
  }
  secretsList.append(row);
}

async function addSecret() {
  const key = newSecretKey.value.trim();
  const val = newSecretValue.value;
  if (!key || !val) { showToast('Both key and value are required.', 'error'); return; }
  try {
    await apiPost('/api/secrets', { [key]: val });
    newSecretKey.value = '';
    newSecretValue.value = '';
    await loadSecrets();
    showToast(`Secret '${key}' saved.`, 'success');
  } catch (e) {
    showToast(`Failed to save secret: ${e.message}`, 'error');
  }
}

function exportSecrets() {
  const a = document.createElement('a');
  a.href = '/api/secrets/export';
  a.download = 'secrets.json';
  a.click();
}

// ═══════════════════════════════════════════════════════════════════════════════
// WebGPU Preview (requires vzglyd_web built separately — not embedded in native)
// ═══════════════════════════════════════════════════════════════════════════════

function previewSlide(index) {
  state.selectedIndex = index;
  renderEntries();

  const entry = state.editablePlaylist?.slides?.[index];
  if (!entry?.path) return;

  previewLabel.textContent = `Preview: ${entry.bundle_manifest?.name ?? entry.path}`;
  previewStatus.textContent = 'Preview requires the web build. Run wasm-pack in VRX-64-web and rebuild.';
}

// ═══════════════════════════════════════════════════════════════════════════════
// Event wiring
// ═══════════════════════════════════════════════════════════════════════════════

saveBtn.addEventListener('click', () => void savePlaylist());

addEntryBtn.addEventListener('click', () => {
  if (!state.editablePlaylist) return;
  state.editablePlaylist.slides.push(emptyEditableEntry());
  renderEditor();
});

defaultDuration.addEventListener('input', () => {
  if (!state.editablePlaylist) return;
  state.editablePlaylist.defaults.duration_seconds = defaultDuration.value;
  updateSaveButton();
});

displayScale.addEventListener('input', () => {
  if (!state.editablePlaylist) return;
  state.editablePlaylist.display_scale = displayScale.value;
  updateSaveButton();
});

defaultTransitionIn.addEventListener('change', () => {
  if (!state.editablePlaylist) return;
  state.editablePlaylist.defaults.transition_in = defaultTransitionIn.value;
  renderEditor();
});

defaultTransitionOut.addEventListener('change', () => {
  if (!state.editablePlaylist) return;
  state.editablePlaylist.defaults.transition_out = defaultTransitionOut.value;
  renderEditor();
});

entryList.addEventListener('input', e => {
  const card = e.target.closest('.editor-entry');
  if (!card) return;
  const idx = +card.dataset.index;
  const entry = state.editablePlaylist?.slides?.[idx];
  if (!entry) return;
  const field = e.target.dataset.field;
  if (field === 'enabled') entry.enabled = e.target.value !== 'false';
  else if (field) entry[field] = e.target.value;
  updateSaveButton();
});

entryList.addEventListener('change', e => {
  const card = e.target.closest('.editor-entry');
  if (!card) return;
  const idx = +card.dataset.index;
  const entry = state.editablePlaylist?.slides?.[idx];
  if (!entry) return;
  const field = e.target.dataset.field;
  if (field === 'enabled') entry.enabled = e.target.value !== 'false';
  else if (field) entry[field] = e.target.value;
  if (field === 'path') {
    void hydrateManifest(entry).then(() => renderEntries());
  }
  updateSaveButton();
});

entryList.addEventListener('click', e => {
  const btn = e.target.closest('button[data-action]');
  if (!btn) return;
  const card = btn.closest('.editor-entry');
  const idx = +card.dataset.index;
  const slides = state.editablePlaylist?.slides;
  if (!slides) return;

  switch (btn.dataset.action) {
    case 'preview': void previewSlide(idx); break;
    case 'move-up':
      if (idx > 0) { [slides[idx-1], slides[idx]] = [slides[idx], slides[idx-1]]; renderEditor(); }
      break;
    case 'move-down':
      if (idx < slides.length-1) { [slides[idx], slides[idx+1]] = [slides[idx+1], slides[idx]]; renderEditor(); }
      break;
    case 'remove':
      slides.splice(idx, 1);
      if (state.selectedIndex === idx) state.selectedIndex = null;
      renderEditor();
      break;
  }
});

refreshLibraryBtn.addEventListener('click', () => void loadLibrary());

dropZone.addEventListener('click', () => uploadInput.click());
uploadInput.addEventListener('change', () => {
  if (uploadInput.files?.length) void uploadFiles([...uploadInput.files]);
  uploadInput.value = '';
});
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('is-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('is-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('is-over');
  if (e.dataTransfer?.files?.length) void uploadFiles([...e.dataTransfer.files]);
});

addSecretBtn.addEventListener('click', () => void addSecret());
newSecretValue.addEventListener('keydown', e => { if (e.key === 'Enter') void addSecret(); });
exportSecretsBtn.addEventListener('click', exportSecrets);

// ═══════════════════════════════════════════════════════════════════════════════
// Boot
// ═══════════════════════════════════════════════════════════════════════════════

async function boot() {
  fillTransitionSelect(defaultTransitionIn);
  fillTransitionSelect(defaultTransitionOut);
  await Promise.all([loadPlaylist(), loadLibrary(), loadSecrets()]);
  setInterval(() => void pollStatus(), 2000);
  void pollStatus();
}

void boot();
}
