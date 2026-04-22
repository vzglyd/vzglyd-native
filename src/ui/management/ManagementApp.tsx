import type { ReactElement } from 'react';

const markup = String.raw`
<div id="layout">
  <!-- Header -->
  <header id="header">
    <div>
      <p class="eyebrow">VRX-64 Management</p>
      <h1>Display Control</h1>
    </div>
    <div class="btn-row">
      <span id="status-badge">connecting…</span>
      <button class="secondary-btn" id="save-btn" disabled>Save playlist</button>
    </div>
  </header>

  <!-- Left panel: editor + secrets + library -->
  <div id="left-panel">

    <!-- Defaults -->
    <div class="card" id="defaults-card">
      <div class="card-head">
        <div>
          <h2>Defaults</h2>
          <p class="hint">Applied when a slide does not specify its own value.</p>
        </div>
      </div>
      <div class="form-grid">
        <label class="field">
          <span>Default duration (s)</span>
          <input id="default-duration" type="number" min="1" max="300" placeholder="7" />
        </label>
        <label class="field">
          <span>Display scale</span>
          <input id="display-scale" type="number" min="0" max="2" step="0.05" placeholder="1.0" />
        </label>
        <label class="field">
          <span>Transition in</span>
          <select id="default-transition-in"></select>
        </label>
        <label class="field">
          <span>Transition out</span>
          <select id="default-transition-out"></select>
        </label>
      </div>
    </div>

    <!-- Slide list -->
    <div class="card">
      <div class="card-head">
        <div><h2>Slides</h2><p class="hint">Drag to reorder. Click a slide to preview it.</p></div>
        <button class="secondary-btn" id="add-entry-btn">+ Add</button>
      </div>
      <div id="entry-list"></div>
    </div>

    <!-- Slide library (upload + available bundles) -->
    <div class="card">
      <div class="card-head">
        <div><h2>Slide Library</h2><p class="hint">Available .vzglyd bundles. Click to add to playlist.</p></div>
        <button class="secondary-btn" id="refresh-library-btn">Refresh</button>
      </div>
      <label class="drop-zone" id="drop-zone">
        <input type="file" id="upload-input" accept=".vzglyd" multiple />
        Drop .vzglyd files here, or click to upload
      </label>
      <div id="library-list" style="margin-top:0.6rem"></div>
    </div>

    <!-- Secrets -->
    <div class="card">
      <div class="card-head">
        <div><h2>Secrets</h2><p class="hint">API keys for slides. Never exposed in logs.</p></div>
        <button class="secondary-btn" id="export-secrets-btn" title="Download secrets.json for the native runtime">Export secrets.json</button>
      </div>
      <div class="secrets-warning">
        ⚠ Do not commit <code>secrets.json</code> to a public repository.
      </div>
      <div id="secrets-list" style="margin-top:0.75rem"></div>
      <div class="secret-row" style="margin-top:0.6rem">
        <label class="field">
          <span>Key</span>
          <input id="new-secret-key" type="text" placeholder="LASTFM_API_KEY" autocomplete="off" />
        </label>
        <label class="field">
          <span>Value</span>
          <input id="new-secret-value" type="password" placeholder="••••••••" autocomplete="new-password" />
        </label>
        <button class="secondary-btn" id="add-secret-btn" style="margin-bottom:0">Set</button>
      </div>
    </div>

  </div>

  <!-- Right panel: preview placeholder -->
  <div id="right-panel">
    <p id="preview-label">Select a slide to inspect</p>
    <div id="preview-placeholder" style="
      width: 640px; max-width: 100%; aspect-ratio: 640/480;
      display: flex; align-items: center; justify-content: center;
      border: 1px solid var(--border); border-radius: 8px;
      background: var(--surface); color: var(--text-dim);
      font-size: 0.9rem; text-align: center; padding: 1rem;
    ">Native-first runtime<br/>watches JSON result files</div>
    <p id="preview-status"></p>
  </div>
</div>

<div id="toast"></div>
`;

export function ManagementApp(): ReactElement {
  return <div dangerouslySetInnerHTML={{ __html: markup }} />;
}
