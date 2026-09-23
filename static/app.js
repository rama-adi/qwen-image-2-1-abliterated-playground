const $ = id => document.getElementById(id);
const state = { socket: null, reconnect: null, failures: 0, defaults: {}, settings: {}, characters: [], reference: '', sourceHistoryId: '', job: null, timer: null, history: [], selectedHistoryId: null, previewStep: 0 };
const modelLabels = { sd_cli: 'sd-cli executable', diffusion: 'Qwen Image 2.1 diffusion model', encoder: 'Heretic text encoder', vision: 'Heretic vision projector GGUF', vae: 'Qwen Image 2.1 VAE' };

function addCharacter(data = {}) {
  state.characters.push({ name: data.name || '', position: data.position || 'auto', description: data.description || '', avoid: data.avoid || '' });
  renderCharacters();
}
function renderCharacters() {
  const host = $('characters');
  host.innerHTML = '';
  state.characters.forEach((character, index) => {
    const card = document.createElement('div');
    card.className = 'character-card';
    card.innerHTML = `<div class="character-top"><strong>Character ${index + 1}</strong><div class="character-actions"><button class="mini-button up" title="Move up" type="button">↑</button><button class="mini-button down" title="Move down" type="button">↓</button><button class="mini-button remove" title="Remove" type="button">✕</button></div></div><div class="character-grid"><input class="name" placeholder="Name or role (optional)" aria-label="Character name"><select class="position" aria-label="Character position"><option value="auto">Auto position</option><option value="left">Left</option><option value="center">Center</option><option value="right">Right</option><option value="foreground">Foreground</option><option value="background">Background</option></select></div><textarea class="description" placeholder="Appearance, clothing, expression, and action..." aria-label="Character description"></textarea><input class="avoid" placeholder="Avoid for this character (optional)" aria-label="Avoid for this character">`;
    card.querySelector('.name').value = character.name;
    card.querySelector('.position').value = character.position;
    card.querySelector('.description').value = character.description;
    card.querySelector('.avoid').value = character.avoid;
    for (const key of ['name', 'position', 'description', 'avoid']) card.querySelector('.' + key).addEventListener('input', e => { character[key] = e.target.value; previewPrompt(); });
    card.querySelector('.up').onclick = () => move(index, -1);
    card.querySelector('.down').onclick = () => move(index, 1);
    card.querySelector('.remove').onclick = () => { state.characters.splice(index, 1); renderCharacters(); previewPrompt(); };
    host.append(card);
  });
}
function move(index, offset) {
  const target = index + offset;
  if (target < 0 || target >= state.characters.length) return;
  [state.characters[index], state.characters[target]] = [state.characters[target], state.characters[index]];
  renderCharacters(); previewPrompt();
}
function payload() {
  return { scene: $('scene').value, negative: $('negative').value, characters: state.characters,
    reference: state.reference, source_history_id: state.sourceHistoryId, input_mode: $('inputMode').value,
    width: Number($('width').value), height: Number($('height').value),
    seed: Number($('seed').value), rewrite: $('rewrite').checked, steps: Number($('steps').value), cfg: Number($('cfg').value), vae_cpu: $('vaeCpu').checked, offload: $('offload').checked, live_preview: $('livePreview').checked, preview_interval: Number($('previewInterval').value),
    settings: state.settings };
}
async function api(path, body) {
  const response = await fetch(path, { method: body === undefined ? 'GET' : 'POST', headers: body === undefined ? {} : { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}
let previewSequence = 0;
async function previewPrompt() {
  const sequence = ++previewSequence;
  if (!$('scene').value.trim()) { $('promptPreview').textContent = 'Enter a scene to preview the prompt.'; return; }
  try { const result = await api('/api/preview', payload()); if (sequence === previewSequence) $('promptPreview').textContent = result.prompt; }
  catch (error) { if (sequence === previewSequence) $('promptPreview').textContent = error.message; }
}
function showError(message) { $('error').textContent = message; $('error').hidden = !message; }
function setStatus(status) { $('statusBadge').textContent = status; }
function setBusy(busy) { $('generate').disabled = busy; $('generate').textContent = busy ? 'Generating…' : '✦ Generate image'; $('cancel').hidden = !busy; }
function showImage(url, item = null) {
  $('previewBadge').hidden = true;
  $('outputImage').src = url;
  $('outputImage').hidden = false;
  $('emptyState').hidden = true;
  $('download').href = url;
  $('resultActions').hidden = false;
  const meta = $('previewMeta');
  meta.replaceChildren();
  if (item) {
    const title = document.createElement('strong');
    title.textContent = item.scene;
    const detail = document.createElement('span');
    detail.textContent = [item.input_mode === 'edit' ? 'Edited image' : '', item.width && item.height ? `${item.width} × ${item.height}` : '', item.steps ? `${item.steps} steps` : ''].filter(Boolean).join(' · ');
    meta.append(title, detail);
    if (item.prompt) {
      const details = document.createElement('details');
      const summary = document.createElement('summary'); summary.textContent = 'View prompt';
      const prompt = document.createElement('pre'); prompt.textContent = item.prompt;
      details.append(summary, prompt); meta.append(details);
    }
    meta.hidden = false;
  } else meta.hidden = true;
}
function updateInputMode() {
  const editing = $('inputMode').value === 'edit';
  $('sceneHeading').textContent = editing ? 'Edit instruction' : 'Scene';
  $('sceneHelp').textContent = editing ? 'Describe what to change in the supplied image.' : 'The setting, action, composition, and overall idea.';
  $('imageHint').textContent = editing
    ? 'Describe the changes above. Qwen uses the image as the starting content and tries to keep unspecified details.'
    : 'Qwen Image 2.1 needs the reference in both its vision and image paths. The style instruction reduces content copying, but cannot fully prevent it.';
  $('scene').placeholder = editing
    ? 'Change the flower petals to deep red, keep the stem, leaves, and background the same.'
    : 'A young witch in a cluttered potion shop, surrounded by glass bottles, herbs, and candles.';
  $('rewrite').disabled = editing;
  if (editing) $('rewrite').checked = false;
  previewPrompt();
}
function showInputImage(src, name) {
  $('referencePreview').src = src;
  $('referencePreview').hidden = false;
  $('referencePlaceholder').hidden = true;
  $('referenceName').textContent = name;
  $('clearReference').hidden = false;
}
function editHistoryImage() {
  const id = state.selectedHistoryId;
  if (!id) return;
  const item = state.history.find(entry => entry.id === id);
  state.sourceHistoryId = id;
  state.reference = '';
  $('referenceInput').value = '';
  $('inputMode').value = 'edit';
  showInputImage(`/api/history/${id}/image`, 'Selected from history');
  if (item && item.width && item.height) {
    $('width').value = item.width;
    $('height').value = item.height;
  }
  $('scene').value = '';
  updateInputMode();
  setStatus('Ready to edit');
  $('scene').focus();
}
function selectHistory(item) {
  state.selectedHistoryId = item.id;
  showImage(item.image, item);
  setStatus('Saved');
  if (!state.job) $('log').textContent = 'Showing a saved render.';
  renderHistory();
}
function renderHistory() {
  const list = $('historyList');
  list.replaceChildren();
  $('historyCount').textContent = String(state.history.length);
  if (!state.history.length) {
    const empty = document.createElement('div'); empty.className = 'history-empty';
    empty.textContent = 'Generated images will appear here.'; list.append(empty); return;
  }
  for (const item of state.history) {
    const button = document.createElement('button'); button.type = 'button'; button.className = 'history-item';
    if (item.id === state.selectedHistoryId) button.classList.add('selected');
    const image = document.createElement('img'); image.src = item.image; image.alt = '';
    const content = document.createElement('div'); content.className = 'history-copy';
    const title = document.createElement('strong'); title.textContent = item.scene;
    const time = document.createElement('small'); time.textContent = (item.input_mode === 'edit' ? 'Edited · ' : '') + new Date(item.created_at * 1000).toLocaleString();
    content.append(title, time); button.append(image, content);
    button.onclick = () => selectHistory(item);
    list.append(button);
  }
}
async function loadHistory() {
  const result = await api('/api/history');
  state.history = result.items;
  renderHistory();
}
async function displayProgress(job) {
  if (job.id !== state.job) return;
  $('log').textContent = job.log || 'Loading model…';
  $('log').scrollTop = $('log').scrollHeight;
  const current = job.step || 0;
  const total = job.total_steps || 0;
  const labels = { loading: 'Loading models', rewriting: 'Rewriting prompt', sampling: 'Sampling', preview: 'Decoding preview', decoding: 'Decoding final image' };
  const label = labels[job.stage] || job.status;
  const progressText = current ? `${label} · step ${current} / ${total}` : label;
  $('progressText').textContent = progressText;
  $('stepProgress').max = total || 1;
  $('stepProgress').value = current;
  $('stepProgress').hidden = !current;
  $('streamProgress').hidden = false;
  $('logStep').textContent = progressText;
  setStatus(current ? `${current} / ${total}` : label);
  if (job.status === 'running' && job.preview && job.preview_step > state.previewStep) {
    state.previewStep = job.preview_step;
    $('outputImage').src = job.preview + '?step=' + job.preview_step;
    $('outputImage').hidden = false;
    $('emptyState').hidden = true;
    $('resultActions').hidden = true;
    $('previewMeta').hidden = true;
    $('previewBadge').textContent = `Live preview · step ${job.preview_step} / ${total}`;
    $('previewBadge').hidden = false;
  }
  if (['done', 'error', 'cancelled'].includes(job.status)) {
    state.job = null;
    stopStream();
    setBusy(false);
    $('previewBadge').hidden = true;
    $('progressText').textContent = job.status === 'done' ? 'Image complete' : job.status;
    $('logStep').textContent = job.status === 'done' ? `Complete · ${total} steps` : job.status;
    setStatus(job.status === 'done' ? 'Done' : job.status);
    if (job.status === 'done') {
      showImage(job.image + '?t=' + Date.now());
      state.selectedHistoryId = job.id;
      await loadHistory();
    } else showError(job.error || (job.status === 'cancelled' ? 'Render cancelled.' : 'Generation failed. See the log for details.'));
  }
}
function stopStream() {
  clearTimeout(state.timer); clearTimeout(state.reconnect);
  if (state.socket) { state.socket.onclose = null; state.socket.close(); state.socket = null; }
}
async function poll() {
  const id = state.job;
  if (!id) return;
  try { await displayProgress(await api('/api/jobs/' + id)); }
  catch (error) { $('logStep').textContent = 'Connection interrupted; retrying…'; }
  if (state.job === id) state.timer = setTimeout(poll, 1000);
}
function streamProgress() {
  stopStream();
  const id = state.job;
  if (!id) return;
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const socket = new WebSocket(`${protocol}//${location.host}/api/jobs/${id}/events`);
  state.socket = socket;
  socket.onopen = () => { state.failures = 0; };
  socket.onmessage = event => {
    if (state.job !== id) return;
    try { displayProgress(JSON.parse(event.data)).catch(error => showError(error.message)); }
    catch { $('logStep').textContent = 'Invalid progress update; reconnecting…'; socket.close(); }
  };
  socket.onerror = () => socket.close();
  socket.onclose = () => {
    if (state.job !== id) return;
    state.failures += 1;
    if (state.failures >= 3) {
      $('logStep').textContent = 'Using HTTP progress fallback';
      poll();
    } else {
      $('logStep').textContent = 'Reconnecting progress stream…';
      state.reconnect = setTimeout(streamProgress, 1000 * state.failures);
    }
  };
}
async function generate() {
  showError('');
  try {
    const result = await api('/api/jobs', payload()); state.job = result.id; state.previewStep = 0; setBusy(true); setStatus('Starting');
    $('log').textContent = 'Starting model…'; $('outputImage').hidden = true; $('emptyState').hidden = false; $('resultActions').hidden = true; $('previewMeta').hidden = true; $('previewBadge').hidden = true;
    state.failures = 0; streamProgress();
  } catch (error) { showError(error.message); }
}
async function referenceFile(file) {
  if (!file) return;
  if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type)) { showError('Choose a PNG, JPG, or WebP image.'); return; }
  if (file.size > 16 * 1024 * 1024) { showError('Reference image exceeds 16 MB.'); return; }
  const reader = new FileReader();
  reader.onload = () => { state.reference = String(reader.result); state.sourceHistoryId = ''; showInputImage(state.reference, file.name); previewPrompt(); showError(''); };
  reader.readAsDataURL(file);
}
function renderSettings() {
  $('modelFields').innerHTML = '';
  for (const [key, label] of Object.entries(modelLabels)) {
    const wrap = document.createElement('label'); wrap.className = 'model-field'; wrap.textContent = label;
    const input = document.createElement('input'); input.id = 'model-' + key; input.value = state.settings[key] || state.defaults[key] || ''; input.autocomplete = 'off'; wrap.append(input); $('modelFields').append(wrap);
  }
}
async function init() {
  const config = await api('/api/config'); state.defaults = config.defaults;
  $('vaeCpu').checked = Boolean(config.vae_cpu_default);
  $('rewriteControl').hidden = !config.rewriter;
  if (config.backend === 'diffusers') {
    $('settingsButton').hidden = true;
    $('vaeCpu').checked = false;
    $('vaeCpu').closest('label').hidden = true;
    $('offload').checked = true;
    $('cfg').value = 1;
    $('steps').value = 40;
    $('width').value = 1024;
    $('height').value = 1024;
  }
  try { state.settings = JSON.parse(localStorage.getItem('qwen-image-2-1-abliterated-playground-model-paths') || '{}'); } catch { state.settings = {}; }
  renderSettings();
  $('settingsButton').onclick = () => $('settingsDialog').showModal();
  $('settingsDialog').addEventListener('close', () => { if ($('settingsDialog').returnValue === 'save') { for (const key of Object.keys(modelLabels)) state.settings[key] = $('model-' + key).value.trim(); localStorage.setItem('qwen-image-2-1-abliterated-playground-model-paths', JSON.stringify(state.settings)); } });
  $('resetSettings').onclick = () => { state.settings = {}; localStorage.removeItem('qwen-image-2-1-abliterated-playground-model-paths'); renderSettings(); };
  $('addCharacter').onclick = () => addCharacter();
  $('inputMode').onchange = updateInputMode;
  $('generate').onclick = generate;
  $('editResult').onclick = editHistoryImage;
  $('refreshHistory').onclick = () => loadHistory().catch(error => showError(error.message));
  $('cancel').onclick = async () => { if (state.job) await api('/api/jobs/' + state.job + '/cancel', {}); };
  $('livePreview').onchange = () => { $('previewInterval').disabled = !$('livePreview').checked; };
  $('referenceInput').onchange = e => referenceFile(e.target.files[0]);
  $('clearReference').onclick = () => { state.reference = ''; state.sourceHistoryId = ''; $('referenceInput').value = ''; $('referencePreview').hidden = true; $('referencePlaceholder').hidden = false; $('referenceName').textContent = 'No reference selected'; $('clearReference').hidden = true; previewPrompt(); };
  const dropzone = $('dropzone');
  dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.style.borderColor = '#c7a2ff'; });
  dropzone.addEventListener('dragleave', () => dropzone.style.borderColor = '');
  dropzone.addEventListener('drop', e => { e.preventDefault(); dropzone.style.borderColor = ''; referenceFile(e.dataTransfer.files[0]); });
  $('scene').addEventListener('input', previewPrompt);
  await loadHistory();
  if (!config.active && state.history.length) selectHistory(state.history[0]);
  if (config.active && config.active !== 'preparing') { state.job = config.active; setBusy(true); streamProgress(); }
}
init().catch(error => showError(error.message));
