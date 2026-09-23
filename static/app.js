const $ = id => document.getElementById(id);
const state = { socket: null, reconnect: null, handshake: null, failures: 0, defaults: {}, settings: {}, characters: [], references: [], referenceCards: [], loras: [], backend: null, job: null, timer: null, history: [], selectedHistoryId: null, previewStep: 0 };
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
    references: state.referenceCards.map(({id, source, role, note}) => ({id, source, role, note})), input_mode: $('inputMode').value,
    width: Number($('width').value), height: Number($('height').value),
    seed: Number($('seed').value), rewrite: $('rewrite').checked, steps: Number($('steps').value), cfg: Number($('cfg').value), vae_cpu: $('vaeCpu').checked, offload: $('offload').checked, live_preview: $('livePreview').checked, preview_interval: Number($('previewInterval').value),
    lora_id: $('loraSelect').value, lora_strength: Number($('loraStrength').value),
    settings: state.settings };
}
async function api(path, body, method = null) {
  const response = await fetch(path, { method: method || (body === undefined ? 'GET' : 'POST'), headers: body === undefined ? {} : { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) });
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
    const title = document.createElement('h3'); title.textContent = 'Experiment settings';
    const description = document.createElement('p'); description.textContent = item.scene;
    const copy = document.createElement('button'); copy.type = 'button'; copy.className = 'secondary'; copy.textContent = 'Reuse all settings';
    const settings = item.settings || {scene:item.scene, prompt:item.prompt, width:item.width, height:item.height, steps:item.steps, seed:item.seed};
    const text = JSON.stringify(settings, null, 2);
    const details = document.createElement('details');
    const summary = document.createElement('summary'); summary.textContent = `${item.width || '?'} × ${item.height || '?'} · ${item.steps || '?'} steps · CFG ${settings.cfg ?? '?'} · Seed ${item.seed ?? '?'}`;
    const values = document.createElement('pre'); values.textContent = text;
    copy.onclick = async () => {
      copy.disabled = true;
      try { await reuseSettings(item); copy.textContent = 'Settings restored'; }
      catch (error) { showError(error.message); }
      finally { copy.disabled = false; }
    };
    details.append(summary, values); meta.append(title, description, copy, details);
    meta.hidden = false;
  } else meta.hidden = true;
}
async function reuseSettings(item) {
  const settings = item.settings || item;
  await loadReferences(); await loadLoras();
  let lora = null;
  if (settings.lora) {
    lora = state.loras.find(entry => entry.id === settings.lora.id || (settings.lora.sha256 && entry.sha256 === settings.lora.sha256));
    if (!lora) throw new Error(`Upload ${settings.lora.name || 'the saved LoRA'} again before reusing these settings.`);
  }
  let refs = settings.references || [];
  if (!refs.length && settings.has_reference) {
    // Older renders retain a copy of their single input alongside the output.
    refs = [{id:item.id, source:'reference', role:'style', note:''}];
  }
  const cards = refs.map(ref => {
    const saved = ref.source === 'history' ? state.history.find(entry=>entry.id===ref.id) : state.references.find(entry=>entry.id===ref.id);
    if (!saved) throw new Error('A saved reference is no longer available. Add it again before reusing these settings.');
    return {...ref, name:saved.name || saved.scene || 'Saved reference', image:saved.image};
  });
  const fields = {scene:'scene', negative:'negative', width:'width', height:'height', steps:'steps', cfg:'cfg', seed:'seed', input_mode:'inputMode', preview_interval:'previewInterval'};
  const missing = [];
  for (const [key,id] of Object.entries(fields)) {
    if (settings[key] !== null && settings[key] !== undefined) $(id).value = settings[key];
    else missing.push(key);
  }
  for (const [key,id] of Object.entries({offload:'offload', vae_cpu:'vaeCpu', live_preview:'livePreview', rewrite:'rewrite'})) {
    if (settings[key] !== null && settings[key] !== undefined) $(id).checked = Boolean(settings[key]);
    else missing.push(key);
  }
  if (Array.isArray(settings.characters)) state.characters = structuredClone(settings.characters);
  else missing.push('characters');
  renderCharacters();
  state.referenceCards = cards;
  $('referenceInput').value = ''; $('referenceSelect').value = '';
  $('loraSelect').value = lora ? lora.id : '';
  $('loraStrength').value = settings.lora?.strength ?? 1;
  if (settings.model_settings && state.backend === 'sd-cpp' && settings.backend === 'sd-cpp') {
    state.settings = {...settings.model_settings}; renderSettings();
    localStorage.setItem('qwen-image-2-1-abliterated-playground-model-paths', JSON.stringify(state.settings));
  }
  if ($('rewriteControl').hidden) $('rewrite').checked = false;
  if (state.backend === 'diffusers') $('vaeCpu').checked = false;
  updateInputMode();
  $('previewInterval').disabled = !$('livePreview').checked;
  document.querySelector('.editor').dispatchEvent(new Event('change', {bubbles:true}));
  showError(missing.length ? `Older render: ${missing.join(', ')} were not recorded; those controls were left unchanged.` : '');
  setStatus('Settings restored'); $('scene').focus();
}
function updateInputMode() {
  const editing = $('inputMode').value === 'edit';
  $('sceneHeading').textContent = editing ? 'Edit instruction' : 'Scene';
  $('sceneHelp').textContent = editing ? 'Describe what to change in the supplied image.' : 'The setting, action, composition, and overall idea.';
  $('imageHint').textContent = editing
    ? 'Image 1 is the source to edit. Additional images guide the changes by their assigned roles.'
    : 'Roles guide the prompt; pose and identity matching are approximate. More images use more memory.';
  renderReferenceCards();
  $('scene').placeholder = editing
    ? 'Change the flower petals to deep red, keep the stem, leaves, and background the same.'
    : 'A young witch in a cluttered potion shop, surrounded by glass bottles, herbs, and candles.';
  $('rewrite').disabled = editing;
  if (editing) $('rewrite').checked = false;
  previewPrompt();
}
function editHistoryImage() {
  const id = state.selectedHistoryId;
  if (!id) return;
  const item = state.history.find(entry => entry.id === id);
  state.referenceCards = [{id, source:'history', role:'identity', note:'', name:'Image from history', image:`/api/history/${id}/image`}];
  $('inputMode').value = 'edit';
  renderReferenceCards();
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
async function deleteHistory(id = null) {
  const message = id ? 'Delete this image and its saved prompt, previews, and logs? This cannot be undone.' : 'Delete ALL saved images, prompts, previews, and logs? This cannot be undone.';
  if (!confirm(message)) return;
  try {
    const result = await api('/api/history' + (id ? '/' + id : ''), undefined, 'DELETE');
    state.referenceCards = state.referenceCards.filter(ref => !result.deleted.includes(ref.id));
    renderReferenceCards(); saveReferenceCards();
    if (result.deleted.includes(state.selectedHistoryId)) {
      state.selectedHistoryId = null;
      $('outputImage').hidden = true;
      $('outputImage').removeAttribute('src');
      $('emptyState').hidden = false;
      $('resultActions').hidden = true;
      $('previewMeta').hidden = true;
    }
    await loadHistory();
  } catch (error) { showError(error.message); }
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
    const row = document.createElement('div'); row.className = 'history-row';
    const remove = document.createElement('button'); remove.type = 'button';
    remove.className = 'history-delete'; remove.textContent = 'Delete';
    remove.setAttribute('aria-label', 'Delete image: ' + item.scene);
    remove.onclick = () => deleteHistory(item.id);
    row.append(button, remove); list.append(row);
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
  const labels = { loading: 'Loading models', encoding: 'Encoding prompt / first step', rewriting: 'Rewriting prompt', sampling: 'Sampling', preview: 'Decoding preview', decoding: 'Decoding final image' };
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
    $('transportStatus').textContent = 'Stream finished';
    setBusy(false);
    $('previewBadge').hidden = true;
    $('progressText').textContent = job.status === 'done' ? 'Image complete' : job.status;
    $('logStep').textContent = job.status === 'done' ? `Complete · ${total} steps` : job.status;
    setStatus(job.status === 'done' ? 'Done' : job.status);
    if (job.status === 'done') {
      showImage(job.image + '?t=' + Date.now());
      state.selectedHistoryId = job.id;
      await loadHistory();
      showImage(job.image, state.history.find(item => item.id === job.id));
    } else showError(job.error || (job.status === 'cancelled' ? 'Render cancelled.' : 'Generation failed. See the log for details.'));
  }
}
function stopStream() {
  clearTimeout(state.timer); clearTimeout(state.reconnect); clearTimeout(state.handshake);
  if (state.socket) { state.socket.onclose = null; state.socket.close(); state.socket = null; }
}
async function poll() {
  const id = state.job;
  if (!id) return;
  try {
    const update = await api('/api/jobs/' + id);
    if (state.job === id && (!state.socket || state.socket.readyState !== WebSocket.OPEN)) await displayProgress(update);
  }
  catch (error) { $('logStep').textContent = 'Connection interrupted; retrying…'; }
  if (state.job === id && (!state.socket || state.socket.readyState !== WebSocket.OPEN)) state.timer = setTimeout(poll, 1000);
}
function streamProgress() {
  stopStream();
  const id = state.job;
  if (!id) return;
  $('transportStatus').textContent = 'Connecting WebSocket…';
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const socket = new WebSocket(`${protocol}//${location.host}/api/jobs/${id}/events`);
  state.socket = socket;
  state.handshake = setTimeout(() => { if (socket.readyState === WebSocket.CONNECTING) socket.close(); }, 10000);
  socket.onopen = () => { clearTimeout(state.handshake); clearTimeout(state.timer); $('transportStatus').textContent = 'WebSocket live'; };
  socket.onmessage = event => {
    if (state.job !== id) return;
    try { const update = JSON.parse(event.data); state.failures = 0; displayProgress(update).catch(error => showError(error.message)); }
    catch { $('logStep').textContent = 'Invalid progress update; reconnecting…'; socket.close(); }
  };
  socket.onerror = () => socket.close();
  socket.onclose = event => {
    clearTimeout(state.handshake);
    if (state.job !== id) return;
    state.failures += 1;
    if (state.failures >= 3) {
      $('transportStatus').textContent = `HTTP fallback · WS closed ${event.code}`;
      poll();
      state.reconnect = setTimeout(streamProgress, 30000);
    } else {
      $('transportStatus').textContent = `Reconnecting WebSocket · ${event.code}`;
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
async function loadReferences(selected = $('referenceSelect').value) {
  const result = await api('/api/references'); state.references = result.items;
  const select = $('referenceSelect'); select.replaceChildren(new Option('None', ''));
  for (const item of result.items) select.add(new Option(item.name, item.id));
  if (result.items.some(item => item.id === selected)) select.value = selected;
}
const referenceRoles = {style:'Copy style', pose:'Copy pose', identity:'Character / identity', clothing:'Clothing / outfit', composition:'Composition / camera', background:'Background / environment', object:'Object / product', lighting:'Lighting / palette'};
function saveReferenceCards() {
  document.querySelector('.editor').dispatchEvent(new Event('change', {bubbles:true}));
  previewPrompt();
}
function renderReferenceCards() {
  const host = $('referenceCards'); host.replaceChildren();
  state.referenceCards.forEach((ref, index) => {
    const card = document.createElement('div'); card.className = 'reference-card';
    const image = document.createElement('img'); image.src = ref.image; image.alt = `Reference ${index+1}: ${ref.name}`;
    const title = document.createElement('strong'); title.textContent = `Image ${index+1} · ${ref.name}`;
    const role = document.createElement('select'); role.setAttribute('aria-label', `Role for image ${index+1}`);
    for (const [value, label] of Object.entries(referenceRoles)) role.add(new Option(label,value));
    role.value = ref.role;
    if ($('inputMode').value === 'edit' && index === 0) { role.add(new Option('Source image to edit','base')); role.value = 'base'; role.disabled = true; }
    role.onchange = () => { ref.role = role.value; saveReferenceCards(); };
    const note = document.createElement('input'); note.placeholder = 'Optional: e.g. pose for the person on the left'; note.maxLength = 500;
    note.setAttribute('aria-label', `Instruction for image ${index+1}`); note.value = ref.note;
    note.oninput = () => { ref.note = note.value; saveReferenceCards(); };
    const actions = document.createElement('div'); actions.className = 'reference-actions';
    for (const [text, offset] of [['↑',-1],['↓',1]]) {
      const button = document.createElement('button'); button.type='button'; button.className='mini-button'; button.textContent=text;
      button.setAttribute('aria-label', `Move image ${index+1} ${offset<0?'up':'down'}`); button.disabled=index+offset<0 || index+offset>=state.referenceCards.length;
      button.onclick=()=>{ const other=index+offset; [state.referenceCards[index],state.referenceCards[other]]=[state.referenceCards[other],state.referenceCards[index]]; renderReferenceCards(); saveReferenceCards(); }; actions.append(button);
    }
    const remove=document.createElement('button'); remove.type='button'; remove.className='text-button'; remove.textContent='Remove';
    remove.onclick=()=>{state.referenceCards.splice(index,1); renderReferenceCards(); saveReferenceCards();}; actions.append(remove);
    card.append(image,title,role,note,actions); host.append(card);
  });
  $('referenceName').textContent = `${state.referenceCards.length} / 4 references`;
  $('clearReference').hidden = !state.referenceCards.length;
  $('referencePreview').hidden = true; $('referencePlaceholder').hidden = false;
}
function addReference(item) {
  if (!item) return;
  if (state.referenceCards.length >= 4) { showError('Use at most 4 reference images.'); return; }
  state.referenceCards.push({...item, source:'reference', role:'style', note:''});
  renderReferenceCards(); saveReferenceCards();
}
async function referenceFiles(files) {
  const selected = [...files];
  if (selected.length + state.referenceCards.length > 4) { showError('Use at most 4 reference images.'); return; }
  for (const file of selected) await referenceFile(file);
}
async function referenceFile(file) {
  if (!file) return;
  if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type)) { showError('Choose a PNG, JPG, or WebP image.'); return; }
  if (file.size > 16 * 1024 * 1024) { showError('Reference image exceeds 16 MB.'); return; }
  $('referenceInput').disabled = true;
  try {
    const reference = await new Promise((resolve, reject) => {
      const reader = new FileReader(); reader.onload = () => resolve(String(reader.result)); reader.onerror = () => reject(new Error('Could not read reference image.')); reader.readAsDataURL(file);
    });
    const saved = await api('/api/references', {name:file.name, reference});
    await loadReferences(saved.id); addReference(saved);
    $('referenceSelect').dispatchEvent(new Event('change', {bubbles:true})); showError('');
  } catch(error) { showError(error.message); }
  finally { $('referenceInput').disabled = false; }
}
function renderSettings() {
  $('modelFields').innerHTML = '';
  for (const [key, label] of Object.entries(modelLabels)) {
    const wrap = document.createElement('label'); wrap.className = 'model-field'; wrap.textContent = label;
    const input = document.createElement('input'); input.id = 'model-' + key; input.value = state.settings[key] || state.defaults[key] || ''; input.autocomplete = 'off'; wrap.append(input); $('modelFields').append(wrap);
  }
}
async function loadLoras(selected = $('loraSelect').value) {
  const result = await api('/api/loras'); state.loras = result.items;
  const select = $('loraSelect'); select.replaceChildren(new Option('None', ''));
  for (const item of result.items) select.add(new Option(item.name, item.id));
  if (result.items.some(item => item.id === selected)) select.value = selected;
}
async function uploadLora(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith('.safetensors') || file.size > 2 * 1024 ** 3) {
    showError('Choose a .safetensors LoRA up to 2 GiB.'); return;
  }
  let id = null;
  $('loraFile').disabled = true;
  try {
    showError('');
    const upload = await api('/api/loras/uploads', {name: file.name, size: file.size}); id = upload.id;
    for (let offset = 0; offset < file.size; offset += upload.chunk_size) {
      const response = await fetch(`/api/loras/uploads/${id}/chunk`, {method: 'POST', headers: {'Content-Type':'application/octet-stream', 'X-Upload-Offset': String(offset)}, body: file.slice(offset, offset + upload.chunk_size)});
      if (!response.ok) { const error = await response.json(); throw new Error(error.error || 'Upload failed.'); }
      $('loraStatus').textContent = `Uploading ${file.name}: ${Math.round(Math.min(offset + upload.chunk_size, file.size) / file.size * 100)}%`;
    }
    $('loraStatus').textContent = 'Checking adapter file…';
    await api(`/api/loras/uploads/${id}/finish`, {});
    await loadLoras(id);
    $('loraSelect').dispatchEvent(new Event('change', {bubbles:true}));
    $('loraStatus').textContent = `${file.name} uploaded. Model compatibility is checked on generation.`;
  } catch (error) {
    showError(error.message); $('loraStatus').textContent = 'Upload failed. Select the file to retry.';
    if (id) await api(`/api/loras/uploads/${id}`, undefined, 'DELETE').catch(() => {});
  } finally { $('loraFile').disabled = false; $('loraFile').value = ''; }
}
function preserveConfig(backend) {
  const key = 'playground-form-' + backend;
  const controls = [...document.querySelectorAll('.editor input[id], .editor textarea[id], .editor select[id]')].filter(el => el.type !== 'file');
  try {
    const saved = JSON.parse(sessionStorage.getItem(key) || 'null');
    if (saved) {
      for (const el of controls) if (Object.hasOwn(saved.fields, el.id)) {
        if (el.type === 'checkbox') el.checked = saved.fields[el.id]; else el.value = saved.fields[el.id];
      }
      state.characters = saved.characters || []; renderCharacters();
      state.referenceCards = (saved.referenceCards || []).slice(0,4); renderReferenceCards();
    }
  } catch {}
  const save = () => {
    const fields = Object.fromEntries(controls.map(el => [el.id, el.type === 'checkbox' ? el.checked : el.value]));
    try { sessionStorage.setItem(key, JSON.stringify({fields, characters: state.characters, referenceCards:state.referenceCards})); } catch {}
  };
  document.querySelector('.editor').addEventListener('input', save);
  document.querySelector('.editor').addEventListener('change', save);
  window.addEventListener('pagehide', save);
}
async function init() {
  const config = await api('/api/config'); state.defaults = config.defaults; state.backend = config.backend;
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
  await loadLoras();
  await loadReferences();
  preserveConfig(config.backend);
  $('addSavedReference').onclick = () => addReference(state.references.find(item=>item.id===$('referenceSelect').value));
  updateInputMode();
  $('loraFile').onchange = e => uploadLora(e.target.files[0]);
  $('settingsButton').onclick = () => $('settingsDialog').showModal();
  $('settingsDialog').addEventListener('close', () => { if ($('settingsDialog').returnValue === 'save') { for (const key of Object.keys(modelLabels)) state.settings[key] = $('model-' + key).value.trim(); localStorage.setItem('qwen-image-2-1-abliterated-playground-model-paths', JSON.stringify(state.settings)); } });
  $('resetSettings').onclick = () => { state.settings = {}; localStorage.removeItem('qwen-image-2-1-abliterated-playground-model-paths'); renderSettings(); };
  $('addCharacter').onclick = () => addCharacter();
  $('inputMode').onchange = updateInputMode;
  $('generate').onclick = generate;
  $('editResult').onclick = editHistoryImage;
  $('deleteAllHistory').onclick = () => deleteHistory();
  $('refreshHistory').onclick = () => loadHistory().catch(error => showError(error.message));
  $('cancel').onclick = async () => { if (state.job) await api('/api/jobs/' + state.job + '/cancel', {}); };
  $('livePreview').onchange = () => { $('previewInterval').disabled = !$('livePreview').checked; };
  $('referenceInput').onchange = e => referenceFiles(e.target.files);
  $('clearReference').onclick = () => { state.referenceCards=[]; $('referenceInput').value=''; renderReferenceCards(); saveReferenceCards(); };
  const dropzone = $('dropzone');
  dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.style.borderColor = '#c7a2ff'; });
  dropzone.addEventListener('dragleave', () => dropzone.style.borderColor = '');
  dropzone.addEventListener('drop', e => { e.preventDefault(); dropzone.style.borderColor = ''; referenceFiles(e.dataTransfer.files); });
  $('scene').addEventListener('input', previewPrompt);
  await loadHistory();
  if (!config.active && state.history.length) selectHistory(state.history[0]);
  if (config.active && config.active !== 'preparing') { state.job = config.active; setBusy(true); streamProgress(); }
}
init().catch(error => showError(error.message));
