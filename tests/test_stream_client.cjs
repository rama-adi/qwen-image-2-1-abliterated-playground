const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

test('WebSocket fallback retries and a live connection stops HTTP polling', async () => {
  const timers = new Map(), sockets = [], elements = new Map(); let next = 0;
  class Socket {
    static OPEN = 1;
    constructor() { this.readyState = 0; sockets.push(this); }
    close() { this.readyState = 3; if (this.onclose) this.onclose({code:1006}); }
  }
  const context = vm.createContext({
    document: {getElementById: id => {if (!elements.has(id)) elements.set(id, {}); return elements.get(id);}},
    location: {protocol:'https:', host:'example.test'}, WebSocket: Socket,
    setTimeout: (fn, delay) => {timers.set(++next, {fn, delay}); return next;},
    clearTimeout: id => timers.delete(id),
  });
  const source = fs.readFileSync('static/app.js', 'utf8').replace('init().catch(error => showError(error.message));', '');
  vm.runInContext(source + '\nstate.job="test"; api=async()=>({}); displayProgress=async()=>{}; streamProgress();', context);
  for (let attempt=0; attempt<3; attempt++) {
    sockets.at(-1).close();
    if (attempt<2) {
      const entry=[...timers].find(([,v])=>v.delay===(attempt+1)*1000);
      timers.delete(entry[0]); entry[1].fn();
    }
  }
  await Promise.resolve();
  assert.match(elements.get('transportStatus').textContent, /HTTP fallback/);
  const retry=[...timers].find(([,v])=>v.delay===30000); assert.ok(retry);
  timers.delete(retry[0]); retry[1].fn();
  const socket=sockets.at(-1); socket.readyState=Socket.OPEN; socket.onopen();
  assert.equal(elements.get('transportStatus').textContent, 'WebSocket live');
  await vm.runInContext('poll()', context);
  assert.equal([...timers.values()].filter(v=>v.delay===1000).length, 0);
});

test('Reuse restores all sidebar controls, assets and actual seed without generating', async () => {
  const elements = new Map(); let persisted = 0;
  const element = id => {
    if (!elements.has(id)) elements.set(id, {value:'stale',checked:true,hidden:false,focus(){}});
    return elements.get(id);
  };
  const context = vm.createContext({document:{getElementById:element,querySelector:()=>({dispatchEvent(){persisted++;}})},
    structuredClone, Event:class {}, localStorage:{setItem(){}}});
  const source=fs.readFileSync('static/app.js','utf8').replace('init().catch(error => showError(error.message));','');
  vm.runInContext(source+`
    loadReferences=async()=>{}; loadLoras=async()=>{}; renderCharacters=()=>{};
    updateInputMode=()=>{}; renderSettings=()=>{};
    state.backend='diffusers'; state.loras=[{id:'new-id',sha256:'hash'}];
    state.references=[{id:'ref',name:'pose',image:'/reference.png'}];
  `,context);
  const settings={scene:'A boat',negative:'',width:1024,height:768,steps:20,cfg:3,seed:79,input_mode:'style',preview_interval:4,
    offload:false,vae_cpu:false,live_preview:false,rewrite:false,characters:[],lora:{id:'old-id',sha256:'hash',strength:0},
    references:[{id:'ref',source:'reference',role:'pose',note:'left person'}]};
  context.fixture={settings};
  await vm.runInContext('reuseSettings(fixture)',context);
  assert.equal(element('scene').value,'A boat'); assert.equal(element('negative').value,'');
  assert.equal(element('seed').value,79); assert.equal(element('cfg').value,3);
  assert.equal(element('offload').checked,false); assert.equal(element('livePreview').checked,false);
  assert.equal(element('previewInterval').disabled,true); assert.equal(element('loraSelect').value,'new-id');
  assert.equal(element('loraStrength').value,0); assert.equal(persisted,1);
  assert.equal(vm.runInContext('state.referenceCards[0].role',context),'pose');
  assert.equal(vm.runInContext('state.characters.length',context),0);
  assert.equal(vm.runInContext('state.job',context),null);
  context.fixture={settings:{...settings,scene:'Should not apply',lora:{id:'missing',name:'lost'}}};
  await assert.rejects(vm.runInContext('reuseSettings(fixture)',context),/Upload lost/);
  assert.equal(element('scene').value,'A boat');
});
