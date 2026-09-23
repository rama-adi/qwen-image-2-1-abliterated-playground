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
