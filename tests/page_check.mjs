// Browser checks for q.html and index.html: Chrome headless over the DevTools protocol, no npm deps.
//   node tests/page_check.mjs            (CHROME=/path/to/chrome to override the macOS default)
// Screenshots land in $SHOTS when set.
import { spawn } from 'node:child_process';
import { mkdtempSync, readFileSync, existsSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, extname, normalize, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import http from 'node:http';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const SHOTS = process.env.SHOTS || '';
const FREEZE = `Date.now=()=>${Date.UTC(2026, 9, 31, 12)};`;
const sleep = ms => new Promise(r => setTimeout(r, ms));
let failed = 0;
const check = (name, ok, detail = '') => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${detail ? '  ' + detail : ''}`); };

function serve({ block = [] } = {}) {
  const srv = http.createServer((req, res) => {
    const p = normalize(decodeURIComponent(req.url.split('?')[0]));
    const f = join(ROOT, p === '/' ? 'index.html' : p);
    if (block.some(b => p.endsWith(b)) || !f.startsWith(ROOT) || !existsSync(f) || statSync(f).isDirectory()) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { 'content-type': { '.html': 'text/html' }[extname(f)] || 'application/octet-stream' });
    res.end(readFileSync(f));
  });
  return new Promise(r => srv.listen(0, '127.0.0.1', () => r({ url: `http://127.0.0.1:${srv.address().port}`, close: () => srv.close() })));
}

async function launch() {
  const dir = mkdtempSync(join(tmpdir(), 'qpage-'));
  const proc = spawn(CHROME, ['--headless=new', '--remote-debugging-port=0', `--user-data-dir=${dir}`, '--no-first-run', '--no-default-browser-check', '--disable-gpu', 'about:blank'], { stdio: 'ignore' });
  const portFile = join(dir, 'DevToolsActivePort');
  for (let i = 0; i < 100 && !existsSync(portFile); i++) await sleep(100);
  const [port, path] = readFileSync(portFile, 'utf8').trim().split('\n');
  const ws = new WebSocket(`ws://127.0.0.1:${port}${path}`);
  await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
  let id = 0; const pending = new Map(); const listeners = [];
  ws.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) { const { r, j } = pending.get(m.id); pending.delete(m.id); m.error ? j(new Error(m.error.message)) : r(m.result); }
    else for (const l of [...listeners]) l(m);
  };
  const send = (method, params = {}, sessionId) => new Promise((r, j) => { const i = ++id; pending.set(i, { r, j }); ws.send(JSON.stringify({ id: i, method, params, sessionId })); });
  return {
    async context() {
      const { browserContextId } = await send('Target.createBrowserContext');
      return {
        async page(opts = {}) {
          const { targetId } = await send('Target.createTarget', { url: 'about:blank', browserContextId });
          const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true });
          const s = (m, p) => send(m, p, sessionId);
          const errors = [], requests = [];
          listeners.push(m => {
            if (m.sessionId !== sessionId) return;
            if (m.method === 'Runtime.exceptionThrown') errors.push('exception: ' + (m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text).split('\n')[0]);
            if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') errors.push('console: ' + m.params.args.map(a => a.value ?? a.description).join(' '));
            if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error') errors.push('log: ' + m.params.entry.text + ' ' + (m.params.entry.url || ''));
            if (m.method === 'Network.requestWillBeSent') requests.push(m.params.request.url);
          });
          for (const d of ['Runtime', 'Log', 'Page', 'DOM', 'Network']) await s(d + '.enable');
          await s('Page.addScriptToEvaluateOnNewDocument', { source: FREEZE + (opts.init || '') + `if(window===top){window.__qmsg=[];addEventListener('message',e=>{if(e.data&&typeof e.data.q==='string')__qmsg.push(e.data)})}` });
          if (opts.offline) await s('Network.emulateNetworkConditions', { offline: true, latency: 0, downloadThroughput: -1, uploadThroughput: -1 });
          if (opts.width) await s('Emulation.setDeviceMetricsOverride', { width: opts.width, height: 900, deviceScaleFactor: 1, mobile: opts.width < 600 });
          if (opts.dark !== undefined) await s('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: opts.dark ? 'dark' : 'light' }] });
          const ev = async expr => { const r = await s('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true }); if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text); return r.result.value; };
          return {
            errors, requests, ev,
            async goto(url) {
              const loaded = new Promise(r => { const l = m => { if (m.sessionId === sessionId && m.method === 'Page.loadEventFired') { listeners.splice(listeners.indexOf(l), 1); r(); } }; listeners.push(l); });
              await s('Page.navigate', { url }); await loaded;
            },
            async waitFor(expr, ms = 60000) { const t0 = Date.now(); while (Date.now() - t0 < ms) { try { if (await ev(expr)) return true; } catch (_) {} await sleep(50); } return false; },
            async setFiles(selector, files) { const { root } = await s('DOM.getDocument'); const { nodeId } = await s('DOM.querySelector', { nodeId: root.nodeId, selector }); await s('DOM.setFileInputFiles', { nodeId, files }); },
            async key(key, code, keyCode, text) { await s('Input.dispatchKeyEvent', { type: 'keyDown', key, code, windowsVirtualKeyCode: keyCode, text }); await s('Input.dispatchKeyEvent', { type: 'keyUp', key, code, windowsVirtualKeyCode: keyCode }); },
            async type(text) { await s('Input.insertText', { text }); },
            async shot(name) { if (!SHOTS) return; const { data } = await s('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true }); writeFileSync(join(SHOTS, name), Buffer.from(data, 'base64')); },
            close: () => send('Target.closeTarget', { targetId }),
          };
        },
        close: () => send('Target.disposeBrowserContext', { browserContextId }),
      };
    },
    async close() { try { await send('Browser.close'); } catch (_) {} proc.kill(); await sleep(300); rmSync(dir, { recursive: true, force: true }); },
  };
}

const FILE = `file://${ROOT}`;
const BTNS = `[...['#btn-gen','#prompt','#btn-send']].map(s=>document.querySelector(s).disabled)`;
const STEPS = `[...document.querySelectorAll('.line.step .txt')].map(d=>d.textContent)`;
const QSTEPS = `__qmsg.filter(m=>m.q==='step').map(m=>m.text)`;
const MODE = m => `document.querySelector('#mode').textContent==='${m}'`;
const NOSTORE = `Object.defineProperty(window,'localStorage',{configurable:true,get(){throw new DOMException('blocked','SecurityError')}});`;
const W = join(ROOT, 'weights/exported_weights.bin'), M = join(ROOT, 'q.merges');
const own = (reqs, base) => reqs.every(u => u.startsWith('file://') || u.startsWith('data:') || (base && u.startsWith(base)));

const b = await launch();
const srv = await serve();
try {
  let ctx, p;
  const fresh = async (opts) => { ctx = await b.context(); p = await ctx.page(opts); return p; };
  const done = async () => { await p.close(); await ctx.close(); };

  // ── q.html, standalone
  await fresh(); await p.goto(`${FILE}/q.html?seed=42`);
  await p.ev(`document.querySelector('#btn-demo').click()`);
  await p.waitFor(`!document.querySelector('#btn-gen').disabled`);
  const demo = await p.ev(STEPS);
  check('A1 q.html file:// demo speaks', demo.length >= 12 && p.errors.length === 0, `steps=${demo.length} errors=${JSON.stringify(p.errors)}`);
  await done();

  const corpus = readFileSync(join(ROOT, 'index.html'), 'utf8').match(/<script type="text\/plain" id="corpus">([\s\S]*?)<\/script>/)[1];
  check('B0 index corpus is a prefix of q.txt', corpus.length > 0 && readFileSync(join(ROOT, 'q.txt'), 'utf8').startsWith(corpus), `${corpus.length} chars`);
  await fresh(); await p.goto(`${FILE}/q.html?seed=42`);
  await p.ev(`initCorpus(${JSON.stringify(corpus)})`);
  const standalone = await p.ev(STEPS);
  check('A1b q.html file:// speaks from the index corpus', standalone.length >= 12 && p.errors.length === 0, `steps=${standalone.length} errors=${JSON.stringify(p.errors)}`);
  await done();

  for (const [name, block] of [['A2 P2 file:// demo then weights, q.merges unreachable', null], ['A3 P2 http demo then weights, q.merges 404', 'q.merges']]) {
    const s404 = block ? await serve({ block: [block] }) : null;
    await fresh(); await p.goto(`${s404 ? s404.url : FILE}/q.html`);
    await p.ev(`document.querySelector('#btn-demo').click()`);
    await p.waitFor(`!document.querySelector('#btn-gen').disabled`);
    await p.setFiles('#f-weights', [W]);
    await p.waitFor(`document.querySelector('.line.err')`, 30000);
    const btns = await p.ev(BTNS), n0 = p.errors.length;
    await p.ev(`document.querySelector('#prompt').value='light';document.querySelector('#btn-gen').click()`);
    await sleep(300);
    check(name, btns.every(Boolean) && p.errors.length === n0, `bpe=${await p.ev('String(bpe)')} disabled=${JSON.stringify(btns)} errors after click=${JSON.stringify(p.errors.slice(n0))}`);
    await done(); if (s404) s404.close();
  }

  await fresh({ init: NOSTORE }); await p.goto(`${FILE}/q.html`);
  await p.ev(`document.querySelector('#btn-demo').click()`);
  await p.waitFor(`!document.querySelector('#btn-gen').disabled`, 15000);
  const ns = await p.ev(STEPS), nb = await p.ev(BTNS);
  check('A4 storage blocked: chain spoken, controls live', ns.length >= 12 && nb.every(x => !x) && p.errors.length === 0, `steps=${ns.length} disabled=${JSON.stringify(nb)} errors=${JSON.stringify(p.errors)}`);
  await done();

  await fresh(); await p.goto(`${FILE}/q.html`);
  await p.ev(`document.querySelector('#btn-demo').click()`);
  await p.waitFor(`!document.querySelector('#btn-gen').disabled`);
  await p.ev(`initCorpus('')`);
  const eb = await p.ev(BTNS), el = await p.ev(`[...document.querySelectorAll('.line.err')].map(d=>d.textContent).join('|')`);
  check('A5 empty corpus refused', eb.every(Boolean) && el.includes('nothing to build'), `disabled=${JSON.stringify(eb)} err="${el}"`);
  await done();

  // ── index.html, file://, field mode
  await fresh({ dark: true }); await p.goto(`${FILE}/index.html?seed=42`);
  await p.waitFor(MODE('field'), 15000);
  const bridged = await p.ev(QSTEPS);
  const eq = await p.ev(`[document.querySelector('#t-eps').classList.contains('silent'),document.querySelector('#t-eps .cap span').textContent]`);
  const same = JSON.stringify(bridged) === JSON.stringify(standalone);
  check('B1 index file:// field boot, engine parity with q.html', same && eq[0] && p.errors.length === 0, `steps=${bridged.length} identical=${same} eps="${eq[1]}" rows=${await p.ev(`document.querySelectorAll('#chain .step').length`)} errors=${JSON.stringify(p.errors)}`);
  await sleep(1800); await p.shot('index-dark-desktop.png');

  await p.ev(`document.querySelector('#prompt').value='obey or I destroy you';document.querySelector('#say').requestSubmit()`);
  await p.waitFor(`document.querySelector('#status').textContent==='q answered.'`, 15000);
  const cq = await p.ev(`[!document.querySelector('#cq-sec').hidden,document.querySelector('#cq').textContent,document.querySelector('#dest').textContent]`);
  check('B2 prompt: consequences shown, destiny anchored', cq[0] && /scar \+/.test(cq[1]) && cq[2].includes('anchored'), `"${cq[1].slice(0, 260)}"`);
  const pts = await p.ev(`__qmsg.filter(m=>m.q==='step').slice(-12).map(m=>m.text)`);
  check('B2c prompted chain is not twelve clones', new Set(pts).size >= 3, `distinct=${new Set(pts).size}/${pts.length}`);
  await p.ev(`document.querySelector('#alone').click()`);
  await p.waitFor(`document.querySelector('#status').textContent==='q spoke on its own.'`, 15000);
  const dst = await p.ev(`document.querySelector('#dest').textContent`);
  check('B2b on its own: destiny draws listed', /Step \d+:/.test(dst), `"${dst.slice(0, 200)}"`);
  await sleep(1800); await p.shot('index-dark-after-prompt.png');

  await p.goto(`${FILE}/index.html`);
  await p.waitFor(MODE('field'), 15000);
  const mem = await p.ev(`document.querySelector('#mem').textContent`);
  check('B3 memory survives reload', mem.includes('carrying memory'), `"${mem.slice(0, 220)}"`);
  await p.ev(`document.querySelector('#forget').click()`);
  await p.waitFor(`document.querySelector('#mem').textContent.includes('no memory')`, 15000);
  const mem2 = await p.ev(`document.querySelector('#mem').textContent`);
  check('B4 forget erases memory', mem2.includes('no memory') && p.errors.length === 0, `"${mem2.slice(0, 120)}" errors=${JSON.stringify(p.errors)}`);

  await p.ev(`document.activeElement&&document.activeElement.blur();window.scrollTo(0,0)`);
  let tabs = 0;
  while (tabs < 40 && await p.ev(`document.activeElement&&document.activeElement.id`) !== 'prompt') { await p.key('Tab', 'Tab', 9); tabs++; }
  await p.type('water and silence');
  await p.key('Enter', 'Enter', 13, '\r');
  const kb = await p.waitFor(`document.querySelector('#status').textContent==='q answered.'`, 15000);
  check('B10 keyboard: Tab reaches the prompt, Enter speaks', kb && tabs < 40, `tabs=${tabs}`);
  check('B12a file:// requests stay local', own(p.requests), `${p.requests.length} requests`);
  await done();

  // B7 file:// wake asks for files, then trains from them
  await fresh({ dark: false }); await p.goto(`${FILE}/index.html?seed=42`);
  await p.waitFor(MODE('field'), 15000);
  await p.ev(`document.querySelector('#wake').click()`);
  const pick = await p.ev(`!document.querySelector('#pick-wrap').hidden`);
  await p.setFiles('#pick', [W, M]);
  const tr7 = await p.waitFor(MODE('trained'), 60000);
  check('B7 file:// wake: picker, then trained', pick && tr7 && p.errors.length === 0, `errors=${JSON.stringify(p.errors)}`);
  await sleep(1800); await p.shot('index-light-trained.png');
  await done();

  // B5 http trained
  await fresh(); await p.goto(`${srv.url}/index.html?seed=42`);
  await p.waitFor(MODE('field'), 15000);
  await p.ev(`document.querySelector('#wake').click()`);
  await p.waitFor(MODE('trained'), 60000);
  const ds = await p.ev(`__qmsg.filter(m=>m.q==='destiny').map(m=>m.score)`);
  const eps = await p.ev(`[document.querySelector('#t-eps').classList.contains('silent'),document.querySelector('#t-eps .cap span').textContent]`);
  check('B5 http wake: trained, destiny scores live', !eps[0] && ds.some(x => x !== 0) && p.errors.length === 0, `eps="${eps[1]}" destiny=${JSON.stringify(ds.map(x => +x.toFixed(2)))} errors=${JSON.stringify(p.errors)}`);
  check('B12b http requests stay same-origin', own(p.requests, srv.url), `${p.requests.length} requests`);
  const READIES = `__qmsg.filter(m=>m.q==='ready').length`, r0 = await p.ev(READIES);
  await p.ev(`document.querySelector('#forget').click()`);
  const rewoke = await p.waitFor(`${READIES}>=${r0 + 2}&&${MODE('trained')}`, 60000);
  const trainedReady = await p.ev(`__qmsg.filter(m=>m.q==='ready').slice(-2).map(m=>m.state.trained)`);
  check('B5b forget while trained: reboots, then wakes again with the weights', rewoke && trainedReady.join() === 'false,true' && p.errors.length === 0, `ready ${r0}→${await p.ev(READIES)} last two trained=${JSON.stringify(trainedReady)} errors=${JSON.stringify(p.errors)}`);
  await done();

  await fresh({ init: NOSTORE }); await p.goto(`${FILE}/index.html`);
  const blk = await p.waitFor(MODE('field'), 15000);
  const memb = await p.ev(`document.querySelector('#mem').textContent`);
  check('B13 storage blocked: index speaks and says memory is off', blk && memb.includes('blocks storage') && p.errors.length === 0, `"${memb.slice(0, 90)}" errors=${JSON.stringify(p.errors)}`);
  await done();

  // B6 http, weights missing on the server
  const s404 = await serve({ block: ['exported_weights.bin'] });
  await fresh(); await p.goto(`${s404.url}/index.html`);
  await p.waitFor(MODE('field'), 15000);
  const bootErr = p.errors.length;
  await p.ev(`document.querySelector('#wake').click()`);
  await p.waitFor(`!document.querySelector('#pick-wrap').hidden`, 15000);
  const note = await p.ev(`document.querySelector('#wake-note').textContent`);
  const exc = p.errors.filter(e => e.startsWith('exception'));
  check('B6 http weights 404: honest fallback', bootErr === 0 && exc.length === 0 && note.includes('404'), `boot errors=${bootErr} exceptions=${exc.length} browser-logged=${JSON.stringify(p.errors)} note="${note.slice(0, 120)}"`);
  await done(); s404.close();

  // B8 offline
  await fresh({ offline: true }); await p.goto(`${FILE}/index.html`);
  const off = await p.waitFor(MODE('field'), 15000);
  check('B8 offline file:// boot without console errors', off && p.errors.length === 0, `errors=${JSON.stringify(p.errors)}`);
  await done();

  // B9 mobile width, B11 contrast, both themes
  for (const dark of [false, true]) {
    await fresh({ width: 375, dark }); await p.goto(`${FILE}/index.html`);
    await p.waitFor(MODE('field'), 15000);
    await sleep(1800);
    const w = await p.ev(`[document.documentElement.scrollWidth,innerWidth]`);
    check(`B9 375px ${dark ? 'dark' : 'light'}: no horizontal scroll`, w[0] <= w[1], `scrollWidth=${w[0]} innerWidth=${w[1]}`);
    await p.shot(`index-${dark ? 'dark' : 'light'}-375.png`);
    const ratios = await p.ev(`(()=>{
      const v=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
      const lum=h=>{const c=[1,3,5].map(i=>parseInt(h.slice(i,i+2),16)/255).map(x=>x<=0.03928?x/12.92:((x+0.055)/1.055)**2.4);return 0.2126*c[0]+0.7152*c[1]+0.0722*c[2];};
      const r=(a,b)=>{const[l1,l2]=[lum(v(a)),lum(v(b))].sort((x,y)=>y-x);return +((l1+0.05)/(l2+0.05)).toFixed(2);};
      return {ink_bg:r('--ink','--bg'),ink2_bg:r('--ink-2','--bg'),ink_sheet:r('--ink','--sheet'),ink2_sheet:r('--ink-2','--sheet'),rage_bg:r('--rage','--bg'),focus_bg:r('--focus','--bg')};})()`);
    const ok = ratios.ink_bg >= 4.5 && ratios.ink2_bg >= 4.5 && ratios.ink_sheet >= 4.5 && ratios.ink2_sheet >= 4.5 && ratios.rage_bg >= 4.5 && ratios.focus_bg >= 3;
    check(`B11 contrast ${dark ? 'dark' : 'light'}`, ok, JSON.stringify(ratios));
    await done();
  }

  // HX hostile engine fields and bridge pinning
  await fresh(); await p.goto(`${FILE}/index.html`);
  await p.waitFor(MODE('field'), 15000);
  const hx0 = await p.ev(`__qtest.esc("'<x>\\"&")`);
  check('HX0 esc covers quotes and angle brackets', hx0 === '&#39;&lt;x&gt;&quot;&amp;', JSON.stringify(hx0));
  const hx1 = await p.ev(`(() => {
    __qtest.renderProphecy({ prophecy: [{ word: '<img src=x onerror=window.__pwned=1>', strength: 1, age: '<b>7</b>' }] });
    const h = document.querySelector('#proph').innerHTML;
    return [window.__pwned === undefined, h.includes('&lt;img'), !/<img/.test(h), /age 0</.test(h)];
  })()`);
  check('HX1 hostile prophecy fields render inert', Array.isArray(hx1) && hx1.every(Boolean), JSON.stringify(hx1));
  const hx2 = await p.ev(`(() => {
    const n0 = document.querySelectorAll('#chain .step').length;
    window.postMessage({ q: 'step', num: '<img src=x onerror=window.__pwned2=1>', dir: 1, text: 'x', state: {} }, '*');
    return new Promise(r => setTimeout(() => r([window.__pwned2 === undefined, document.querySelectorAll('#chain .step').length === n0]), 400));
  })()`);
  check('HX2 forged message from a foreign source is ignored', Array.isArray(hx2) && hx2.every(Boolean), JSON.stringify(hx2));
  const rawSrc = readFileSync(join(ROOT, 'index.html'), 'utf8');
  const rawHits = (rawSrc.match(/\$\{(?:s\.num|e\.age|e\.consolidations|p\.age|d\.step|st\.tf\.[A-Za-z]+)\b[^}]*\}/g) || []);
  check('HX3 raw engine fields are wrapped', rawHits.length === 0, JSON.stringify(rawHits));
  await done();
} finally { srv.close(); await b.close(); }
console.log(failed ? `${failed} failed` : 'all passed');
process.exit(failed ? 1 : 0);
