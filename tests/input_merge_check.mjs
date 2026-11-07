// The words you type and the text q prints after them: where they join in q.html. Reads files only: no browser, no npm deps, no code from the page is executed.
//   node tests/input_merge_check.mjs [path/to/q.html]     (default: q.html next to tests/)
// The pivot step of a chain starts from your words (anchoredPromptFromInput) and the sampler continues them. The input arrives trimmed, so its last
// word had no space after it, and q's tokens carry their word-end space inside ("y ", "s "): the first sampled token fused with your last word,
// "hey you" + "s " printed "hey yous conversation ...", "hello" printed "helloDreams ...". The window also cut your words in half ("who are y").
// Bpe and anchoredPromptFromInput below are a byte-identical copy of the page's code; the test cuts the same text out of q.html and out of this
// file and compares the strings, so a page that drifts from the copy (or still has the old rule) fails.
import { readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

class Bpe {
  constructor(vocabBytes, merges) { this.vocabBytes = vocabBytes; this.merges = merges; this.vocabSize = vocabBytes.length; }
  encode(text){
    let ids=Array.from(new TextEncoder().encode(text));
    for(const{a,b,nid}of this.merges){
      const r=[];
      for(let i=0;i<ids.length;i++){
        if(i<ids.length-1&&ids[i]===a&&ids[i+1]===b){r.push(nid);i++;}
        else r.push(ids[i]);
      }
      ids=r;
    }
    return ids;
  }
  startsWithSpace(id){
    const b=this.vocabBytes[id];
    if(!b||!b.length)return false;
    return b[0]===32;
  }
  endsOnSpace(id){
    const b=this.vocabBytes[id],c=b&&b.length?b[b.length-1]:-1;
    return c===32||(c>=9&&c<=13);
  }
  isCleanSeedToken(id){
    const b=this.vocabBytes[id];
    return !!(b&&b.length&&(b[0]===32||b[0]<97||b[0]>122));
  }
}
let bpe = null;
function anchoredPromptFromInput(text,maxTokens=4){
  const inpIds=bpe.encode(text+' '); // the input comes trimmed: the space closes its last word, else the first sampled token fuses with it ("you"+"s " -> "yous ")
  if(!inpIds.length)return null;
  const body=inpIds[inpIds.length-1]===32?inpIds.length-1:inpIds.length; // a bare space token at the tail is the one appended above, no place to start
  let start=0;
  if(!bpe.isCleanSeedToken(inpIds[start])){
    let found=false;
    for(let i=1;i<body;i++)if(bpe.startsWithSpace(inpIds[i])&&bpe.isCleanSeedToken(inpIds[i])){start=i;found=true;break;}
    if(!found)for(let i=1;i<body;i++)if(bpe.startsWithSpace(inpIds[i])){start=i;break;}
  }
  let end=Math.min(inpIds.length,start+maxTokens);
  while(end<inpIds.length&&end-start<maxTokens*8&&!bpe.endsOnSpace(inpIds[end-1]))end++; // the window ends on a space, never inside a word of the input
  return inpIds.slice(start,end);
}

const SELF = readFileSync(fileURLToPath(import.meta.url), 'utf8');
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const HTML = process.argv[2] ? resolve(process.argv[2]) : join(ROOT, 'q.html');
const html = readFileSync(HTML, 'utf8');
let failed = 0;
const check = (name, ok, detail = '') => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${detail ? '  ' + detail : ''}`); };

// a class method (two-space indent, closed by "\n  }\n") or a top-level function (closed by "\n}\n"); null when the source has none
const method = (src, name) => { const s = src.indexOf('\n  ' + name + '('); const e = s < 0 ? -1 : src.indexOf('\n  }\n', s); return e < 0 ? null : src.slice(s, e + 5); };
const func = (src, name) => { const s = src.indexOf('\nfunction ' + name + '('); const e = s < 0 ? -1 : src.indexOf('\n}\n', s + 1); return e < 0 ? null : src.slice(s, e + 3); };
for (const [what, take] of [['encode', s => method(s, 'encode')], ['startsWithSpace', s => method(s, 'startsWithSpace')], ['endsOnSpace', s => method(s, 'endsOnSpace')],
  ['isCleanSeedToken', s => method(s, 'isCleanSeedToken')], ['anchoredPromptFromInput', s => func(s, 'anchoredPromptFromInput')]]) {
  const p = take(html), c = take(SELF);
  check(`K q.html carries ${what}, byte for byte the copy in this test`, p !== null && p === c, p === null ? `q.html has no ${what}` : `page ${p.length} chars, copy ${c.length} chars`);
}

// the page before the fix, kept as the reference: the window is the first tokens of the input as typed
const legacyAnchored = (b, text, maxTokens = 4) => {
  const inpIds = b.encode(text); if (!inpIds.length) return null;
  let start = 0;
  if (!b.isCleanSeedToken(inpIds[start])) {
    let found = false;
    for (let i = 1; i < inpIds.length; i++) if (b.startsWithSpace(inpIds[i]) && b.isCleanSeedToken(inpIds[i])) { start = i; found = true; break; }
    if (!found) for (let i = 1; i < inpIds.length; i++) if (b.startsWithSpace(inpIds[i])) { start = i; break; }
  }
  return inpIds.slice(start, Math.min(inpIds.length, start + maxTokens));
};
const decode = (b, ids) => new TextDecoder('utf-8', { fatal: false }).decode(new Uint8Array(ids.flatMap(i => b.vocabBytes[i] || [])));

// vocabularies: q.merges as shipped, and the one the page learns from the 16 KB corpus of index.html
const fromMerges = buf => {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength), n = dv.getUint32(0, true), vb = [], merges = [];
  for (let i = 0; i < 256; i++) vb[i] = [i];
  for (let i = 0; i < n; i++) {
    const a = dv.getUint32(4 + i * 12, true), b = dv.getUint32(8 + i * 12, true), nid = dv.getUint32(12 + i * 12, true);
    merges.push({ a, b, nid });
    const bA = vb[a] || [a], bB = vb[b] || [b];
    if (bA.length + bB.length < 64) vb[nid] = [...bA, ...bB];
  }
  return new Bpe(vb, merges);
};
const learn = (text, nMerges) => {
  let ids = Array.from(new TextEncoder().encode(text)); const vb = [], merges = [];
  for (let i = 0; i < 256; i++) vb[i] = [i];
  for (let m = 0; m < nMerges; m++) {
    const pairs = new Map();
    for (let i = 0; i < ids.length - 1; i++) { const k = ids[i] << 16 | ids[i + 1]; pairs.set(k, (pairs.get(k) || 0) + 1); }
    if (!pairs.size) break;
    let bestK = 0, bestC = 0; for (const [k, c] of pairs) if (c > bestC) { bestC = c; bestK = k; }
    const a = bestK >> 16, b = bestK & 0xffff, nid = 256 + m;
    merges.push({ a, b, nid }); vb[nid] = [...(vb[a] || [a]), ...(vb[b] || [b])];
    const next = []; for (let i = 0; i < ids.length; i++) { if (i < ids.length - 1 && ids[i] === a && ids[i + 1] === b) { next.push(nid); i++; } else next.push(ids[i]); }
    ids = next;
  }
  return new Bpe(vb, merges);
};
const indexCorpus = readFileSync(join(ROOT, 'index.html'), 'utf8').match(/<script type="text\/plain" id="corpus">([\s\S]*?)<\/script>/)[1];
const vocabs = [['q.merges', fromMerges(readFileSync(join(ROOT, 'q.merges')))], ['16 KB corpus', learn(indexCorpus, 1024)]];

const inputs = ['hey you', 'hello', 'hello!', 'the sea', 'dreams', 'thanks!', 'who are you', 'good morning', 'nice to meet you', 'extraordinary things happen', 'tell me a story',
  'I wonder about wonderful things', 'hello, world', "don't stop believing", 'what is water', 'can you hear me?', 'привет мир', 'שלום', '你好'];
const words = s => s.split(/\s+/).filter(Boolean);
const bare = w => w.replace(/[.,;:!?)"'\]}]+$/, '');
// the continuation token t fuses with the typed text when the last word of the prompt is no longer a word of the printed line
const fuses = (b, P, t) => { const lw = bare(words(P).pop() || ''); return !!lw && !words(P + decode(b, [t])).map(bare).includes(lw); };

for (const [vname, b] of vocabs) {
  bpe = b;
  const ends = [], whole = [], empty = [], fusedNew = [], fusedOld = [], cutOld = [];
  for (const input of inputs) {
    const P = decode(b, anchoredPromptFromInput(input)), Pold = decode(b, legacyAnchored(b, input));
    if (!/\s$/.test(P)) ends.push(JSON.stringify(P));
    const inWords = words(input);
    if (!words(P).every(w => inWords.includes(w))) whole.push(JSON.stringify(P));
    if (!P.trim()) empty.push(input);
    let nNew = 0, nOld = 0;
    for (let t = 0; t < b.vocabSize; t++) { if (fuses(b, P, t)) nNew++; if (fuses(b, Pold, t)) nOld++; }
    if (nNew) fusedNew.push(`${JSON.stringify(P)}:${nNew}`);
    if (nOld) fusedOld.push(input);
    if (!words(Pold).every(w => inWords.includes(w))) cutOld.push(input);
  }
  check(`E ${vname}: the prompt ends on whitespace for all ${inputs.length} inputs`, ends.length === 0, ends.slice(0, 3).join(' '));
  check(`M ${vname}: every word of the prompt is a whole word the user typed`, whole.length === 0, whole.slice(0, 3).join(' '));
  check(`R ${vname}: the user's words are not dropped (prompt not blank)`, empty.length === 0, empty.slice(0, 3).join(' '));
  check(`F ${vname}: no vocabulary token as first continuation fuses with the typed word`, fusedNew.length === 0, fusedNew.slice(0, 3).join(' '));
  check(`T ${vname}: the old window fused on a token of the vocabulary, or cut a word, for most inputs`, fusedOld.length >= 10 && cutOld.length >= 3, `${fusedOld.length}/${inputs.length} fused, ${cutOld.length} cut`);
}

// the report: "hey you" then the page's first sampled tokens printed "hey yous conversation between water and land that has been ongoing for billions of years."
{
  bpe = vocabs[1][1];
  const cont = 's conversation between water and land that has been ongoing for billions of years.';
  const oldLine = decode(bpe, legacyAnchored(bpe, 'hey you')) + cont, newLine = decode(bpe, anchoredPromptFromInput('hey you')) + cont;
  check('X the reported line: old window prints "hey yous conversation ...", the new one keeps "you" a word',
    oldLine === 'hey yous conversation between water and land that has been ongoing for billions of years.' && words(newLine).includes('you'), `old ${JSON.stringify(oldLine.slice(0, 20))}  new ${JSON.stringify(newLine.slice(0, 20))}`);
}

// a long word has no space inside: the window stops growing at maxTokens*8 tokens instead of swallowing the room the sampler needs (out.length<256)
{
  bpe = vocabs[0][1];
  const p = anchoredPromptFromInput('x'.repeat(300));
  check('L a 300-letter word keeps the prompt at 32 tokens or fewer', p.length >= 1 && p.length <= 32, `${p.length} tokens`);
}

// the page still hands the pivot step this prompt, and the ingest of your words keeps its own unspaced encode
check('W pivot prompt is anchoredPromptFromInput(inputText,4)', html.includes('prompt=anchoredPromptFromInput(inputText,4);'));
check('W ingest still encodes the typed words as typed', html.includes('const uids=bpe.encode(inputText);'));

console.log(failed ? `${failed} failed` : 'all passed');
process.exit(failed ? 1 : 0);
