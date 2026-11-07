// Sentence ends in q.html: where a spoken sentence stops and what the page prints for it. Reads files only: no browser, no npm deps.
//   node tests/sentence_check.mjs [path/to/q.html]     (default: q.html next to tests/)
// BPE fuses a sentence end with the next opening (". It", ".\n\nThe "), generation stops on that token, and the page used to print
// it whole: "…is not always straight. It". Seg below carries the page's methods as a byte-identical copy; the test cuts the same block
// out of q.html and out of this file and compares the two strings, so a page that drifts from the copy (or still has the old rule) fails.
import { readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

class Seg {
  constructor(vocabBytes, vocabSize) { this.vocabBytes = vocabBytes; this.vocabSize = vocabSize; }
  // byte offset where a sentence ends inside token id, -1 if none: after a [.!?] run and the quotes/brackets closing it,
  // when the token ends there or whitespace follows. BPE fuses an end with the next opening (". It", ".\n\nThe "), so the
  // token that stops a sentence can carry the first bytes of the next one past this offset.
  boundaryEnd(id){
    const b=this.vocabBytes[id];if(!b)return -1;
    const term=c=>c===46||c===33||c===63,closer=c=>c===34||c===39||c===41||c===93||c===125;
    for(let i=0;i<b.length;i++){
      if(!term(b[i]))continue;
      let j=i+1;
      while(j<b.length&&(term(b[j])||closer(b[j])))j++;
      if(j===b.length||b[j]===32||b[j]===10||b[j]===13)return j;
      i=j-1;
    }
    return -1;
  }
  isBoundary(id){return this.boundaryEnd(id)>=0;}
  decodeSentence(ids){
    const bytes=[];
    for(const id of ids){const b=this.vocabBytes[id];if(b)bytes.push(...b);}
    const last=ids.length?ids[ids.length-1]:-1,e=this.boundaryEnd(last);
    if(e>=0)bytes.length-=this.vocabBytes[last].length-e;
    return new TextDecoder('utf-8',{fatal:false}).decode(new Uint8Array(bytes));
  }
}

const SELF = readFileSync(fileURLToPath(import.meta.url), 'utf8');
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const HTML = process.argv[2] ? resolve(process.argv[2]) : join(ROOT, 'q.html');
const html = readFileSync(HTML, 'utf8');
let failed = 0;
const check = (name, ok, detail = '') => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${detail ? '  ' + detail : ''}`); };

// the block of methods, from its first comment line to the closing brace of decodeSentence; null when the source has no such block
const take = src => {
  const s = src.indexOf('  // byte offset where a sentence ends inside token id');
  const d = s < 0 ? -1 : src.indexOf('  decodeSentence(ids){', s);
  const e = d < 0 ? -1 : src.indexOf('\n  }\n', d);
  return e < 0 ? null : src.slice(s, e + 5);
};
const pageBlock = take(html), copyBlock = take(SELF);
check('K q.html carries the sentence-end block, byte for byte the copy in this test', pageBlock !== null && pageBlock === copyBlock,
  pageBlock === null ? 'q.html has no boundaryEnd/decodeSentence block' : `page ${pageBlock.length} chars, copy ${copyBlock.length} chars`);

// the page before the fix, kept here as the reference: its stop rule and its print through a plain decode
const legacyBoundary = (seg, id) => {
  const b = seg.vocabBytes[id]; if (!b) return false;
  for (let i = 0; i < b.length; i++) if (b[i] === 46 || b[i] === 33 || b[i] === 63) { if (i === b.length - 1) return true; if (b[i + 1] === 32 || b[i + 1] === 10 || b[i + 1] === 13) return true; }
  return false;
};
const decodeAll = (seg, ids) => {
  const bytes = []; for (const id of ids) { const b = seg.vocabBytes[id]; if (b) bytes.push(...b); }
  return new TextDecoder('utf-8', { fatal: false }).decode(new Uint8Array(bytes));
};

// genSentence: the prompt tokens, then every sampled one; the first boundary token after step 8 ends the sentence
const speak = (isBoundary, ids, promptLen) => {
  for (let i = 0; i < ids.length; i++) if (isBoundary(ids[i]) && i - promptLen > 8) return ids.slice(0, i + 1);
  return ids;
};

// vocabularies: q.merges as shipped, the one the page learns from the 16 KB corpus, and a hand-built one with fused quote/bracket tokens
const fromMerges = buf => {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength), n = dv.getUint32(0, true), vb = [], merges = [];
  for (let i = 0; i < 256; i++) vb[i] = [i];
  for (let i = 0; i < n; i++) {
    const a = dv.getUint32(4 + i * 12, true), b = dv.getUint32(8 + i * 12, true), nid = dv.getUint32(12 + i * 12, true);
    merges.push({ a, b, nid });
    const bA = vb[a] || [a], bB = vb[b] || [b];
    if (bA.length + bB.length < 64) vb[nid] = [...bA, ...bB];
  }
  return { seg: new Seg(vb, 256 + n), merges };
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
  return { seg: new Seg(vb, 256 + merges.length), merges };
};
const encode = (text, merges) => {
  let ids = Array.from(new TextEncoder().encode(text));
  for (const { a, b, nid } of merges) { const r = []; for (let i = 0; i < ids.length; i++) { if (i < ids.length - 1 && ids[i] === a && ids[i + 1] === b) { r.push(nid); i++; } else r.push(ids[i]); } ids = r; }
  return ids;
};
const real = fromMerges(readFileSync(join(ROOT, 'q.merges')));
const indexCorpus = readFileSync(join(ROOT, 'index.html'), 'utf8').match(/<script type="text\/plain" id="corpus">([\s\S]*?)<\/script>/)[1];
const learned = learn(indexCorpus, 1024);
const toy = new Seg(Array.from({ length: 256 }, (_, i) => [i]), 256);
const toyIds = parts => parts.map(p => { const bytes = [...new TextEncoder().encode(p)]; if (bytes.length === 1) return bytes[0]; toy.vocabBytes[toy.vocabSize] = bytes; return toy.vocabSize++; });

const cases = [
  ['straight. It', real, 'The road along the river is not always straight, and it turns a little more with every season of rain. It bends toward the sea.', 'The road along the river is not always straight, and it turns a little more with every season of rain.'],
  ['distance. S', real, 'The light of a star that crosses the dark between worlds diminishes with distance and with time. Some stars fade first.', 'The light of a star that crosses the dark between worlds diminishes with distance and with time.'],
  ['away. L', real, 'The nearest star is about four light-years away. Light takes years to arrive.', 'The nearest star is about four light-years away.'],
  ['paragraph break', real, 'Most of the surface of the planet is covered by deep water in the oceans and the seas.\n\nThe rest is land.', 'Most of the surface of the planet is covered by deep water in the oceans and the seas.'],
  ['abbreviation inside the sentence', real, 'Dr. Smith measured the field strength of the planet slowly, then wrote it down. Rain fell.', 'Dr. Smith measured the field strength of the planet slowly, then wrote it down.'],
  ['ellipsis', { seg: toy }, ['The', ' tide', ' goes', ' out', ' and', ' the', ' shore', ' keeps', ' waiting', '... It', ' comes', ' back'], 'The tide goes out and the shore keeps waiting...'],
  ['question and closing quote', { seg: toy }, ['He', ' asked', ' ', '"', 'Is', ' it', ' really', ' very', ' far', '?" It', ' was', ' far', '.'], 'He asked "Is it really very far?"'],
  ['bracket after the period', { seg: toy }, ['The', ' old', ' map', ' of', ' the', ' coast', ' (', 'see', ' the', ' map', '.) The', ' road', ' ends', '.'], 'The old map of the coast (see the map.)'],
];
const olds = {};
for (const [name, v, input, want] of cases) {
  const ids = typeof input === 'string' ? encode(input, v.merges) : toyIds(input), promptLen = typeof input === 'string' ? 5 : 0;
  const old = decodeAll(v.seg, speak(id => legacyBoundary(v.seg, id), ids, promptLen));
  const got = v.seg.decodeSentence(speak(id => v.seg.isBoundary(id), ids, promptLen));
  olds[name] = old;
  check(`S ${name}`, got === want, `new ${JSON.stringify(got)}  old ${JSON.stringify(old)}`);
}
// the old page printed the next sentence's opening; the cases are traps only if it did
check('T old page prints the tail: "straight. It" and "away. L"', olds['straight. It'] !== 'The road along the river is not always straight.' && olds['away. L'] !== 'The nearest star is about four light-years away.');
const live = cases.filter(([n, , , want]) => olds[n] !== want).length;
check('T the old rule fails at least six of the eight cases', live >= 6, `${live}/8`);

// the fix changes what is printed, not when q stops: same boundary tokens as the old rule on both shipped vocabularies
for (const [name, v] of [['q.merges', real], ['16 KB corpus', learned]]) {
  const diff = []; let boundary = 0, tail = 0;
  for (let id = 0; id < v.seg.vocabSize; id++) {
    if (v.seg.isBoundary(id) !== legacyBoundary(v.seg, id)) diff.push(id);
    if (legacyBoundary(v.seg, id)) { boundary++; if (/[.!?]["')\]]*\s+\S/.test(decodeAll(v.seg, [id]))) tail++; }
  }
  check(`P ${name}: stop set equals the old [.!?] rule`, diff.length === 0, `${v.seg.vocabSize} tokens, ${boundary} boundary, ${tail} carry a next-sentence tail, ${diff.length} differ`);
}

// a sentence the cap cut short has no boundary token to trim
const capped = toyIds(['no', ' end', ' here']);
check('N capped sentence is printed whole', toy.decodeSentence(capped) === 'no end here', JSON.stringify(toy.decodeSentence(capped)));

// the page prints through decodeSentence at both places a step is written, and q still stops where the cases assume
check('W chain prints decodeSentence', /const text=bpe\.decodeSentence\(/.test(html) && /const rText=bpe\.decodeSentence\(/.test(html));
check('W generation stops on bpe.isBoundary(ch)&&step>8', html.includes('if(bpe.isBoundary(ch)&&step>8)break;'));

console.log(failed ? `${failed} failed` : 'all passed');
process.exit(failed ? 1 : 0);
