"""
postgpt_q.py — PostGPT-Q: Resonant Reasoning Engine (Python inference)

Faithful port of postgpt_q.c. Same architecture, same constants, same pipeline.
Triple attention: Content (QK^T) + RRPRAM (x@Wr) + Janus echo (W^T·W)
Dario equation: bigram + trigram + hebbian + destiny.
Transformer gate: untrained = silent, trained = speaks.
6 Kuramoto chambers. Calendar drift. Schumann resonance. 12 bidirectional steps.

python3 postgpt_q.py [weights.bin] q.merges q.txt

(c) 2026 arianna method
resonance is unbreakable.
"""

import math
import random
import struct
import os
import re
import sys
import time

# ── constants ──
MAX_VOCAB   = 1280
MAX_CTX     = 128
MAX_BPE     = 1024
MAX_SEQ     = 4096
MAX_BIGRAM  = 65536
MAX_TRIGRAM = 65536
MAX_HEBBIAN = 131072
MAX_PROPHECY = 32
N_CHAMBERS  = 6
CHAIN_STEPS = 12
TOP_K       = 15
MAX_PERIODIC = 4096
QPTQ_MAGIC  = 0x51505451
QMEM_SOMA   = 0x414D4F53
SPA_DIM     = 32
SPA_NH      = 4
SPA_HD      = SPA_DIM // SPA_NH
MAX_EXPERTS = 16
DOE_RANK    = 4
DOE_ALPHA   = 0.05
VEL_WALK    = 0
VEL_RUN     = 1
VEL_STOP    = 2
VEL_BREATHE = 3
VEL_UP      = 4
VEL_DOWN    = 5
VEL_N = ["WALK", "RUN", "STOP", "BREATHE", "UP", "DOWN"]

# ── chamber indices ──
CH_FEAR  = 0
CH_LOVE  = 1
CH_RAGE  = 2
CH_VOID  = 3
CH_FLOW  = 4
CH_CMPLX = 5
CH_N = ["FEAR", "LOVE", "RAGE", "VOID", "FLOW", "CMPLX"]
CH_D = [0.90, 0.93, 0.85, 0.97, 0.88, 0.94]
COU = [
    [ 0.0, -0.3,  0.5,  0.4, -0.2,  0.1],
    [-0.3,  0.0, -0.4, -0.5,  0.5,  0.2],
    [ 0.5, -0.3,  0.0,  0.2, -0.3,  0.3],
    [ 0.4, -0.5,  0.3,  0.0, -0.3,  0.4],
    [-0.2,  0.4, -0.2, -0.3,  0.0,  0.3],
    [ 0.1,  0.2,  0.3,  0.4,  0.3,  0.0],
]

ANCHORS = {
    "fear": CH_FEAR, "terror": CH_FEAR, "panic": CH_FEAR, "threat": CH_FEAR,
    "danger": CH_FEAR, "horror": CH_FEAR, "dread": CH_FEAR, "alarm": CH_FEAR,
    "love": CH_LOVE, "warmth": CH_LOVE, "gentle": CH_LOVE, "care": CH_LOVE,
    "heart": CH_LOVE, "mother": CH_LOVE, "child": CH_LOVE, "touch": CH_LOVE,
    "embrace": CH_LOVE, "tenderness": CH_LOVE, "affection": CH_LOVE,
    "rage": CH_RAGE, "fury": CH_RAGE, "anger": CH_RAGE, "fire": CH_RAGE,
    "war": CH_RAGE, "hate": CH_RAGE, "destroy": CH_RAGE, "burn": CH_RAGE,
    "violence": CH_RAGE, "storm": CH_RAGE, "fight": CH_RAGE,
    "nothing": CH_VOID, "silence": CH_VOID, "empty": CH_VOID, "void": CH_VOID,
    "darkness": CH_VOID, "shadow": CH_VOID, "death": CH_VOID, "cold": CH_VOID,
    "lost": CH_VOID, "forgotten": CH_VOID, "absence": CH_VOID, "alone": CH_VOID,
    "flow": CH_FLOW, "rhythm": CH_FLOW, "wave": CH_FLOW, "dance": CH_FLOW,
    "pulse": CH_FLOW, "breath": CH_FLOW, "emergence": CH_FLOW, "harmony": CH_FLOW,
    "resonance": CH_FLOW, "coherence": CH_FLOW, "synchronize": CH_FLOW,
    "paradox": CH_CMPLX, "contradiction": CH_CMPLX, "tension": CH_CMPLX,
    "chaos": CH_CMPLX, "mystery": CH_CMPLX, "transform": CH_CMPLX,
    "strange": CH_CMPLX, "ambiguity": CH_CMPLX, "uncertain": CH_CMPLX,
}
SOMATIC_SEEDS = {
    "pulse":      [0.4, 0.0, 0.8, 0.0, 0.3, 0.2],
    "tremor":     [0.8, 0.0, 0.2, 0.2, 0.0, 0.3],
    "burning":    [0.3, 0.1, 0.9, 0.0, 0.1, 0.2],
    "clenching":  [0.4, 0.0, 0.8, 0.1, 0.0, 0.3],
    "tingling":   [0.5, 0.2, 0.1, 0.0, 0.4, 0.5],
    "throbbing":  [0.3, 0.0, 0.7, 0.1, 0.3, 0.2],
    "aching":     [0.2, 0.1, 0.2, 0.7, 0.0, 0.3],
    "tightness":  [0.6, 0.0, 0.5, 0.3, 0.0, 0.3],
    "sinking":    [0.5, 0.0, 0.0, 0.9, 0.0, 0.2],
    "nausea":     [0.5, 0.0, 0.2, 0.7, 0.0, 0.3],
    "heaviness":  [0.2, 0.0, 0.1, 0.8, 0.0, 0.2],
    "weakness":   [0.5, 0.0, 0.0, 0.7, 0.0, 0.2],
    "shaking":    [0.8, 0.0, 0.4, 0.1, 0.0, 0.3],
    "freezing":   [0.7, 0.0, 0.0, 0.5, 0.0, 0.2],
    "sweating":   [0.6, 0.0, 0.3, 0.1, 0.2, 0.2],
    "warmth":     [0.0, 0.9, 0.0, 0.0, 0.6, 0.1],
    "softness":   [0.0, 0.8, 0.0, 0.0, 0.5, 0.2],
    "floating":   [0.0, 0.3, 0.0, 0.2, 0.8, 0.3],
    "pressure":   [0.4, 0.0, 0.5, 0.5, 0.0, 0.4],
    "vibrating":  [0.2, 0.1, 0.3, 0.0, 0.4, 0.8],
    "chest":      [0.4, 0.5, 0.4, 0.3, 0.2, 0.3],
    "throat":     [0.6, 0.2, 0.3, 0.5, 0.0, 0.3],
    "stomach":    [0.5, 0.1, 0.3, 0.6, 0.1, 0.3],
    "jaw":        [0.2, 0.0, 0.9, 0.1, 0.0, 0.2],
    "fists":      [0.1, 0.0, 0.9, 0.0, 0.0, 0.2],
    "spine":      [0.7, 0.0, 0.2, 0.2, 0.1, 0.4],
    "temples":    [0.4, 0.0, 0.3, 0.3, 0.0, 0.6],
    "shoulders":  [0.3, 0.0, 0.4, 0.4, 0.0, 0.3],
}
def extract_words(text):
    return re.findall(r"[a-z']+", text.lower())
# ── math ──
def clampf(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)
def rmsnorm(x, n):
    ms = sum(x[i] * x[i] for i in range(n))
    ms = 1.0 / math.sqrt(ms / n + 1e-6)
    return [x[i] * ms for i in range(n)]
def matmul(x, w, n_in, d_out):
    out = [0.0] * d_out
    for d in range(d_out):
        v = 0.0
        base = d * n_in
        for j in range(n_in):
            v += x[j] * w[base + j]
        out[d] = v
    return out
def softmax_inplace(x, n):
    mx = max(x[:n])
    s = 0.0
    for i in range(n):
        x[i] = math.exp(x[i] - mx)
        s += x[i]
    if s > 0:
        for i in range(n):
            x[i] /= s
def sample_nucleus(logits, V, temp, top_p):
    idx = [0] * TOP_K
    val = [-1e30] * TOP_K
    for i in range(V):
        if logits[i] > val[TOP_K - 1]:
            val[TOP_K - 1] = logits[i]
            idx[TOP_K - 1] = i
            for k in range(TOP_K - 2, -1, -1):
                if val[k + 1] > val[k]:
                    val[k], val[k + 1] = val[k + 1], val[k]
                    idx[k], idx[k + 1] = idx[k + 1], idx[k]
                else:
                    break
    mx = val[0]
    pr = [0.0] * TOP_K
    tot = 0.0
    for k in range(TOP_K):
        pr[k] = math.exp((val[k] - mx) / temp)
        tot += pr[k]
    cum = 0.0
    nk = TOP_K
    for k in range(TOP_K):
        cum += pr[k] / tot
        if cum >= top_p:
            nk = k + 1
            break
    ntot = sum(pr[:nk])
    r = random.random() * ntot
    cum = 0.0
    for k in range(nk):
        cum += pr[k]
        if cum >= r:
            return idx[k]
    return idx[0]
# ── BPE ──
class BPE:
    def __init__(self):
        self.merges = []          # list of (a, b, new_id)
        self.n_merges = 0
        self.vocab_size = 0
        self.vocab_bytes = {}     # id -> bytes
        self.vocab_len = {}       # id -> int
def bpe_load(bpe, path):
    try:
        f = open(path, "rb")
    except OSError:
        print("ERROR: " + path, file=sys.stderr)
        return False
    data = f.read(4)
    n = struct.unpack("<I", data)[0]
    bpe.n_merges = n
    bpe.vocab_size = 256 + n
    for i in range(256):
        bpe.vocab_bytes[i] = bytes([i])
        bpe.vocab_len[i] = 1
    for i in range(min(n, MAX_BPE)):
        a, b, nid = struct.unpack("<III", f.read(12))
        bpe.merges.append((a, b, nid))
        ba = bpe.vocab_bytes.get(a, b"")
        bb = bpe.vocab_bytes.get(b, b"")
        if len(ba) + len(bb) < 64:
            bpe.vocab_bytes[nid] = ba + bb
            bpe.vocab_len[nid] = len(ba) + len(bb)
    f.close()
    return True
def bpe_encode(bpe, text_bytes, tlen, maxo):
    out = []
    for i in range(min(tlen, maxo)):
        out.append(text_bytes[i])
    for m in range(bpe.n_merges):
        a, b, nid = bpe.merges[m]
        j = 0
        new_out = []
        i = 0
        n = len(out)
        while i < n:
            if i < n - 1 and out[i] == a and out[i + 1] == b:
                new_out.append(nid)
                i += 2
            else:
                new_out.append(out[i])
                i += 1
        out = new_out
    return out
def bpe_decode_token(bpe, tid):
    if tid < 0 or tid >= bpe.vocab_size:
        return ""
    b = bpe.vocab_bytes.get(tid, b"")
    try:
        return b.decode("utf-8", errors="replace")
    except Exception:
        return b.decode("latin-1")
# ── MetaWeights ──
class MetaW:
    def __init__(self):
        self.unigram = [0.0] * MAX_VOCAB
        self.bigrams = []     # list of [a, b, prob]
        self.n_bi = 0
        self.trigrams = []    # list of [a, b, c, prob]
        self.n_tri = 0
        self.hebbs = []       # list of [a, b, str]
        self.n_hebb = 0
        self.prophecies = []  # list of [target, strength, age]
class PeriodicTable:
    def __init__(self):
        self.elements = {}
        for word, chamber in ANCHORS.items():
            self.elements[word] = {"ch": chamber, "mass": 0.6}

    def discover(self, words, idx, window=4):
        word = words[idx]
        if word in self.elements:
            return
        profile = [0.0] * N_CHAMBERS
        total = 0.0
        lo = max(0, idx - window)
        hi = min(len(words), idx + window + 1)
        for j in range(lo, hi):
            if j == idx:
                continue
            neighbor = words[j]
            if neighbor not in self.elements:
                continue
            el = self.elements[neighbor]
            decay = 1.0 / (1.0 + abs(j - idx))
            profile[el["ch"]] += el["mass"] * decay
            total += decay
        if total <= 0.1:
            return
        dom = max(range(N_CHAMBERS), key=lambda i: profile[i])
        mass = min(profile[dom] / total, 0.8)
        if mass > 0.05:
            self.elements[word] = {"ch": dom, "mass": mass}

    def build_from_text(self, text):
        words = extract_words(text)
        for idx in range(len(words)):
            self.discover(words, idx)

    def classify(self, word):
        return self.elements.get(word.lower())
def meta_build(mw, ids, n, V):
    for i in range(n):
        if ids[i] < V:
            mw.unigram[ids[i]] += 1.0
    tot = sum(mw.unigram[:V])
    if tot > 0:
        for i in range(V):
            mw.unigram[i] /= tot

    # bigram
    bc = []  # list of [a, b, count]
    for i in range(n - 1):
        if len(bc) >= MAX_BIGRAM - 1:
            break
        a, b = ids[i], ids[i + 1]
        found = False
        for j in range(len(bc)):
            if bc[j][0] == a and bc[j][1] == b:
                bc[j][2] += 1
                found = True
                break
        if not found:
            bc.append([a, b, 1])
    for i in range(len(bc)):
        t = 0.0
        for j in range(len(bc)):
            if bc[j][0] == bc[i][0]:
                t += bc[j][2]
        if t > 0:
            mw.bigrams.append([bc[i][0], bc[i][1], bc[i][2] / t])
            mw.n_bi += 1

    # trigram
    tc = []  # list of [a, b, c, count]
    for i in range(n - 2):
        if len(tc) >= MAX_TRIGRAM - 1:
            break
        a, b, c = ids[i], ids[i + 1], ids[i + 2]
        found = False
        for j in range(len(tc)):
            if tc[j][0] == a and tc[j][1] == b and tc[j][2] == c:
                tc[j][3] += 1
                found = True
                break
        if not found:
            tc.append([a, b, c, 1])
    for i in range(len(tc)):
        if mw.n_tri >= MAX_TRIGRAM:
            break
        t = 0.0
        for j in range(len(tc)):
            if tc[j][0] == tc[i][0] and tc[j][1] == tc[i][1]:
                t += tc[j][3]
        if t > 0:
            mw.trigrams.append([tc[i][0], tc[i][1], tc[i][2], tc[i][3] / t])
            mw.n_tri += 1

    # hebbian
    hn = min(n, 8000)
    win = 5
    for i in range(hn):
        if mw.n_hebb >= MAX_HEBBIAN - 1:
            break
        jstart = max(0, i - win)
        jend = min(hn, i + win + 1)
        for j in range(jstart, jend):
            if i == j:
                continue
            a = min(ids[i], ids[j])
            b = max(ids[i], ids[j])
            decay = 1.0 / (1.0 + abs(i - j))
            found = False
            for k in range(mw.n_hebb):
                if mw.hebbs[k][0] == a and mw.hebbs[k][1] == b:
                    mw.hebbs[k][2] += decay
                    found = True
                    break
            if not found and mw.n_hebb < MAX_HEBBIAN - 1:
                mw.hebbs.append([a, b, decay])
                mw.n_hebb += 1
    mx = 0.0
    for i in range(mw.n_hebb):
        if mw.hebbs[i][2] > mx:
            mx = mw.hebbs[i][2]
    if mx > 0:
        for i in range(mw.n_hebb):
            mw.hebbs[i][2] /= mx
    print("  metaweights: %d bi, %d tri, %d hebb" % (mw.n_bi, mw.n_tri, mw.n_hebb))
def meta_bi(mw, prev, nxt):
    for i in range(mw.n_bi):
        if mw.bigrams[i][0] == prev and mw.bigrams[i][1] == nxt:
            return mw.bigrams[i][2]
    return 1e-10
def meta_tri(mw, p2, p1, nxt):
    for i in range(mw.n_tri):
        if mw.trigrams[i][0] == p2 and mw.trigrams[i][1] == p1 and mw.trigrams[i][2] == nxt:
            return mw.trigrams[i][3]
    return 1e-10
def meta_hebb(mw, ctx, cl, V):
    out = [0.0] * V
    for ci in range(cl):
        c = ctx[ci]
        for k in range(mw.n_hebb):
            if mw.hebbs[k][0] == c and mw.hebbs[k][1] < V:
                out[mw.hebbs[k][1]] += mw.hebbs[k][2]
            elif mw.hebbs[k][1] == c and mw.hebbs[k][0] < V:
                out[mw.hebbs[k][0]] += mw.hebbs[k][2]
    mx = 0.0
    for i in range(V):
        if out[i] > mx:
            mx = out[i]
    if mx > 0:
        for i in range(V):
            out[i] /= mx
    return out
def meta_prophecy(mw, ctx, cl, V):
    out = [0.0] * V
    appeared = [0] * 256
    na = min(cl, 256)
    for i in range(cl - na, cl):
        if ctx[i] < 256:
            appeared[ctx[i]] = 1
    start = max(0, cl - 12)  # extended window: 12 tokens back instead of 4
    for ci in range(start, cl):
        c = ctx[ci]
        decay = 1.0 / (1.0 + float(cl - 1 - ci))  # recent tokens contribute more
        for k in range(mw.n_bi):
            if mw.bigrams[k][0] == c and mw.bigrams[k][1] < V and not appeared[mw.bigrams[k][1] % 256]:
                out[mw.bigrams[k][1]] += mw.bigrams[k][2] * decay
    # trigram prophecy: predict from last 2 tokens as pair context
    if cl >= 2:
        p0, p1 = ctx[cl - 2], ctx[cl - 1]
        for k in range(mw.n_tri):
            if (mw.trigrams[k][0] == p0 and mw.trigrams[k][1] == p1
                    and mw.trigrams[k][2] < V and not appeared[mw.trigrams[k][2] % 256]):
                out[mw.trigrams[k][2]] += mw.trigrams[k][3] * 1.5  # trigrams are more specific
    for target, strength, age in mw.prophecies:
        if 0 <= target < V and not appeared[target % 256]:
            out[target] += strength * math.log1p(float(age))
    mx = 0.0
    for i in range(V):
        if out[i] > mx:
            mx = out[i]
    if mx > 0:
        for i in range(V):
            out[i] /= mx
    return out
def prophecy_add(mw, target, strength):
    if target < 0:
        return
    for item in mw.prophecies:
        if item[0] == target:
            item[1] = max(item[1], strength)
            item[2] = 0
            return
    if len(mw.prophecies) >= MAX_PROPHECY:
        oldest = max(range(len(mw.prophecies)), key=lambda i: mw.prophecies[i][2])
        mw.prophecies.pop(oldest)
    mw.prophecies.append([target, strength, 0])
def prophecy_update(mw, token):
    kept = []
    for target, strength, age in mw.prophecies:
        if target == token:
            continue
        age += 1
        strength *= 0.995
        if age < 50 and strength > 0.01:
            kept.append([target, strength, age])
    mw.prophecies = kept
def prophecy_pressure(mw):
    total = 0.0
    for _target, strength, age in mw.prophecies:
        total += strength * math.log1p(float(age))
    return clampf(total / 4.0, 0.0, 1.0)
def ingest_ids(mw, ids, amount=0.02):
    ulen = len(ids)
    if ulen <= 1:
        return
    for i in range(ulen - 1):
        a, b = ids[i], ids[i + 1]
        found = False
        for j in range(mw.n_bi):
            if mw.bigrams[j][0] == a and mw.bigrams[j][1] == b:
                mw.bigrams[j][2] += amount
                found = True
                break
        if not found and mw.n_bi < MAX_BIGRAM:
            mw.bigrams.append([a, b, max(0.05, amount)])
            mw.n_bi += 1
    for i in range(ulen - 2):
        a, b, c = ids[i], ids[i + 1], ids[i + 2]
        found = False
        for j in range(mw.n_tri):
            if mw.trigrams[j][0] == a and mw.trigrams[j][1] == b and mw.trigrams[j][2] == c:
                mw.trigrams[j][3] += amount
                found = True
                break
        if not found and mw.n_tri < MAX_TRIGRAM:
            mw.trigrams.append([a, b, c, max(0.05, amount)])
            mw.n_tri += 1
    for i in range(ulen):
        for j in range(max(0, i - 6), min(ulen, i + 7)):
            if i == j:
                continue
            a = min(ids[i], ids[j])
            b = max(ids[i], ids[j])
            decay = 1.0 / (1.0 + abs(i - j))
            found = False
            for k in range(mw.n_hebb):
                if mw.hebbs[k][0] == a and mw.hebbs[k][1] == b:
                    mw.hebbs[k][2] += decay * amount * 0.5
                    found = True
                    break
            if not found and mw.n_hebb < MAX_HEBBIAN:
                mw.hebbs.append([a, b, decay * max(0.01, amount)])
                mw.n_hebb += 1
def load_memory(mw, path, periodic=None, chambers=None):
    if not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as mf:
            magic = struct.unpack("<I", mf.read(4))[0]
            if magic != 0x514D454D:
                return False
            nb_m, nt_m, nh_m = struct.unpack("<3i", mf.read(12))
            for _ in range(min(nb_m, MAX_BIGRAM)):
                a, b, p = struct.unpack("<2if", mf.read(12))
                ingest_ids(mw, [a, b], 0.0)
                for item in mw.bigrams:
                    if item[0] == a and item[1] == b:
                        item[2] = max(item[2], p)
                        break
            for _ in range(min(nt_m, MAX_TRIGRAM)):
                a, b, c, p = struct.unpack("<3if", mf.read(16))
                found = False
                for item in mw.trigrams:
                    if item[0] == a and item[1] == b and item[2] == c:
                        item[3] = max(item[3], p)
                        found = True
                        break
                if not found and mw.n_tri < MAX_TRIGRAM:
                    mw.trigrams.append([a, b, c, p])
                    mw.n_tri += 1
            for _ in range(min(nh_m, MAX_HEBBIAN)):
                a, b, p = struct.unpack("<2if", mf.read(12))
                found = False
                for item in mw.hebbs:
                    if item[0] == a and item[1] == b:
                        item[2] = max(item[2], p)
                        found = True
                        break
                if not found and mw.n_hebb < MAX_HEBBIAN:
                    mw.hebbs.append([a, b, p])
                    mw.n_hebb += 1
            # load periodic table elements
            try:
                npe_data = mf.read(4)
                if len(npe_data) == 4 and periodic is not None:
                    npe = struct.unpack("<I", npe_data)[0]
                    if npe > 0 and npe <= MAX_PERIODIC:
                        for _ in range(npe):
                            wlen = struct.unpack("B", mf.read(1))[0]
                            if wlen > 31:
                                wlen = 31
                            w = mf.read(wlen).decode("utf-8", errors="replace")
                            chamber = struct.unpack("B", mf.read(1))[0]
                            mass = struct.unpack("<f", mf.read(4))[0]
                            if chamber < 6:
                                periodic.elements[w] = {"ch": chamber, "mass": mass}
                        print("  [periodic: %d elements loaded]" % len(periodic.elements))
            except Exception:
                pass  # no periodic data in older memory files
            try:
                soma_tag = mf.read(4)
                if len(soma_tag) == 4 and chambers is not None:
                    if struct.unpack("<I", soma_tag)[0] == QMEM_SOMA:
                        soma_vals = struct.unpack("<6f", mf.read(24))
                        presence, debt, trauma = struct.unpack("<3f", mf.read(12))
                        chambers.soma = [clampf(v, 0.0, 1.0) for v in soma_vals]
                        chambers.presence = clampf(presence, 0.0, 1.0)
                        chambers.debt = clampf(max(chambers.debt, debt), 0.0, 1.0)
                        chambers.trauma = clampf(max(chambers.trauma, trauma), 0.0, 1.0)
                        for i in range(N_CHAMBERS):
                            chambers.act[i] = clampf(max(chambers.act[i], 0.25 * chambers.soma[i]), 0.0, 1.0)
            except Exception:
                pass
        return True
    except Exception:
        return False
def save_memory(mw, path, periodic=None, chambers=None):
    with open(path, "wb") as mf:
        mf.write(struct.pack("<I", 0x514D454D))
        mf.write(struct.pack("<3i", mw.n_bi, mw.n_tri, mw.n_hebb))
        for i in range(mw.n_bi):
            mf.write(struct.pack("<2if", mw.bigrams[i][0], mw.bigrams[i][1], mw.bigrams[i][2]))
        for i in range(mw.n_tri):
            mf.write(struct.pack("<3if", mw.trigrams[i][0], mw.trigrams[i][1], mw.trigrams[i][2], mw.trigrams[i][3]))
        for i in range(mw.n_hebb):
            mf.write(struct.pack("<2if", mw.hebbs[i][0], mw.hebbs[i][1], mw.hebbs[i][2]))
        # save periodic table
        if periodic is not None:
            elements = list(periodic.elements.items())
            mf.write(struct.pack("<I", len(elements)))
            for word, elem in elements:
                wbytes = word.encode("utf-8")[:31]
                mf.write(struct.pack("B", len(wbytes)))
                mf.write(wbytes)
                mf.write(struct.pack("B", elem["ch"]))
                mf.write(struct.pack("<f", elem["mass"]))
        else:
            mf.write(struct.pack("<I", 0))
        if chambers is not None:
            mf.write(struct.pack("<I", QMEM_SOMA))
            mf.write(struct.pack("<6f", *[clampf(v, 0.0, 1.0) for v in chambers.soma]))
            mf.write(struct.pack("<3f", clampf(chambers.presence, 0.0, 1.0), clampf(chambers.debt, 0.0, 1.0), clampf(chambers.trauma, 0.0, 1.0)))
# ── Chambers ──
class Chambers:
    def __init__(self):
        self.act = [0.0] * 6
        self.soma = [0.0] * 6
        self.debt = 0.0
        self.trauma = 0.0
        self.presence = 0.0

    def feel(self, text, periodic=None):
        soma_hits = 0
        soma_mix = [0.0] * 6
        for word in extract_words(text):
            anchor = ANCHORS.get(word)
            if anchor is not None:
                self.act[anchor] += 0.15
            if periodic is not None:
                el = periodic.classify(word)
                if el is not None:
                    self.act[el["ch"]] += 0.08 * el["mass"]
            seed = SOMATIC_SEEDS.get(word)
            if seed is not None:
                soma_hits += 1
                for i in range(N_CHAMBERS):
                    soma_mix[i] += seed[i]
        if soma_hits > 0:
            inv = 1.0 / soma_hits
            for i in range(N_CHAMBERS):
                avg = soma_mix[i] * inv
                self.soma[i] = clampf(0.82 * self.soma[i] + 0.18 * avg, 0.0, 1.0)
                self.act[i] += 0.06 * avg
            intensity = clampf(sum(soma_mix) * inv / 2.4, 0.0, 1.0)
            self.presence = clampf(0.86 * self.presence + 0.14 * intensity, 0.0, 1.0)
        else:
            for i in range(N_CHAMBERS):
                self.soma[i] *= 0.98
            self.presence *= 0.99
        somatic_trauma = 0.45 * self.soma[CH_FEAR] + 0.35 * self.soma[CH_RAGE] + 0.20 * self.soma[CH_VOID]
        somatic_debt = 0.35 * self.soma[CH_CMPLX] + 0.25 * self.soma[CH_FLOW] + 0.20 * self.presence
        self.trauma = clampf(0.92 * self.trauma + 0.08 * somatic_trauma, 0.0, 1.0)
        self.debt = clampf(0.96 * self.debt + 0.04 * somatic_debt, 0.0, 1.0)
        for i in range(N_CHAMBERS):
            self.act[i] = clampf(self.act[i], 0.0, 1.0)

    def dominant(self):
        return max(range(N_CHAMBERS), key=lambda i: self.act[i])

    def emergence(self):
        return (1.0 - max(self.act[CH_VOID], 0.10)) * min(self.act[CH_FLOW], 0.95)

    def modulate(self):
        a = clampf(1.0 + 0.4 * self.act[CH_LOVE] - 0.2 * self.act[CH_RAGE] + 0.3 * self.act[CH_FLOW], 0.3, 2.0)
        b = clampf(1.0 + 0.4 * self.act[CH_FLOW] - 0.2 * self.act[CH_FEAR], 0.3, 2.0)
        g = clampf(1.0 + 0.5 * self.act[CH_CMPLX] + 0.2 * self.act[CH_LOVE] - 0.1 * self.act[CH_VOID], 0.3, 2.0)
        t = clampf(1.0 - 0.2 * self.act[CH_FLOW] + 0.1 * self.act[CH_FEAR], 0.3, 2.0)
        a *= clampf(1.0 + 0.14 * self.soma[CH_LOVE] + 0.08 * self.soma[CH_FLOW] + 0.05 * self.presence, 0.7, 1.5)
        b *= clampf(1.0 + 0.10 * self.soma[CH_FLOW] + 0.08 * self.soma[CH_CMPLX] + 0.04 * self.presence, 0.7, 1.5)
        g *= clampf(1.0 + 0.10 * self.soma[CH_CMPLX] + 0.05 * self.soma[CH_VOID] + 0.06 * self.presence, 0.7, 1.5)
        t *= clampf(1.0 - 0.10 * self.soma[CH_FLOW] + 0.08 * self.soma[CH_FEAR] + 0.06 * self.soma[CH_RAGE], 0.7, 1.5)
        a = clampf(a, 0.3, 2.0)
        b = clampf(b, 0.3, 2.0)
        g = clampf(g, 0.3, 2.0)
        t = clampf(t, 0.3, 2.0)
        return a, b, g, t

    def summary(self):
        parts = []
        for i in range(N_CHAMBERS):
            if self.act[i] > 0.05:
                parts.append("%s:%.0f%%" % (CH_N[i], self.act[i] * 100.0))
        if self.presence > 0.05:
            parts.append("SOMA:%.0f%%" % (self.presence * 100.0))
        return " ".join(parts) if parts else "quiet"
def ch_init(c):
    c.act = [0.0] * 6
    c.soma = [0.0] * 6
    c.act[CH_LOVE] = 0.2
    c.act[CH_FLOW] = 0.15
    c.debt = 0.0
    c.trauma = 0.0
    c.presence = 0.0
def ch_xfire(c, it):
    for _ in range(it):
        old = list(c.act)
        for i in range(6):
            c.act[i] *= CH_D[i]
            for j in range(6):
                if i != j:
                    c.act[i] += 0.03 * COU[i][j] * math.sin(old[j] - old[i])
            c.act[i] = clampf(c.act[i], 0.0, 1.0)
            c.soma[i] = clampf(0.94 * c.soma[i] + 0.02 * c.act[i], 0.0, 1.0)
        c.presence = clampf(0.95 * c.presence + 0.03 * c.emergence(), 0.0, 1.0)
def velocity_profile(ch, dissonance):
    mode = VEL_WALK
    if dissonance > 0.8:
        mode = VEL_UP
    elif dissonance > 0.6:
        mode = VEL_RUN
    elif dissonance < 0.2:
        mode = VEL_STOP
    elif ch.trauma > 0.5:
        mode = VEL_BREATHE
    elif ch.debt > 0.55:
        mode = VEL_DOWN

    prof = {
        "mode": mode,
        "name": VEL_N[mode],
        "temp_mul": 1.0,
        "heb_mul": 1.0,
        "pro_mul": 1.0,
        "ds_mul": 1.0,
        "bg_mul": 1.0,
        "tg_mul": 1.0,
        "interf_bonus": 0.0,
        "wormhole_bonus": 0.0,
        "debt_decay": 1.0,
        "trauma_decay": 1.0,
    }
    if mode == VEL_RUN:
        prof["temp_mul"] = 1.12
        prof["bg_mul"] = 1.15
        prof["interf_bonus"] = 0.05
    elif mode == VEL_STOP:
        prof["temp_mul"] = 0.72
        prof["ds_mul"] = 1.25
        prof["debt_decay"] = 0.75
    elif mode == VEL_BREATHE:
        prof["temp_mul"] = 0.9
        prof["debt_decay"] = 0.65
        prof["trauma_decay"] = 0.75
    elif mode == VEL_UP:
        prof["temp_mul"] = 1.22
        prof["pro_mul"] = 1.25
        prof["bg_mul"] = 0.9
        prof["wormhole_bonus"] = 0.05
        prof["interf_bonus"] = 0.1
    elif mode == VEL_DOWN:
        prof["temp_mul"] = 0.82
        prof["heb_mul"] = 1.1
        prof["bg_mul"] = 1.1
        prof["pro_mul"] = 0.9
    return prof
class Interference:
    def __init__(self):
        self.docs = []

    def _summarize_ids(self, ids, bpe, heavy_limit=32, keyword_limit=16):
        tmp = MetaW()
        meta_build(tmp, ids, len(ids), bpe.vocab_size)
        heavy = []
        counts = {}
        for a, b, _p in tmp.bigrams:
            counts[a] = counts.get(a, 0) + 1
            counts[b] = counts.get(b, 0) + 1
        ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        keywords = []
        for tok, _score in ranked:
            dec = bpe_decode_token(bpe, tok).strip()
            if len(dec) > 2 and any(ch.isalpha() for ch in dec):
                heavy.append(tok)
                keywords.append(dec.lower())
            if len(heavy) >= heavy_limit:
                break
        return {"heavy": heavy or ids[:heavy_limit], "keywords": keywords[:keyword_limit]}

    def load_docs(self, docs_dir, bpe):
        self.docs = []
        if not os.path.isdir(docs_dir):
            return
        for fn in sorted(os.listdir(docs_dir)):
            if not fn.endswith(".txt"):
                continue
            path = os.path.join(docs_dir, fn)
            try:
                with open(path, "rb") as f:
                    raw = f.read()
            except OSError:
                continue
            ids = bpe_encode(bpe, raw, len(raw), len(raw))
            doc = {"name": fn, **self._summarize_ids(ids, bpe, 32, 16), "chunks": []}
            chunk_len = 64
            stride = 32
            for start in range(0, len(ids), stride):
                part = ids[start:start + chunk_len]
                if len(part) < 12:
                    continue
                chunk = self._summarize_ids(part, bpe, 16, 8)
                if chunk["heavy"]:
                    chunk["start"] = start
                    doc["chunks"].append(chunk)
                if len(doc["chunks"]) >= 8:
                    break
            if not doc["chunks"] and doc["heavy"]:
                doc["chunks"].append({"start": 0, "heavy": doc["heavy"][:16], "keywords": doc["keywords"][:8]})
            self.docs.append(doc)

    def _score_item(self, item, words, dom, periodic, prophecy_words):
        score = 0.01 * len(item.get("heavy", []))
        for word in item.get("keywords", []):
            if word in words:
                score += 1.2
            if word in ANCHORS and ANCHORS[word] == dom:
                score += 0.6
            if periodic is not None:
                el = periodic.classify(word)
                if el is not None and el["ch"] == dom:
                    score += 0.35 * el["mass"]
            if word in prophecy_words:
                score += 0.9 * prophecy_words[word]
        score += random.random() * 0.05
        return score

    def _prophecy_words(self, mw, bpe):
        out = {}
        if mw is None:
            return out
        for target, strength, _age in mw.prophecies:
            word = bpe_decode_token(bpe, target).strip().lower()
            if len(word) > 2 and any(ch.isalpha() for ch in word):
                age_boost = strength * math.log1p(float(_age))
                out[word] = max(out.get(word, 0.0), age_boost)
        return out

    def choose_doc(self, text=None, chambers=None, periodic=None, mw=None, bpe=None):
        if not self.docs:
            return None
        if chambers is None and not text:
            return random.choice(self.docs)
        words = set(extract_words(text or ""))
        dom = chambers.dominant() if chambers is not None else CH_FLOW
        prophecy_words = self._prophecy_words(mw, bpe) if bpe is not None else {}
        best_doc = None
        best_score = -1e30
        for doc in self.docs:
            score = self._score_item(doc, words, dom, periodic, prophecy_words)
            if score > best_score:
                best_score = score
                best_doc = doc
        return best_doc

    def choose_chunk(self, doc, text=None, chambers=None, periodic=None, mw=None, bpe=None):
        if doc is None:
            return None
        chunks = doc.get("chunks") or []
        if not chunks:
            return doc
        words = set(extract_words(text or ""))
        dom = chambers.dominant() if chambers is not None else CH_FLOW
        prophecy_words = self._prophecy_words(mw, bpe) if bpe is not None else {}
        best_chunk = chunks[0]
        best_score = -1e30
        for chunk in chunks:
            score = self._score_item(chunk, words, dom, periodic, prophecy_words)
            if score > best_score:
                best_score = score
                best_chunk = chunk
        return best_chunk

    def inject_seed(self, chambers=None, bpe=None, periodic=None, doc=None):
        if not self.docs and doc is None:
            return None
        doc = doc or random.choice(self.docs)
        if not doc["heavy"]:
            return None
        if chambers is None or bpe is None:
            return random.choice(doc["heavy"])
        dom = chambers.dominant()
        scored = []
        for tid in doc["heavy"]:
            token = bpe_decode_token(bpe, tid).strip().lower()
            score = 0.1
            if token in ANCHORS and ANCHORS[token] == dom:
                score += 1.0
            if periodic is not None:
                el = periodic.classify(token)
                if el is not None and el["ch"] == dom:
                    score += 0.5 * el["mass"]
            score += random.random() * 0.05
            scored.append((score, tid))
        scored.sort(reverse=True)
        top = scored[:5]
        total = sum(max(0.01, s) for s, _tid in top)
        r = random.random() * total
        cum = 0.0
        for score, tid in top:
            cum += max(0.01, score)
            if cum >= r:
                return tid
        return top[0][1]
def interference_signal(doc, V):
    out = [0.0] * V
    if doc is None:
        return out
    for rank, tid in enumerate(doc.get("heavy", [])[:16]):
        if 0 <= tid < V:
            out[tid] += 1.0 / (1.0 + rank)
    mx = max(out) if out else 0.0
    if mx > 1e-8:
        for i in range(V):
            out[i] /= mx
    return out
# ── DOE Parliament ──
class Expert:
    def __init__(self):
        self.A = []
        self.B = []
        self.d_in = 0
        self.d_out = 0
        self.rank = 0
        self.vitality = 1.0
        self.age = 0
        self.low_steps = 0
def expert_init(e, d_in, d_out, rank):
    e.d_in = d_in
    e.d_out = d_out
    e.rank = rank
    e.A = [0.01 * (random.random() - 0.5) for _ in range(rank * d_in)]
    e.B = [0.01 * (random.random() - 0.5) for _ in range(d_out * rank)]
    e.vitality = 1.0
    e.age = 0
    e.low_steps = 0
def expert_forward(e, x):
    mid = [0.0] * e.rank
    for r in range(e.rank):
        s = 0.0
        base = r * e.d_in
        for d in range(e.d_in):
            s += e.A[base + d] * x[d]
        mid[r] = s
    out = [0.0] * e.d_out
    for o in range(e.d_out):
        s = 0.0
        base = o * e.rank
        for r in range(e.rank):
            s += e.B[base + r] * mid[r]
        out[o] = s
    return out
def expert_hebbian(e, x, dy, lr):
    for r in range(e.rank):
        u = 0.0
        for o in range(e.d_out):
            u += e.B[o * e.rank + r] * dy[o]
        u += 0.01 * (random.random() - 0.5)
        base_a = r * e.d_in
        for d in range(e.d_in):
            e.A[base_a + d] += lr * x[d] * u
        for o in range(e.d_out):
            e.B[o * e.rank + r] *= 0.999
class Parliament:
    def __init__(self):
        self.ex = []
        self.n = 0
        self.d_model = 0
        self.alpha = DOE_ALPHA
        self.step = 0
def parl_init(p, d_model, n_init):
    p.d_model = d_model
    p.alpha = DOE_ALPHA
    p.step = 0
    p.n = min(n_init, MAX_EXPERTS)
    p.ex = []
    for _ in range(p.n):
        e = Expert()
        expert_init(e, d_model, d_model, DOE_RANK)
        p.ex.append(e)
def parl_election(p, x):
    result = [0.0] * p.d_model
    if p.n == 0:
        return result
    votes = [0.0] * p.n
    outs = []
    for i in range(p.n):
        o = expert_forward(p.ex[i], x)
        outs.append(o)
        dot = sum(o[d] * x[d] for d in range(p.d_model))
        votes[i] = dot
    mx = max(votes)
    mn = min(votes)
    cons = (mx - mn) / (abs(mx) + abs(mn) + 1e-8)
    k = int(p.n * (1.0 - cons))
    if k < 1:
        k = 1
    if k > p.n:
        k = p.n
    sel = list(range(p.n))
    for i in range(p.n - 1):
        for j in range(i + 1, p.n):
            if votes[sel[j]] > votes[sel[i]]:
                sel[i], sel[j] = sel[j], sel[i]
    sv = votes[sel[0]]
    exps = [0.0] * p.n
    tot = 0.0
    for i in range(k):
        exps[i] = math.exp(votes[sel[i]] - sv)
        tot += exps[i]
    for i in range(k):
        w = exps[i] / tot
        for d in range(p.d_model):
            result[d] += w * outs[sel[i]][d]
        p.ex[sel[i]].vitality = 0.9 * p.ex[sel[i]].vitality + 0.1 * abs(w)
    for i in range(k, p.n):
        p.ex[sel[i]].vitality *= 0.95
        p.ex[sel[i]].low_steps += 1
    return result
def parl_inject(p, logits, x, V):
    delta = parl_election(p, x)
    n = min(V, p.d_model)
    for i in range(n):
        logits[i] += p.alpha * delta[i]
def parl_notorch(p, x, debt, dlen):
    n = min(dlen, p.d_model)
    ds = [0.0] * p.d_model
    for i in range(n):
        ds[i] = debt[i]
    for i in range(p.n):
        expert_hebbian(p.ex[i], x, ds, 0.001)
        p.ex[i].age += 1
def parl_lifecycle(p):
    # apoptosis
    alive = []
    for i in range(p.n):
        if p.ex[i].low_steps >= 8 and p.ex[i].vitality < 0.1 and p.n > 2:
            continue
        alive.append(p.ex[i])
    p.ex = alive
    p.n = len(alive)
    # mitosis
    births = []
    for i in range(p.n):
        if p.n + len(births) >= MAX_EXPERTS:
            break
        if p.ex[i].vitality > 0.8 and p.ex[i].age > 50:
            child = Expert()
            expert_init(child, p.ex[i].d_in, p.ex[i].d_out, p.ex[i].rank)
            for j in range(child.rank * child.d_in):
                child.A[j] = p.ex[i].A[j] + 0.005 * (random.random() - 0.5)
            for j in range(child.d_out * child.rank):
                child.B[j] = p.ex[i].B[j] + 0.005 * (random.random() - 0.5)
            child.vitality = 0.5
            births.append(child)
            p.ex[i].vitality *= 0.6
    p.ex.extend(births)
    p.n = len(p.ex)
    p.step += 1
# ── Transformer ──
class TFLayer:
    def __init__(self):
        self.wq = None
        self.wk = None
        self.vc = None
        self.wr = None
        self.vr = None
        self.wj = None
        self.vj = None
        self.gw = None
        self.gb = None
        self.wo = None
        self.up = None
        self.dn = None
class TF:
    def __init__(self):
        self.V = 0
        self.D = 0
        self.NH = 0
        self.NL = 0
        self.CTX = 0
        self.NC = 0
        self.NR = 0
        self.NJ = 0
        self.HD = 0
        self.tok = None
        self.pos = None
        self.L = []
        self.kc = []
        self.vcc = []
        self.vrc = []
        self.clen = 0
        self.logits = None
def _read_floats(f, count):
    data = f.read(count * 4)
    if len(data) < count * 4:
        return list(struct.unpack("<%df" % (len(data) // 4), data))
    return list(struct.unpack("<%df" % count, data))
def tf_load(t, path):
    try:
        f = open(path, "rb")
    except OSError:
        print("ERROR: " + path, file=sys.stderr)
        return False
    magic = struct.unpack("<I", f.read(4))[0]
    if magic != QPTQ_MAGIC:
        print("bad magic", file=sys.stderr)
        f.close()
        return False
    hdr = struct.unpack("<10I", f.read(40))
    ver, v, d, nh, nl, ctx, nc, nr, nj, hd = hdr
    t.V = v; t.D = d; t.NH = nh; t.NL = nl; t.CTX = ctx
    t.NC = nc; t.NR = nr; t.NJ = nj; t.HD = hd
    nm = (1 if nc > 0 else 0) + (1 if nr > 0 else 0) + (1 if nj > 0 else 0)
    print("  model: V=%d D=%d H=%d L=%d nc=%d nr=%d nj=%d" % (v, d, nh, nl, nc, nr, nj))

    t.tok = _read_floats(f, v * d)
    t.pos = _read_floats(f, ctx * d)

    t.L = []
    for li in range(nl):
        layer = TFLayer()
        if nc > 0:
            layer.wq = _read_floats(f, nc * hd * d)
            layer.wk = _read_floats(f, nc * hd * d)
            layer.vc = _read_floats(f, nc * hd * d)
        if nr > 0:
            layer.wr = _read_floats(f, nr * d * ctx)
            layer.vr = _read_floats(f, nr * hd * d)
        if nj > 0:
            layer.wj = _read_floats(f, nj * hd * d)
            layer.vj = _read_floats(f, nj * hd * d)
        if nm > 1:
            layer.gw = _read_floats(f, nm * d)
            layer.gb = _read_floats(f, nm)
        layer.wo = _read_floats(f, d * d)
        layer.up = _read_floats(f, 4 * d * d)
        layer.dn = _read_floats(f, d * 4 * d)
        t.L.append(layer)

    t.kc = []
    t.vcc = []
    t.vrc = []
    for li in range(nl):
        t.kc.append([0.0] * (ctx * (nc * hd if nc > 0 else 1)))
        t.vcc.append([0.0] * (ctx * (nc * hd if nc > 0 else 1)))
        t.vrc.append([0.0] * (ctx * (nr * hd if nr > 0 else 1)))
    t.clen = 0
    t.logits = [0.0] * v
    f.close()
    return True
def tf_reset(t):
    t.clen = 0
def tf_forward(t, tok, pos):
    D = t.D; HD = t.HD; NC = t.NC; NR = t.NR; NJ = t.NJ
    nm = (1 if NC > 0 else 0) + (1 if NR > 0 else 0) + (1 if NJ > 0 else 0)
    sl = pos + 1

    x = [0.0] * D
    for d in range(D):
        x[d] = t.tok[tok * D + d] + t.pos[pos * D + d]

    for li in range(t.NL):
        xr = list(x)
        xn = rmsnorm(x, D)

        co = None; ro = None; jo = None

        # content attention
        if NC > 0:
            co = [0.0] * (NC * HD)
            q = matmul(xn, t.L[li].wq, D, NC * HD)
            k = matmul(xn, t.L[li].wk, D, NC * HD)
            vc = matmul(xn, t.L[li].vc, D, NC * HD)
            # store in KV cache
            for d in range(NC * HD):
                t.kc[li][pos * NC * HD + d] = k[d]
                t.vcc[li][pos * NC * HD + d] = vc[d]
            for h in range(NC):
                sc = [0.0] * sl
                for p in range(sl):
                    dot = 0.0
                    for d in range(HD):
                        dot += q[h * HD + d] * t.kc[li][p * NC * HD + h * HD + d]
                    sc[p] = dot / math.sqrt(float(HD))
                softmax_inplace(sc, sl)
                for d in range(HD):
                    v = 0.0
                    for p in range(sl):
                        v += sc[p] * t.vcc[li][p * NC * HD + h * HD + d]
                    co[h * HD + d] = v

        # RRPRAM
        if NR > 0:
            ro = [0.0] * (NR * HD)
            vr = matmul(xn, t.L[li].vr, D, NR * HD)
            for d in range(NR * HD):
                t.vrc[li][pos * NR * HD + d] = vr[d]
            for h in range(NR):
                sc = [0.0] * sl
                for p in range(sl):
                    s = 0.0
                    for d in range(D):
                        s += xn[d] * t.L[li].wr[(h * D + d) * t.CTX + p]
                    sc[p] = s
                softmax_inplace(sc, sl)
                for d in range(HD):
                    v = 0.0
                    for p in range(sl):
                        v += sc[p] * t.vrc[li][p * NR * HD + h * HD + d]
                    ro[h * HD + d] = v

        # Janus echo
        if NJ > 0:
            jo = [0.0] * (NJ * HD)
            wjp = matmul(xn, t.L[li].wj, D, NJ * HD)
            vjp = matmul(xn, t.L[li].vj, D, NJ * HD)
            norm = sum(wjp[d] * wjp[d] for d in range(NJ * HD))
            norm = 1.0 / math.sqrt(norm + 1e-8)
            for d in range(NJ * HD):
                jo[d] = vjp[d] * (wjp[d] * norm)

        # gating + output
        comb = [0.0] * D
        if nm > 1 and t.L[li].gw is not None:
            gl = matmul(xn, t.L[li].gw, D, nm)
            gates = [0.0] * nm
            for g in range(nm):
                gates[g] = 1.0 / (1.0 + math.exp(-(gl[g] + t.L[li].gb[g])))
            off = 0; gi = 0
            if NC > 0:
                for d in range(NC * HD):
                    comb[off + d] = gates[gi] * co[d]
                off += NC * HD; gi += 1
            if NR > 0:
                for d in range(NR * HD):
                    comb[off + d] = gates[gi] * ro[d]
                off += NR * HD; gi += 1
            if NJ > 0:
                for d in range(NJ * HD):
                    comb[off + d] = gates[gi] * jo[d]
                off += NJ * HD; gi += 1
        else:
            off = 0
            if NC > 0 and co is not None:
                for d in range(NC * HD):
                    comb[off + d] = co[d]
                off += NC * HD
            if NR > 0 and ro is not None:
                for d in range(NR * HD):
                    comb[off + d] = ro[d]
                off += NR * HD
            if NJ > 0 and jo is not None:
                for d in range(NJ * HD):
                    comb[off + d] = jo[d]
                off += NJ * HD

        proj = matmul(comb, t.L[li].wo, D, D)
        for d in range(D):
            x[d] = xr[d] + proj[d]

        # MLP
        xr = list(x)
        xn = rmsnorm(x, D)
        up = matmul(xn, t.L[li].up, D, 4 * D)
        for d in range(4 * D):
            if up[d] < 0:
                up[d] = 0.0
        dn = matmul(up, t.L[li].dn, 4 * D, D)
        for d in range(D):
            x[d] = xr[d] + dn[d]

    xn = rmsnorm(x, D)
    for v in range(t.V):
        dot = 0.0
        for d in range(D):
            dot += xn[d] * t.tok[v * D + d]
        t.logits[v] = dot

    # transformer gate
    mag = sum(abs(t.logits[v]) for v in range(t.V)) / (t.V if t.V > 0 else 1)
    tg = clampf((mag - 0.5) / 1.5, 0.0, 1.0)
    for v in range(t.V):
        t.logits[v] *= tg
    t.clen = sl
# ── coherence score ──
def coherence_score(mw, ids, n, V):
    if n < 2:
        return 0.0
    bi_sum = 0.0
    for i in range(n - 1):
        for j in range(mw.n_bi):
            if mw.bigrams[j][0] == ids[i] and mw.bigrams[j][1] == ids[i + 1]:
                bi_sum += mw.bigrams[j][2]
                break
    # trigram continuity: stronger signal than bigrams for coherence
    tri_sum = 0.0
    for i in range(n - 2):
        for j in range(mw.n_tri):
            if (mw.trigrams[j][0] == ids[i] and mw.trigrams[j][1] == ids[i + 1]
                    and mw.trigrams[j][2] == ids[i + 2]):
                tri_sum += mw.trigrams[j][3]
                break
    hb_sum = 0.0
    for i in range(min(n - 1, 20)):
        a = min(ids[i], ids[i + 1])
        b = max(ids[i], ids[i + 1])
        for k in range(mw.n_hebb):
            if mw.hebbs[k][0] == a and mw.hebbs[k][1] == b:
                hb_sum += mw.hebbs[k][2]
                break
    if n > 15:
        len_bonus = 1.5
    elif n > 10:
        len_bonus = 0.8
    elif n > 6:
        len_bonus = 0.2
    else:
        len_bonus = -0.5
    tri_norm = tri_sum / (n - 2) if n > 2 else 0
    return bi_sum / (n - 1) + 0.5 * hb_sum / (n - 1) + 0.8 * tri_norm + len_bonus
# ── boundary check ──
def is_boundary(bpe, tid):
    if tid < 0 or tid >= bpe.vocab_size:
        return False
    b = bpe.vocab_bytes.get(tid, b"")
    blen = len(b)
    for i in range(blen):
        c = b[i]
        if c in (ord('.'), ord('!'), ord('?')):
            if i == blen - 1:
                return True
            nc = b[i + 1]
            if nc in (ord(' '), ord('\n'), ord('\r')):
                return True
    return False
def starts_with_space(bpe, tid):
    if tid < 0 or tid >= bpe.vocab_size:
        return False
    b = bpe.vocab_bytes.get(tid, b"")
    if len(b) == 0:
        return False
    return b[0] == ord(' ')
# ── generate sentence ──
def gen_sent(t, bpe, mw, prompt, plen, temp, maxo, parl, global_destiny, ch_ptr, velocity=None, doc_signal=None):
    tf_reset(t)
    V = t.V; D = t.D
    destiny = [0.0] * D
    if global_destiny is not None:
        for d in range(D):
            destiny[d] = 0.3 * global_destiny[d]
    prev_logits = [0.0] * V
    prev_chosen = -1
    ctx = [0] * MAX_SEQ
    cl = 0
    out = []
    gl = 0
    for i in range(min(plen, t.CTX - 1)):
        tf_forward(t, prompt[i], i)
        ctx[cl] = prompt[i]; cl += 1
        out.append(prompt[i]); gl += 1

    am, bm, gm, tm = (1.0, 1.0, 1.0, 1.0)
    if ch_ptr is not None:
        am, bm, gm, tm = ch_ptr.modulate()

    for step in range(120):
        if gl >= maxo:
            break
        pos = cl - 1
        if pos >= t.CTX - 1:
            break
        tf_forward(t, ctx[cl - 1], pos)
        raw = list(t.logits)

        # DOE Parliament injection
        if parl is not None:
            xn = rmsnorm(t.tok[ctx[cl - 1] * D: ctx[cl - 1] * D + D], D)
            parl_inject(parl, raw, xn, V)
            if step > 0 and prev_chosen >= 0:
                debt = [0.0] * D
                top3 = [0, 0, 0]
                tv = [-1e30, -1e30, -1e30]
                for i in range(V):
                    if prev_logits[i] > tv[2]:
                        tv[2] = prev_logits[i]; top3[2] = i
                        for k in range(1, -1, -1):
                            if tv[k + 1] > tv[k]:
                                tv[k], tv[k + 1] = tv[k + 1], tv[k]
                                top3[k], top3[k + 1] = top3[k + 1], top3[k]
                for k in range(3):
                    if top3[k] != prev_chosen and top3[k] < V:
                        if top3[k] < t.V:
                            for d in range(D):
                                debt[d] += 0.1 * t.tok[top3[k] * D + d]
                if prev_chosen < t.V:
                    for d in range(D):
                        debt[d] -= 0.1 * t.tok[prev_chosen * D + d]
                parl_notorch(parl, xn, debt, D)
                if step % 20 == 0:
                    parl_lifecycle(parl)

        prev_logits = list(raw)
        last = ctx[cl - 1]
        # adaptive destiny momentum: faster update early, stable later
        d_mom = 0.85 if step < 20 else 0.92
        d_lr = 1.0 - d_mom
        if last < V:
            for d in range(D):
                destiny[d] = d_mom * destiny[d] + d_lr * t.tok[last * D + d]
        dn = math.sqrt(sum(destiny[d] * destiny[d] for d in range(D)) + 1e-10)

        hs = max(0, cl - 8)
        heb = meta_hebb(mw, ctx[hs:cl], cl - hs, V)
        pro = meta_prophecy(mw, ctx[:cl], cl, V)
        p_debt = prophecy_pressure(mw)

        # trauma gravity
        if ch_ptr is not None and ch_ptr.trauma > 0.1:
            for i in range(V):
                raw[i] /= (1.0 + ch_ptr.trauma)

        # detect if transformer is active
        tmag = sum(abs(raw[v]) for v in range(V)) / (V if V > 0 else 1)
        has_tf = tmag > 0.1

        # Dario field coefficients — boosted for better coherence
        c_heb = (0.6 if has_tf else 1.0) * am
        c_pro = (0.4 if has_tf else 0.7) * bm
        c_ds  = (0.3 if has_tf else 0.15) * gm
        c_bg  = 5.0 if has_tf else 15.0
        c_tg  = 3.0 if has_tf else 10.0
        if velocity is not None:
            c_heb *= velocity["heb_mul"]
            c_pro *= velocity["pro_mul"] * (1.0 + 0.35 * p_debt)
            c_ds *= velocity["ds_mul"]
            c_bg *= velocity["bg_mul"]
            c_tg *= velocity["tg_mul"]
        c_doc = 0.18 if has_tf else 0.32

        for i in range(V):
            bg = meta_bi(mw, ctx[cl - 1], i)
            tg_val = meta_tri(mw, ctx[cl - 2], ctx[cl - 1], i) if cl >= 2 else 1e-10
            ds = 0.0
            if dn > 1e-8:
                en = math.sqrt(sum(t.tok[i * D + d] * t.tok[i * D + d] for d in range(D)) + 1e-10)
                if en > 1e-8:
                    dot = sum(destiny[d] * t.tok[i * D + d] for d in range(D))
                    ds = dot / (dn * en)
            raw[i] += c_heb * heb[i] + c_pro * pro[i] + c_ds * ds + c_bg * bg + c_tg * tg_val
            if doc_signal is not None:
                raw[i] += c_doc * doc_signal[i]
            if mw.unigram[i] < 1e-6:
                raw[i] -= 2.0
            elif mw.unigram[i] > 0.01:
                raw[i] -= 0.3 * (mw.unigram[i] - 0.01) * 100.0

        # repetition penalty
        for ri in range(cl - 1, max(-1, cl - 21), -1):
            if ctx[ri] < V:
                age_factor = float(cl - ri)
                pen = 0.3 + 0.035 * age_factor
                raw[ctx[ri]] *= pen

        # bigram blocking
        if cl >= 2:
            for ri in range(cl - 1):
                if ctx[ri] == ctx[cl - 2] and ctx[ri + 1] < V:
                    raw[ctx[ri + 1]] *= 0.2

        # hybrid decode
        if not has_tf:
            if step < 6:
                ch_tok = 0
                mx_val = raw[0]
                for i in range(1, V):
                    if raw[i] > mx_val:
                        mx_val = raw[i]
                        ch_tok = i
            else:
                ch_tok = sample_nucleus(raw, V, 0.5, 0.7)
        elif step < 4:
            ch_tok = 0
            mx_val = raw[0]
            for i in range(1, V):
                if raw[i] > mx_val:
                    mx_val = raw[i]
                    ch_tok = i
        else:
            vel_temp = velocity["temp_mul"] if velocity is not None else 1.0
            ch_tok = sample_nucleus(raw, V, clampf(temp * tm * vel_temp, 0.25, 1.35), 0.85)

        prev_chosen = ch_tok
        out.append(ch_tok); gl += 1
        ctx[cl] = ch_tok; cl += 1
        prophecy_update(mw, ch_tok)

        # word capture
        if cl >= 2:
            prev_tok = ctx[cl - 2]
            cur = ctx[cl - 1]
            found = False
            for i in range(mw.n_bi):
                if mw.bigrams[i][0] == prev_tok and mw.bigrams[i][1] == cur:
                    mw.bigrams[i][2] += 0.005
                    found = True
                    break
            if not found and mw.n_bi < MAX_BIGRAM:
                mw.bigrams.append([prev_tok, cur, 0.01])
                mw.n_bi += 1
            best_pred = -1
            best_prob = 0.0
            for i in range(mw.n_tri):
                if mw.trigrams[i][0] == prev_tok and mw.trigrams[i][1] == cur and mw.trigrams[i][3] > best_prob:
                    best_prob = mw.trigrams[i][3]
                    best_pred = mw.trigrams[i][2]
            if best_pred < 0:
                for i in range(mw.n_bi):
                    if mw.bigrams[i][0] == cur and mw.bigrams[i][2] > best_prob:
                        best_prob = mw.bigrams[i][2]
                        best_pred = mw.bigrams[i][1]
            if best_pred >= 0:
                prophecy_add(mw, best_pred, 0.2 + 0.5 * best_prob)

            hw = max(0, cl - 6)
            for ri in range(hw, cl - 1):
                a = min(ctx[ri], cur)
                b = max(ctx[ri], cur)
                decay = 1.0 / (1.0 + abs((cl - 1) - ri))
                found_h = False
                for k in range(mw.n_hebb):
                    if mw.hebbs[k][0] == a and mw.hebbs[k][1] == b:
                        mw.hebbs[k][2] += decay * 0.005
                        found_h = True
                        break
                if not found_h and mw.n_hebb < MAX_HEBBIAN:
                    mw.hebbs.append([a, b, decay * 0.01])
                    mw.n_hebb += 1

        if is_boundary(bpe, ch_tok) and step > 8:
            break

    # export destiny
    if global_destiny is not None:
        for d in range(D):
            global_destiny[d] = 0.7 * global_destiny[d] + 0.3 * destiny[d]
    return out
# ── SPA ──
class SPACtx:
    def __init__(self):
        self.W_embed = []
        self.r_bias = []
        self.alpha = 0.85
def spa_init(s, V):
    s.alpha = 0.85
    s.W_embed = []
    for i in range(min(V, MAX_VOCAB)):
        s.W_embed.append([0.02 * (random.random() - 0.5) for _ in range(SPA_DIM)])
    s.r_bias = [0.1 / (1.0 + i) for i in range(CHAIN_STEPS + 1)]
def spa_embed_sentence(s, ids, n):
    out = [0.0] * SPA_DIM
    if n == 0:
        return out
    total_w = 0.0
    for i in range(n):
        w = s.alpha ** (n - 1 - i)
        if 0 <= ids[i] < len(s.W_embed):
            for d in range(SPA_DIM):
                out[d] += w * s.W_embed[ids[i]][d]
        total_w += w
    if total_w > 0:
        for d in range(SPA_DIM):
            out[d] /= total_w
    norm = math.sqrt(sum(out[d] * out[d] for d in range(SPA_DIM)) + 1e-8)
    inv = 1.0 / norm
    for d in range(SPA_DIM):
        out[d] *= inv
    return out
def spa_cross_attend(s, embs, S):
    scores = [0.0] * S
    for i in range(S):
        total_attn = 0.0
        for j in range(S):
            if i == j:
                continue
            dot = sum(embs[i][d] * embs[j][d] for d in range(SPA_DIM))
            dot /= math.sqrt(float(SPA_DIM))
            dist = abs(i - j)
            if dist > CHAIN_STEPS:
                dist = CHAIN_STEPS
            dot += s.r_bias[dist]
            total_attn += math.exp(dot)
        scores[i] = total_attn
    return scores
# ── chain ──
def gen_chain(t, bpe, mw, ch, cids, clen, has_weights, parl, periodic=None, interference=None, input_text=None):
    # calendar dissonance
    try:
        epoch_t = time.mktime((2024, 10, 3, 12, 0, 0, 0, 0, -1))
    except Exception:
        epoch_t = 0
    days = (time.time() - epoch_t) / 86400.0 if epoch_t > 0 else 0.0
    y = days / 365.25
    drift = y * 11.25
    full = int(y / 19)
    corr = full * 7 * 30.0
    partial = y % 19
    yic = int(partial) + 1
    met = [3, 6, 8, 11, 14, 17, 19]
    for m in met:
        if m <= yic:
            corr += 30
    drift -= corr
    cd = clampf(abs(drift % 33) / 33.0, 0, 1)

    nb = int(CHAIN_STEPS * (0.3 + 0.4 * ch.debt + 0.1 * cd))
    if nb < 1:
        nb = 1
    if nb >= CHAIN_STEPS:
        nb = CHAIN_STEPS - 1

    if input_text:
        inp_bytes = input_text.encode("utf-8", errors="replace")
        ingest_ids(mw, bpe_encode(bpe, inp_bytes, len(inp_bytes), 512))
        ch.feel(input_text, periodic)
        ch.act[CH_FLOW] = clampf(ch.act[CH_FLOW] + 0.1, 0.0, 1.0)
        ch_xfire(ch, 8)
    vel = velocity_profile(ch, cd)
    ch.debt = clampf((0.88 * ch.debt + 0.12 * prophecy_pressure(mw)) * vel["debt_decay"], 0.0, 1.0)
    ch.trauma = clampf(ch.trauma * vel["trauma_decay"], 0.0, 1.0)

    mode_str = "[TRAINED]" if has_weights else "[METAWEIGHTS ONLY]"
    print("\n  diss=%.3f debt=%.3f emrg=%.3f vel=%s %s" % (cd, ch.debt, ch.emergence(), vel["name"], mode_str))
    print("  chambers: %s" % ch.summary())
    if parl is not None:
        av = sum(e.vitality for e in parl.ex) / (parl.n if parl.n > 0 else 1)
        print("  parliament: %d experts, avg_vitality=%.2f" % (parl.n, av))
    if interference is not None and interference.docs:
        print("  interference: %d docs loaded" % len(interference.docs))
    print()

    gdest = [0.0] * t.D
    spa = SPACtx()
    spa_init(spa, t.V)
    chain_ids = [[] for _ in range(CHAIN_STEPS)]
    chain_lens = [0] * CHAIN_STEPS

    for si in range(CHAIN_STEPS):
        if si < nb:
            direction = -1
        elif si == nb:
            direction = 0
        else:
            direction = 1

        active_doc = interference.choose_doc(input_text, ch, periodic, mw, bpe) if interference is not None and interference.docs else None
        active_chunk = interference.choose_chunk(active_doc, input_text, ch, periodic, mw, bpe) if active_doc is not None else None
        if active_chunk is not None and t.D > 0 and active_chunk.get("heavy"):
            seed_tok = active_chunk["heavy"][0]
            if 0 <= seed_tok < t.V:
                for d in range(t.D):
                    gdest[d] = 0.97 * gdest[d] + 0.03 * t.tok[seed_tok * t.D + d]
        doc_signal = interference_signal(active_chunk, t.V) if active_chunk is not None else None

        prompt = None
        used_interference = False
        if interference is not None and interference.docs and random.random() < clampf(0.3 + vel["interf_bonus"], 0.05, 0.5):
            seed = interference.inject_seed(ch, bpe, periodic, active_chunk or active_doc)
            if seed is not None:
                prompt = [seed]
                used_interference = True
        if prompt is None and input_text:
            inp_ids = bpe_encode(bpe, input_text.encode("utf-8", errors="replace"), len(input_text.encode("utf-8", errors="replace")), 128)
            if inp_ids:
                st = random.randint(0, max(0, len(inp_ids) - 2))
                prompt = inp_ids[st:st + 2]
        pl = len(prompt) if prompt is not None else 0
        if prompt is None:
            start = -1
            if direction >= 0 and si > 0:
                best_score = -1e30
                best_pos = -1
                for _try in range(50):
                    r = random.randint(0, max(0, clen - 6))
                    if is_boundary(bpe, cids[r]) and r + 3 < clen and starts_with_space(bpe, cids[r + 1]):
                        sc = 0.0
                        tok_id = cids[r + 1]
                        if tok_id < t.V:
                            for d in range(t.D):
                                sc += t.tok[tok_id * t.D + d] * gdest[d]
                        if sc > best_score:
                            best_score = sc
                            best_pos = r + 1
                if best_pos >= 0:
                    start = best_pos
            if start < 0:
                for _try in range(200):
                    r = random.randint(0, max(0, clen - 6))
                    if is_boundary(bpe, cids[r]) and r + 3 < clen and starts_with_space(bpe, cids[r + 1]):
                        start = r + 1
                        break
            if start < 0:
                start = random.randint(0, max(0, clen - 6))
            pl = 5 if start + 5 <= clen else 3
            prompt = [cids[start], cids[start + 1], cids[start + 2],
                      cids[start + 3] if pl > 3 else 0,
                      cids[start + 4] if pl > 4 else 0][:pl]
        else:
            pl = len(prompt)

        # Schumann resonance
        t_sec = float(si) / float(CHAIN_STEPS)
        schumann = (0.4 * math.sin(2 * math.pi * 7.83 * t_sec)
                    + 0.2 * math.sin(2 * math.pi * 14.3 * t_sec)
                    + 0.1 * math.sin(2 * math.pi * 20.8 * t_sec)
                    + 0.05 * math.sin(2 * math.pi * 27.3 * t_sec))
        base_temp = 0.6 if has_weights else 0.75
        temp = clampf((base_temp + 0.08 * schumann) * vel["temp_mul"], 0.35, 0.95)

        # best-of-3
        best_out = []
        best_ol = 0
        best_sc = -1e30
        gdest_save = list(gdest) if t.D <= 256 else None

        for cand in range(3):
            if cand > 0 and gdest_save is not None:
                gdest[:] = list(gdest_save)
            result = gen_sent(t, bpe, mw, prompt, pl, temp, 256, parl, gdest, ch, vel, doc_signal)
            sc = coherence_score(mw, result, len(result), t.V)
            if sc > best_sc:
                best_sc = sc
                best_ol = len(result)
                best_out = list(result)
            if best_sc > 1.0 and best_ol > 12:
                break

        wormhole = False
        if si < CHAIN_STEPS - 1:
            wh_prob = 0.02
            if cd > 0.3:
                wh_prob += ((cd - 0.3) / 0.7) * 0.15
            wh_prob = clampf(wh_prob + vel["wormhole_bonus"], 0.0, 0.3)
            wormhole = random.random() < wh_prob
            if wormhole and interference is not None and interference.docs:
                longest = max(interference.docs, key=lambda d: len(d["heavy"]))
                if longest["heavy"]:
                    prompt = [random.choice(longest["heavy"])]
                    direction = -direction if direction != 0 else 1
                    best_out = gen_sent(t, bpe, mw, prompt, len(prompt), 0.55 if has_weights else 0.7, 256, parl, gdest, ch, vel, doc_signal)
                    best_ol = len(best_out)
                    best_sc = coherence_score(mw, best_out, best_ol, t.V)

        mk = '<' if direction < 0 else ('*' if direction == 0 else '>')
        marker = mk + ('+' if wormhole else ' ')
        sys.stdout.write("  [%2d] %s " % (si + 1, marker))

        if best_ol < 5 or (best_sc < 0.01 and best_ol < 8):
            print("[...]")
        else:
            text_parts = []
            printed = 0
            for i in range(best_ol):
                if printed >= 200:
                    break
                s = bpe_decode_token(bpe, best_out[i])
                if s:
                    text_parts.append(s)
                    sys.stdout.write(s)
                    printed += len(s)
            if used_interference:
                sys.stdout.write("  {interf}")
            if wormhole:
                sys.stdout.write("  {wormhole}")
            print()
            out_text = "".join(text_parts)
            ch.feel(out_text, periodic)
            if out_text:
                ingest_ids(mw, bpe_encode(bpe, out_text.encode("utf-8", errors="replace"), len(out_text.encode("utf-8", errors="replace")), 256), 0.005)

        chain_ids[si] = list(best_out)
        chain_lens[si] = best_ol
        ch_xfire(ch, 3)
        ch.debt = 0.9 * ch.debt + 0.05

    # SPA: iterative cross-attention — reseed weak sentences, verify improvement
    spa_embs = [None] * CHAIN_STEPS
    spa_scores = [0.0] * CHAIN_STEPS
    for spa_pass in range(2):
        for i in range(CHAIN_STEPS):
            spa_embs[i] = spa_embed_sentence(spa, chain_ids[i], chain_lens[i])
        spa_scores = spa_cross_attend(spa, spa_embs, CHAIN_STEPS)

        min_sc = spa_scores[0]
        weak_idx = 0
        for i in range(1, CHAIN_STEPS):
            if spa_scores[i] < min_sc:
                min_sc = spa_scores[i]
                weak_idx = i
        avg_sc = sum(spa_scores) / CHAIN_STEPS

        if min_sc < avg_sc * 0.6:  # slightly more aggressive threshold
            print("  [SPA-%d] reseeding step %d (score=%.2f, avg=%.2f)" % (spa_pass + 1, weak_idx + 1, min_sc, avg_sc))
            # use neighbor sentences as context for better continuity
            seed_src = weak_idx - 1 if weak_idx > 0 else (weak_idx + 1 if weak_idx < CHAIN_STEPS - 1 else 0)
            nprom = min(3, chain_lens[seed_src])
            prompt = chain_ids[seed_src][chain_lens[seed_src] - nprom:chain_lens[seed_src]]
            reseed_temp = 0.55 if has_weights else 0.7
            result = gen_sent(t, bpe, mw, prompt, nprom, reseed_temp, 256, parl, gdest, ch, vel, doc_signal)
            new_sc = coherence_score(mw, result, len(result), t.V)
            old_sc = coherence_score(mw, chain_ids[weak_idx], chain_lens[weak_idx], t.V)
            if new_sc > old_sc * 0.7 or len(result) > chain_lens[weak_idx]:  # accept if reasonable
                chain_ids[weak_idx] = list(result)
                chain_lens[weak_idx] = len(result)
                sys.stdout.write("  [%2d] + " % (weak_idx + 1))
                printed = 0
                for i in range(len(result)):
                    if printed >= 200:
                        break
                    s = bpe_decode_token(bpe, result[i])
                    if s:
                        sys.stdout.write(s)
                        printed += len(s)
                print("  {reseeded}")
                # feed reseeded text back into metaweights
                text_parts = []
                for i in range(len(result)):
                    s = bpe_decode_token(bpe, result[i])
                    if s:
                        text_parts.append(s)
                out_text = "".join(text_parts)
                ch.feel(out_text, periodic)
                ingest_ids(mw, result, 0.003)
        else:
            break  # no weak sentences, stop iterating

    # Hebbian decay
    for i in range(mw.n_hebb):
        mw.hebbs[i][2] *= 0.998
# ── main ──
def main():
    print("PostGPT-Q \u2014 Resonant Reasoning Engine (Python)")
    print("theta = epsilon + gamma + alpha*delta")
    print("resonance is unbreakable.\n")

    if len(sys.argv) < 3:
        print("Usage: %s [weights.bin] corpus.merges corpus.txt" % sys.argv[0])
        sys.exit(1)

    random.seed(int(time.time()))

    has_weights = False
    wpath = None
    if len(sys.argv) >= 4:
        wpath = sys.argv[1]
        mpath = sys.argv[2]
        cpath = sys.argv[3]
        has_weights = True
    else:
        mpath = sys.argv[1]
        cpath = sys.argv[2]

    print("[1] BPE...")
    bpe = BPE()
    if not bpe_load(bpe, mpath):
        sys.exit(1)
    print("  %d merges, vocab=%d" % (bpe.n_merges, bpe.vocab_size))

    print("[2] Corpus...")
    try:
        with open(cpath, "rb") as cf:
            craw = cf.read()
    except OSError:
        print("ERROR: " + cpath, file=sys.stderr)
        sys.exit(1)
    csz = len(craw)
    cids = bpe_encode(bpe, craw, csz, csz)
    clen_corpus = len(cids)
    print("  %d bytes -> %d tokens" % (csz, clen_corpus))

    print("[3] MetaWeights...")
    mw = MetaW()
    meta_build(mw, cids, clen_corpus, bpe.vocab_size)
    periodic = PeriodicTable()
    periodic.build_from_text(craw.decode("utf-8", errors="ignore"))
    print("  periodic table: %d elements" % len(periodic.elements))

    t = TF()
    if has_weights:
        print("[4] Transformer...")
        if not tf_load(t, wpath):
            sys.exit(1)
    else:
        print("[4] No weights \u2014 MetaWeights only mode")
        t.V = bpe.vocab_size; t.D = 48; t.NH = 4; t.NL = 1
        t.CTX = 64; t.NC = 2; t.NR = 2; t.NJ = 0; t.HD = 12
        t.tok = [0.0] * (t.V * t.D)
        t.pos = [0.0] * (t.CTX * t.D)
        layer = TFLayer()
        layer.wq = [0.0] * (t.NC * t.HD * t.D)
        layer.wk = [0.0] * (t.NC * t.HD * t.D)
        layer.vc = [0.0] * (t.NC * t.HD * t.D)
        layer.wr = [0.0] * (t.NR * t.D * t.CTX)
        layer.vr = [0.0] * (t.NR * t.HD * t.D)
        layer.wo = [0.0] * (t.D * t.D)
        layer.up = [0.0] * (4 * t.D * t.D)
        layer.dn = [0.0] * (t.D * 4 * t.D)
        t.L = [layer]
        t.kc = [[0.0] * (t.CTX * t.NC * t.HD)]
        t.vcc = [[0.0] * (t.CTX * t.NC * t.HD)]
        t.vrc = [[0.0] * (t.CTX * t.NR * t.HD)]
        t.clen = 0
        t.logits = [0.0] * t.V

    if load_memory(mw, "q.memory", periodic, ch):
        print("  [memory loaded: %d bi, %d tri, %d hebb from q.memory]" % (mw.n_bi, mw.n_tri, mw.n_hebb))

    interference = Interference()
    interference.load_docs("docs", bpe)
    if interference.docs:
        print("[4.5] Interference...")
        print("  %d docs, %d heavy seeds" % (len(interference.docs), sum(len(doc["heavy"]) for doc in interference.docs)))

    ch = Chambers()
    ch_init(ch)

    print("[5] DOE Parliament...")
    parl = Parliament()
    parl_init(parl, t.D, 4)
    print("  %d experts, rank=%d, d_model=%d, alpha=%.2f" % (parl.n, DOE_RANK, t.D, parl.alpha))

    print("\n========== 12 BIDIRECTIONAL STEPS ==========")
    gen_chain(t, bpe, mw, ch, cids, clen_corpus, has_weights, parl, periodic, interference)

    print("\ntype -> 12 sentences. 'quit' to exit.\n")

    try:
        while True:
            sys.stdout.write("  q> ")
            sys.stdout.flush()
            line = sys.stdin.readline()
            if not line:
                break
            inp = line.rstrip("\n")
            if not inp or inp == "quit" or inp == "exit":
                break

            ubytes = inp.encode("utf-8", errors="replace")
            uids = bpe_encode(bpe, ubytes, len(ubytes), 512)
            ingest_ids(mw, uids)
            if len(uids) > 1:
                print("  [ingested %d tokens: +bi +tri +hebb]" % len(uids))
            gen_chain(t, bpe, mw, ch, cids, clen_corpus, has_weights, parl, periodic, interference, inp)
    except (KeyboardInterrupt, EOFError):
        pass

    # save memory
    try:
        save_memory(mw, "q.memory", periodic, ch)
        print("  [memory saved: %d bi, %d tri, %d hebb, %d periodic \u2192 q.memory]" % (mw.n_bi, mw.n_tri, mw.n_hebb, len(periodic.elements)))
    except Exception:
        pass

    print("\nresonance is unbreakable.")
if __name__ == "__main__":
    main()
