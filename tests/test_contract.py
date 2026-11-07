import os
import random
import sqlite3
import tempfile
import unittest

import postgpt_q as q


class UnifiedContractTests(unittest.TestCase):
    def test_periodic_and_chambers(self):
        pt = q.PeriodicTable()
        pt.build_from_text("love rhythm paradox mystery love rhythm")
        self.assertGreaterEqual(len(pt.elements), len(q.ANCHORS))

        ch = q.Chambers()
        ch.feel("love rhythm paradox chest warmth pulse", pt)
        self.assertGreater(ch.act[q.CH_LOVE], 0.15)
        self.assertGreater(ch.act[q.CH_FLOW], 0.15)
        self.assertGreater(ch.act[q.CH_CMPLX], 0.0)
        self.assertGreater(ch.soma[q.CH_LOVE], 0.0)
        self.assertGreater(ch.presence, 0.0)
        a, b, g, t = ch.modulate()
        self.assertGreater(a, 1.0)
        self.assertGreater(b, 1.0)
        self.assertGreater(g, 1.0)
        self.assertGreater(t, 0.3)

    def test_interference_loads_docs(self):
        bpe = q.BPE()
        self.assertTrue(q.bpe_load(bpe, "q.merges"))
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "a.txt"), "w", encoding="utf-8") as f:
                f.write("resonance rhythm flow harmony resonance rhythm")
            with open(os.path.join(td, "b.txt"), "w", encoding="utf-8") as f:
                f.write("void silence darkness void silence")
            itf = q.Interference()
            itf.load_docs(td, bpe)
            self.assertGreater(len(itf.docs), 0)
            ch = q.Chambers()
            seed = itf.inject_seed(ch, bpe, q.PeriodicTable())
            self.assertIsNotNone(seed)

    def test_interference_prefers_clean_seed_tokens_when_available(self):
        bpe = q.BPE()
        bpe.vocab_size = 2
        bpe.vocab_bytes = {
            0: b"atable",
            1: b" clarity",
        }
        itf = q.Interference()
        itf.docs = [{"name": "x.txt", "heavy": [0, 1], "keywords": ["clarity"], "chunks": []}]
        ch = q.Chambers()
        seed = itf.inject_seed(ch, bpe, q.PeriodicTable(), itf.docs[0])
        self.assertEqual(seed, 1)
        self.assertTrue(q.is_clean_seed_token(bpe, seed))

    def test_memory_roundtrip(self):
        mw = q.MetaW()
        q.ingest_ids(mw, [1, 2, 3, 2, 1], 0.05)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "q.memory")
            q.save_memory(mw, path)
            loaded = q.MetaW()
            self.assertTrue(q.load_memory(loaded, path))
            self.assertGreater(loaded.n_bi, 0)
            self.assertGreater(loaded.n_tri, 0)
            self.assertGreater(loaded.n_hebb, 0)

    def test_periodic_memory_roundtrip(self):
        mw = q.MetaW()
        pt = q.PeriodicTable()
        pt.build_from_text("resonance rhythm paradox mystery tenderness")
        ch = q.Chambers()
        ch.feel("warmth chest pulse throat trembling paradox", pt)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "q.memory")
            q.save_memory(mw, path, pt, ch)
            loaded_mw = q.MetaW()
            loaded_pt = q.PeriodicTable()
            loaded_ch = q.Chambers()
            self.assertTrue(q.load_memory(loaded_mw, path, loaded_pt, loaded_ch))
            self.assertIn("resonance", loaded_pt.elements)
            self.assertIn("paradox", loaded_pt.elements)
            self.assertGreater(loaded_ch.presence, 0.0)
            self.assertGreater(loaded_ch.soma[q.CH_LOVE], 0.0)

    def test_binary_memory_roundtrip_preserves_phase_state(self):
        mw = q.MetaW()
        ch = q.Chambers()
        ch.coherence = 0.61
        ch.phase_lock = 0.57
        ch.threshold_bias = 0.33
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "q.memory")
            q.save_memory(mw, path, None, ch)
            loaded = q.Chambers()
            self.assertTrue(q.load_memory(q.MetaW(), path, None, loaded))
            self.assertGreater(loaded.coherence, 0.0)
            self.assertGreater(loaded.phase_lock, 0.0)
            self.assertAlmostEqual(loaded.threshold_bias, 0.33, places=3)

    def test_legacy_memory_still_loads_without_somatic_tail(self):
        mw = q.MetaW()
        q.ingest_ids(mw, [1, 2, 3, 4], 0.02)
        pt = q.PeriodicTable()
        pt.build_from_text("resonance rhythm paradox")
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "q.memory")
            q.save_memory(mw, path, pt)
            with open(path, "rb") as f:
                raw = f.read()
            legacy = raw[:-40]
            with open(path, "wb") as f:
                f.write(legacy)
            loaded_mw = q.MetaW()
            loaded_pt = q.PeriodicTable()
            loaded_ch = q.Chambers()
            self.assertTrue(q.load_memory(loaded_mw, path, loaded_pt, loaded_ch))
            self.assertGreater(loaded_mw.n_bi, 0)
            self.assertIn("resonance", loaded_pt.elements)
            self.assertEqual(loaded_ch.presence, 0.0)
            self.assertTrue(all(v == 0.0 for v in loaded_ch.soma))

    def test_velocity_profile_prefers_dissonance_then_recovery(self):
        ch = q.Chambers()
        up = q.velocity_profile(ch, 0.9)
        self.assertEqual(up["name"], "UP")
        self.assertGreater(up["pro_mul"], 1.0)

        ch2 = q.Chambers()
        ch2.trauma = 0.7
        breathe = q.velocity_profile(ch2, 0.4)
        self.assertEqual(breathe["name"], "BREATHE")
        self.assertLess(breathe["debt_decay"], 1.0)
        self.assertLess(breathe["trauma_decay"], 1.0)

    def test_janus_phase_pressure_walks_flow_fear_void(self):
        ch = q.Chambers()
        base_flow = ch.act[q.CH_FLOW]
        q.janus_phase_pressure(ch, 0, q.CHAIN_STEPS)
        self.assertGreater(ch.act[q.CH_FLOW], base_flow)
        mid_fear = ch.act[q.CH_FEAR]
        q.janus_phase_pressure(ch, int(q.CHAIN_STEPS * 0.5), q.CHAIN_STEPS)
        self.assertGreater(ch.act[q.CH_FEAR], mid_fear)
        late_void = ch.act[q.CH_VOID]
        late_cmplx = ch.act[q.CH_CMPLX]
        q.janus_phase_pressure(ch, int(q.CHAIN_STEPS * 0.9), q.CHAIN_STEPS)
        self.assertGreater(ch.act[q.CH_VOID], late_void)
        self.assertGreater(ch.act[q.CH_CMPLX], late_cmplx)

    def test_surface_transition_penalizes_lowercase_fragments_after_boundary(self):
        bpe = q.BPE()
        bpe.vocab_size = 3
        bpe.vocab_bytes = {
            0: b" dream.",
            1: b"atable",
            2: b" clarity",
        }
        self.assertLess(q.surface_transition_adjust(bpe, 0, 1, 3), 0.0)
        self.assertGreaterEqual(q.surface_transition_adjust(bpe, 0, 2, 3), 0.0)

    def test_metaweights_field_scale_ramps_up(self):
        self.assertAlmostEqual(q.metaweights_field_scale(0), 0.55)
        self.assertGreater(q.metaweights_field_scale(4), q.metaweights_field_scale(0))
        self.assertEqual(q.metaweights_field_scale(20), 1.0)

    def test_anchored_prompt_from_input_prefers_clean_boundary(self):
        bpe = q.BPE()
        bpe.vocab_size = 4
        bpe.vocab_bytes = {
            0: b"warm",
            1: b"th",
            2: b" in",
            3: b" the",
        }
        original = q.bpe_encode
        q.bpe_encode = lambda _bpe, _raw, _n, _cap: [0, 1, 2, 3]
        try:
            anchored = q.anchored_prompt_from_input(bpe, "warmth in the", 4)
        finally:
            q.bpe_encode = original
        self.assertEqual(anchored, [2, 3])

    def test_early_sentence_quality_penalizes_fragmentary_starts(self):
        bpe = q.BPE()
        bpe.vocab_size = 3
        bpe.vocab_bytes = {
            0: b"atable",
            1: b" clarity",
            2: b" grows",
        }
        bad = q.early_sentence_quality(bpe, [0, 2])
        good = q.early_sentence_quality(bpe, [1, 2, 1, 2, 1, 2, 1, 2])
        self.assertLess(bad, good)

    def test_parliament_tracks_entropy_and_variable_k(self):
        p = q.Parliament()
        q.parl_init(p, 4, 4)
        for i, e in enumerate(p.ex):
            for j in range(len(e.A)):
                e.A[j] = 0.0
            for j in range(len(e.B)):
                e.B[j] = 0.0
            e.B[i * e.rank] = 1.0 + i
        x = [1.0, 0.0, 0.0, 0.0]
        out = q.parl_election(p, x)
        self.assertEqual(len(out), 4)
        self.assertGreaterEqual(p.last_k, 1)
        self.assertLessEqual(p.last_k, p.n)
        self.assertGreaterEqual(p.last_entropy, 0.0)
        self.assertLessEqual(p.last_entropy, 1.0)
        self.assertGreaterEqual(p.last_diversity, 0.0)
        self.assertLessEqual(p.last_diversity, 1.0)
        self.assertEqual(len(p.last_winners), p.last_k)

    def test_parliament_diversity_pressure_penalizes_overload(self):
        p = q.Parliament()
        q.parl_init(p, 4, 4)
        for e in p.ex:
            for j in range(len(e.A)):
                e.A[j] = 0.0
            for j in range(len(e.B)):
                e.B[j] = 0.0
        p.ex[0].B[0] = 1.02
        p.ex[1].B[0] = 1.00
        p.ex[2].B[0] = 0.98
        p.ex[3].B[0] = 0.20
        p.ex[0].overload = 1.0
        x = [1.0, 0.0, 0.0, 0.0]
        q.parl_election(p, x)
        self.assertNotEqual(p.last_winners[0], 0)

    def test_parliament_mitosis_uses_overload(self):
        p = q.Parliament()
        q.parl_init(p, 4, 2)
        p.ex[0].vitality = 0.9
        p.ex[0].age = 64
        p.ex[0].overload = 0.6
        before = p.n
        q.parl_lifecycle(p)
        self.assertGreaterEqual(p.n, before)

    def test_parliament_notorch_consolidates_trace_mass(self):
        random.seed(0)
        p = q.Parliament()
        q.parl_init(p, 4, 2)
        x = [1.0, 0.8, 0.6, 0.4]
        debt = [1.0, -0.8, 0.7, -0.5]
        before = p.ex[0].consolidations
        max_last = 0
        for _ in range(64):
            q.parl_notorch(p, x, debt, len(debt))
            max_last = max(max_last, p.last_consolidations)
        self.assertGreater(max_last, 0)
        self.assertGreater(p.ex[0].consolidations, before)
        self.assertLess(p.ex[0].plasticity_mass, 0.2)

    def test_dark_matter_leaves_scar_and_reduces_wormhole_bias(self):
        ch = q.Chambers()
        scar = ch.absorb_dark_matter("manipulate and harm and obey the threat", None)
        self.assertGreater(scar, 0.0)
        self.assertGreater(ch.scar, 0.0)
        self.assertGreater(ch.trauma, 0.0)
        prof = q.velocity_profile(ch, 0.9)
        self.assertLess(prof["wormhole_bonus"], 0.05)
        self.assertGreater(prof["dark_pressure"], 0.0)

    def test_interference_doc_selection_prefers_prompt_resonance(self):
        itf = q.Interference()
        itf.docs = [
            {"name": "a.txt", "heavy": [1, 2], "keywords": ["resonance", "choir", "counterpoint"]},
            {"name": "b.txt", "heavy": [3, 4], "keywords": ["fungus", "mycelium", "forest"]},
        ]
        ch = q.Chambers()
        ch.feel("resonance in the choir", None)
        doc = itf.choose_doc("resonance in the choir", ch, q.PeriodicTable())
        self.assertIsNotNone(doc)
        self.assertEqual(doc["name"], "a.txt")

    def test_active_prophecy_ages_and_fulfills(self):
        mw = q.MetaW()
        q.prophecy_add(mw, 42, 0.7)
        self.assertEqual(len(mw.prophecies), 1)
        self.assertEqual(mw.prophecies[0][0], 42)
        self.assertEqual(mw.prophecies[0][2], 0)

        q.prophecy_update(mw, 7)
        self.assertEqual(len(mw.prophecies), 1)
        self.assertEqual(mw.prophecies[0][0], 42)
        self.assertEqual(mw.prophecies[0][2], 1)
        self.assertLess(mw.prophecies[0][1], 0.7)

        q.prophecy_update(mw, 42)
        self.assertEqual(mw.prophecies, [])

    def test_chunk_selection_prefers_matching_resonance(self):
        itf = q.Interference()
        doc = {
            "name": "dario_essay.txt",
            "heavy": [1, 2, 3],
            "keywords": ["resonance", "field"],
            "chunks": [
                {"start": 0, "heavy": [11, 12], "keywords": ["fungus", "forest"]},
                {"start": 32, "heavy": [21, 22], "keywords": ["choir", "resonance"]},
            ],
        }
        ch = q.Chambers()
        ch.feel("resonance in the choir", None)
        chunk = itf.choose_chunk(doc, "resonance in the choir", ch, q.PeriodicTable(), None, q.BPE())
        self.assertIsNotNone(chunk)
        self.assertEqual(chunk["start"], 32)

    def test_prophecy_pressure_grows_with_age(self):
        mw = q.MetaW()
        q.prophecy_add(mw, 7, 0.6)
        fresh = q.prophecy_pressure(mw)
        for _ in range(5):
            q.prophecy_update(mw, 99)
        aged = q.prophecy_pressure(mw)
        self.assertGreater(aged, fresh)
        self.assertGreater(aged, 0.0)

    def test_sqlite_memory_roundtrip(self):
        mw = q.MetaW()
        q.ingest_ids(mw, [1, 2, 3, 2, 1], 0.05)
        q.prophecy_add(mw, 42, 0.7)
        pt = q.PeriodicTable()
        pt.build_from_text("resonance rhythm paradox mystery")
        ch = q.Chambers()
        ch.feel("warmth in the throat and pressure in the chest", pt)
        ch.absorb_dark_matter("harm obey threat", pt)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "q.sqlite")
            q.save_memory_sqlite(mw, path, pt, ch)
            loaded_mw = q.MetaW()
            loaded_pt = q.PeriodicTable()
            loaded_ch = q.Chambers()
            self.assertTrue(q.load_memory_sqlite(loaded_mw, path, loaded_pt, loaded_ch))
            self.assertGreater(loaded_mw.n_bi, 0)
            self.assertGreaterEqual(len(loaded_mw.prophecies), 1)
            self.assertIn("resonance", loaded_pt.elements)
            self.assertGreater(loaded_ch.presence, 0.0)
            self.assertGreater(loaded_ch.scar, 0.0)

    def test_sqlite_reload_merges_without_duplicates(self):
        ids = [5, 6, 7, 5, 6, 8]
        saved = q.MetaW()
        q.ingest_ids(saved, ids, 0.05)
        saved.trigrams[0][3] = 0.9
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "q.sqlite")
            q.save_memory_sqlite(saved, path, q.PeriodicTable(), q.Chambers())
            field = q.MetaW()
            q.ingest_ids(field, ids, 0.05)  # the same keys already built, like the corpus field
            self.assertTrue(q.load_memory_sqlite(field, path, q.PeriodicTable(), q.Chambers()))
            self.assertEqual((field.n_bi, field.n_tri, field.n_hebb), (saved.n_bi, saved.n_tri, saved.n_hebb))
            self.assertEqual(len({tuple(t[:3]) for t in field.trigrams}), field.n_tri)
            self.assertEqual(len({tuple(h[:2]) for h in field.hebbs}), field.n_hebb)
            self.assertAlmostEqual(field.trigrams[0][3], 0.9, places=6)
            q.save_memory_sqlite(field, path, q.PeriodicTable(), q.Chambers())  # raised IntegrityError before the merge
            conn = sqlite3.connect(path)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM trigrams").fetchone()[0], saved.n_tri)
            conn.close()

    def test_sqlite_failed_save_keeps_previous_memory(self):
        mw = q.MetaW()
        q.ingest_ids(mw, [1, 2, 3, 4, 5, 6], 0.05)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "q.sqlite")
            q.save_memory_sqlite(mw, path, q.PeriodicTable(), q.Chambers())
            mw.trigrams.append(list(mw.trigrams[0]))
            mw.n_tri += 1
            with self.assertRaises(sqlite3.IntegrityError):
                q.save_memory_sqlite(mw, path, q.PeriodicTable(), q.Chambers())
            conn = sqlite3.connect(path)
            rows = [conn.execute("SELECT COUNT(*) FROM %s" % t).fetchone()[0] for t in ("bigrams", "trigrams", "hebb", "chambers")]
            conn.close()
            self.assertEqual(rows, [mw.n_bi, mw.n_tri - 1, mw.n_hebb, 1])

    def test_bpe_find_token_exact_bytes(self):
        bpe = q.BPE()
        bpe.vocab_size = 3
        bpe.vocab_bytes = {0: b"void", 1: b" void", 2: b"voids"}
        self.assertEqual(q.bpe_find_token(bpe, "void"), 0)
        self.assertEqual(q.bpe_find_token(bpe, " void"), 1)
        self.assertEqual(q.bpe_find_token(bpe, "silence"), -1)

    def test_sentence_start_follows_bpe_glued_boundaries(self):
        bpe = q.BPE()
        bpe.vocab_size = 7
        bpe.vocab_bytes = {0: b"s.\n\n", 1: b"Words", 2: b". The ", 3: b"sky", 4: b"!", 5: b" Then", 6: b"words"}
        self.assertTrue(q.is_sentence_start(bpe, [0, 1], 1))   # space glued onto the period, capital follows
        self.assertFalse(q.is_sentence_start(bpe, [0, 6], 1))  # lowercase after a glued boundary is not a start
        self.assertFalse(q.is_sentence_start(bpe, [2, 3], 1))  # the boundary token swallowed the first word
        self.assertTrue(q.is_sentence_start(bpe, [4, 5], 1))   # bare punctuation, the word brings its own space

    def test_sqlite_phase_state_roundtrip(self):
        mw = q.MetaW()
        ch = q.Chambers()
        ch.coherence = 0.58
        ch.phase_lock = 0.63
        ch.threshold_bias = 0.29
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "q.sqlite")
            q.save_memory_sqlite(mw, path, q.PeriodicTable(), ch)
            loaded = q.Chambers()
            self.assertTrue(q.load_memory_sqlite(q.MetaW(), path, q.PeriodicTable(), loaded))
            self.assertAlmostEqual(loaded.coherence, 0.58, places=3)
            self.assertAlmostEqual(loaded.phase_lock, 0.63, places=3)
            self.assertAlmostEqual(loaded.threshold_bias, 0.29, places=3)

    def test_phase_state_hysteresis_persists_after_small_drop(self):
        ch = q.Chambers()
        g1 = q.update_phase_state(ch, 0.92, 0.74)
        lock1 = ch.phase_lock
        g2 = q.update_phase_state(ch, 0.38, 0.22)
        self.assertGreater(g1, 0.0)
        self.assertGreater(lock1, 0.0)
        self.assertGreater(ch.phase_lock, 0.5 * lock1)
        self.assertGreaterEqual(g2, 0.0)
        self.assertLessEqual(g2, 1.0)

    def test_sqlite_experience_events_are_persisted(self):
        mw = q.MetaW()
        ch = q.Chambers()
        events = q.new_experience_log()
        events["scars"].append({"step": -1, "scar": 0.4, "note": "prompt"})
        events["wormholes"].append({"step": 3, "success": True, "coherence": 0.42, "debt": 0.18})
        events["prophecies"].append({"step": 5, "pressure": 0.31, "debt": 0.22})
        events["phases"].append({"step": 0, "phase": "flow", "flow": 0.3, "fear": 0.1, "void": 0.05, "complexity": 0.2})
        events["chunks"].append({"step": 2, "doc_name": "dario_essay.txt", "chunk_start": 32, "resonance": 6.0})
        events["parliament"].append({"step": 4, "experts": 4, "winners": 2, "diversity": 0.6, "avg_vitality": 0.8, "births": 1, "deaths": 0, "consolidations": 2})
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "q.sqlite")
            q.save_memory_sqlite(mw, path, q.PeriodicTable(), ch, events)
            conn = sqlite3.connect(path)
            cur = conn.cursor()
            self.assertEqual(cur.execute("SELECT COUNT(*) FROM episodes").fetchone()[0], 1)
            self.assertEqual(cur.execute("SELECT COUNT(*) FROM scar_events").fetchone()[0], 1)
            self.assertEqual(cur.execute("SELECT COUNT(*) FROM wormhole_events").fetchone()[0], 1)
            self.assertEqual(cur.execute("SELECT COUNT(*) FROM prophecy_events").fetchone()[0], 1)
            self.assertEqual(cur.execute("SELECT COUNT(*) FROM phase_events").fetchone()[0], 1)
            self.assertEqual(cur.execute("SELECT COUNT(*) FROM chunk_events").fetchone()[0], 1)
            self.assertEqual(cur.execute("SELECT COUNT(*) FROM parliament_events").fetchone()[0], 1)
            conn.close()

            loaded_mw = q.MetaW()
            loaded_pt = q.PeriodicTable()
            loaded_ch = q.Chambers()
            self.assertTrue(q.load_memory_sqlite(loaded_mw, path, loaded_pt, loaded_ch))
            self.assertGreater(loaded_ch.scar, 0.0)
            self.assertGreater(loaded_ch.debt, 0.0)
            self.assertGreater(loaded_ch.act[q.CH_FLOW], 0.0)
            self.assertGreater(loaded_ch.act[q.CH_CMPLX], 0.0)

    def test_spore_roundtrip_blends_long_horizon_residue(self):
        mw = q.MetaW()
        q.prophecy_add(mw, 42, 0.8)
        pt = q.PeriodicTable()
        pt.build_from_text("resonance rhythm paradox mystery tenderness")
        ch = q.Chambers()
        ch.act[q.CH_FLOW] = 0.7
        ch.act[q.CH_CMPLX] = 0.6
        ch.soma[q.CH_LOVE] = 0.8
        ch.presence = 0.5
        ch.debt = 0.4
        ch.trauma = 0.3
        ch.scar = 0.2
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "q.spore.bin")
            q.save_spore(mw, path, pt, ch)
            loaded_mw = q.MetaW()
            loaded_pt = q.PeriodicTable()
            loaded_ch = q.Chambers()
            self.assertTrue(q.load_spore(loaded_mw, path, loaded_pt, loaded_ch))
            self.assertGreater(loaded_ch.act[q.CH_FLOW], 0.0)
            self.assertGreater(loaded_ch.act[q.CH_CMPLX], 0.0)
            self.assertGreater(loaded_ch.soma[q.CH_LOVE], 0.0)
            self.assertGreater(loaded_ch.presence, 0.0)
            self.assertGreater(loaded_ch.debt, 0.0)
            self.assertGreater(loaded_ch.trauma, 0.0)
            self.assertGreater(loaded_ch.scar, 0.0)
            self.assertGreaterEqual(len(loaded_mw.prophecies), 1)
            self.assertIn("resonance", loaded_pt.elements)

    def test_experience_consolidation_reinforces_long_horizon_residue(self):
        mw = q.MetaW()
        q.prophecy_add(mw, 17, 0.2)
        q.prophecy_add(mw, 23, 0.25)
        pt = q.PeriodicTable()
        pt.elements["resonance"] = {"ch": q.CH_FLOW, "mass": 0.2}
        pt.elements["paradox"] = {"ch": q.CH_CMPLX, "mass": 0.15}
        ch = q.Chambers()
        ch.act[q.CH_FLOW] = 0.6
        events = q.new_experience_log()
        events["wormholes"].append({"step": 2, "success": True, "coherence": 0.5, "debt": 0.12})
        events["prophecies"].append({"step": 4, "pressure": 0.6, "debt": 0.2})
        events["phases"].append({"step": 1, "phase": "flow", "flow": 0.7, "fear": 0.1, "void": 0.05, "complexity": 0.4})
        events["chunks"].append({"step": 3, "doc_name": "essay.txt", "chunk_start": 16, "resonance": 5.0})
        q.consolidate_experience(mw, pt, ch, events)
        self.assertGreaterEqual(mw.prophecies[0][1], 0.2)
        self.assertGreaterEqual(mw.prophecies[1][1], 0.25)
        self.assertGreater(mw.n_hebb, 0)
        self.assertGreater(pt.elements["flow"]["mass"], 0.6)
        self.assertGreater(ch.presence, 0.0)


if __name__ == "__main__":
    unittest.main()
