import os
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

    @unittest.skipUnless(hasattr(q, 'save_memory_sqlite'), "sqlite memory not yet implemented")
    def test_sqlite_memory_roundtrip(self):
        mw = q.MetaW()
        q.ingest_ids(mw, [1, 2, 3, 2, 1], 0.05)
        q.prophecy_add(mw, 42, 0.7)
        pt = q.PeriodicTable()
        pt.build_from_text("resonance rhythm paradox mystery")
        ch = q.Chambers()
        ch.feel("warmth in the throat and pressure in the chest", pt)
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


if __name__ == "__main__":
    unittest.main()
