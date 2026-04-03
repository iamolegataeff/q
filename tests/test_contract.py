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


if __name__ == "__main__":
    unittest.main()
