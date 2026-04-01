# tests/test_eval_store.py
from eval_store import EvalReference, pick_eval_references, get_all_eval_references


class TestEvalStore:
    def test_eval_reference_fields(self):
        refs = get_all_eval_references()
        assert len(refs) >= 2
        for ref in refs:
            assert isinstance(ref, EvalReference)
            assert ref.image_path
            assert ref.description
            assert isinstance(ref.is_good, bool)
            assert 0 <= ref.score <= 10

    def test_good_refs_have_high_score(self):
        refs = get_all_eval_references()
        good = [r for r in refs if r.is_good]
        assert len(good) >= 1
        for r in good:
            assert r.score >= 8.0
            assert len(r.issues) == 0

    def test_bad_refs_have_low_score(self):
        refs = get_all_eval_references()
        bad = [r for r in refs if not r.is_good]
        assert len(bad) >= 1
        for r in bad:
            assert r.score < 6.0
            assert len(r.issues) > 0

    def test_pick_eval_references(self):
        good, bad = pick_eval_references(good_count=1, bad_count=1)
        assert len(good) == 1
        assert len(bad) == 1
        assert good[0].is_good is True
        assert bad[0].is_good is False

    def test_load_bytes(self):
        refs = get_all_eval_references()
        for ref in refs:
            data = ref.load_bytes()
            assert isinstance(data, bytes)
            assert len(data) > 0
