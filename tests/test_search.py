import database
from conftest import VALID, FakeGenerator
from rag_engine import HashingEmbedder, RAGEngine


def titles(results):
    return [r["title"] for r in results]


def test_exact_error_search(engine):
    res = engine.search("ModuleNotFoundError: No module named 'pandas'", 5)
    assert "pandas" in res[0]["title"] and res[0]["score"] > 0.25


def test_similar_wording_semantic_search(engine):
    res = engine.search("Why is Python unable to import pandas even though I installed it?", 5)
    assert any("pandas" in t for t in titles(res)[:3])
    res = engine.search("docker container will not start, port already in use", 5)
    assert any("port is already allocated" in t for t in titles(res)[:3])


def test_filters_narrow_results(engine):
    res = engine.search("permission denied", 5, technology="Docker")
    assert res and all(r["technology"] == "Docker" for r in res)
    res = engine.search("error", 10, operating_system="macOS")
    assert all(r["operating_system"] == "macOS" for r in res)


def test_empty_query_returns_nothing(engine):
    assert engine.search("   ") == []


def test_index_persists_and_reloads_without_rebuild(seeded):
    RAGEngine(embedder=HashingEmbedder(), vector_dir=seeded / "vs")
    again = RAGEngine(embedder=HashingEmbedder(), vector_dir=seeded / "vs")
    assert again.loaded_from_disk and not again.rebuilt and again.index.ntotal == 47


def test_index_rebuilt_when_missing_or_corrupt(seeded):
    RAGEngine(embedder=HashingEmbedder(), vector_dir=seeded / "vs")
    (seeded / "vs" / "faiss.index").write_bytes(b"corrupted")
    eng = RAGEngine(embedder=HashingEmbedder(), vector_dir=seeded / "vs")
    assert eng.rebuilt and eng.index.ntotal == 47
    (seeded / "vs" / "faiss.index").unlink()
    assert RAGEngine(embedder=HashingEmbedder(), vector_dir=seeded / "vs").rebuilt


def test_new_experience_becomes_searchable(engine, user):
    exp = database.create_experience(user["id"], {**VALID, "title": "Pillow import fails: No module named PIL",
                                                  "error_message": "ModuleNotFoundError: No module named 'PIL'"})
    assert engine.index_experience(exp["id"]) is True
    res = engine.search("No module named PIL", 3)
    assert res[0]["id"] == exp["id"]
    assert database.experiences_needing_index() == []


class FlakyEmbedder(HashingEmbedder):
    fail = False

    def encode(self, texts):
        if FlakyEmbedder.fail:
            raise RuntimeError("embedding service down")
        return super().encode(texts)


def test_embedding_failure_keeps_record_and_allows_retry(seeded, user):
    eng = RAGEngine(embedder=FlakyEmbedder(), vector_dir=seeded / "vs")
    exp = database.create_experience(user["id"], VALID)
    FlakyEmbedder.fail = True
    try:
        assert eng.index_experience(exp["id"]) is False
        assert database.get_experience(exp["id"]) is not None          # record is NOT lost
        assert exp["id"] in database.experiences_needing_index()       # flagged as pending
    finally:
        FlakyEmbedder.fail = False
    assert eng.index_pending() == (1, 0)
    assert eng.search("No module named requests", 3)[0]["id"] == exp["id"]


def test_accepting_solution_refreshes_knowledge_record(engine, user):
    exp = database.create_experience(user["id"], VALID)
    engine.index_experience(exp["id"])
    sol = database.add_solution(exp["id"], user["id"], "Use the quarantine-zebra workaround for this.")
    database.accept_solution(sol["id"], user["id"])
    engine.index_experience(exp["id"])
    assert "quarantine-zebra" in database.get_knowledge_content(exp["id"])
