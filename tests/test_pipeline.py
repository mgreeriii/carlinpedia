from carlinpedia.ingest.pipeline import run_ingest
from carlinpedia.llm.client import LLMError


def count(conn, sql):
    return conn.execute(sql).fetchone()[0]


def test_full_ingest_builds_the_library(conn, project, fake_llm, embedder):
    report = run_ingest(conn, project, fake_llm, embedder)
    assert [(w.slug, w.status, w.steps_run) for w in report.works] == [
        ("test-album-1992", "ok", ["normalize", "segment", "enrich", "embed"]),
        ("test-special-1990", "ok", ["normalize", "segment", "enrich", "embed"]),
    ]
    assert count(conn, "SELECT count(*) FROM bit") == 8
    assert count(conn, "SELECT count(*) FROM passage") == 13
    assert count(conn, "SELECT count(*) FROM passage WHERE thesis IS NULL") == 0
    assert count(conn, "SELECT count(*) FROM vec_passage_thesis") == 13
    special = next(w for w in report.works if w.slug == "test-special-1990")
    assert special.proposed == ["theme: Security theater"]


def test_second_run_skips_everything(conn, project, fake_llm, embedder):
    run_ingest(conn, project, fake_llm, embedder)
    calls = len(fake_llm.calls)
    report = run_ingest(conn, project, fake_llm, embedder)
    assert [w.status for w in report.works] == ["skipped", "skipped"]
    assert len(fake_llm.calls) == calls


def test_force_from_enrich_reruns_enrich_and_embed_only(conn, project, fake_llm, embedder):
    run_ingest(conn, project, fake_llm, embedder)
    report = run_ingest(conn, project, fake_llm, embedder, slugs=["test-album-1992"], force_from="enrich")
    assert report.works[0].steps_run == ["enrich", "embed"]


def test_changed_transcript_resegments_and_keeps_favorites(conn, project, fake_llm, embedder):
    run_ingest(conn, project, fake_llm, embedder)
    conn.execute("UPDATE passage SET favorite = 1 WHERE text LIKE 'Ever notice your stuff%'")
    conn.commit()
    transcript = project.corpus_dir / "test-special-1990" / "transcript.txt"
    transcript.write_text(transcript.read_text() + "\n")  # new checksum, same normalized text
    report = run_ingest(conn, project, fake_llm, embedder, slugs=["test-special-1990"])
    assert report.works[0].steps_run == ["normalize"]  # normalized text unchanged, so nothing downstream reruns
    transcript.write_text(transcript.read_text().replace("good night", "goodnight"))
    report = run_ingest(conn, project, fake_llm, embedder, slugs=["test-special-1990"])
    assert report.works[0].steps_run == ["normalize", "segment", "enrich", "embed"]
    assert report.works[0].unmatched_favorites == 0
    assert count(conn, "SELECT count(*) FROM passage WHERE favorite = 1") == 1


def test_one_failing_work_does_not_stop_the_others(conn, project, fake_llm, embedder):
    fake_llm.responses["enrich_v1:test-album-1992:1"] = [LLMError("simulated outage")]
    report = run_ingest(conn, project, fake_llm, embedder)
    album, special = report.works
    assert album.status == "failed" and "simulated outage" in album.error
    assert special.status == "ok"
    row = conn.execute("SELECT status, error FROM pipeline_step s JOIN work w ON w.id = s.work_id "
                       "WHERE w.slug = 'test-album-1992' AND step = 'enrich'").fetchone()
    assert row["status"] == "failed"
    fake_llm.responses["enrich_v1:test-album-1992:1"] = [{"passages": [{
        "ordinal": 0, "thesis": "Renaming.", "themes": [], "proposed_themes": [], "targets": [],
        "proposed_targets": [], "tone": "observational", "profanity": False}]}]
    report = run_ingest(conn, project, fake_llm, embedder, slugs=["test-album-1992"])
    assert report.works[0].steps_run == ["enrich", "embed"]


def test_bad_segmentation_marks_needs_review(conn, project, fake_llm, embedder):
    fake_llm.responses["segment_v1:test-album-1992"] = [{"bits": []}]
    report = run_ingest(conn, project, fake_llm, embedder)
    assert report.works[0].status == "needs_review"
    assert "no bits returned" in report.works[0].error
    assert report.works[1].status == "ok"


def test_missing_transcript_fails_only_that_work(conn, project, fake_llm, embedder):
    (project.corpus_dir / "test-album-1992" / "transcript.txt").unlink()
    report = run_ingest(conn, project, fake_llm, embedder)
    assert report.works[0].status == "failed" and "does not exist" in report.works[0].error
    assert report.works[1].status == "ok"
