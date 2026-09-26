import pytest

pytestmark = pytest.mark.slow


def test_bge_embedder_links_a_news_story_to_the_right_thesis():
    from carlinpedia.ingest.embed import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder("BAAI/bge-base-en-v1.5", 768)
    theses = [
        "People soften language to avoid facing unpleasant truths.",
        "Consumer life is an endless cycle of buying more things and finding room for them.",
        "Voting is a ritual that gives people the illusion of choice.",
    ]
    docs = embedder.embed_documents(theses)
    query = embedder.embed_query("Company announces layoffs, calls them a workforce optimization initiative")
    assert len(query) == 768
    scores = [sum(a * b for a, b in zip(query, d)) for d in docs]
    assert scores.index(max(scores)) == 0
