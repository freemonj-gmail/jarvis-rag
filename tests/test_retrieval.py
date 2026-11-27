"""Retrieval checks: vector-store retention and exact search, embedding-model
loading and batching, query phrasing, and evidence ordering.

Uses mocked stores and models with synthetic text, never real documents or a
live Milvus. Run with ``make retrieval-test`` or directly through uv.
"""

import contextlib
import io
import os
import re
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery
from jarvisrag.vectorstore import MilvusVectorStore, create_milvus_vector_store


class RetentionTests(unittest.TestCase):
    def test_add_never_evicts_or_queries_capacity(self):
        store = create_milvus_vector_store(dim=8)
        store._client = Mock()
        store._collection = store._client
        store._client.has_collection.return_value = True
        nodes = [TextNode(id_=f"synthetic-{ordinal}", text="Synthetic retention check.", embedding=[0.5] * 8) for ordinal in range(6000)]
        with patch.object(MilvusVectorStore, "_get_collection_size", side_effect=AssertionError("unexpected capacity query")):
            self.assertEqual(len(store.add(nodes)), 6000)
        self.assertEqual(len(store._client.insert.call_args.args[1]), 6000)
        store._client.delete.assert_not_called()
        store._client.query.assert_not_called()

    def test_factory_has_no_automatic_eviction_setting(self):
        self.assertNotIn("max_vectors", MilvusVectorStore.model_fields)

    def test_failed_child_enumeration_is_not_an_empty_index(self):
        store = create_milvus_vector_store()
        store._client = Mock()
        store._client.has_collection.return_value = True
        iterator = store._client.query_iterator.return_value
        iterator.next.side_effect = RuntimeError("synthetic read failure")
        with self.assertRaisesRegex(RuntimeError, "read failure"):
            store.indexed_node_ids()
        iterator.close.assert_called_once()

    def test_strict_deletion_propagates_failures(self):
        store = create_milvus_vector_store()
        store._client = Mock()
        store._client.has_collection.return_value = True
        store._client.delete.side_effect = RuntimeError("synthetic deletion failure")
        with self.assertRaisesRegex(RuntimeError, "deletion failure"):
            store.delete_nodes(["synthetic-child"], strict=True)
        store.delete_nodes(["synthetic-child"])


class CacheFirstModelLoadTests(unittest.TestCase):
    """Cached models load with the Hub off; only an uncached model goes online."""

    def test_cached_model_loads_without_the_hub_and_restores_the_setting(self):
        from huggingface_hub import constants
        from jarvisrag.answer.bootstrap import _load_from_cache_first

        before = constants.HF_HUB_OFFLINE
        seen = []
        result = _load_from_cache_first(lambda: seen.append(constants.HF_HUB_OFFLINE) or "model")
        self.assertEqual(result, "model")
        self.assertEqual(seen, [True])
        self.assertEqual(constants.HF_HUB_OFFLINE, before)

    def test_uncached_model_falls_back_to_an_online_load(self):
        from huggingface_hub import constants
        from jarvisrag.answer.bootstrap import _load_from_cache_first

        attempts = []

        def load():
            attempts.append(constants.HF_HUB_OFFLINE)
            if constants.HF_HUB_OFFLINE:
                raise OSError("synthetic: not in the local cache")
            return "downloaded"

        # The suite imports a benchmark script that sets HF_HUB_OFFLINE=1, so
        # start from the online setting a real first run has.
        with patch.object(constants, "HF_HUB_OFFLINE", False):
            self.assertEqual(_load_from_cache_first(load), "downloaded")
            self.assertEqual(attempts, [True, False])
            self.assertFalse(constants.HF_HUB_OFFLINE)


class BoundedEmbeddingShapeTests(unittest.TestCase):
    """Fixed-shape MPS batching: bounded shapes, original order, CPU untouched."""

    def _fake_model(self, device):
        class Module:
            max_seq_length = 8192

            def __init__(self):
                self.tokenizer = lambda texts, **kw: {"input_ids": [[0] * len(t.split()) for t in texts]}
                self.tokenize = lambda texts, *a, **kw: {}

        class SentenceModel(list):
            pass

        sentence = SentenceModel([Module()])
        sentence.device = device

        class Model:
            embed_batch_size = 32

        model = Model()
        model._model = sentence
        model.calls = []

        def stock_embed(inputs, prompt_name=None):
            model.calls.append((len(inputs), model.embed_batch_size))
            return [[float(len(text.split()))] for text in inputs]

        model._embed = stock_embed
        return model

    def test_padded_lengths_form_a_small_fixed_set(self):
        from jarvisrag.embeddings import _padded_length

        sizes = {_padded_length(n, 8192) for n in range(1, 8193)}
        self.assertEqual(sorted(sizes), [128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192])

    def test_non_mps_model_is_left_untouched(self):
        from jarvisrag.embeddings import bound_mps_embedding_shapes

        model = self._fake_model("cpu")
        original = model._embed
        self.assertFalse(bound_mps_embedding_shapes(model))
        self.assertIs(model._embed, original)

    def test_batches_use_bounded_sizes_and_keep_input_order(self):
        from jarvisrag.embeddings import bound_mps_embedding_shapes

        model = self._fake_model("mps")
        self.assertTrue(bound_mps_embedding_shapes(model))
        lengths = [5, 300, 40, 900, 7, 1300, 120, 64] * 7
        texts = [" ".join(["w"] * n) for n in lengths]
        vectors = model._embed(texts)
        self.assertEqual([v[0] for v in vectors], [float(n) for n in lengths])
        self.assertEqual(model.embed_batch_size, 32)
        for count, size in model.calls:
            self.assertEqual(count % size, 0, "every batch is full at its own size")
            self.assertTrue(size & (size - 1) == 0 or size in {32, 21, 16, 10, 8, 5, 4, 2, 1})


class ExactVectorSearchTests(unittest.TestCase):
    def _existing_store(self, index_type):
        store = create_milvus_vector_store(dim=8)
        store._client = Mock()
        store._client.has_collection.return_value = True
        store._client.describe_collection.return_value = {"fields": [{"name": "embedding", "params": {"dim": 8}}]}
        store._client.list_indexes.return_value = [f"embedding_{index_type.lower()}"]
        store._client.describe_index.return_value = {"index_type": index_type}
        return store

    def test_existing_ann_index_is_replaced_by_exact_index_in_place(self):
        store = self._existing_store("HNSW")
        with contextlib.redirect_stdout(io.StringIO()):
            store._ensure_collection(8)
        store._client.release_collection.assert_called_once()
        store._client.drop_index.assert_called_once_with(store.collection_name, "embedding_hnsw")
        params = store._client.create_index.call_args.kwargs["index_params"]
        self.assertEqual([p.index_type for p in params], ["FLAT"])
        store._client.load_collection.assert_called()
        store._client.drop_collection.assert_not_called()
        store._client.delete.assert_not_called()

    def test_existing_exact_index_is_left_alone(self):
        store = self._existing_store("FLAT")
        store._ensure_collection(8)
        store._client.drop_index.assert_not_called()
        store._client.create_index.assert_not_called()

    def test_search_is_exact(self):
        store = self._existing_store("FLAT")
        store._client.search.return_value = [[{"node_id": "synthetic", "distance": 0.9}]]
        result = store.query(VectorStoreQuery(query_embedding=[1.0] * 8, similarity_top_k=32))
        self.assertEqual(result.ids, ["synthetic"])
        self.assertEqual(store._client.search.call_args.kwargs["search_params"], {"metric_type": "IP"})

    def test_first_search_on_a_warm_start_replaces_an_ann_index(self):
        store = self._existing_store("HNSW")
        store._client.search.return_value = [[{"node_id": "synthetic", "distance": 0.9}]]
        with contextlib.redirect_stdout(io.StringIO()):
            store.query(VectorStoreQuery(query_embedding=[1.0] * 8, similarity_top_k=32))
            store.query(VectorStoreQuery(query_embedding=[1.0] * 8, similarity_top_k=32))
        store._client.drop_index.assert_called_once_with(store.collection_name, "embedding_hnsw")
        self.assertEqual(store._client.create_index.call_count, 1)


class DetailSnippetScoringTests(unittest.TestCase):
    """Detail snippets are chosen by structure (values, labels, readable words), not keyword lists."""

    def test_labeled_values_beat_filler_prose(self):
        from jarvisrag.text.query import _best_detail_candidate, _detail_score

        prose = "Thank you for taking the time to read this message carefully today."
        detail = "Reference number: 48213; Renewal date: 03/15/2027; Fee: $1,240.00 per period"
        self.assertGreater(_detail_score(detail), _detail_score(prose))
        _score, best = _best_detail_candidate(prose + "\n" + detail)
        self.assertIn("48213", best)

    def test_number_only_noise_is_rejected(self):
        from jarvisrag.text.query import _detail_score

        self.assertEqual(_detail_score("12 34 56 78 90 11 22 33 44 55 66 77"), 0)


class QueryLexiconTests(unittest.TestCase):
    """Retrieval phrasings and the evidence gate use only the question's own words:
    no hardcoded aliases (state names, form numbers) or padding lists."""

    QUESTION = "What is the state payment total for the 2026 filing?"

    def test_search_phrasings_use_only_the_question_words(self):
        from jarvisrag.text.query import _retrieval_query_variants

        allowed = set(re.findall(r"[a-z0-9]+", self.QUESTION.lower())) | {
            "find", "indexed", "evidence", "for", "original", "question",
        }
        for variant in _retrieval_query_variants(self.QUESTION):
            extra = set(re.findall(r"[a-z0-9]+", variant.lower())) - allowed
            self.assertEqual(extra, set(), variant)

    def test_gate_matches_the_question_words_not_aliases(self):
        from jarvisrag.retrieval import _query_evidence_is_supported
        from jarvisrag.text.query import _query_anchor_terms, _query_terms

        terms = _query_terms(self.QUESTION)
        anchors = _query_anchor_terms(self.QUESTION, terms)
        on_topic = "The state payment total for the 2026 filing was $1,250.00."
        alias_only = "Comptroller remittance by direct debit: $1,250.00."
        self.assertTrue(_query_evidence_is_supported(on_topic, terms, anchors, True))
        self.assertFalse(_query_evidence_is_supported(alias_only, terms, anchors, True))

    def test_plural_forms_are_added_once_and_only_for_a_single_trailing_s(self):
        from jarvisrag.text.query import _retrieval_query_variants, _term_family

        self.assertEqual(_term_family("standard"), ["standard"])
        self.assertEqual(_term_family("positions"), ["positions", "position"])
        self.assertEqual(_term_family("address"), ["address"])
        for variant in _retrieval_query_variants("What is the standard deduction?"):
            if "original question" in variant.lower():
                continue
            words = variant.lower().split()
            self.assertEqual(len(words), len(set(words)), variant)


class HomeReserveSemanticOrderTests(unittest.TestCase):
    """The reserved launch-folder slots go to the chunks closest in meaning to the
    question, even when they share no words with it and search never returned them."""

    class _Node:
        def __init__(self, node_id, path, text):
            self.node_id, self.metadata, self._text = node_id, {"file_path": path, "file_name": os.path.basename(path)}, text

        def get_content(self):
            return self._text

    def _nodes(self):
        home = [self._Node(f"h{i}", f"/tmp/homedir/note{i}.txt", f"Household note {i}: water the plants, buy milk, item {i * 7}.")
                for i in range(4)]
        home.append(self._Node("h-cv", "/tmp/homedir/cv.txt",
                               "Professional experience. Senior engineer at Acme Corporation from 2001 to 2005, "
                               "leading the platform team. Analyst at Beta Consulting from 2005 to 2010, "
                               "responsible for data modeling and reporting for regional clients."))
        other = [self._Node(f"o{i}", f"/tmp/other/doc{i}.txt", f"Jobs held record {i}: position {i} held at site {i * 11}.")
                 for i in range(7)]
        return other + home

    def _select(self, similarities):
        import jarvisrag.retrieval as R
        nodes = self._nodes()
        sims = None if similarities is None else (lambda _q, texts: [similarities(t) for t in texts])
        with patch.object(R, "_semantic_similarities", side_effect=sims or (lambda *_a: None)):
            return R._source_nodes_to_query_evidence_lines(
                "Which jobs have I held?", nodes, max_items=5, home_corpus_dir="/tmp/homedir"
            )

    def test_closest_home_chunk_fills_the_reserve(self):
        lines = self._select(lambda text: 0.9 if "Acme" in text else 0.1)
        self.assertTrue(any("cv.txt" in line for line in lines))

    def test_without_similarities_previous_order_is_kept(self):
        lines = self._select(None)
        self.assertFalse(any("cv.txt" in line for line in lines))

    def test_similarity_helper_embeds_each_text_once(self):
        import jarvisrag.retrieval as R
        from llama_index.core import Settings

        calls = []

        class FakeModel:
            def get_text_embedding_batch(self, texts):
                calls.append(len(texts))
                return [[1.0, 0.0] if "match" in t else [0.0, 1.0] for t in texts]

            def get_query_embedding(self, _q):
                return [1.0, 0.0]

        R._TEXT_EMBEDDING_CACHE.clear()
        with patch.object(Settings, "_embed_model", FakeModel()):
            first = R._semantic_similarities("q", ["a match", "other", "a match"])
            second = R._semantic_similarities("q", ["a match", "other"])
        self.assertEqual([round(x, 3) for x in first], [1.0, 0.0, 1.0])
        self.assertEqual([round(x, 3) for x in second], [1.0, 0.0])
        self.assertEqual(calls, [2])
        R._TEXT_EMBEDDING_CACHE.clear()


if __name__ == "__main__":
    unittest.main()
