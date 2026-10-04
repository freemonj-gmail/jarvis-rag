"""Email verification: sync, extraction, preparation, retention, and checkpoints.

Uses synthetic files and local test embeddings, never real mail, a network
service, or an LLM. Run with ``make mail-test`` or directly through uv.
These checks prove lifecycle behavior, not model throughput or answer accuracy.
"""

import glob
import contextlib
import io
import json
import os
import shutil
import sqlite3
import struct
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from jarvisrag.docsync import DocSet, refresh_mail_set
from jarvisrag.extraction.extractors import EmlxReader
import jarvisrag.docsync as docsync
from jarvisrag.vectorstore import MilvusVectorStore, create_milvus_vector_store
from llama_index.core import Settings
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.ingestion import run_transformations
from llama_index.core.schema import Document, MetadataMode, TextNode
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.vector_stores.types import VectorStoreQuery
from jarvisrag.extraction import indexing, owned_index
from scripts.bench_mail_embedding import AlignedTokenizer, copy_sample, embedding_order, measure_pipeline, padded_tokens, revise_sample_message

# An exported JARVIS_MAIL_ACCOUNTS scopes the walk to one real account, which
# would reject every synthetic fixture and trip the empty-scan guard instead.
_ACCOUNT_SCOPE = patch.dict(os.environ, {"JARVIS_MAIL_ACCOUNTS": ""})


def setUpModule():
    _ACCOUNT_SCOPE.start()


def tearDownModule():
    _ACCOUNT_SCOPE.stop()


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

        before = constants.HF_HUB_OFFLINE
        attempts = []

        def load():
            attempts.append(constants.HF_HUB_OFFLINE)
            if constants.HF_HUB_OFFLINE:
                raise OSError("synthetic: not in the local cache")
            return "downloaded"

        self.assertEqual(_load_from_cache_first(load), "downloaded")
        self.assertEqual(attempts, [True, before])
        self.assertEqual(constants.HF_HUB_OFFLINE, before)


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


class MailStartupTests(unittest.TestCase):
    def test_unbuilt_mail_index_never_starts_a_long_build_on_launch(self):
        from jarvisrag.answer import bootstrap

        with tempfile.TemporaryDirectory() as corpus, tempfile.TemporaryDirectory() as persist:
            mail = DocSet(name="mail", corpus_dir=corpus, type="mail", match_any=["email"])
            with patch.object(bootstrap, "PERSIST_DIR", persist), \
                    patch.object(bootstrap, "load_document_sources", return_value=[mail]), \
                    patch.object(bootstrap, "_load_corpus_documents", side_effect=AssertionError("loaded the corpus")), \
                    patch.dict(os.environ, {}, clear=False):
                os.environ.pop("JARVIS_BUILD_MAIL_INDEX", None)
                self.assertEqual(bootstrap._build_domain_retrievers(Mock(), Mock()), [])

            ledger = Path(persist, "domains", "mail")
            ledger.mkdir(parents=True)
            (ledger / "owned_index.sqlite3").touch()
            with patch.object(bootstrap, "PERSIST_DIR", persist), \
                    patch.object(bootstrap, "load_document_sources", return_value=[mail]), \
                    patch.object(bootstrap, "_load_corpus_documents", return_value=[Document(text="Synthetic staged message.")]), \
                    patch.object(bootstrap, "_build_index", return_value=(Mock(), Mock(), "query", "milvus")) as build, \
                    patch.object(bootstrap, "build_hybrid_retriever", return_value=Mock()):
                self.assertEqual(len(bootstrap._build_domain_retrievers(Mock(), Mock())), 1)
            self.assertEqual(build.call_args.kwargs["owned_corpus_dir"], corpus)

    def test_explicit_opt_in_builds_the_mail_index(self):
        from jarvisrag.answer import bootstrap

        with tempfile.TemporaryDirectory() as corpus, tempfile.TemporaryDirectory() as persist:
            mail = DocSet(name="mail", corpus_dir=corpus, type="mail", match_any=["email"])
            with patch.object(bootstrap, "PERSIST_DIR", persist), \
                    patch.object(bootstrap, "load_document_sources", return_value=[mail]), \
                    patch.object(bootstrap, "_load_corpus_documents", return_value=[Document(text="Synthetic staged message.")]), \
                    patch.object(bootstrap, "_build_index", return_value=(Mock(), Mock(), "query", "milvus")) as build, \
                    patch.object(bootstrap, "build_hybrid_retriever", return_value=Mock()), \
                    patch.dict(os.environ, {"JARVIS_BUILD_MAIL_INDEX": "1"}):
                self.assertEqual(len(bootstrap._build_domain_retrievers(Mock(), Mock())), 1)
            self.assertEqual(build.call_args.kwargs["owned_corpus_dir"], corpus)

    def test_only_mail_sets_request_owned_indexing(self):
        from jarvisrag.answer import bootstrap

        with tempfile.TemporaryDirectory() as directory:
            mail = DocSet(name="mail", corpus_dir=directory, type="mail", match_any=["email"])
            general = DocSet(name="reference", corpus_dir=directory, match_any=["reference"])
            with patch.object(bootstrap, "load_document_sources", return_value=[mail, general]), \
                    patch.object(bootstrap, "_load_corpus_documents", return_value=[Document(text="Synthetic reference content.")]), \
                    patch.object(bootstrap, "_build_index", return_value=(Mock(), Mock(), "query", "milvus")) as build, \
                    patch.object(bootstrap, "build_hybrid_retriever", return_value=Mock()), \
                    patch.dict(os.environ, {"JARVIS_BUILD_MAIL_INDEX": "1"}):
                self.assertEqual(len(bootstrap._build_domain_retrievers(Mock(), Mock())), 2)
            self.assertEqual(build.call_args_list[0].kwargs["owned_corpus_dir"], directory)
            self.assertNotIn("owned_corpus_dir", build.call_args_list[1].kwargs)

    def test_mail_preserves_source_inventory(self):
        from jarvisrag.answer import bootstrap

        documents = [Document(text="...", metadata={"file_path": "/synthetic/one.emlx"})]
        with patch.dict(os.environ, {"JARVIS_VECTOR_BACKEND": "milvus", "JARVIS_CONTEXT_MODE": "chunk"}), \
                patch.object(bootstrap, "filter_informative_documents", side_effect=AssertionError("premature filtering")), \
                patch.object(bootstrap, "_is_port_open", return_value=True), \
                patch.object(bootstrap, "create_milvus_vector_store") as factory, \
                patch.object(bootstrap, "build_or_load_persistent_index", return_value=(Mock(), Mock())) as build:
            bootstrap._build_index(documents, Mock(), owned_corpus_dir="/synthetic")
        self.assertNotIn("max_vectors", factory.call_args.kwargs)
        self.assertEqual(build.call_args.args[0], documents)
        self.assertEqual(build.call_args.kwargs["owned_extensions"], (".emlx",))

    def test_failed_mail_build_never_falls_back_to_unverified_index(self):
        from jarvisrag.answer import bootstrap

        with patch.dict(os.environ, {"JARVIS_VECTOR_BACKEND": "milvus", "JARVIS_CONTEXT_MODE": "chunk", "JARVIS_REQUIRE_MILVUS": "0"}), \
                patch.object(bootstrap, "_is_port_open", return_value=True), \
                patch.object(bootstrap, "create_milvus_vector_store"), \
                patch.object(bootstrap, "build_or_load_persistent_index", side_effect=RuntimeError("synthetic interruption")) as build:
            with self.assertRaisesRegex(RuntimeError, "verification failed"):
                bootstrap._build_index([Document(text="Synthetic content for testing.")], Mock(), owned_corpus_dir="/synthetic")
        self.assertEqual(build.call_count, 1)


class OwnedIndexTests(unittest.TestCase):
    def test_length_bucketed_embedding_preserves_order_and_batches_by_length(self):
        nodes = [TextNode(text="word " * length) for length in (300, 5, 120, 40, 900, 70, 8, 250, 15)]
        seen_batches, seen_sizes = [], []

        class Recorder(MockEmbedding):
            def get_text_embedding_batch(self, texts, **kwargs):
                seen_batches.append(list(texts))
                seen_sizes.append(self.embed_batch_size)
                return [[float(len(t)), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for t in texts]

        embedder = Recorder(embed_dim=8, embed_batch_size=32)
        result = owned_index._embed_length_bucketed(embedder, nodes)
        expected = [node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes]
        self.assertEqual([vector[0] for vector in result], [float(len(t)) for t in expected])
        self.assertEqual(sorted(seen_batches[0]), sorted(expected))
        self.assertEqual(seen_batches[0], sorted(expected, key=len))
        self.assertEqual(seen_sizes, [owned_index._BUCKET_BATCH])
        self.assertEqual(embedder.embed_batch_size, 32)

    def setUp(self):
        self.work = tempfile.TemporaryDirectory(prefix="jarvis_owned_test_")
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name, "corpus")
        self.root.mkdir()
        self.persist = str(Path(self.work.name, "index"))
        self.embedder = MockEmbedding(embed_dim=8)

    def document(self, name, text):
        path = self.root / name
        path.write_text(text)
        return Document(text=text, metadata={"file_path": str(path), "file_name": name})

    def build(self, documents, store=None):
        return indexing.build_or_load_persistent_index(
            documents, store or SimpleVectorStore(), self.persist, self.embedder,
            owned_corpus_dir=str(self.root), owned_extensions=(".txt",),
        )

    def test_warm_load_skips_preparation_and_embedding(self):
        documents = [self.document("one.txt", "The committee approved the project schedule after the final review.")]
        first, _ = self.build(documents)
        with patch.object(owned_index, "_prepare_children", side_effect=AssertionError("unexpected preparation")), \
                patch.object(MockEmbedding, "get_text_embedding_batch", side_effect=AssertionError("unexpected embedding")):
            second, _ = self.build(documents)
        self.assertEqual(set(first.docstore.docs), set(second.docstore.docs))

    def test_multiple_write_batches_flush_once_and_remain_recoverable(self):
        documents = [self.document(f"message-{ordinal}.txt", "The committee approved the schedule after reviewing the project report.") for ordinal in range(65)]
        with patch.object(indexing, "_flush_vector_store", side_effect=RuntimeError("synthetic final flush failure")) as flush:
            with self.assertRaisesRegex(RuntimeError, "final flush failure"):
                self.build(documents)
        flush.assert_called_once()
        with patch.object(MockEmbedding, "_get_text_embedding", side_effect=AssertionError("unexpected re-embedding")), \
                patch.object(indexing, "_flush_vector_store", wraps=indexing._flush_vector_store) as flush:
            recovered, _ = self.build(documents)
        flush.assert_called_once()
        self.assertEqual(len(recovered.vector_store.data.embedding_dict), 65)

    def test_failed_insert_resumes_without_reembedding(self):
        documents = [self.document("one.txt", "The committee approved the project schedule after the final review.")]
        with patch.object(SimpleVectorStore, "add", side_effect=RuntimeError("synthetic interruption")):
            with self.assertRaises(RuntimeError):
                self.build(documents)

        def forbid_text_embedding(texts, **kwargs):
            self.assertEqual(texts, [], "Cached vectors must not be re-embedded.")
            return []

        with patch.object(MockEmbedding, "get_text_embedding_batch", side_effect=forbid_text_embedding):
            index, _ = self.build(documents)
        self.assertEqual(len(index.docstore.docs), 1)

    def test_vectors_round_trip_and_legacy_float64_blobs_still_load(self):
        vector = [0.5, -0.25, 0.125, 2.0]
        current = owned_index._embedding_blob(vector)
        legacy = struct.pack(f"<{len(vector)}d", *vector)
        self.assertEqual(len(current), len(vector) * 4)
        self.assertEqual(owned_index._embedding_values(current, len(vector)), vector)
        self.assertEqual(owned_index._embedding_values(legacy, None), vector)

    def test_checkpoints_written_before_float32_are_reused_not_reembedded(self):
        documents = [self.document("one.txt", "The committee approved the project schedule after the final review.")]
        with patch.object(SimpleVectorStore, "add", side_effect=RuntimeError("synthetic interruption")):
            with self.assertRaises(RuntimeError):
                self.build(documents)
        with contextlib.closing(sqlite3.connect(os.path.join(self.persist, "owned_index.sqlite3"))) as ledger:
            stored = ledger.execute("SELECT id, embedding, dims FROM children").fetchall()
            self.assertTrue(stored)
            with ledger:
                for identity, blob, dims in stored:
                    values = owned_index._embedding_values(blob, dims)
                    ledger.execute(
                        "UPDATE children SET embedding=?, dims=NULL WHERE id=?",
                        (struct.pack(f"<{len(values)}d", *values), identity),
                    )

        def forbid_text_embedding(texts, **kwargs):
            self.assertEqual(texts, [], "Legacy vectors must not be re-embedded.")
            return []

        with patch.object(MockEmbedding, "get_text_embedding_batch", side_effect=forbid_text_embedding):
            index, _ = self.build(documents)
        self.assertEqual(len(index.docstore.docs), 1)

    def test_changed_and_removed_sources_replace_exact_children(self):
        first = self.document("one.txt", "The initial schedule has been approved for review by the committee.")
        second = self.document("two.txt", "The second report describes the completed inspection and its findings.")
        original, _ = self.build([first, second])
        old_ids = set(original.docstore.docs)
        Path(second.metadata["file_path"]).unlink()
        changed = self.document("one.txt", "The revised schedule has been rejected after further committee review.")
        updated, _ = self.build([changed])
        self.assertEqual(len(updated.docstore.docs), 1)
        self.assertFalse(old_ids & set(updated.docstore.docs))
        self.assertEqual(set(updated.vector_store.data.embedding_dict), set(updated.docstore.docs))

    def test_partial_extraction_cannot_prune(self):
        first = self.document("one.txt", "The initial schedule has been approved for review by the committee.")
        second = self.document("two.txt", "The second report describes the completed inspection and its findings.")
        self.build([first, second])
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.build([first])

    def test_missing_child_is_restored_from_cached_embedding(self):
        document = self.document("one.txt", "The committee reviewed the project schedule and recorded its decision. " * 350)
        first, storage = self.build([document])
        identities = set(first.docstore.docs)
        self.assertGreater(len(identities), 1)
        first.vector_store.delete_nodes([next(iter(identities))])
        storage.persist(self.persist)
        with patch.object(MockEmbedding, "_get_text_embedding", side_effect=AssertionError("unexpected embedding")):
            recovered, _ = self.build([document])
        self.assertEqual(set(recovered.vector_store.data.embedding_dict), identities)

    def test_changed_docstore_text_is_recovered_without_embedding(self):
        document = self.document("one.txt", "The committee approved the schedule after reviewing the final report.")
        first, storage = self.build([document])
        identity = next(iter(first.docstore.docs))
        original = first.docstore.get_node(identity)
        corrupted = original.model_copy(deep=True)
        corrupted.text = "This is unrelated corrupted text with the same node identity."
        storage.docstore.add_documents([corrupted], allow_update=True)
        storage.persist(self.persist)
        with patch.object(MockEmbedding, "_get_text_embedding", side_effect=AssertionError("unexpected embedding")):
            recovered, _ = self.build([document])
        self.assertEqual(recovered.docstore.get_node(identity).text, original.text)

    def test_failed_deletion_cannot_report_success(self):
        first = self.document("one.txt", "The committee approved the schedule after reviewing the final report.")
        self.build([first])
        changed = self.document("one.txt", "The committee rejected the schedule after reviewing a revised report.")
        with patch.object(SimpleVectorStore, "delete_nodes", side_effect=RuntimeError("synthetic deletion failure")):
            with self.assertRaisesRegex(RuntimeError, "deletion failure"):
                self.build([changed])
        with patch.object(MockEmbedding, "_get_text_embedding", side_effect=AssertionError("unexpected embedding")):
            recovered, _ = self.build([changed])
        self.assertEqual(len(recovered.vector_store.data.embedding_dict), 1)
        self.assertIn("rejected", next(iter(recovered.docstore.docs.values())).text)

    def test_missing_vector_storage_is_recovered_without_embedding(self):
        document = self.document("one.txt", "The committee approved the schedule after reviewing the final report.")
        first, _ = self.build([document])
        identities = set(first.docstore.docs)
        for path in Path(self.persist).glob("*vector_store.json"):
            path.unlink()
        with patch.object(MockEmbedding, "_get_text_embedding", side_effect=AssertionError("unexpected embedding")):
            recovered, _ = self.build([document])
        self.assertEqual(set(recovered.vector_store.data.embedding_dict), identities)


class PreparedCacheTests(unittest.TestCase):
    def test_preparation_preserves_original_embedding_inputs(self):
        text = "The committee reviewed the project schedule and recorded its decision. " * 350
        document = Document(text=text, metadata={"file_path": "/synthetic/report.txt", "file_name": "report.txt"})
        originals = [document.model_copy(deep=True)]
        indexing.ensure_document_metadata(originals[0])
        prepared = indexing.filter_informative_documents(indexing.split_large_docs(originals))
        baseline = run_transformations(prepared, Settings.transformations)
        first = owned_index._prepare_children([document], "test-signature")
        second = owned_index._prepare_children([document], "test-signature")
        self.assertEqual([node.get_content(MetadataMode.EMBED) for node in baseline], [node.get_content(MetadataMode.EMBED) for node in first])
        self.assertEqual([node.node_id for node in first], [node.node_id for node in second])
        self.assertEqual([node.metadata for node in baseline], [node.metadata for node in first])
        self.assertNotIn("source", document.metadata)

    def test_cache_reuses_preparation_and_owner_is_exclusive(self):
        document = Document(text="The committee approved the final schedule after reviewing the report.", metadata={"file_path": "/synthetic/report.txt"})
        with tempfile.TemporaryDirectory() as directory:
            os.chmod(directory, 0o755)
            with owned_index._ledger(directory, "owner") as ledger:
                self.assertEqual(os.stat(directory).st_mode & 0o777, 0o700)
                self.assertEqual(os.stat(os.path.join(directory, "owned_index.sqlite3")).st_mode & 0o777, 0o600)
                owned_index._prepare_cache(ledger, {"source": [document]}, "signature")
                with patch.object(owned_index, "_prepare_children", side_effect=AssertionError("unexpected preparation")):
                    owned_index._prepare_cache(ledger, {"source": [document]}, "signature")
                with self.assertRaises(RuntimeError):
                    with owned_index._ledger(directory, "owner"):
                        pass
            with self.assertRaises(ValueError):
                with owned_index._ledger(directory, "different-owner"):
                    pass


class ScheduleTests(unittest.TestCase):
    def test_pilot_paths_and_prepared_inputs_repeat_without_shared_state(self):
        from scripts.bench_mail_embedding import pilot_directory, prepare_batches

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory, "staged")
            source = root / "message.emlx"
            _write_emlx(str(source), "sender@example.com", "Schedule", "The committee approved the project schedule. " * 180)
            documents = EmlxReader().load_data(str(source))
            snapshots = []
            for _ in range(2):
                with pilot_directory(Path(directory)) as workspace:
                    self.assertEqual(workspace.stat().st_mode & 0o777, 0o700)
                    with self.assertRaises(FileExistsError):
                        with pilot_directory(Path(directory)):
                            self.fail("Concurrent pilot was allowed.")
                    copied = copy_sample(documents, root, workspace / "corpus")
                    batches, _ = prepare_batches(copied, 64)
                    snapshots.append([(node.get_content(MetadataMode.EMBED), node.metadata) for batch in batches for node in batch])
                self.assertFalse(workspace.exists())
            self.assertEqual(snapshots[0], snapshots[1])

    def test_pipeline_measurement_forwards_actual_embedding_calls(self):
        embedder = MockEmbedding(embed_dim=8)
        with measure_pipeline(embedder) as measured:
            vectors = embedder.get_text_embedding_batch(["Synthetic first input.", "Synthetic second input."])
            empty = embedder.get_text_embedding_batch([])
        self.assertEqual(len(vectors), 2)
        self.assertEqual(empty, [])
        self.assertEqual(measured["embedded_texts"], 2)
        self.assertEqual(measured["calls"]["embedding"], 2)
        self.assertGreaterEqual(measured["wall_seconds"], measured["seconds"]["embedding"])

    def test_pilot_revision_keeps_body_and_changes_only_copied_message(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory, "staged")
            source = root / "message.emlx"
            _write_emlx(str(source), "sender@example.com", "Schedule", "The committee approved the project schedule.")
            documents = EmlxReader().load_data(str(source))
            copied_root = Path(directory, "pilot")
            copied = copy_sample(documents, root, copied_root)
            updated = revise_sample_message(Path(copied[0].metadata["file_path"]), copied_root)
            self.assertEqual(EmlxReader().load_data(str(source))[0].metadata["email_subject"], "Schedule")
            self.assertEqual(updated[0].metadata["email_subject"], "Schedule [pilot revision]")
            self.assertIn("The committee approved the project schedule.", updated[0].text)
            with self.assertRaisesRegex(ValueError, "outside"):
                revise_sample_message(source, copied_root)

    def test_pilot_copy_preserves_content_and_original_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory, "staged")
            source = root / "message.emlx"
            _write_emlx(str(source), "sender@example.com", "Schedule", "The committee approved the project schedule.")
            documents = EmlxReader().load_data(str(source))
            documents[0].metadata["file_path"] = str(source)
            before = documents[0].model_copy(deep=True)
            copied = copy_sample(documents, root, Path(directory, "pilot"))
            self.assertEqual(documents[0].to_json(), before.to_json())
            self.assertEqual(copied[0].text, before.text)
            self.assertEqual(Path(copied[0].metadata["file_path"]).read_bytes(), source.read_bytes())
            self.assertNotEqual(copied[0].metadata["file_path"], before.metadata["file_path"])
            Path(copied[0].metadata["file_path"]).unlink()
            self.assertTrue(source.exists())

    def test_pilot_copy_rejects_sources_outside_staged_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            document = Document(text="Synthetic message.", metadata={"file_path": "/outside/message.emlx"})
            with self.assertRaisesRegex(ValueError, "outside"):
                copy_sample([document], Path(directory, "staged"), Path(directory, "pilot"))

    def test_padding_adapter_preserves_tokenizer_arguments(self):
        tokenizer = Mock()
        aligned = AlignedTokenizer(tokenizer, 128)
        texts = ["Every source token stays in the input."]
        aligned(texts, padding=True, truncation=False, return_tensors="pt")
        tokenizer.assert_called_once_with(
            texts, padding=True, truncation=False, return_tensors="pt", pad_to_multiple_of=128,
        )

    def test_baseline_and_bucketed_keep_every_index_and_boundary(self):
        lengths = [8, 2, 6, 4, 9, 3]
        baseline = embedding_order(lengths, [4, 2], False)
        bucketed = embedding_order(lengths, [4, 2], True)
        self.assertEqual(baseline, [[0, 1, 2, 3], [4, 5]])
        self.assertEqual(bucketed, [[1, 3, 2, 0], [5, 4]])
        for original, reordered in zip(baseline, bucketed):
            self.assertEqual(sorted(original), sorted(reordered))
        self.assertLess(padded_tokens(bucketed, lengths, 2), padded_tokens(baseline, lengths, 2))

    def test_incomplete_coverage_rejected(self):
        with self.assertRaises(ValueError):
            embedding_order([8, 2, 6], [2], True)


class MailRefreshSafetyTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory(prefix="jarvis_mail_safety_")
        self.addCleanup(self.work.cleanup)
        self.mail_root = os.path.join(self.work.name, "mail")
        self.corpus = os.path.join(self.work.name, "corpus")
        self.first = os.path.join(self.mail_root, "Inbox", "one.emlx")
        self.second = os.path.join(self.mail_root, "Archive", "two.emlx")
        _write_emlx(self.first, "sender@example.com", "Schedule", "Initial schedule.", "<one@example.com>")
        _write_emlx(self.second, "sender@example.com", "Decision", "A separate decision.", "<two@example.com>")
        self.doc_set = DocSet(name="mail", corpus_dir=self.corpus, type="mail", mail_root=self.mail_root)
        self.assertEqual(refresh_mail_set(self.doc_set)[2], [])
        self.dest = os.path.join(self.corpus, docsync._msgid_key("<one@example.com>"))
        self.state_path = os.path.join(self.corpus, ".docsync_state.json")

    def test_partial_scan_leaves_corpus_and_state_untouched(self):
        real_stat = os.stat
        prior_state = os.stat(self.state_path).st_mtime_ns

        def fail_one(path, *args, **kwargs):
            if os.fspath(path) == self.second:
                raise PermissionError("synthetic stat failure")
            return real_stat(path, *args, **kwargs)

        with patch.object(docsync.os, "stat", side_effect=fail_one):
            changed, _, failed = refresh_mail_set(self.doc_set)
        self.assertEqual(changed, [])
        self.assertTrue(failed)
        self.assertEqual(len(_corpus_emlx(self.corpus)), 2)
        self.assertEqual(os.stat(self.state_path).st_mtime_ns, prior_state)

    def test_walk_failure_leaves_corpus_untouched(self):
        def incomplete_walk(root, onerror=None):
            if onerror:
                onerror(PermissionError("synthetic traversal failure"))
            yield os.path.dirname(self.first), [], [os.path.basename(self.first)]

        with patch.object(docsync.os, "walk", side_effect=incomplete_walk):
            changed, _, failed = refresh_mail_set(self.doc_set)
        self.assertEqual(changed, [])
        self.assertTrue(failed)
        self.assertEqual(len(_corpus_emlx(self.corpus)), 2)

    def test_failed_update_retries_next_refresh(self):
        _write_emlx(self.first, "sender@example.com", "Schedule", "The updated schedule has new details.", "<one@example.com>")
        with patch.object(docsync.shutil, "copyfile", side_effect=OSError("synthetic copy failure")):
            self.assertTrue(refresh_mail_set(self.doc_set)[2])
        changed, _, failed = refresh_mail_set(self.doc_set)
        self.assertIn("<one@example.com>", changed)
        self.assertEqual(failed, [])
        with open(self.dest, "rb") as handle:
            self.assertIn(b"updated schedule", handle.read())

    def test_header_is_parsed_only_after_copy(self):
        original = docsync._emlx_message_id
        parsed_paths = []

        def observe(path):
            parsed_paths.append(path)
            self.assertEqual(os.path.dirname(path), self.corpus)
            return original(path)

        with patch.object(docsync, "_emlx_message_id", side_effect=observe):
            changed, _, failed = refresh_mail_set(self.doc_set, force=True)
        self.assertEqual(failed, [])
        self.assertEqual(len(changed), 2)
        self.assertEqual(len(parsed_paths), 2)

    def test_unchanged_scan_neither_copies_nor_rewrites_state(self):
        before = os.stat(self.state_path).st_mtime_ns
        with patch.object(docsync.shutil, "copyfile", side_effect=AssertionError("unexpected copy")):
            changed, unchanged, failed = refresh_mail_set(self.doc_set)
        self.assertEqual((changed, len(unchanged), failed), ([], 2, []))
        self.assertEqual(os.stat(self.state_path).st_mtime_ns, before)

    def test_same_bytes_with_new_mtime_preserves_destination(self):
        before = os.stat(self.dest).st_mtime_ns
        source_stat = os.stat(self.first)
        os.utime(self.first, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns + 2_000_000_000))
        changed, unchanged, failed = refresh_mail_set(self.doc_set)
        self.assertEqual(changed, [])
        self.assertIn("<one@example.com>", unchanged)
        self.assertEqual(failed, [])
        self.assertEqual(os.stat(self.dest).st_mtime_ns, before)

    def test_same_size_subsecond_change_is_detected(self):
        source_stat = os.stat(self.first)
        _write_emlx(self.first, "sender@example.com", "Schedule", "Revised schedule.", "<one@example.com>")
        os.utime(self.first, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns + 1))
        self.assertEqual(os.stat(self.first).st_size, source_stat.st_size)
        self.assertIn("<one@example.com>", refresh_mail_set(self.doc_set)[0])

    def test_missing_state_never_erases_completed_copy(self):
        os.remove(self.state_path)
        with patch.object(docsync.shutil, "copyfile", side_effect=OSError("synthetic copy failure")):
            changed, _, failed = refresh_mail_set(self.doc_set)
        self.assertEqual(changed, [])
        self.assertTrue(failed)
        self.assertEqual(len(_corpus_emlx(self.corpus)), 2)

    def test_legacy_signature_upgrade_keeps_destination_mtime(self):
        with open(self.state_path, encoding="utf-8") as handle:
            state = json.load(handle)
        for entry in state["_sources"].values():
            entry.pop("mtime_ns", None)
        with open(self.state_path, "w", encoding="utf-8") as handle:
            json.dump(state, handle)
        before = os.stat(self.dest).st_mtime_ns
        changed, _, failed = refresh_mail_set(self.doc_set)
        self.assertEqual((changed, failed), ([], []))
        self.assertEqual(os.stat(self.dest).st_mtime_ns, before)

    def test_copy_race_is_retried(self):
        source_stat = os.stat(self.first)
        os.utime(self.first, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns + 1))
        original_copy = shutil.copyfile

        def mutate_after_copy(source, destination):
            result = original_copy(source, destination)
            source_stat = os.stat(source)
            os.utime(source, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns + 1))
            return result

        before = os.stat(self.dest).st_mtime_ns
        with patch.object(docsync.shutil, "copyfile", side_effect=mutate_after_copy):
            changed, _, failed = refresh_mail_set(self.doc_set)
        self.assertEqual(changed, [])
        self.assertTrue(failed)
        self.assertEqual(os.stat(self.dest).st_mtime_ns, before)
        self.assertEqual(refresh_mail_set(self.doc_set)[2], [])

    def test_unchanged_duplicate_does_not_mask_changed_copy(self):
        duplicate = os.path.join(os.path.dirname(self.first), "duplicate.emlx")
        shutil.copyfile(self.first, duplicate)
        self.assertEqual(refresh_mail_set(self.doc_set)[2], [])
        _write_emlx(duplicate, "sender@example.com", "Schedule", "Updated schedule for review.", "<one@example.com>")
        with patch.object(docsync.os, "walk", return_value=[
            (os.path.dirname(self.first), [], ["one.emlx", "duplicate.emlx"]),
            (os.path.dirname(self.second), [], ["two.emlx"]),
        ]):
            changed, unchanged, failed = refresh_mail_set(self.doc_set)
        self.assertEqual(changed, ["<one@example.com>"])
        self.assertNotIn("<one@example.com>", unchanged)
        self.assertEqual(failed, [])

    def test_state_save_failure_is_reported_and_previous_state_survives(self):
        with open(self.state_path, "rb") as handle:
            before = handle.read()
        with patch.object(docsync.os, "replace", side_effect=OSError("synthetic replace failure")):
            self.assertFalse(docsync._save_state(self.corpus, {"replacement": True}))
        with open(self.state_path, "rb") as handle:
            self.assertEqual(handle.read(), before)
        self.assertFalse(glob.glob(os.path.join(self.corpus, ".docsync-*")))

    def test_cli_failure_exit_and_no_false_index_claim(self):
        output = io.StringIO()
        with patch.object(docsync, "load_document_sources", return_value=[self.doc_set]), \
                patch.object(docsync, "refresh_mail_set", return_value=([], [], ["synthetic failure"])), \
                contextlib.redirect_stdout(output):
            self.assertEqual(docsync.main([]), 1)
        self.assertIn("Refresh incomplete", output.getvalue())
        self.assertNotIn("index is up to date", output.getvalue())


class RemoteRefreshTests(unittest.TestCase):
    def test_conditional_downloads_and_identical_bytes_keep_mtime(self):
        import httpx

        with tempfile.TemporaryDirectory(prefix="jarvis_remote_sync_") as corpus:
            source = docsync.DocSource(name="reference.pdf", url="https://example.com/reference.pdf")
            doc_set = DocSet(name="reference", corpus_dir=corpus, documents=[source])
            request = httpx.Request("GET", source.url)
            with patch.object(httpx, "Client") as factory:
                client = factory.return_value.__enter__.return_value
                client.get.side_effect = [
                    httpx.Response(200, content=b"synthetic reference bytes", headers={"ETag": '"v1"'}, request=request),
                    httpx.Response(304, request=request),
                    httpx.Response(200, content=b"synthetic reference bytes", headers={"ETag": '"v2"'}, request=request),
                ]
                self.assertEqual(docsync.refresh_document_set(doc_set), ([source.name], [], []))
                destination = os.path.join(corpus, source.name)
                before = os.stat(destination).st_mtime_ns
                self.assertEqual(docsync.refresh_document_set(doc_set), ([], [source.name], []))
                self.assertEqual(docsync.refresh_document_set(doc_set), ([], [source.name], []))
                self.assertEqual(os.stat(destination).st_mtime_ns, before)
                self.assertEqual(client.get.call_args.kwargs["headers"]["If-None-Match"], '"v1"')


class MailExclusionTests(unittest.TestCase):
    """Excluded mailboxes must drop only messages filed NOWHERE else."""

    def setUp(self):
        self.work = tempfile.TemporaryDirectory(prefix="jarvis_mail_exclude_")
        self.addCleanup(self.work.cleanup)
        self.mail_root = os.path.join(self.work.name, "mail")
        self.corpus = os.path.join(self.work.name, "corpus")
        account = os.path.join(self.mail_root, "V10", "acct")
        self.inbox = os.path.join(account, "INBOX.mbox", "Data", "Messages")
        self.trash = os.path.join(account, "Trash.mbox", "Data", "Messages")
        self.spam = os.path.join(account, "Spam.mbox", "Data", "Messages")
        # "Trashed Drafts" must survive: it merely CONTAINS an excluded word.
        self.lookalike = os.path.join(account, "Trashed Drafts.mbox", "Data", "Messages")

    def _write(self, folder, name, message_id):
        _write_emlx(os.path.join(folder, name), "s@example.com", "Subject", "Body text.", message_id)

    def _refresh(self, exclude):
        doc_set = DocSet(name="mail", corpus_dir=self.corpus, type="mail",
                         mail_root=self.mail_root, exclude_folders=exclude)
        changed, unchanged, failed = refresh_mail_set(doc_set)
        self.assertEqual(failed, [])
        return changed, unchanged

    def test_excludes_only_messages_with_no_other_copy(self):
        self._write(self.inbox, "1.emlx", "<keep@example.com>")
        self._write(self.trash, "1.emlx", "<keep@example.com>")   # same message, also in Trash
        self._write(self.trash, "2.emlx", "<drop@example.com>")   # Trash only
        self._write(self.spam, "3.emlx", "<spam@example.com>")    # Spam only
        self._write(self.lookalike, "4.emlx", "<lookalike@example.com>")

        self._refresh(["Trash", "Spam"])
        staged = {os.path.basename(p) for p in _corpus_emlx(self.corpus)}
        self.assertIn(docsync._msgid_key("<keep@example.com>"), staged)
        self.assertIn(docsync._msgid_key("<lookalike@example.com>"), staged)
        self.assertNotIn(docsync._msgid_key("<drop@example.com>"), staged)
        self.assertNotIn(docsync._msgid_key("<spam@example.com>"), staged)
        self.assertEqual(len(staged), 2)

    def test_enabling_exclusion_prunes_already_staged_messages(self):
        self._write(self.inbox, "1.emlx", "<keep@example.com>")
        self._write(self.trash, "2.emlx", "<drop@example.com>")
        self._refresh([])
        self.assertEqual(len(_corpus_emlx(self.corpus)), 2)

        changed, _ = self._refresh(["Trash"])
        self.assertIn("removed:<drop@example.com>", changed)
        staged = {os.path.basename(p) for p in _corpus_emlx(self.corpus)}
        self.assertEqual(staged, {docsync._msgid_key("<keep@example.com>")})

    def test_no_exclusion_configured_keeps_everything(self):
        self._write(self.inbox, "1.emlx", "<keep@example.com>")
        self._write(self.trash, "2.emlx", "<drop@example.com>")
        self._refresh([])
        self.assertEqual(len(_corpus_emlx(self.corpus)), 2)

    def test_segment_matching_is_case_insensitive_and_ignores_mbox_suffix(self):
        wanted = docsync._normalized_mailbox_names(["Trash", " spam "])
        self.assertTrue(docsync._excluded_mailbox(os.path.join("acct", "TRASH.mbox", "m.emlx"), wanted))
        self.assertTrue(docsync._excluded_mailbox(os.path.join("acct", "Spam", "m.emlx"), wanted))
        self.assertFalse(docsync._excluded_mailbox(os.path.join("acct", "Trashed Drafts.mbox", "m.emlx"), wanted))
        self.assertFalse(docsync._excluded_mailbox(os.path.join("acct", "Inbox.mbox", "m.emlx"), wanted))
        # An empty configuration must never exclude anything.
        self.assertFalse(docsync._excluded_mailbox(os.path.join("Trash.mbox", "m.emlx"), set()))


class MailAccountScopeTests(unittest.TestCase):
    """Account scope is env-driven so no machine-specific id is ever committed."""

    def setUp(self):
        self.work = tempfile.TemporaryDirectory(prefix="jarvis_mail_account_")
        self.addCleanup(self.work.cleanup)
        self.mail_root = os.path.join(self.work.name, "mail")
        self.corpus = os.path.join(self.work.name, "corpus")
        self.keep = os.path.join(self.mail_root, "V10", "ACCT-KEEP", "INBOX.mbox", "Data", "Messages")
        self.other = os.path.join(self.mail_root, "V10", "ACCT-OTHER", "INBOX.mbox", "Data", "Messages")
        _write_emlx(os.path.join(self.keep, "1.emlx"), "s@example.com", "S", "Body.", "<keep@example.com>")
        _write_emlx(os.path.join(self.other, "2.emlx"), "s@example.com", "S", "Body.", "<other@example.com>")
        # One message filed in BOTH accounts must survive scoping.
        _write_emlx(os.path.join(self.keep, "3.emlx"), "s@example.com", "S", "Body.", "<both@example.com>")
        _write_emlx(os.path.join(self.other, "3.emlx"), "s@example.com", "S", "Body.", "<both@example.com>")

    def _staged(self):
        return {os.path.basename(p) for p in _corpus_emlx(self.corpus)}

    def _refresh(self, accounts):
        doc_set = DocSet(name="mail", corpus_dir=self.corpus, type="mail",
                         mail_root=self.mail_root, include_accounts=accounts)
        changed, unchanged, failed = refresh_mail_set(doc_set)
        self.assertEqual(failed, [])
        return changed

    def test_scoping_to_one_account_drops_only_the_others(self):
        self._refresh(["ACCT-KEEP"])
        self.assertEqual(self._staged(), {
            docsync._msgid_key("<keep@example.com>"),
            docsync._msgid_key("<both@example.com>"),
        })

    def test_unset_scope_indexes_every_account(self):
        with patch.dict(os.environ, {"JARVIS_MAIL_ACCOUNTS": ""}, clear=False):
            self._refresh([])
        self.assertEqual(len(self._staged()), 3)

    def test_environment_supplies_the_account_when_the_set_is_empty(self):
        with patch.dict(os.environ, {"JARVIS_MAIL_ACCOUNTS": "acct-keep"}, clear=False):
            self._refresh([])
        self.assertEqual(self._staged(), {
            docsync._msgid_key("<keep@example.com>"),
            docsync._msgid_key("<both@example.com>"),
        })

    def test_narrowing_scope_prunes_previously_staged_accounts(self):
        with patch.dict(os.environ, {"JARVIS_MAIL_ACCOUNTS": ""}, clear=False):
            self._refresh([])
        self.assertEqual(len(self._staged()), 3)
        changed = self._refresh(["ACCT-KEEP"])
        self.assertIn("removed:<other@example.com>", changed)
        self.assertNotIn(docsync._msgid_key("<other@example.com>"), self._staged())


def _write_emlx(path: str, sender: str, subject: str, body: str, message_id: str | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = (
        f"From: {sender}\n"
        f"To: me@example.com\n"
        f"Subject: {subject}\n"
    )
    if message_id:
        header += f"Message-ID: {message_id}\n"
    msg = (
        header
        + "Date: Mon, 1 Jan 2026 10:00:00 -0500\n"
        + "Content-Type: text/plain; charset=utf-8\n"
        + "\n"
        + f"{body}\n"
    ).encode("utf-8")
    framed = str(len(msg)).encode("ascii") + b"\n" + msg + b'<?xml version="1.0"?><plist></plist>'
    with open(path, "wb") as handle:
        handle.write(framed)


def _corpus_emlx(corpus_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(corpus_dir, "*.emlx")))


def run_tests() -> None:
    work = tempfile.mkdtemp(prefix="jarvis_mailtest_")
    try:
        mail_root = os.path.join(work, "mail")
        corpus = os.path.join(work, "corpus")
        inbox = os.path.join(mail_root, "V10", "acct", "INBOX.mbox", "Data", "Messages")
        archive = os.path.join(mail_root, "V10", "acct", "Archive.mbox", "Data", "Messages")

        # Message A exists TWICE (Gmail stores one email under two folders).
        _write_emlx(os.path.join(inbox, "1.emlx"), "Alice <alice@example.com>", "Quarterly report", "The quarterly report body.", "<a@example.com>")
        _write_emlx(os.path.join(archive, "1.emlx"), "Alice <alice@example.com>", "Quarterly report", "The quarterly report body.", "<a@example.com>")
        # Message B is unique.
        _write_emlx(os.path.join(inbox, "2.emlx"), "Bob <bob@example.com>", "Lunch plans", "Lunch Friday works.", "<b@example.com>")

        ds = DocSet(name="mail", corpus_dir=corpus, type="mail", mail_root=mail_root)

        # First run: 3 source files, 2 Message-IDs -> 2 corpus documents (deduped).
        changed, unchanged, failed = refresh_mail_set(ds)
        assert len(changed) == 2 and failed == [], f"first run c={changed} f={failed}"
        assert len(_corpus_emlx(corpus)) == 2, "dedup: 3 files with 2 Message-IDs -> 2 docs"

        # Idempotent: nothing changes on a second run.
        changed, unchanged, failed = refresh_mail_set(ds)
        assert changed == [] and len(unchanged) == 2 and failed == [], \
            f"idempotent c={changed} u={unchanged} f={failed}"

        # New unique message is added.
        _write_emlx(os.path.join(inbox, "3.emlx"), "Carol <carol@example.com>", "Invoice 42", "Invoice 42 mentions invoices.", "<c@example.com>")
        changed, unchanged, failed = refresh_mail_set(ds)
        assert len(changed) == 1 and len(_corpus_emlx(corpus)) == 3, f"add-one c={changed}"

        # Remove ONE copy of A: still referenced by the other copy -> NOT pruned.
        os.remove(os.path.join(archive, "1.emlx"))
        changed, unchanged, failed = refresh_mail_set(ds)
        assert not any(k.startswith("removed:") for k in changed), f"should not prune A yet: {changed}"
        assert len(_corpus_emlx(corpus)) == 3, "one duplicate removed, doc retained"

        # Remove the LAST copy of A: now it is pruned.
        os.remove(os.path.join(inbox, "1.emlx"))
        changed, unchanged, failed = refresh_mail_set(ds)
        assert any(k.startswith("removed:") for k in changed), f"expected prune of A: {changed}"
        assert len(_corpus_emlx(corpus)) == 2, "A fully gone -> 2 docs"

        # FDA guard: an empty scan must never wipe the corpus.
        for folder in (inbox, archive):
            for leftover in glob.glob(os.path.join(folder, "*.emlx")):
                os.remove(leftover)
        changed, unchanged, failed = refresh_mail_set(ds)
        assert changed == [] and failed, f"guard should report failure, not deletions: c={changed} f={failed}"
        assert len(_corpus_emlx(corpus)) == 2, "guard must preserve the corpus"

        # EmlxReader parses the deduped corpus into text + header metadata.
        subjects, bodies = set(), ""
        for path in _corpus_emlx(corpus):
            docs = EmlxReader().load_data(path)
            assert len(docs) == 1, f"one Document per message, got {len(docs)}"
            doc = docs[0]
            subjects.add(doc.metadata.get("email_subject", ""))
            assert doc.metadata.get("email_from") and doc.metadata.get("extracted_by") == "emlx"
            bodies += doc.text
        assert "Invoice 42" in subjects, f"subjects={subjects}"
        assert "invoices" in bodies, "message body text should be present"

        # No-Message-ID fallback: a message without a Message-ID is kept (not merged).
        work2 = tempfile.mkdtemp(prefix="jarvis_mailtest2_")
        try:
            mr2 = os.path.join(work2, "mail")
            cp2 = os.path.join(work2, "corpus")
            box2 = os.path.join(mr2, "V10", "acct", "INBOX.mbox", "Data", "Messages")
            _write_emlx(os.path.join(box2, "9.emlx"), "Dave <dave@example.com>", "No id here", "Body without a message id.", None)
            ds2 = DocSet(name="mail", corpus_dir=cp2, type="mail", mail_root=mr2)
            changed, unchanged, failed = refresh_mail_set(ds2)
            assert len(changed) == 1 and len(_corpus_emlx(cp2)) == 1 and failed == [], \
                f"no-msgid fallback c={changed} f={failed}"
        finally:
            shutil.rmtree(work2, ignore_errors=True)

    finally:
        shutil.rmtree(work, ignore_errors=True)


class MailFlowTests(unittest.TestCase):
    def test_deduplication_delta_and_reader_flow(self):
        run_tests()


class MailCitationTests(unittest.TestCase):
    """A domain-routed answer must cite the message, not its sha1 filename."""

    def _node(self, **metadata):
        return TextNode(text="body", metadata=metadata)

    def test_email_evidence_cites_subject_sender_and_date(self):
        from jarvisrag.answer.answer import _domain_reference_lines

        node = self._node(
            file_name="3f8a1c9e2b7d4a6f8c0e1d2b3a4c5d6e7f8a9b0c.emlx",
            email_subject="Quarterly budget review",
            email_from="Alex Example <alex@example.com>",
            email_date="Tue, 3 Sep 2026 09:14:00 -0400",
        )
        lines = ["- 3f8a1c9e2b7d4a6f8c0e1d2b3a4c5d6e7f8a9b0c.emlx: the budget was approved"]
        references, superseded = _domain_reference_lines(lines, [node])
        self.assertEqual(
            references,
            ['Email: "Quarterly budget review" from Alex Example <alex@example.com> '
             "(Tue, 3 Sep 2026 09:14:00 -0400)"],
        )
        self.assertNotIn("3f8a1c9e", references[0])
        self.assertIn("3f8a1c9e2b7d4a6f8c0e1d2b3a4c5d6e7f8a9b0c.emlx", superseded)

    def test_non_email_domain_evidence_keeps_its_file_label(self):
        from jarvisrag.answer.answer import _domain_reference_lines

        node = self._node(file_name="OPM_C050.pdf")
        lines = ["- OPM_C050.pdf, page 40: components of the basic annuity"]
        self.assertEqual(
            _domain_reference_lines(lines, [node]), (["OPM_C050.pdf, page 40"], set())
        )

    def test_a_message_without_a_subject_is_still_never_cited_by_filename(self):
        from jarvisrag.answer.answer import _domain_reference_lines

        node = self._node(file_name="abc123.emlx", email_subject="", email_from="a@example.com")
        references, superseded = _domain_reference_lines(["- abc123.emlx: body"], [node])
        self.assertEqual(references, ["Email (no subject) from a@example.com"])
        self.assertIn("abc123.emlx", superseded)

    def test_only_delivered_evidence_is_cited(self):
        from jarvisrag.answer import answer as answer_module

        # The domain cap now follows JARVIS_MAX_EVIDENCE_LINES (default 30) so it
        # never sits tighter than the retriever's own output. Pin a small cap here
        # so truncation is exercised deterministically regardless of the default.
        cap = 5
        nodes = [self._node(file_name=f"m{i}.emlx", email_subject=f"Subject {i}") for i in range(cap + 8)]
        lines = [f"- m{i}.emlx: body {i}" for i in range(cap + 8)]
        with patch.dict(os.environ, {"JARVIS_MAX_EVIDENCE_LINES": str(cap)}), \
                patch.object(answer_module, "_retrieve_query_supported_evidence", return_value=(lines, nodes)):
            delivered, references, superseded = answer_module._retrieve_domain_evidence(
                "what did the email say", [{"name": "mail", "retriever": object()}]
            )
        self.assertEqual(len(delivered), cap)
        self.assertEqual(len(references), cap)
        self.assertEqual(len(superseded), cap)
        self.assertIn('Email: "Subject 0"', references)
        self.assertNotIn(f'Email: "Subject {cap}"', references)

    def test_every_routed_domain_is_represented_under_the_shared_cap(self):
        from jarvisrag.answer import answer as answer_module

        cap = 6
        first = [f"- a{i}.pdf: alpha {i}" for i in range(20)]
        second = [f"- b{i}.pdf: beta {i}" for i in range(2)]
        retrievers = {"first": object(), "second": object()}
        by_retriever = {id(retrievers["first"]): first, id(retrievers["second"]): second}

        def retrieve(_query, retriever, **_kwargs):
            return by_retriever[id(retriever)], []

        domains = [{"name": name, "retriever": retriever} for name, retriever in retrievers.items()]
        with patch.dict(os.environ, {"JARVIS_MAX_EVIDENCE_LINES": str(cap)}), \
                patch.object(answer_module, "_retrieve_query_supported_evidence", side_effect=retrieve):
            delivered, _refs, _superseded = answer_module._retrieve_domain_evidence("question", domains)
            single, _r, _s = answer_module._retrieve_domain_evidence("question", domains[:1])
        self.assertEqual(len(delivered), cap)
        self.assertEqual(delivered[:4], [first[0], second[0], first[1], second[1]])
        self.assertEqual(delivered[4:], first[2:4], "unused slots go to the domain that has more")
        self.assertEqual(single, first[:cap], "a single domain keeps its own order")

    def test_a_failing_domain_does_not_block_the_others(self):
        from jarvisrag.answer import answer as answer_module

        good = [f"- g{i}.pdf: good {i}" for i in range(3)]

        def retrieve(_query, retriever, **_kwargs):
            if retriever == "broken":
                raise RuntimeError("synthetic lookup failure")
            return good, []

        domains = [{"name": "broken", "retriever": "broken"}, {"name": "good", "retriever": "good"}]
        with patch.object(answer_module, "_retrieve_query_supported_evidence", side_effect=retrieve):
            delivered, _refs, _superseded = answer_module._retrieve_domain_evidence("question", domains)
        self.assertEqual(delivered, good)

    def test_references_survive_when_the_model_cites_nothing_in_prose(self):
        from jarvisrag.answer.answer import _format_response_for_display

        rendered = _format_response_for_display(
            "The budget was approved.", None, ['Email: "Quarterly budget review"']
        )
        self.assertIn("References:", rendered)
        self.assertIn('Email: "Quarterly budget review"', rendered)

    def test_domain_lookup_failure_yields_no_evidence_and_no_references(self):
        from jarvisrag.answer.answer import _retrieve_domain_evidence

        self.assertEqual(_retrieve_domain_evidence("question", None), ([], [], set()))

    def test_message_filenames_are_never_spoken_or_printed(self):
        """A sha1(Message-ID).emlx name must not survive into the spoken or the
        printed answer, and must not reappear as its own References entry once a
        Subject/From/Date reference has replaced it."""
        from jarvisrag.answer.answer import _format_response_for_display
        from jarvisrag.speech import _strip_inline_citations, _strip_source_filenames_for_speech

        name = "3f8a1c9e2b7d4a6f8c0e1d2b3a4c5d6e7f8a9b0c.emlx"
        prose = f"The budget was approved in {name}, and a follow-up was sent ({name})."

        spoken = _strip_source_filenames_for_speech(_strip_inline_citations(prose))
        self.assertNotIn(".emlx", spoken)
        self.assertNotIn("3f8a1c9e", spoken)
        self.assertIn("budget was approved", spoken)

        rendered = _format_response_for_display(
            prose, None, ['Email: "Quarterly budget review"'], {name.lower()}
        )
        self.assertNotIn(".emlx", rendered)
        self.assertIn('Email: "Quarterly budget review"', rendered)

    def test_every_ingested_extension_is_citation_suppressed(self):
        """The spoken/printed citation regex must cover the whole ingest list;
        mail (.emlx) and web archives (.html/.mhtml) were previously missed."""
        import re as _re
        from jarvisrag.answer.bootstrap import DOCUMENT_EXTENSIONS, IMAGE_EXTENSIONS
        from jarvisrag.speech import _CITATION_FILE_EXT

        pattern = _re.compile(rf"^(?:{_CITATION_FILE_EXT})$", _re.IGNORECASE)
        for ext in list(DOCUMENT_EXTENSIONS) + list(IMAGE_EXTENSIONS):
            with self.subTest(ext=ext):
                self.assertTrue(pattern.match(ext.lstrip(".")), f"{ext} not citation-suppressed")


def verify_live_milvus():
    """Explicit local-service validation using only a uniquely named disposable collection."""
    from jarvisrag.iotools import _is_port_open

    if not _is_port_open("127.0.0.1", "19530", timeout=2.0):
        raise RuntimeError("Live email validation requires local Milvus on 127.0.0.1:19530.")
    collection = "jarvis_mail_validation_" + uuid.uuid4().hex
    store = create_milvus_vector_store(collection_name=collection, dim=8)
    if store.has_collection():
        raise RuntimeError("Refusing to reuse an existing validation collection.")
    try:
        with tempfile.TemporaryDirectory(prefix="jarvis_mail_live_") as directory:
            root = Path(directory, "corpus")
            root.mkdir()
            persist = str(Path(directory, "index"))
            path = root / "one.txt"
            text = "The committee reviewed the project schedule and recorded its decision. " * 350
            path.write_text(text)
            document = Document(text=text, metadata={"file_path": str(path), "file_name": path.name})
            embedder = MockEmbedding(embed_dim=8)

            def build(documents, active_store=store):
                return indexing.build_or_load_persistent_index(
                    documents, active_store, persist, embedder,
                    owned_corpus_dir=str(root), owned_extensions=(".txt",),
                )

            original_add = MilvusVectorStore.add

            def partially_insert(active_store, nodes, **kwargs):
                original_add(active_store, nodes[:max(1, len(nodes) // 2)], **kwargs)
                raise RuntimeError("synthetic interruption after partial insertion")

            with patch.object(MilvusVectorStore, "add", new=partially_insert):
                try:
                    build([document])
                except RuntimeError as error:
                    assert "synthetic interruption" in str(error)
                else:
                    raise AssertionError("The interruption probe did not execute.")
            partial = store.indexed_node_ids()
            assert partial
            with patch.object(MockEmbedding, "_get_text_embedding", side_effect=AssertionError("unexpected re-embedding")):
                index, _ = build([document])
                expected = set(index.docstore.docs)
                assert partial < expected
                assert store.indexed_node_ids() == expected
                reconnected = create_milvus_vector_store(collection_name=collection, dim=8)
                try:
                    warm, _ = build([document], reconnected)
                    assert set(warm.docstore.docs) == expected
                    result = reconnected.query(VectorStoreQuery(query_embedding=[0.5] * 8, similarity_top_k=2))
                    assert result.ids and set(result.ids) <= expected
                finally:
                    if reconnected._client is not None:
                        reconnected._client.close()
                store.delete_nodes([next(iter(expected))], strict=True)
                store.flush()
                repaired, _ = build([document])
                assert set(repaired.docstore.docs) == store.indexed_node_ids() == expected
            extra = [TextNode(id_=f"synthetic-retention-{ordinal}", text="Synthetic retention check.", embedding=[0.5] * 8) for ordinal in range(5001)]
            store.add(extra, upsert=True)
            retained = store.indexed_node_ids()
            assert len(retained) == len(expected) + 5001
            store.add(extra[:3], upsert=True)
            assert store.indexed_node_ids() == retained
            with patch.object(MockEmbedding, "_get_text_embedding", side_effect=AssertionError("unexpected re-embedding")):
                build([document])
            assert store.indexed_node_ids() == expected
            changed_text = "The committee rejected the revised schedule after further review."
            path.write_text(changed_text)
            changed, _ = build([Document(text=changed_text, metadata=dict(document.metadata))])
            assert store.indexed_node_ids() == set(changed.docstore.docs)
            assert not expected & set(changed.docstore.docs)
            path.unlink()
            empty, _ = build([])
            assert not store.indexed_node_ids() and not empty.docstore.docs
            print("Live email validation passed: partial-write resume, warm search, child repair, >5000 retention, idempotent upsert, deltas and deletion.")
    finally:
        if store._client is not None:
            try:
                if store._client.has_collection(collection):
                    store._client.drop_collection(collection)
                assert not store._client.has_collection(collection)
                print("Disposable email validation collection removed.")
            finally:
                store._client.close()

class AnswerBudgetTests(unittest.TestCase):
    """The per-turn wall-clock budget must bound the LLM fallback chain: once the
    budget is spent, no further distillation (a slow model call) is STARTED, yet a
    non-empty answer still comes back from the fast pure-Python extractive
    fallbacks. This is the guard against the stacked-generation hang where a slow
    synthesis returned empty and the fallback chain then launched more multi-minute
    model calls with no time bound.
    """

    def _run_with_empty_synthesis(self, budget):
        from jarvisrag.answer import answer as answer_module

        evidence = ["- doc.pdf: The 2026 elective deferral limit is $24,500."]
        distill_calls = {"n": 0}

        def fake_distill(*_a, **_k):
            distill_calls["n"] += 1
            return "DISTILLED ANSWER"

        with patch.dict(os.environ, {"JARVIS_ANSWER_TIME_BUDGET_SECONDS": str(budget)}), \
                patch.object(answer_module, "_synthesize_evidence_answer", return_value=""), \
                patch.object(answer_module, "_extractive_reasoning_answer_from_evidence", return_value=""), \
                patch.object(answer_module, "_extractive_summary_answer_from_evidence", return_value=""), \
                patch.object(answer_module, "_distill_answer_from_evidence", side_effect=fake_distill), \
                patch.object(answer_module, "_forced_extractive_summary_from_evidence", return_value="FORCED SUMMARY"):
            answer, ok, _reason = answer_module._answer_with_review_loop("what is the limit", evidence, object())
        return answer, ok, distill_calls["n"]

    def test_spent_budget_skips_the_slow_distillation_call(self):
        answer, ok, distills = self._run_with_empty_synthesis(budget=0)
        self.assertEqual(distills, 0)  # no slow LLM call started once the budget is gone
        self.assertTrue(ok)
        self.assertEqual(answer, "FORCED SUMMARY")  # fast fallback still yields an answer

    def test_available_budget_still_uses_distillation(self):
        answer, ok, distills = self._run_with_empty_synthesis(budget=1000)
        self.assertEqual(distills, 1)  # with budget remaining, the distill IS attempted
        self.assertEqual(answer, "DISTILLED ANSWER")

class ReviewerCorroborationTests(unittest.TestCase):
    """A reviewer's hallucination claim discards an answer only when the evidence agrees."""

    EVIDENCE = [
        "- limits.html: For 2026 the 401(k) elective deferral limit is $24,500 and the catch-up is $8,000.",
        "- ira.html: The 2026 IRA limit is $7,500 ($8,600 if age 50 or older).",
    ]

    def _run(self, answer_text):
        from jarvisrag.answer import answer as answer_module

        reviewer = lambda *_a, **_k: (False, "these values are not present in the evidence")
        with patch.object(answer_module, "_synthesize_evidence_answer", return_value=answer_text), \
                patch.object(answer_module, "_review_answer_against_evidence", side_effect=reviewer), \
                patch.object(answer_module, "_extractive_value_answer_from_evidence", return_value="EXTRACTIVE"), \
                patch.object(answer_module, "_extractive_reasoning_answer_from_evidence", return_value=""), \
                patch.object(answer_module, "_extractive_summary_answer_from_evidence", return_value=""), \
                patch.object(answer_module, "_distill_answer_from_evidence", return_value=""), \
                patch.object(answer_module, "_forced_extractive_summary_from_evidence", return_value=""):
            return answer_module._answer_with_review_loop("are the limits the same", self.EVIDENCE, object())

    def test_answer_is_kept_when_every_value_is_in_the_evidence(self):
        text = "No. The 2026 IRA limit is $7,500 ($8,600 at 50+); the 401(k) limit is $24,500 plus $8,000."
        answer, ok, reason = self._run(text)
        self.assertEqual(answer, text)
        self.assertTrue(ok)
        self.assertIn("not corroborated", reason)

    def test_answer_is_discarded_when_a_value_is_missing_from_the_evidence(self):
        answer, ok, _reason = self._run("The 2026 IRA limit is $9,999.")
        self.assertEqual(answer, "EXTRACTIVE")
        self.assertTrue(ok)

    def test_value_extraction_normalizes_formatting(self):
        from jarvisrag.answer.answer import _unsupported_answer_values

        self.assertEqual(_unsupported_answer_values("Limits: $7,500, $24,500.", self.EVIDENCE), set())
        self.assertEqual(_unsupported_answer_values("A rise of 12% to $1,234.50.", self.EVIDENCE), {"12", "1234.50"})


class EvidenceBudgetTests(unittest.TestCase):
    """The global evidence ceiling must bound the diluting per-source pile WITHOUT
    ever truncating the authoritative (MCP + domain) tiers -- so a value the domain
    retriever ranked deep in its own fully-retained list can never be cut here.
    Pure, domain-agnostic list assembly.
    """

    def _lines(self, tag, n):
        return [f"- {tag}{i}.pdf: {tag} evidence {i}" for i in range(n)]

    def test_domain_is_never_truncated_even_when_ranked_deep(self):
        from jarvisrag.answer.answer import _budget_combined_evidence

        mcp = self._lines("mcp", 1)
        domain = self._lines("dom", 30)  # answer sits at domain position 21 (index 20)
        main = self._lines("main", 30)
        out = _budget_combined_evidence(mcp, domain, main, budget=40, main_reserve=6)
        # every domain line survives, in order, right after MCP
        self.assertEqual(out[: len(mcp) + len(domain)], mcp + domain)
        self.assertIn("- dom20.pdf: dom evidence 20", out)  # the deep linking chunk
        self.assertLessEqual(len(out), 40)

    def test_main_is_reserved_when_a_domain_routed(self):
        from jarvisrag.answer.answer import _budget_combined_evidence

        out = _budget_combined_evidence(self._lines("mcp", 1), self._lines("dom", 30),
                                        self._lines("main", 30), budget=40, main_reserve=6)
        main_kept = [l for l in out if l.startswith("- main")]
        self.assertEqual(len(main_kept), 6)  # main capped to the reserve, not flooding

    def test_authoritative_over_budget_is_kept_and_main_dropped(self):
        from jarvisrag.answer.answer import _budget_combined_evidence

        # MCP(12) + domain(30) = 42 authoritative lines > budget 40
        out = _budget_combined_evidence(self._lines("mcp", 12), self._lines("dom", 30),
                                        self._lines("main", 30), budget=40, main_reserve=6)
        self.assertEqual(len(out), 42)  # authoritative never sacrificed
        self.assertFalse([l for l in out if l.startswith("- main")])  # main dropped to fit

    def test_no_domain_leaves_main_untouched(self):
        from jarvisrag.answer.answer import _budget_combined_evidence

        out = _budget_combined_evidence([], [], self._lines("main", 30), budget=40, main_reserve=6)
        self.assertEqual(len(out), 30)  # single-corpus path unchanged, no reserve applied

class PeriodClarificationTests(unittest.TestCase):
    """Deterministic year clarification: a quantity question with no year, over
    evidence spanning multiple years, must ask WHICH year -- agnostically (keyed
    only on quantity-intent + multiple years, never on a topic/value/corpus).
    """

    def _ev(self, *years):
        return [f"- src{y}.pdf: the limit for {y} is a value" for y in years]

    def test_asks_year_when_quantity_query_lacks_year_and_evidence_spans_years(self):
        from jarvisrag.answer.answer import _period_clarification_from_evidence

        q = _period_clarification_from_evidence(
            "what is the contribution limit for people over 50", self._ev(2024, 2025, 2026)
        )
        self.assertTrue(q.endswith("?") or "?" in q)
        self.assertIn("2026", q)
        self.assertIn("2024", q)

    def test_silent_when_query_already_names_a_year(self):
        from jarvisrag.answer.answer import _period_clarification_from_evidence

        self.assertEqual(
            _period_clarification_from_evidence(
                "what is the 2026 contribution limit", self._ev(2024, 2025, 2026)
            ),
            "",
        )

    def test_silent_when_evidence_spans_one_year(self):
        from jarvisrag.answer.answer import _period_clarification_from_evidence

        self.assertEqual(
            _period_clarification_from_evidence("what is the limit", self._ev(2026)), ""
        )

    def test_silent_for_non_quantity_question(self):
        from jarvisrag.answer.answer import _period_clarification_from_evidence

        self.assertEqual(
            _period_clarification_from_evidence(
                "who is eligible to participate", self._ev(2024, 2025, 2026)
            ),
            "",
        )

    def test_currency_symbol_counts_as_quantity_intent(self):
        from jarvisrag.answer.answer import _period_clarification_from_evidence

        q = _period_clarification_from_evidence(
            "is the $8,000 the same for everyone", self._ev(2025, 2026)
        )
        self.assertIn("year", q.lower())


class FallbackConcisionTests(unittest.TestCase):
    """The last-resort evidence-preview fallbacks must stay CONCISE: an empty
    synthesis must never emit a wall of raw snippets (the 'answer entirely too
    long' failure). Bounded to a few short snippets, env-overridable."""

    def _long_lines(self, n):
        return [f"- Doc{i}.pdf: " + ("word " * 800) for i in range(n)]

    def test_forced_summary_is_capped_in_items_and_length(self):
        from jarvisrag.answer.answer import _forced_extractive_summary_from_evidence

        out = _forced_extractive_summary_from_evidence("what is the limit", self._long_lines(10))
        self.assertLessEqual(out.count("\n- "), 3)          # at most 3 snippet bullets
        self.assertLess(len(out), 2000)                     # nowhere near the old ~8KB dump

    def test_summary_respects_env_budget(self):
        from jarvisrag.answer.answer import _forced_extractive_summary_from_evidence
        import os

        os.environ["JARVIS_FALLBACK_SUMMARY_ITEMS"] = "2"
        os.environ["JARVIS_FALLBACK_SUMMARY_CHARS"] = "150"
        try:
            out = _forced_extractive_summary_from_evidence("what is the limit", self._long_lines(10))
            self.assertLessEqual(out.count("\n- "), 2)
        finally:
            del os.environ["JARVIS_FALLBACK_SUMMARY_ITEMS"]
            del os.environ["JARVIS_FALLBACK_SUMMARY_CHARS"]


class EmlxHtmlCleaningTests(unittest.TestCase):
    """An HTML-only email body must reduce to readable text: no <style>/<script>
    rule/code text, entities decoded, zero-width chars gone, whitespace
    collapsed -- while the From/To/Date/Subject metadata is preserved. Regression
    for the case where CSS ('a:hover { ... }') leaked into the answer."""

    _HTML = (
        "<html><head>"
        "<style>a:hover { border-style: none; text-decoration: underline rgb(57,60,65); }</style>"
        "<script>var t = 1; document.write(\"x\");</script>"
        "</head><body>\ufeff"
        "Your RMA is <b>ABC-0000EXAMPLE</b>. Return label&nbsp;tracking: 1Z999&amp;AA."
        "<!-- tracking pixel --></body></html>"
    )

    def test_html_to_text_strips_style_script_and_decodes(self):
        out = EmlxReader._html_to_text(self._HTML)
        self.assertIn("Your RMA is ABC-0000EXAMPLE", out)
        self.assertIn("1Z999&AA", out)          # &amp; decoded to &
        self.assertIn("tracking:", out)
        self.assertNotIn("a:hover", out)         # CSS rule text gone
        self.assertNotIn("text-decoration", out)
        self.assertNotIn("document.write", out)  # script code gone
        self.assertNotIn("<", out)               # no residual tags
        self.assertNotIn("\ufeff", out)          # BOM/zero-width removed
        self.assertNotIn("&nbsp;", out)          # entity decoded, not literal
        self.assertNotIn("  ", out)              # whitespace collapsed

    def _write_html_emlx(self, path: str, html: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        msg = (
            "From: sender@example.com\n"
            "To: me@example.com\n"
            "Subject: Your Order Update\n"
            "Date: Mon, 1 Jan 2026 10:00:00 -0500\n"
            "Content-Type: text/html; charset=utf-8\n"
            "\n"
            f"{html}\n"
        ).encode("utf-8")
        framed = str(len(msg)).encode("ascii") + b"\n" + msg + b'<?xml version="1.0"?><plist></plist>'
        with open(path, "wb") as handle:
            handle.write(framed)

    def test_load_data_cleans_html_body_and_keeps_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "staged", "label.emlx")
            self._write_html_emlx(source, self._HTML)
            docs = EmlxReader().load_data(source)
            self.assertEqual(len(docs), 1)
            doc = docs[0]
            self.assertIn("ABC-0000EXAMPLE", doc.text)
            self.assertNotIn("a:hover", doc.text)
            self.assertNotIn("text-decoration", doc.text)
            # Header metadata preserved for citation.
            self.assertEqual(doc.metadata.get("email_subject"), "Your Order Update")
            self.assertEqual(doc.metadata.get("email_from"), "sender@example.com")


class SynthesisBoundedOutputTests(unittest.TestCase):
    """The synthesis prompt always carries the bounded-output directive (the fix
    for the runaway 'Synthesizing 10m00s' -> empty -> raw-dump)."""

    class _CapturingLLM:
        def __init__(self):
            self.additional_kwargs = {}
            self.captured = ""

        def chat(self, messages):
            self.captured = messages[0].content
            resp = Mock()
            resp.message = Mock()
            resp.message.content = "A grounded answer."
            return resp

    def _run(self):
        from jarvisrag.answer.answer import _synthesize_evidence_answer
        llm = self._CapturingLLM()
        out = _synthesize_evidence_answer(
            "What is the RMA number?",
            ["- Email.emlx: RMA is ECR-1 (Email.emlx)"],
            llm,
        )
        return out, llm.captured

    def test_directive_is_always_present(self):
        out, prompt = self._run()
        self.assertEqual(out, "A grounded answer.")
        self.assertIn("ANSWER SHAPE", prompt)


class DomainCascadeTests(unittest.TestCase):
    """The failure-triggered domain cascade (_cascade_domain_answer) consults
    domains left un-consulted this turn ONLY when the primary answer did not fully
    answer, and adopts a re-synthesis only when it is supported and supplies at
    least part of what was asked. Keyed on answer completeness, not topic
    keywords, so it stays domain-agnostic."""

    @staticmethod
    def _domains():
        return [{"name": "mail", "retriever": object(), "match_any": ["email"]}]

    def test_no_unconsulted_domains_returns_none(self):
        from jarvisrag.answer.answer import _cascade_domain_answer
        self.assertIsNone(_cascade_domain_answer("q", [], [], llm=Mock()))

    def test_no_domain_evidence_returns_none(self):
        import jarvisrag.answer.answer as ans
        with patch.object(ans, "_retrieve_domain_evidence", return_value=([], [], set())):
            self.assertIsNone(
                ans._cascade_domain_answer("q", self._domains(), [], llm=Mock())
            )

    def test_supported_answer_is_adopted(self):
        import jarvisrag.answer.answer as ans
        with patch.object(
            ans,
            "_retrieve_domain_evidence",
            return_value=(["- Email.emlx: RMA is ECR-1 (Email.emlx)"], ["Email.emlx"], set()),
        ), patch.object(
            ans, "_answer_with_review_loop", return_value=("The RMA number is ECR-1.", True, "ok")
        ):
            out = ans._cascade_domain_answer("q", self._domains(), [], llm=Mock())
        self.assertIsNotNone(out)
        self.assertEqual(out["response_text"], "The RMA number is ECR-1.")
        self.assertEqual(out["refs"], ["Email.emlx"])
        self.assertEqual(out["domain_names"], ["mail"])
        self.assertTrue(out["evidence_lines"])

    def test_dismissive_resynthesis_returns_none(self):
        import jarvisrag.answer.answer as ans
        with patch.object(
            ans,
            "_retrieve_domain_evidence",
            return_value=(["- Email.emlx: unrelated text (Email.emlx)"], ["Email.emlx"], set()),
        ), patch.object(
            ans,
            "_answer_with_review_loop",
            return_value=("The evidence does not contain an RMA number.", True, "ok"),
        ):
            self.assertIsNone(
                ans._cascade_domain_answer("q", self._domains(), [], llm=Mock())
            )

    def test_failed_review_returns_none(self):
        import jarvisrag.answer.answer as ans
        with patch.object(
            ans,
            "_retrieve_domain_evidence",
            return_value=(["- Email.emlx: x (Email.emlx)"], ["Email.emlx"], set()),
        ), patch.object(ans, "_answer_with_review_loop", return_value=("", False, "no")):
            self.assertIsNone(
                ans._cascade_domain_answer("q", self._domains(), [], llm=Mock())
            )

    @staticmethod
    def _two_domains():
        return [
            {"name": "reference", "retriever": object(), "index": "ref-index"},
            {"name": "mail", "retriever": object(), "index": "mail-index"},
        ]

    @staticmethod
    def _per_domain_evidence(query, domains):
        name = domains[0]["name"]
        return ([f"- {name}.doc: {name} text ({name}.doc)"], [f"{name}.doc"], set())

    def test_each_domain_is_retrieved_in_isolation(self):
        # Pooled retrieval let one corpus fill the shared cap and starve the
        # next; every retrieval call must carry exactly one domain.
        import jarvisrag.answer.answer as ans
        calls = []

        def record(query, domains):
            calls.append([d["name"] for d in domains])
            return self._per_domain_evidence(query, domains)

        with patch.object(ans, "_retrieve_domain_evidence", side_effect=record), patch.object(
            ans, "_answer_with_review_loop", return_value=("The evidence does not contain it.", True, "ok")
        ):
            ans._cascade_domain_answer("q", self._two_domains(), [], llm=Mock())
        self.assertEqual(sorted(calls), [["mail"], ["reference"]])

    def test_falls_through_to_domain_that_answers(self):
        import jarvisrag.answer.answer as ans

        def answer(query, evidence, llm):
            if any("mail" in line for line in evidence):
                return ("The RMA number is ECR-1.", True, "ok")
            return ("The evidence does not contain an RMA number.", True, "ok")

        with patch.object(
            ans, "_retrieve_domain_evidence", side_effect=self._per_domain_evidence
        ), patch.object(ans, "_answer_with_review_loop", side_effect=answer), patch.object(
            ans, "_corpus_relevance", side_effect=lambda q, idx: {"ref-index": 0.9, "mail-index": 0.1}[idx]
        ):
            out = ans._cascade_domain_answer("q", self._two_domains(), [], llm=Mock())
        self.assertIsNotNone(out)
        self.assertEqual(out["domain_names"], ["mail"])
        self.assertEqual(out["refs"], ["mail.doc"])
        self.assertFalse(any("reference" in line for line in out["evidence_lines"]))

    def test_most_relevant_domain_is_tried_first(self):
        import jarvisrag.answer.answer as ans
        order = []

        def record(query, domains):
            order.append(domains[0]["name"])
            return self._per_domain_evidence(query, domains)

        with patch.object(ans, "_retrieve_domain_evidence", side_effect=record), patch.object(
            ans, "_answer_with_review_loop", return_value=("The RMA number is ECR-1.", True, "ok")
        ), patch.object(
            ans, "_corpus_relevance", side_effect=lambda q, idx: {"ref-index": 0.2, "mail-index": 0.7}[idx]
        ):
            out = ans._cascade_domain_answer("q", self._two_domains(), [], llm=Mock())
        self.assertEqual(order, ["mail"])  # first domain answered; the other is never paid for
        self.assertEqual(out["domain_names"], ["mail"])

    def test_attachment_passages_join_evidence_before_one_synthesis(self):
        # Attachments are read BEFORE synthesis, not by a retry keyed on the
        # answer's hedge wording (which missed "is not provided").
        import jarvisrag.answer.answer as ans
        seen = []

        def synth(query, evidence, llm):
            seen.append(list(evidence))
            return ("The tracking code is ZX-1.", True, "ok")

        attachment = "- mail.doc: [Attachment: label.pdf] TRACKING #: ZX-1"
        with patch.object(
            ans, "_retrieve_domain_evidence", side_effect=self._per_domain_evidence
        ), patch.object(ans, "_attachment_evidence_lines", return_value=[attachment]), patch.object(
            ans, "_answer_with_review_loop", side_effect=synth
        ):
            out = ans._cascade_domain_answer("q", self._domains(), [], llm=Mock())
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0], attachment)
        self.assertEqual(out["response_text"], "The tracking code is ZX-1.")

    def test_complete_answer_from_later_domain_beats_earlier_partial(self):
        import jarvisrag.answer.answer as ans
        order = []

        def answer(query, evidence, llm):
            name = "mail" if any("mail" in line for line in evidence) else "reference"
            order.append(name)
            return (f"{name} answer", True, "ok")

        with patch.object(
            ans, "_retrieve_domain_evidence", side_effect=self._per_domain_evidence
        ), patch.object(ans, "_answer_with_review_loop", side_effect=answer), patch.object(
            ans, "_corpus_relevance", side_effect=lambda q, idx: {"ref-index": 0.9, "mail-index": 0.1}[idx]
        ), patch.object(
            ans,
            "_answer_completeness",
            side_effect=lambda q, text, llm: ans.ANSWER_PARTIAL if text.startswith("reference") else ans.ANSWER_COMPLETE,
        ):
            out = ans._cascade_domain_answer("q", self._two_domains(), [], llm=Mock())
        self.assertEqual(order, ["reference", "mail"])
        self.assertEqual(out["domain_names"], ["mail"])
        self.assertEqual(out["completeness"], ans.ANSWER_COMPLETE)

    def test_first_partial_is_kept_when_no_domain_answers_fully(self):
        import jarvisrag.answer.answer as ans
        with patch.object(
            ans, "_retrieve_domain_evidence", side_effect=self._per_domain_evidence
        ), patch.object(
            ans, "_answer_with_review_loop", side_effect=lambda q, ev, llm: (f"{ev[0][2:8]} answer", True, "ok")
        ), patch.object(
            ans, "_corpus_relevance", side_effect=lambda q, idx: {"ref-index": 0.9, "mail-index": 0.1}[idx]
        ), patch.object(ans, "_answer_completeness", return_value=ans.ANSWER_PARTIAL):
            out = ans._cascade_domain_answer("q", self._two_domains(), [], llm=Mock())
        self.assertEqual(out["domain_names"], ["reference"])
        self.assertEqual(out["completeness"], ans.ANSWER_PARTIAL)

    def test_long_non_answer_is_not_adopted_whatever_its_wording(self):
        # Longer than the dismissive-length guard and free of hedge markers, so
        # only the meaning-based verdict can reject it.
        import jarvisrag.answer.answer as ans
        non_answer = (
            "These records cover retirement plan contribution limits, survivor annuities and "
            "earnings tests for several years; a lamp return and its tracking number are "
            "nowhere among them, so that part of your question is outside this material entirely."
        )
        self.assertFalse(ans._response_is_dismissive(non_answer))
        self.assertFalse(ans._answer_hedges_about_missing_detail(non_answer))
        with patch.object(
            ans, "_retrieve_domain_evidence", side_effect=self._per_domain_evidence
        ), patch.object(
            ans, "_answer_with_review_loop", return_value=(non_answer, True, "ok")
        ), patch.object(ans, "_answer_completeness", return_value=ans.ANSWER_UNANSWERED):
            self.assertIsNone(ans._cascade_domain_answer("q", self._domains(), [], llm=Mock()))


class AnswerCompletenessTests(unittest.TestCase):
    """_answer_completeness judges whether an answer supplies what was asked by
    meaning, so a non-answer is caught however it is worded."""

    class _VerdictLLM:
        def __init__(self, reply=None, error=None):
            self.reply, self.error, self.prompts = reply, error, []

        def chat(self, messages):
            self.prompts.append(messages[0].content)
            if self.error:
                raise self.error
            resp = Mock()
            resp.message = Mock()
            resp.message.content = self.reply
            return resp

    # Hedges in wordings the phrase list never contained.
    _PARAPHRASED = "Your blender order number is BX-1042. The delivery date isn't listed in these records."

    def _verdict(self, reply, answer=_PARAPHRASED, error=None):
        from jarvisrag.answer.answer import _answer_completeness
        llm = self._VerdictLLM(reply, error)
        return _answer_completeness("What is the order number and delivery date?", answer, llm), llm

    def test_unanswered_is_not_mistaken_for_answered(self):
        from jarvisrag.answer.answer import ANSWER_UNANSWERED
        self.assertEqual(self._verdict("UNANSWERED")[0], ANSWER_UNANSWERED)

    def test_partial_and_answered_verdicts(self):
        from jarvisrag.answer.answer import ANSWER_COMPLETE, ANSWER_PARTIAL
        self.assertEqual(self._verdict("PARTIAL")[0], ANSWER_PARTIAL)
        self.assertEqual(self._verdict(" answered.\n")[0], ANSWER_COMPLETE)

    def test_prompt_carries_question_and_answer(self):
        _, llm = self._verdict("PARTIAL")
        self.assertIn("What is the order number and delivery date?", llm.prompts[0])
        self.assertIn(self._PARAPHRASED, llm.prompts[0])

    def test_empty_or_dismissive_answer_needs_no_model_call(self):
        from jarvisrag.answer.answer import ANSWER_UNANSWERED
        for answer in ("", "I could not find that."):
            verdict, llm = self._verdict("ANSWERED", answer=answer)
            self.assertEqual(verdict, ANSWER_UNANSWERED)
            self.assertEqual(llm.prompts, [])

    def test_unavailable_model_falls_back_to_phrase_markers(self):
        from jarvisrag.answer.answer import ANSWER_COMPLETE, ANSWER_PARTIAL
        hedged = "The order number is BX-1042, but the evidence does not state a delivery date."
        self.assertEqual(self._verdict(None, answer=hedged, error=RuntimeError("down"))[0], ANSWER_PARTIAL)
        self.assertEqual(self._verdict("maybe", answer="The order number is BX-1042.")[0], ANSWER_COMPLETE)


class CascadeTriggerTests(unittest.TestCase):
    """handle_query cascades when the primary answer passed review but did not
    supply what was asked -- judged by meaning, not by hedge phrases."""

    _EVIDENCE = [
        "- receipt.pdf: Lamp purchased June 2 for $45 (receipt.pdf)",
        "- receipt.pdf: Shipped from the Ohio warehouse (receipt.pdf)",
        "- receipt.pdf: Return label follows in a separate message (receipt.pdf)",
    ]
    # Passes review, is too long for the dismissive guard, and uses no hedge marker.
    _PRIMARY = (
        "The lamp was purchased on June 2 for $45 and shipped from the Ohio warehouse. A return "
        "label is mentioned as arriving in a separate message, which isn't among these records, "
        "so the tracking number for the return isn't something I can see here."
    )

    def _run(self, completeness, cascade_result=None):
        import jarvisrag.answer.answer as ans
        runtime = ans.QueryRuntime(
            all_docs=[],
            llm=Mock(),
            doc_retriever=object(),
            domain_retrievers=[{"name": "mail", "retriever": object(), "match_any": ["email"]}],
        )
        cascade = Mock(return_value=cascade_result)
        judge = Mock(return_value=completeness)
        with patch.object(
            ans, "_retrieve_query_supported_evidence", return_value=(list(self._EVIDENCE), [])
        ), patch.object(ans, "fetch_mcp_evidence", return_value=[]), patch.object(
            ans, "_extractive_value_answer_from_evidence", return_value=""
        ), patch.object(
            ans, "_extractive_value_answer_from_documents", return_value=""
        ), patch.object(
            ans, "_answer_with_review_loop", return_value=(self._PRIMARY, True, "ok")
        ), patch.object(ans, "_answer_completeness", judge), patch.object(
            ans, "_cascade_domain_answer", cascade
        ), patch.object(
            ans, "_clarifying_question_from_evidence", return_value=""
        ), contextlib.redirect_stdout(io.StringIO()):
            turn = ans.handle_query("What is the tracking number for the lamp return?", runtime)
        return turn, cascade, judge

    def test_paraphrased_partial_answer_cascades_and_adopts(self):
        import jarvisrag.answer.answer as ans
        adopted = {
            "response_text": "The return tracking number is ZX-1.",
            "review_reason": "ok",
            "evidence_lines": ["- mail.emlx: tracking ZX-1 (mail.emlx)"],
            "refs": ["mail.emlx"],
            "superseded": set(),
            "domain_names": ["mail"],
            "completeness": ans.ANSWER_COMPLETE,
        }
        turn, cascade, judge = self._run(ans.ANSWER_PARTIAL, adopted)
        cascade.assert_called_once()
        self.assertEqual([d["name"] for d in cascade.call_args.args[1]], ["mail"])
        self.assertEqual(turn["response_text"], "The return tracking number is ZX-1.")
        self.assertEqual(judge.call_count, 1)  # cascade's verdict is reused, not recomputed

    def test_complete_answer_never_cascades(self):
        import jarvisrag.answer.answer as ans
        turn, cascade, judge = self._run(ans.ANSWER_COMPLETE)
        cascade.assert_not_called()
        self.assertEqual(turn["response_text"], self._PRIMARY)
        self.assertEqual(judge.call_count, 1)

    def test_partial_answer_is_kept_when_cascade_finds_nothing(self):
        import jarvisrag.answer.answer as ans
        turn, cascade, judge = self._run(ans.ANSWER_PARTIAL, None)
        cascade.assert_called_once()
        self.assertEqual(turn["response_text"], self._PRIMARY)
        self.assertEqual(judge.call_count, 1)  # the clarification check reuses the verdict


def _write_attachment_emlx(path: str, message_id: str, parts: list[tuple[str, str, bytes, str]]) -> None:
    """Multipart .emlx like Apple Mail writes. Each part is (filename, content
    type, payload, disposition); an empty payload models an attachment Apple
    Mail stored outside the message."""
    import base64

    os.makedirs(os.path.dirname(path), exist_ok=True)
    boundary = "BOUNDARY"
    body = [f"--{boundary}\nContent-Type: text/plain; charset=utf-8\n\nSee attached.\n"]
    for filename, ctype, payload, disposition in parts:
        cid = "Content-ID: <logo@example.com>\n" if disposition == "inline" else ""
        body.append(
            f"--{boundary}\nContent-Type: {ctype}; name=\"{filename}\"\n"
            f"Content-Disposition: {disposition}; filename=\"{filename}\"\n"
            f"Content-Transfer-Encoding: base64\n{cid}\n"
            f"{base64.b64encode(payload).decode('ascii')}\n"
        )
    msg = (
        "From: sender@example.com\nTo: me@example.com\nSubject: Documents\n"
        f"Message-ID: {message_id}\nDate: Mon, 1 Jan 2026 10:00:00 -0500\n"
        f"Content-Type: multipart/mixed; boundary=\"{boundary}\"\n\n"
        + "".join(body) + f"--{boundary}--\n"
    ).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(str(len(msg)).encode("ascii") + b"\n" + msg + b'<?xml version="1.0"?><plist></plist>')


class EmailAttachmentPointerTests(unittest.TestCase):
    """Attachments are never indexed. Mail refresh mirrors Apple Mail's external
    attachment files beside the corpus, the staged .emlx's MIME headers act as
    the pointer, and an attachment is extracted only when an answer needs it."""

    def setUp(self):
        self.work = tempfile.TemporaryDirectory(prefix="jarvis_attach_")
        self.addCleanup(self.work.cleanup)
        self.mail_root = os.path.join(self.work.name, "mail")
        self.corpus = os.path.join(self.work.name, "corpus")
        data = os.path.join(self.mail_root, "Account", "INBOX.mbox", "Data", "1")
        self.source = os.path.join(data, "Messages", "7.partial.emlx")
        self.external = os.path.join(data, "Attachments", "7", "2", "notes.txt")
        _write_attachment_emlx(self.source, "<att@example.com>", [
            ("notes.txt", "text/plain", b"", "attachment"),
            ("memo.txt", "text/plain", b"Embedded memo text.", "attachment"),
            ("logo.png", "image/png", b"\x89PNG", "inline"),
        ])
        os.makedirs(os.path.dirname(self.external), exist_ok=True)
        with open(self.external, "w", encoding="utf-8") as handle:
            handle.write("The tracking code for the shipment is ZX-1.")
        self.doc_set = DocSet(name="mail", corpus_dir=self.corpus, type="mail", mail_root=self.mail_root)
        self.assertEqual(refresh_mail_set(self.doc_set)[2], [])
        self.staged = os.path.join(self.corpus, docsync._msgid_key("<att@example.com>"))
        self.stem = os.path.basename(self.staged)[: -len(".emlx")]
        self.mirror = os.path.join(docsync.mail_attachment_dir(self.corpus), self.stem)

    def test_refresh_mirrors_external_attachment_beside_corpus(self):
        mirrored = os.path.join(self.mirror, "2", "notes.txt")
        with open(mirrored, encoding="utf-8") as handle:
            self.assertIn("ZX-1", handle.read())
        self.assertFalse(os.path.abspath(self.mirror).startswith(os.path.abspath(self.corpus) + os.sep))
        self.assertEqual(glob.glob(os.path.join(self.corpus, "**", "notes.txt"), recursive=True), [])

    def test_refresh_prunes_mirror_when_message_leaves(self):
        shutil.rmtree(os.path.join(self.mail_root, "Account"))
        _write_emlx(os.path.join(self.mail_root, "Inbox", "other.emlx"), "sender@example.com", "Other", "Text.", "<other@example.com>")
        self.assertEqual(refresh_mail_set(self.doc_set)[2], [])
        self.assertFalse(os.path.exists(self.mirror))

    def test_pointer_resolves_external_and_embedded_but_skips_inline(self):
        from jarvisrag.extraction.extractors import staged_email_attachments

        found = dict(staged_email_attachments(self.staged))
        self.assertEqual(sorted(found), ["memo.txt", "notes.txt"])
        self.assertTrue(all(os.path.isfile(path) for path in found.values()))
        with open(found["memo.txt"], encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "Embedded memo text.")

    def test_attachment_text_is_dereferenced_on_demand_and_cites_its_email(self):
        import jarvisrag.answer.answer as ans
        import jarvisrag.answer.bootstrap as boot

        cache = os.path.join(self.work.name, "cache")
        domain = {"name": "mail", "corpus_dir": os.path.abspath(self.corpus)}
        evidence = [f"- {self.stem}.emlx: See attached.", "- other.pdf: unrelated text"]
        with patch.object(boot, "EXTRACTION_CACHE_DIR", cache), contextlib.redirect_stdout(io.StringIO()):
            lines = ans._attachment_evidence_lines("What is the tracking code?", evidence, [domain])
        self.assertTrue(any("[Attachment: notes.txt]" in line and "ZX-1" in line for line in lines))
        self.assertTrue(all(line.startswith(f"- {self.stem}.emlx: ") for line in lines))
        self.assertTrue(os.listdir(cache))  # extraction went to the on-disk cache, not the index

    def test_emails_outside_the_domain_corpus_are_ignored(self):
        import jarvisrag.answer.answer as ans

        other = {"name": "mail", "corpus_dir": os.path.join(self.work.name, "elsewhere")}
        self.assertEqual(ans._evidence_email_paths([f"- {self.stem}.emlx: text"], [other]), [])

    def test_identical_attachment_on_two_emails_is_read_once(self):
        import jarvisrag.answer.answer as ans
        import jarvisrag.answer.bootstrap as boot

        twin_stem = "f" * 24
        shutil.copyfile(self.staged, os.path.join(self.corpus, f"{twin_stem}.emlx"))
        shutil.copytree(self.mirror, os.path.join(docsync.mail_attachment_dir(self.corpus), twin_stem))
        domain = {"name": "mail", "corpus_dir": os.path.abspath(self.corpus)}
        evidence = [f"- {self.stem}.emlx: See attached.", f"- {twin_stem}.emlx: See attached."]
        with patch.object(boot, "EXTRACTION_CACHE_DIR", os.path.join(self.work.name, "cache")), \
                contextlib.redirect_stdout(io.StringIO()):
            lines = ans._attachment_evidence_lines("What is the tracking code?", evidence, [domain])
        tracking = [line for line in lines if "[Attachment: notes.txt]" in line]
        self.assertEqual(len(tracking), 1)
        self.assertTrue(tracking[0].startswith(f"- {self.stem}.emlx: "))  # kept from the higher-ranked email

    def test_unrelated_attachment_is_not_added_to_evidence(self):
        import jarvisrag.answer.answer as ans
        import jarvisrag.answer.bootstrap as boot

        domain = {"name": "mail", "corpus_dir": os.path.abspath(self.corpus)}
        evidence = [f"- {self.stem}.emlx: See attached."]
        with patch.object(boot, "EXTRACTION_CACHE_DIR", os.path.join(self.work.name, "cache")), \
                contextlib.redirect_stdout(io.StringIO()):
            lines = ans._attachment_evidence_lines("Which city hosts the annual gardening festival?", evidence, [domain])
        self.assertEqual(lines, [])


if __name__ == "__main__":
    unittest.main()
