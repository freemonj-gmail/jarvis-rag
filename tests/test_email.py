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


if __name__ == "__main__":
    unittest.main()
