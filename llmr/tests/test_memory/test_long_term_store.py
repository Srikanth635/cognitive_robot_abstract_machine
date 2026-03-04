"""Tests for LongTermMemoryStore using a mocked MongoDB client."""

from unittest.mock import MagicMock, patch

import pytest

from llmr.workflows.lg_memory.memory.long_term_store import LongTermMemoryStore


@pytest.fixture
def store() -> LongTermMemoryStore:
    with patch(
        "src.workflows.lg_memory.memory.long_term_store.MongoClient"
    ) as mock_client_cls:
        mock_col = MagicMock()
        mock_db = MagicMock()
        mock_db.__getitem__.return_value = mock_col
        mock_client_cls.return_value.__getitem__.return_value = mock_db

        s = LongTermMemoryStore(client=mock_client_cls.return_value)
        s.col = mock_col
        return s


class TestLongTermMemoryStoreSave:
    def test_save_returns_string_id(self, store: LongTermMemoryStore) -> None:
        store.col.insert_one.return_value.inserted_id = "507f1f77bcf86cd799439011"
        result = store.save(user_id="u1", content="User likes coffee")
        assert isinstance(result, str)
        assert result == "507f1f77bcf86cd799439011"

    def test_save_calls_insert_once(self, store: LongTermMemoryStore) -> None:
        store.col.insert_one.return_value.inserted_id = "abc"
        store.save(user_id="u1", content="A fact")
        store.col.insert_one.assert_called_once()

    def test_save_includes_user_id_in_doc(self, store: LongTermMemoryStore) -> None:
        store.col.insert_one.return_value.inserted_id = "xyz"
        store.save(user_id="test_user", content="some content")
        call_args = store.col.insert_one.call_args[0][0]
        assert call_args["user_id"] == "test_user"
        assert call_args["content"] == "some content"

    def test_save_defaults_memory_type_to_fact(self, store: LongTermMemoryStore) -> None:
        store.col.insert_one.return_value.inserted_id = "id1"
        store.save(user_id="u1", content="c")
        doc = store.col.insert_one.call_args[0][0]
        assert doc["memory_type"] == "fact"

    def test_save_with_custom_memory_type(self, store: LongTermMemoryStore) -> None:
        store.col.insert_one.return_value.inserted_id = "id2"
        store.save(user_id="u1", content="c", memory_type="preference")
        doc = store.col.insert_one.call_args[0][0]
        assert doc["memory_type"] == "preference"


class TestLongTermMemoryStoreStats:
    def test_stats_aggregates_by_type(self, store: LongTermMemoryStore) -> None:
        store.col.aggregate.return_value = [
            {"_id": "fact", "count": 3},
            {"_id": "preference", "count": 1},
        ]
        result = store.stats("u1")
        assert result == {"fact": 3, "preference": 1}

    def test_stats_empty_result(self, store: LongTermMemoryStore) -> None:
        store.col.aggregate.return_value = []
        result = store.stats("u_new")
        assert result == {}


class TestLongTermMemoryStoreUpsert:
    def test_upsert_calls_save_when_no_existing(self, store: LongTermMemoryStore) -> None:
        store.col.find_one.return_value = None
        store.col.insert_one.return_value.inserted_id = "new_id"
        result = store.upsert(user_id="u1", content="new fact")
        store.col.insert_one.assert_called_once()
        assert result == "new_id"

    def test_upsert_updates_when_existing(self, store: LongTermMemoryStore) -> None:
        store.col.find_one.return_value = {"_id": "existing_id", "tags": [], "embedding": None, "metadata": {}}
        result = store.upsert(user_id="u1", content="existing fact")
        store.col.update_one.assert_called_once()
        assert result == "existing_id"
