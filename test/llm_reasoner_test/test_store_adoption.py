"""
Tests of a store adopting the hypotheses another store admitted: the records it lacks
are copied over whole, so a later request is served from it without asking a model.
"""

from __future__ import annotations

import shutil

from llm_reasoner.knowledge.storage.store import INCOMING_PREFIX, HypothesisStore

from .test_generation import GenerationCase
from .test_hypotheses import pouring_query, service_answering

# %% helpers


def store_holding_one(directory) -> HypothesisStore:
    """
    :return: A store into which one hypothesis about pouring was generated.
    """
    store = HypothesisStore(directory=directory)
    service_answering(store, GenerationCase.LIFT_AND_POUR).serve(pouring_query())
    return store


def record_ids(store: HypothesisStore) -> set[str]:
    return {record.record_id for record in store.records()}


# %% adopting


class TestAdoption:
    def test_the_records_a_store_lacks_are_adopted(self, tmp_path):
        source = store_holding_one(tmp_path / "source")
        target = HypothesisStore(directory=tmp_path / "target")
        adopted = target.adopt_new_records(source)
        assert set(adopted) == record_ids(source)
        assert record_ids(target) == record_ids(source)

    def test_an_adopted_hypothesis_is_the_same_hypothesis(self, tmp_path):
        source = store_holding_one(tmp_path / "source")
        target = HypothesisStore(directory=tmp_path / "target")
        (record_id,) = target.adopt_new_records(source)
        assert set(target.get(record_id).graph) == set(source.get(record_id).graph)

    def test_an_adopted_hypothesis_is_served_without_asking_a_model(self, tmp_path):
        source = store_holding_one(tmp_path / "source")
        target = HypothesisStore(directory=tmp_path / "target")
        (record_id,) = target.adopt_new_records(source)
        assert service_answering(target).serve(pouring_query()) == record_id

    def test_records_the_store_holds_are_not_adopted_again(self, tmp_path):
        source = store_holding_one(tmp_path / "source")
        target = HypothesisStore(directory=tmp_path / "target")
        target.adopt_new_records(source)
        assert target.adopt_new_records(source) == []

    def test_nothing_but_the_records_is_left_in_the_store(self, tmp_path):
        source = store_holding_one(tmp_path / "source")
        target = HypothesisStore(directory=tmp_path / "target")
        target.adopt_new_records(source)
        assert {path.name for path in target.directory.iterdir()} == record_ids(source)

    def test_a_store_with_no_directory_gives_nothing_to_adopt(self, tmp_path):
        target = HypothesisStore(directory=tmp_path / "target")
        empty = HypothesisStore(directory=tmp_path / "missing")
        assert target.adopt_new_records(empty) == []

    def test_a_hypothesis_still_being_copied_in_is_not_listed(self, tmp_path):
        source = store_holding_one(tmp_path / "source")
        target = HypothesisStore(directory=tmp_path / "target")
        (record_id,) = record_ids(source)
        shutil.copytree(
            source.directory / record_id,
            target.directory / (INCOMING_PREFIX + record_id),
        )
        assert target.records() == []
