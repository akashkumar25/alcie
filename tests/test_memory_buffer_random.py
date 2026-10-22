import random

import torch

from alcie.scripts.memory_managements.random_sampling import (
    MemoryBufferRandom,
    RandomMemoryReplayTrainer,
)


def make_sample(image_id):
    return {
        "image_ids": image_id,
        "input_ids": torch.tensor([[1, 2, 3]]),
        "labels": torch.tensor([[1, 2, 3]]),
    }


def fill(buf, cluster, n, prefix=None):
    for i in range(n):
        buf.push(make_sample(f"{prefix or cluster}-{i}"), cluster_id=cluster)


def count(buf, cluster):
    return sum(1 for v in buf.buffer.values() if v["cluster_id"] == cluster)


def test_push_tags_cluster_and_ignores_duplicates():
    buf = MemoryBufferRandom(capacity=10)
    buf.push(make_sample("a"), cluster_id=1)
    buf.push(make_sample("a"), cluster_id=2)
    assert len(buf) == 1
    assert buf.buffer["a"]["cluster_id"] == 1


def test_capacity_is_not_exceeded():
    buf = MemoryBufferRandom(capacity=5)
    fill(buf, 1, 20)
    assert len(buf) == 5
    buf.set_total_capacity(8)
    fill(buf, 2, 20)
    assert len(buf) == 8


def test_free_space_deletes_fraction_from_each_previous_cluster_only():
    buf = MemoryBufferRandom(capacity=100)
    fill(buf, 1, 20)
    fill(buf, 2, 20)
    fill(buf, 3, 20)
    buf.free_space_for_new_cluster(current_cluster=3, delete_percent=0.25)
    assert count(buf, 1) == 15
    assert count(buf, 2) == 15
    assert count(buf, 3) == 20  # the current cluster is untouched


def test_free_space_with_tiny_fraction_deletes_nothing():
    buf = MemoryBufferRandom(capacity=100)
    fill(buf, 1, 3)
    buf.free_space_for_new_cluster(current_cluster=2, delete_percent=0.1)
    assert len(buf) == 3


def test_sample_by_cluster_excludes_current_cluster_and_caps_per_cluster():
    random.seed(0)
    buf = MemoryBufferRandom(capacity=100)
    fill(buf, 1, 5)
    fill(buf, 2, 1)
    fill(buf, 3, 5)
    batch = buf.sample_by_cluster(2, exclude_cluster=3)
    assert sorted(s["cluster_id"] for s in batch) == [1, 1, 2]


def test_save_and_load_roundtrip(tmp_path):
    buf = MemoryBufferRandom(capacity=10)
    fill(buf, 1, 4)
    path = tmp_path / "buffer.pkl"
    buf.save(path)
    other = MemoryBufferRandom(capacity=10)
    other.load(path)
    assert set(other.buffer) == set(buf.buffer)
    assert torch.equal(other.buffer["1-0"]["input_ids"], buf.buffer["1-0"]["input_ids"])


def test_collate_samples_pads_unequal_sequences():
    samples = [
        {"input_ids": torch.tensor([[1, 2, 3]]), "image_ids": "a"},
        {"input_ids": torch.tensor([[4, 5]]), "image_ids": "b"},
    ]
    batch = RandomMemoryReplayTrainer.collate_samples(
        type("T", (), {"device": "cpu"})(), samples
    )
    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 5, 0]]
    assert batch["image_ids"] == ["a", "b"]
