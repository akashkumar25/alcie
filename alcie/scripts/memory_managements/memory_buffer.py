import pickle
import random
import os
import torch
from torch.nn.functional import softmax
from transformers import Trainer
import clip

# MemoryBuffer Class
class MemoryBuffer:
    def __init__(self, capacity=550, sampling_strategy="random"):
        self.buffer = {}
        self.capacity = capacity
        self.sampling_strategy = sampling_strategy
        self.log_file = {
            "random": "logs/memory_tracking_random.log",
            "diversity": "logs/memory_tracking_diversity.log",
            "uncertainty": "logs/memory_tracking_uncertainty.log"
        }[sampling_strategy]
        
        with open(self.log_file, 'a') as f:
            f.write(f"\nMemory Buffer Tracking Log - {sampling_strategy.capitalize()}\n")
            f.write("===========================\n\n")

    def set_total_capacity(self, capacity):
        self.capacity = capacity
        print(f"Total memory buffer capacity set to {self.capacity}.")

    def _hash_batch(self, batch):
        image_data = batch['patch_images'].cpu().numpy().tobytes()
        input_data = batch['input_ids'].cpu().numpy().tobytes()
        return hash((image_data, input_data))

    def push(self, batch, cluster_id):
        clean_batch = {k: v for k, v in batch.items() if k != 'caption'}
        clean_batch['return_loss'] = True
        clean_batch['cluster_id'] = cluster_id

        batch_hash = self._hash_batch(clean_batch)

        if batch_hash in self.buffer:
            print(f"Duplicate batch detected, not adding to memory buffer.")
        else:
            if len(self.buffer) >= self.capacity:
                print(f"Memory buffer full. No more space to add new batches.")
            else:
                self.buffer[batch_hash] = clean_batch
                print(f"Added batch to memory buffer. Current buffer size: {len(self.buffer)} batches.")
                with open(self.log_file, 'a') as f:
                    f.write(f"Added batch from cluster {cluster_id}. Current buffer size: {len(self.buffer)} batches.\n")

    def sample(self, sample_size, exclude_cluster=None):
        if exclude_cluster is not None:
            available_hashes = [k for k, v in self.buffer.items() if v['cluster_id'] != exclude_cluster]
        else:
            available_hashes = list(self.buffer.keys())

        sampled_hashes = random.sample(available_hashes, min(sample_size, len(available_hashes)))
        with open(self.log_file, 'a') as f:
            f.write(f"Replaying memory batch with samples.\n")

        return [self.buffer[hash] for hash in sampled_hashes]
    
    def sample_by_cluster(self, num_samples_per_cluster, exclude_cluster=None):
        """
        Samples a fixed number of batches from each cluster in the memory buffer,
        excluding the specified current cluster.

        Args:
            num_samples_per_cluster (int): Number of samples to draw from each cluster.
            exclude_cluster (int, optional): Cluster ID to exclude (e.g., the current cluster).

        Returns:
            list: A list of sampled batches from all previous clusters.
        """
        clusters = {}
        # Group keys by their cluster_id
        for key, value in self.buffer.items():
            cluster_id = value['cluster_id']
            if exclude_cluster is not None and cluster_id == exclude_cluster:
                continue
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(key)
        
        sampled_batches = []
        for cluster_id, keys in clusters.items():
            n = min(num_samples_per_cluster, len(keys))
            sampled_keys = random.sample(keys, n)
            for key in sampled_keys:
                sampled_batches.append(self.buffer[key])
        
        return sampled_batches


    def __len__(self):
        return len(self.buffer)

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.buffer, f)
        print(f"Memory buffer saved to {file_path}")
        with open(self.log_file, 'a') as f:
            f.write(f"Memory buffer saved to {file_path}\n")

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.buffer = pickle.load(f)
        print(f"Memory buffer loaded with {len(self.buffer)} items.")
        with open(self.log_file, 'a') as f:
            f.write(f"Memory buffer loaded with {len(self.buffer)} items from {file_path}\n")

    def free_space_for_new_cluster(self, current_cluster, delete_percent_per_cluster):
        previous_clusters = list(range(1, current_cluster))
        total_deleted = 0

        for cluster in previous_clusters:
            deleted = self._delete_from_cluster(cluster, delete_percent_per_cluster)
            total_deleted += deleted
            with open(self.log_file, 'a') as f:
                f.write(f"Deleted {deleted} items from cluster {cluster} to make space for cluster {current_cluster}\n")

        print(f"Total of {total_deleted} items deleted to make space for cluster {current_cluster}.")

    def _delete_from_cluster(self, cluster, delete_percent):
        cluster_entries = [key for key, value in self.buffer.items() if value['cluster_id'] == cluster]
        num_to_delete = int(len(cluster_entries) * delete_percent)

        for key in cluster_entries[:num_to_delete]:
            del self.buffer[key]
        print(f"Deleted {num_to_delete} items from cluster {cluster}.")
        return num_to_delete
