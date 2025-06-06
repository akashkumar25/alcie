import torch
import clip
from transformers import Trainer
from sklearn.cluster import KMeans
import numpy as np
import pickle
import random
import os
from torch.nn.utils.rnn import pad_sequence


# ===============================
# MemoryBuffer with Score-based Deletion
# ===============================
class MemoryBufferDiversity:
    def __init__(self, capacity=550, sampling_strategy="diversity"):
        self.buffer = {}
        self.capacity = capacity
        self.sampling_strategy = sampling_strategy
        self.log_file = f"logs/memory_tracking_{sampling_strategy}.log"

        with open(self.log_file, 'a') as f:
            f.write(f"\nMemory Buffer Tracking Log - {sampling_strategy.capitalize()}\n")
            f.write("===========================\n\n")

    def set_total_capacity(self, capacity):
        self.capacity = capacity
        print(f"Total memory buffer capacity set to {self.capacity}.")

    def _hash_sample(self, sample):
        image_data = sample['patch_image'].half().cpu().numpy().tobytes()
        return hash(image_data)
    
    def _get_lowest_score(self, cluster_id):
        # Get only the samples in the current cluster.
        cluster_samples = [sample for sample in self.buffer.values() if sample.get('cluster_id') == cluster_id]
        if not cluster_samples:
            return -1  # Return a sentinel value if no sample exists for this cluster.
        lowest_score_sample = min(cluster_samples, key=lambda x: x['diversity_score'])
        return lowest_score_sample['diversity_score']
        
    def push(self, sample, cluster_id, score):
        sample['cluster_id'] = cluster_id
        sample['diversity_score'] = score
        
        image_id = sample['image_ids']
        lowest_score = self._get_lowest_score(cluster_id)

        if image_id in self.buffer:
            print(f"Duplicate sample detected, not adding to memory buffer.")
        else:
            if len(self.buffer) >= self.capacity:
                if score > lowest_score:
                    print(f"Freeing space: Current sample score {score} > lowest score {lowest_score}.")
                    self.free_space(cluster_id, num_to_delete=1) # Free space before adding the new sample
                    self._add_sample(image_id, sample, score, cluster_id)
                else:
                    print(f"Memory Buffer is full. Current size is {len(self.buffer)} samples")     
            else:
                self._add_sample(image_id, sample, score, cluster_id)
                
               
    def _add_sample(self, image_id, sample, score, cluster_id):
        self.buffer[image_id] = sample
        print(f"Added sample with diversity score {score} from cluster {cluster_id}. Current buffer size: {len(self.buffer)} samples.\n")
        with open(self.log_file, 'a') as f:
            f.write(f"Added sample with diversity score {score} from cluster {cluster_id}. Current buffer size: {len(self.buffer)} samples.\n")
            
    def free_space(self, cluster_id, num_to_delete=1):
        """
        Free a fraction of the memory by deleting samples with the highest diversity scores
        from the current cluster only (i.e. the ones with the largest distances), thereby preserving the best (lowest distance) samples.
        """
        # Filter the items to only those from the current cluster.
        cluster_items = [(key, sample) for key, sample in self.buffer.items() if sample.get('cluster_id') == cluster_id]
        if not cluster_items:
            print(f"No samples found in cluster {cluster_id} to delete.")
            return

        # Sort the samples in descending order (highest diversity score first).
        # sorted_samples = sorted(cluster_items, key=lambda x: x[1]['diversity_score'], reverse=True)
        sorted_samples = sorted(cluster_items, key=lambda x: x[1]['diversity_score'])  # ascending
        
        # Delete up to num_to_delete items from this sorted list.
        for i in range(min(num_to_delete, len(sorted_samples))):
            key_to_delete = sorted_samples[i][0]
            del self.buffer[key_to_delete]
        print(f"Deleted {num_to_delete} sample(s) with the highest diversity scores from cluster {cluster_id}.")
        with open(self.log_file, 'a') as f:
            f.write(f"Deleted {num_to_delete} sample(s) based on highest diversity scores from cluster {cluster_id}.\n")
            
    def free_space_for_new_cluster(self, current_cluster, delete_percent):
        """
            Free space by deleting a fraction of the samples from each previous cluster.
            
            The total deletion fraction is defined as 1/current_cluster.
            This total is split evenly among all previously trained clusters.
            
            For example:
            - If current_cluster == 3, total deletion fraction = 1/3 (~33.33%).
                There are 2 previous clusters, so each deletes (1/3)/2 ≈ 16.67%.
            - If current_cluster == 4, total deletion fraction = 1/4 (25%).
                There are 3 previous clusters, so each deletes (1/4)/3 ≈ 8.33%.
        """
        previous_clusters = list(range(1, current_cluster))
        total_deleted = 0
        for cluster in previous_clusters:
            deleted = self._delete_from_cluster(cluster, delete_percent)
            total_deleted += deleted
            with open(self.log_file, 'a') as f:
                f.write(f"Deleted {deleted} items i.e. {delete_percent}% from cluster {cluster} to make space for cluster {current_cluster}\n")

        print(f"Total of {total_deleted} items deleted to make space for cluster {current_cluster}.")

    def _delete_from_cluster(self, cluster, delete_percent):
        """
        Delete a percentage of the samples from the specified cluster, based on the highest diversity scores.
      
        (i.e., the largest distances), thereby preserving the best (lowest distance) samples.
        
        """  
        cluster_entries = [key for key, value in self.buffer.items() if value['cluster_id'] == cluster]
        if not cluster_entries:
            print(f"No samples found for cluster {cluster}. Skipping deletion.")
            return 0

        # Sort samples by diversity score in ascending order
        sorted_entries = sorted(cluster_entries, key=lambda key: self.buffer[key]['diversity_score'])
        
        # Determine how many samples to delete
        num_to_delete = int(len(sorted_entries) * delete_percent)
        if num_to_delete == 0:
            print(f"Calculated 0 samples to delete for cluster {cluster}. Skipping.")
            return 0

        # Delete the selected samples
        for i in range(num_to_delete):
            del self.buffer[sorted_entries[i]]

        print(f"Deleted {num_to_delete} items i.e {delete_percent}% from cluster {cluster}.")        
        
        return num_to_delete

    def sample_by_cluster(self, num_samples_per_cluster, exclude_cluster=None):
        """
        Sample a fixed number of data points per cluster.
        """
        clusters = {}
        for key, value in self.buffer.items():
            cluster_id = value['cluster_id']
            if exclude_cluster is not None and cluster_id == exclude_cluster:
                continue
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(key)

        sampled_batches = []
        for cluster_id, keys in clusters.items():
            sampled_keys = random.sample(keys, min(num_samples_per_cluster, len(keys)))
            sampled_batches.extend([self.buffer[key] for key in sampled_keys])

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

# ===============================
# Diversity Memory Replay Trainer
# ===============================
class DiversityMemoryReplayTrainer(Trainer):
    def __init__(self, *args, memory_buffer=None, replay_freq=200, use_memory_replay=False,
                 current_cluster=1, clip_model=None, preprocess=None, device="cuda", **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_buffer = memory_buffer
        self.replay_freq = replay_freq
        self.use_memory_replay = use_memory_replay
        self.current_cluster = current_cluster
        self.clip_model = clip_model
        self.preprocess = preprocess
        self.device = device
        self.total_samples_seen = 0  # Counter to track samples processed

    def training_step(self, model, inputs, num_items_in_batch=None):
        inputs_for_model = {k: v for k, v in inputs.items() if k not in ['captions','image_ids']}
        loss = super().training_step(model, inputs_for_model, num_items_in_batch)

        self.total_samples_seen += len(inputs_for_model['input_ids'])

        if self.memory_buffer is not None:
            self.select_diverse_samples_with_clip(inputs)

        if self.use_memory_replay and len(self.memory_buffer) > 0:
            if self.total_samples_seen >= self.replay_freq:
                self.total_samples_seen = 0
                memory_samples = self.memory_buffer.sample_by_cluster(num_samples_per_cluster=1, exclude_cluster=self.current_cluster)
                if memory_samples:
                    memory_batch = self.collate_samples(memory_samples)
                else:
                    print('Not able to sample data from previous clusters')
                clean_memory_batch = {k: v for k, v in memory_batch.items() if k not in ['cluster_id', 'diversity_score','image_ids']}
                print(f"Replaying memory batch from previous clusters. Sample size: {len(clean_memory_batch['input_ids'])} samples.")
                memory_loss = super().training_step(model, clean_memory_batch, num_items_in_batch)
                loss += memory_loss
                print("Replayed sample from memory.")

        return loss

    def select_diverse_samples_with_clip(self, batch, k=16):
        # patch_images = batch['patch_images']
        pixel_values = batch['pixel_values'].to(self.device)
        captions = batch['captions']
        batch_size = pixel_values.size(0)

        patch_images_device = pixel_values.to(self.device)
        tokenized_captions = clip.tokenize(captions, context_length=77, truncate=True).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(patch_images_device)
            text_features = self.clip_model.encode_text(tokenized_captions)
            features = (image_features + text_features) / 2

        features_np = features.cpu().numpy()
        num_clusters = min(k, batch_size)
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(features_np)
        
        for cluster in range(num_clusters):
            indices = np.where(cluster_labels == cluster)[0]
            if len(indices) == 0:
                continue

            cluster_features = features_np[indices]
            centroid = kmeans.cluster_centers_[cluster]
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            # best_index = indices[np.argmin(distances)]
            
            # # min_dist, max_dist = np.min(distances), np.max(distances)
            # # normalized_distances = (distances - min_dist) / (max_dist - min_dist + 1e-8)  # Normalize to [0, 1]     
            # score = distances[np.argmin(distances)]  # 0 is highest score 
            best_index = indices[np.argmax(distances)]
            score = distances[np.argmax(distances)]

            sample = {
                'pixel_values': pixel_values[best_index].unsqueeze(0).cpu().half(),
                'input_ids': batch['input_ids'][best_index].unsqueeze(0).cpu(),
                # 'decoder_input_ids': batch['decoder_input_ids'][best_index].unsqueeze(0).cpu(),
                'labels': batch['labels'][best_index].unsqueeze(0).cpu(),
                'attention_mask': batch['attention_mask'][best_index].unsqueeze(0).cpu(),
                'image_ids':batch['image_ids'][best_index],
                # 'return_loss': True
                }

            self.memory_buffer.push(sample, cluster_id=self.current_cluster, score=score)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = {k: v for k, v in inputs.items() if k not in ['captions','image_ids']}
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
    

    def collate_samples(self, sample_list):
        batch = {}

        device = self.device
        # Iterate over keys (assumes all samples have the same keys)
        for key in sample_list[0].keys():
            # If the first sample is a tensor, assume all samples for this key are tensors
            if isinstance(sample_list[0][key], torch.Tensor):
                try:
                    # Try concatenating directly along dimension 0
                    concatenated = torch.cat([sample[key] for sample in sample_list], dim=0)
                except RuntimeError:
                    # If the sizes don't match, squeeze the extra dimension and pad the sequences
                    tensors = [sample[key].squeeze(0) for sample in sample_list]
                    concatenated = pad_sequence(tensors, batch_first=True)
                    # Optionally, if you need the extra batch dimension to be present (it will be after stacking)
                    # no further unsqueeze is required because pad_sequence already returns shape [batch, ...]
                # Move the concatenated tensor to the desired device
                batch[key] = concatenated.to(device)
            else:
                # For non-tensor values (e.g., strings), just collect them into a list
                batch[key] = [sample[key] for sample in sample_list]
                
        return batch
