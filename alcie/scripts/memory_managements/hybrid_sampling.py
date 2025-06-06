import torch
import clip
from transformers import Trainer
from sklearn.cluster import KMeans
import numpy as np
import pickle
import random
import os
from torch.nn.utils.rnn import pad_sequence

# ----------------------------------------------------------------------------
# Hybrid Memory Buffer:
# ----------------------------------------------------------------------------
class MemoryBufferHybrid:
    def __init__(self, capacity=550, sampling_strategy="hybrid"):
        self.buffer = {}
        self.capacity = capacity
        self.sampling_strategy = sampling_strategy
        self.log_file = f"logs/memory_tracking_{sampling_strategy}.log"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, 'a') as f:
            f.write(f"\nMemory Buffer Tracking Log - {sampling_strategy.capitalize()} Sampling\n")
            f.write("===========================\n\n")

    def set_total_capacity(self, capacity):
        self.capacity = capacity
        print(f"Total memory buffer capacity set to {self.capacity}.")

    def push(self, sample, cluster_id):
        # Mark the sample with its cluster and the hybrid score.
        sample['cluster_id'] = cluster_id
        hybrid_score=sample['hybrid_score']
        
        image_id = sample['image_ids']
        # For our hybrid strategy, a higher hybrid score is better.
        # We compare against the current lowest hybrid score in the same cluster.
        cluster_samples = [s for s in self.buffer.values() if s.get('cluster_id') == cluster_id]
        if cluster_samples:
            lowest_score = min(cluster_samples, key=lambda x: x['hybrid_score'])['hybrid_score']
        else:
            lowest_score = -1  # If no sample exists, accept the new one.

        if image_id in self.buffer:
            print("Duplicate sample detected; not adding to memory buffer.")
        else:
            if len(self.buffer) >= self.capacity:
                if hybrid_score > lowest_score:
                    print(f"Buffer full: New hybrid score {hybrid_score:.4f} > lowest score {lowest_score:.4f}; freeing space.")
                    self.free_space(cluster_id, num_to_delete=1)
                    self._add_sample(image_id, sample, hybrid_score, cluster_id)
                else:
                    print(f"Buffer full; sample hybrid score {hybrid_score:.4f} not higher than lowest {lowest_score:.4f} in cluster {cluster_id}.")
            else:
                self._add_sample(image_id, sample, hybrid_score, cluster_id)

    def _add_sample(self, image_id, sample, hybrid_score, cluster_id):
        self.buffer[image_id] = sample
        print(f"Added sample with hybrid score {hybrid_score:.4f} to cluster {cluster_id}. Buffer size: {len(self.buffer)}.")
        with open(self.log_file, 'a') as f:
            f.write(f"Added sample with hybrid score {hybrid_score:.4f} to cluster {cluster_id}. Buffer size: {len(self.buffer)}.\n")

    def free_space(self, cluster_id, num_to_delete=1):
        # Delete samples with the lowest hybrid scores in the given cluster.
        cluster_items = [(key, sample) for key, sample in self.buffer.items() if sample.get('cluster_id') == cluster_id]
        if not cluster_items:
            print(f"No samples in cluster {cluster_id} to delete.")
            return
        # Sorting in ascending order so that samples with the lowest hybrid score come first.
        sorted_samples = sorted(cluster_items, key=lambda x: x[1]['hybrid_score'])
        for i in range(min(num_to_delete, len(sorted_samples))):
            key_to_delete = sorted_samples[i][0]
            del self.buffer[key_to_delete]
        print(f"Deleted {min(num_to_delete, len(sorted_samples))} sample(s) from cluster {cluster_id}.")
        
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
        sorted_entries = sorted(cluster_entries, key=lambda key: self.buffer[key]['hybrid_score'])
        
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
        clusters = {}
        for key, sample in self.buffer.items():
            cluster_id = sample['cluster_id']
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
        print(f"Memory buffer saved to {file_path}.")
        with open(self.log_file, 'a') as f:
            f.write(f"Memory buffer saved to {file_path}.\n")

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.buffer = pickle.load(f)
        print(f"Memory buffer loaded with {len(self.buffer)} items.")
        with open(self.log_file, 'a') as f:
            f.write(f"Memory buffer loaded with {len(self.buffer)} items from {file_path}.\n")

# ----------------------------------------------------------------------------
# Hybrid Memory Replay Trainer:
# ----------------------------------------------------------------------------
class HybridMemoryReplayTrainer(Trainer):
    def __init__(self, *args, memory_buffer=None, replay_freq=200, use_memory_replay=False,
                 current_cluster=1, clip_model=None, alpha=0.5, preprocess=None, device="cuda", **kwargs):
        """
        :param alpha: Weight for uncertainty score (between 0 and 1); (1 - alpha) is applied to diversity.
        """
        super().__init__(*args, **kwargs)
        self.memory_buffer = memory_buffer
        self.replay_freq = replay_freq
        self.use_memory_replay = use_memory_replay
        self.current_cluster = current_cluster
        self.clip_model = clip_model  # Used to compute diversity via image-text embeddings.
        self.preprocess = preprocess
        self.alpha = alpha            # Trade-off weight between uncertainty and diversity.
        self.device = device
        self.total_samples_seen = 0   # Counter for samples processed, to schedule memory replay.

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Remove keys not used for computing the loss.
        inputs_for_model = {k: v for k, v in inputs.items() if k not in ['captions', 'image_ids']}
        loss = super().training_step(model, inputs_for_model)
        self.total_samples_seen += len(inputs_for_model['input_ids'])

        # Calculate hybrid scores and select samples from the current batch.
        if self.memory_buffer is not None:
            self.select_hybrid_samples(inputs, k=16)
        
        # Memory replay: every replay_freq steps, sample from previous clusters.
        if self.use_memory_replay and len(self.memory_buffer) > 0:
            if self.total_samples_seen >= self.replay_freq:
                self.total_samples_seen = 0
                memory_samples = self.memory_buffer.sample_by_cluster(num_samples_per_cluster=1, exclude_cluster=self.current_cluster)
                if memory_samples:
                    memory_batch = self.collate_samples(memory_samples)
                    # Remove extra keys before training.
                    clean_memory_batch = {k: v for k, v in memory_batch.items() if k not in ['cluster_id', 'hybrid_score', 'image_ids']}
                    print(f"Replaying memory batch. Sample size: {len(clean_memory_batch.get('input_ids', []))}.")
                    memory_loss = super().training_step(model, clean_memory_batch)
                    loss += memory_loss
                    print("Replayed memory samples.")
                else:
                    print("No memory samples available for replay.")

        return loss

    # def select_hybrid_samples(self, batch, k=16):
    #     """
    #     Select samples from the current batch using a hybrid score that fuses uncertainty and diversity.
    #     """
    #     # ------------------------
    #     # Compute Diversity Scores via CLIP:
    #     # ------------------------
    #     pixel_values = batch['pixel_values'].to(self.device)
    #     captions = batch['captions']  # List of caption strings.
    #     batch_size = pixel_values.size(0)
    #     # Tokenize captions for CLIP
    #     tokenized_captions = clip.tokenize(captions, context_length=77, truncate=True).to(self.device)
    #     with torch.no_grad():
    #         image_features = self.clip_model.encode_image(pixel_values)
    #         text_features = self.clip_model.encode_text(tokenized_captions)
    #         fused_features = (image_features + text_features) / 2  # Fusing modalities.
    #     features_np = fused_features.cpu().numpy()
    #     num_clusters = min(k, batch_size)
    #     kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    #     cluster_labels = kmeans.fit_predict(features_np)

    #     # ------------------------
    #     # Compute Uncertainty Scores using the model:
    #     # ------------------------
    #     input_ids = batch['input_ids'].to(self.device)
    #     attention_mask = batch['attention_mask'].to(self.device)
    #     labels = batch['labels'].to(self.device)
    #     # Set model to evaluation mode for uncertainty estimation.
    #     self.model.eval()
    #     with torch.no_grad():
    #         outputs = self.model(
    #             pixel_values=pixel_values,
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             labels=labels
    #         )
    #     logits = outputs.logits  # Shape: [batch_size, seq_length, vocab_size]
    #     probabilities = torch.softmax(logits, dim=-1)
    #     # Compute token-level uncertainty (1 - max probability)
    #     token_uncertainty = 1 - torch.max(probabilities, dim=-1).values  # Shape: [batch_size, seq_length]
    #     # Aggregate uncertainty over the sequence (mean uncertainty per sample)
    #     sequence_uncertainty = token_uncertainty.mean(dim=1)  # Shape: [batch_size]
    #     uncertainty_scores = sequence_uncertainty.cpu().numpy()  # Higher score means more uncertain.

    #     # ------------------------
    #     # Compute Diversity Scores (distance from cluster centroid)
    #     # ------------------------
    #     diversity_scores = np.zeros(batch_size)
    #     for i in range(batch_size):
    #         cluster = cluster_labels[i]
    #         centroid = kmeans.cluster_centers_[cluster]
    #         diversity_scores[i] = np.linalg.norm(features_np[i] - centroid)

    #     # ------------------------
    #     # Normalize both scores to the range [0,1]:
    #     # ------------------------
    #     norm_diversity = (diversity_scores - diversity_scores.min()) / (diversity_scores.max() - diversity_scores.min() + 1e-8)
    #     norm_uncertainty = (uncertainty_scores - uncertainty_scores.min()) / (uncertainty_scores.max() - uncertainty_scores.min() + 1e-8)

    #     # ------------------------
    #     # Compute the Hybrid Score:
    #     #   hybrid_score = alpha * normalized_uncertainty + (1 - alpha) * normalized_diversity
    #     # ------------------------
    #     hybrid_scores = self.alpha * norm_uncertainty + (1 - self.alpha) * norm_diversity

    #     # ------------------------
    #     # Cluster-wise Selection of Samples:
    #     # For each cluster, select the sample with the highest hybrid score.
    #     # ------------------------
    #     hybrid_samples = []
    #     for cluster in range(num_clusters):
    #         indices = np.where(cluster_labels == cluster)[0]
    #         if len(indices) == 0:
    #             continue
    #         cluster_hybrid_scores = hybrid_scores[indices]
    #         best_idx_in_cluster = indices[np.argmax(cluster_hybrid_scores)]
    #         sample = {
    #             'pixel_values': pixel_values[best_idx_in_cluster].unsqueeze(0).cpu().half(),
    #             'input_ids': batch['input_ids'][best_idx_in_cluster].unsqueeze(0).cpu(),
    #             'labels': batch['labels'][best_idx_in_cluster].unsqueeze(0).cpu(),
    #             'attention_mask': batch['attention_mask'][best_idx_in_cluster].unsqueeze(0).cpu(),
    #             'image_ids': batch['image_ids'][best_idx_in_cluster]
    #         }
    #         sample['hybrid_score'] = hybrid_scores[best_idx_in_cluster]
    #         hybrid_samples.append(sample)

    #     # ------------------------
    #     # Push each selected sample into the hybrid memory buffer.
    #     # ------------------------
    #     for sample in hybrid_samples:
    #         self.memory_buffer.push(sample, cluster_id=self.current_cluster)
    
    def select_hybrid_samples(self, batch, k=16):
        """
        Select samples from the current batch using a hybrid score that fuses uncertainty and diversity,
        following uncertainty-first stratification like HUDS, and push exactly 'k' samples into memory.
        """
        patch_images = batch['patch_images'].to(self.device)
        # pixel_values = batch['pixel_values'].to(self.device)
        captions = batch['captions']
        batch_size = patch_images.size(0)
        tokenized_captions = clip.tokenize(captions, context_length=77, truncate=True).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(patch_images)
            text_features = self.clip_model.encode_text(tokenized_captions)
            fused_features = (image_features + text_features) / 2

        features_np = fused_features.cpu().numpy()

        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        decoder_input_ids = batch['decoder_input_ids'].to(self.device)
        # labels = batch['labels'].to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                patch_images=patch_images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                # labels=labels,
                decoder_input_ids=decoder_input_ids,
                return_loss=True
            )
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        token_uncertainty = 1 - torch.max(probabilities, dim=-1).values
        sequence_uncertainty = token_uncertainty.mean(dim=1)
        uncertainty_scores = sequence_uncertainty.cpu().numpy()

        # ----------------- STRATIFICATION by uncertainty -----------------
        n_strata = 5  # number of uncertainty bins
        strata = [[] for _ in range(n_strata)]
        u_min, u_max = uncertainty_scores.min(), uncertainty_scores.max()
        strata_edges = np.linspace(u_min, u_max, n_strata + 1)

        for idx in range(batch_size):
            for i in range(n_strata):
                if strata_edges[i] <= uncertainty_scores[idx] <= strata_edges[i + 1]:
                    strata[i].append(idx)
                    break

        hybrid_samples = []

        # ----------------- Process each stratum separately -----------------
        for stratum in strata:
            if len(stratum) == 0:
                continue

            stratum_features = features_np[stratum]
            stratum_uncertainty = uncertainty_scores[stratum]

            num_clusters = min(k, len(stratum))
            if num_clusters <= 1:
                # not enough samples to cluster, take directly
                for idx in stratum:
                    sample = {
                        # 'pixel_values': pixel_values[idx].unsqueeze(0).cpu().half(),
                        'patch_images': patch_images[idx].unsqueeze(0).cpu().half(),
                        'input_ids': batch['input_ids'][idx].unsqueeze(0).cpu(),
                        # 'labels': batch['labels'][idx].unsqueeze(0).cpu(),
                        'decoder_input_ids': batch['decoder_input_ids'][idx].unsqueeze(0).cpu(),
                        'attention_mask': batch['attention_mask'][idx].unsqueeze(0).cpu(),
                        'image_ids': batch['image_ids'][idx],
                        'return_loss': True
                    }
                    sample['hybrid_score'] = 1.0  # maximum score if only one sample
                    hybrid_samples.append(sample)
                continue

            # clustering within stratum
            kmeans = KMeans(n_clusters=num_clusters, random_state=0)
            cluster_labels = kmeans.fit_predict(stratum_features)

            # compute diversity inside stratum
            diversity_scores = np.zeros(len(stratum))
            for i in range(len(stratum)):
                cluster = cluster_labels[i]
                centroid = kmeans.cluster_centers_[cluster]
                diversity_scores[i] = np.linalg.norm(stratum_features[i] - centroid)

            # normalize uncertainty and diversity inside stratum
            norm_diversity = (diversity_scores - diversity_scores.min()) / (diversity_scores.max() - diversity_scores.min() + 1e-8)
            norm_uncertainty = (stratum_uncertainty - stratum_uncertainty.min()) / (stratum_uncertainty.max() - stratum_uncertainty.min() + 1e-8)

            hybrid_scores = self.alpha * norm_uncertainty + (1 - self.alpha) * norm_diversity

            # select highest hybrid score sample per cluster
            for cluster in range(num_clusters):
                indices = np.where(cluster_labels == cluster)[0]
                if len(indices) == 0:
                    continue
                cluster_hybrid_scores = hybrid_scores[indices]
                best_idx_in_cluster = stratum[np.argmax(cluster_hybrid_scores)]
                sample = {
                    # 'pixel_values': pixel_values[best_idx_in_cluster].unsqueeze(0).cpu().half(),
                    'patch_images': patch_images[best_idx_in_cluster].unsqueeze(0).cpu().half(),
                    'input_ids': batch['input_ids'][best_idx_in_cluster].unsqueeze(0).cpu(),
                    # 'labels': batch['labels'][best_idx_in_cluster].unsqueeze(0).cpu(),
                    'decoder_input_ids': batch['decoder_input_ids'][best_idx_in_cluster].unsqueeze(0).cpu(),
                    'attention_mask': batch['attention_mask'][best_idx_in_cluster].unsqueeze(0).cpu(),
                    'image_ids': batch['image_ids'][best_idx_in_cluster],
                    'return_loss': True
                }
                sample['hybrid_score'] = hybrid_scores[np.argmax(cluster_hybrid_scores)]
                hybrid_samples.append(sample)

        # ----------------- Limit final pushed samples to top-k (16) -----------------
        if len(hybrid_samples) > k:
            # Sort hybrid samples by hybrid_score (descending)
            hybrid_samples = sorted(hybrid_samples, key=lambda x: x['hybrid_score'], reverse=True)
            hybrid_samples = hybrid_samples[:k]  # Keep only top-k

        # Push to memory
        for sample in hybrid_samples:
            self.memory_buffer.push(sample, cluster_id=self.current_cluster)


    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = {k: v for k, v in inputs.items() if k not in ['captions', 'image_ids']}
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    def collate_samples(self, sample_list):
        batch = {}
        device = self.device
        for key in sample_list[0].keys():
            if isinstance(sample_list[0][key], torch.Tensor):
                try:
                    concatenated = torch.cat([sample[key] for sample in sample_list], dim=0)
                except RuntimeError:
                    tensors = [sample[key].squeeze(0) for sample in sample_list]
                    concatenated = pad_sequence(tensors, batch_first=True)
                batch[key] = concatenated.to(device)
            else:
                batch[key] = [sample[key] for sample in sample_list]
        return batch
