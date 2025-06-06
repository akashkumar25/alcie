#!/bin/bash

set -e  # Exit immediately if a command fails

# ========================
# CONFIGURABLE PARAMETERS
# ========================

BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR:-"/home/akkumar/ALCIE/fine_tuned_models/blip2"}
TRAIN_ARGS_TEMPLATE="alcie/scripts/trainer/train_blip2_template.json"
MEMORY_DIR="/home/akkumar/ALCIE/memory/blip2"

# Path to your initial BLIP-2 checkpoint (could be a HF model name or local folder)
INITIAL_BASE_MODEL="Salesforce/blip2-opt-2.7b"
# Or something like "/home/akkumar/ALCIE/BLIP2-checkpoint" if you have a local dir

TRAIN_SCRIPT="/home/akkumar/ALCIE/alcie/scripts/trainer/train_blip2.py"

# Clusters
START_CLUSTER=1
END_CLUSTER=6
INITIAL_TOTAL_CAPACITY=10000

SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-"uncertainty"}
MEMORY_FILE="$MEMORY_DIR/${SAMPLING_STRATEGY}_sampling/memory_buffer.pkl"

echo "Using Sampling Strategy: $SAMPLING_STRATEGY"
echo "Using Base Output Directory: $BASE_OUTPUT_DIR"

# Cluster directories (adjust names as needed)
declare -A CLUSTER_DIRS
CLUSTER_DIRS[1]="accessories"
CLUSTER_DIRS[2]="bottoms"
CLUSTER_DIRS[3]="dresses"
CLUSTER_DIRS[4]="outerwear"
CLUSTER_DIRS[5]="shoes"
CLUSTER_DIRS[6]="tops"

# ========================
# FUNCTION DEFINITIONS
# ========================

# Function to update the train_blip2.json file
update_train_blip2_json() {
  local cluster_name="$1"
  local model_checkpoint="$2"

  local output_dir="$BASE_OUTPUT_DIR/$SAMPLING_STRATEGY/BLIP2_trained_model_memory/$cluster_name"
  local train_caption_file="/home/akkumar/ALCIE/cluster_facade_training_combined/$cluster_name/train_caption.jsonl"
  local train_image_file="/home/akkumar/ALCIE/cluster_facade_training_combined/$cluster_name/train_image.tsv"

  # Copy the BLIP-2 template to a working file
  cp "$TRAIN_ARGS_TEMPLATE" alcie/scripts/trainer/train_blip2.json

  # Use jq to update the fields
  jq --arg output_dir "$output_dir" \
     --arg model_name_or_path "$model_checkpoint" \
     --arg train_caption_file "$train_caption_file" \
     --arg train_image_file "$train_image_file" \
     '.output_dir = $output_dir |
      .model_name_or_path = $model_name_or_path |
      .train_caption_file = $train_caption_file |
      .train_image_file = $train_image_file' \
     alcie/scripts/trainer/train_blip2.json > alcie/scripts/trainer/train_blip2_temp.json

  mv alcie/scripts/trainer/train_blip2_temp.json alcie/scripts/trainer/train_blip2.json
}

# Function to calculate delete percentage for the current cluster
calculate_delete_percent() {
  local cluster_num="$1"

  # No deletion if random_no_delete
  if [ "$SAMPLING_STRATEGY" = "random_no_delete" ]; then
    echo "0"
  else
    python - <<END
cluster_num = int(${cluster_num})
delete_percent_per_cluster = 1.0 / cluster_num
print(delete_percent_per_cluster)
END
  fi
}

# Function to get the current memory buffer size
get_memory_buffer_size() {
  python - <<END
import pickle
memory_file = "$MEMORY_FILE"
try:
    with open(memory_file, 'rb') as f:
        buffer = pickle.load(f)
        print(len(buffer))
except FileNotFoundError:
    print(0)
END
}

# Function to run the training process for a specific cluster
run_cluster() {
  local cluster_name="$1"
  local model_checkpoint="$2"
  local cluster_num="$3"
  local use_memory_replay="$4"

  echo "Starting training for cluster $cluster_name (cluster_num=$cluster_num) loading checkpoint $model_checkpoint"

  update_train_blip2_json "$cluster_name" "$model_checkpoint"

  if [ "$SAMPLING_STRATEGY" = "random_no_delete" ]; then
    total_capacity=$INITIAL_TOTAL_CAPACITY
    delete_percent_per_cluster=0
  else
    if [ "$cluster_num" -gt 1 ]; then
      delete_percent_per_cluster=$(calculate_delete_percent "$cluster_num")
      total_capacity=$(get_memory_buffer_size)
      echo "Updated memory buffer size for cluster $cluster_name: $total_capacity"
    else
      delete_percent_per_cluster=0
      total_capacity=$INITIAL_TOTAL_CAPACITY
    fi
  fi

  # Construct the training command
  command="python $TRAIN_SCRIPT --train_args_file alcie/scripts/trainer/train_blip2.json \
            --memory_file $MEMORY_FILE \
            --current_cluster $cluster_num \
            --total_capacity $total_capacity \
            --delete_percent $delete_percent_per_cluster \
            --training_mode $SAMPLING_STRATEGY"

  # Conditionally add --use_memory_replay
  if [ "$use_memory_replay" = true ]; then
    command="$command --use_memory_replay"
  fi

  echo "Executing command: $command"
  eval $command || exit 1

  echo "Completed training for cluster $cluster_name."

  echo "Waiting for a minute before saving the checkpoint-best"
  sleep 60

  # Update the model checkpoint for next cluster
  model_checkpoint="$BASE_OUTPUT_DIR/$SAMPLING_STRATEGY/BLIP2_trained_model_memory/$cluster_name/checkpoint-best"
  echo "Model checkpoint updated to: $model_checkpoint"
}

# ========================
# MAIN TRAINING LOOP
# ========================

for cluster_num in $(seq $START_CLUSTER $END_CLUSTER); do
  cluster_name="${CLUSTER_DIRS[$cluster_num]}"
  if [ -z "$cluster_name" ]; then
    echo "Cluster $cluster_num not found in CLUSTER_DIRS."
    exit 1
  fi

  if [ "$cluster_num" -eq 1 ]; then
    model_checkpoint="$INITIAL_BASE_MODEL"
    use_memory_replay=false
  else
    prev_cluster=$((cluster_num - 1))
    model_checkpoint="$BASE_OUTPUT_DIR/$SAMPLING_STRATEGY/BLIP2_trained_model_memory/${CLUSTER_DIRS[$prev_cluster]}/checkpoint-best"
    use_memory_replay=true
  fi

  run_cluster "$cluster_name" "$model_checkpoint" "$cluster_num" "$use_memory_replay"

  if [ "$cluster_num" -lt "$END_CLUSTER" ]; then
    echo "Waiting for 30 seconds before starting the next cluster..."
    sleep 30
  fi
done

echo "Training for all clusters completed."
