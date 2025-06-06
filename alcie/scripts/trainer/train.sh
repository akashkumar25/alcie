#!/bin/bash

set -e  # Exit immediately if a command fails

# ========================
# CONFIGURABLE PARAMETERS
# ========================

BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR:-"/home/akkumar/ALCIE/fine_tuned_models/AL"}
TRAIN_ARGS_TEMPLATE="alcie/scripts/trainer/train_ofa_template.json"
MEMORY_DIR="/home/akkumar/ALCIE/memory"
INITIAL_BASE_MODEL="/home/akkumar/ALCIE/OFA-large-caption"
TRAIN_SCRIPT="/home/akkumar/ALCIE/alcie/scripts/trainer/train.py"

START_CLUSTER=1  # Set the starting cluster number
END_CLUSTER=6    # Set the last cluster to train
total_capacity=10000  # Initial total capacity for the first cluster

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

# Function to update the train_ofa.json file
update_train_ofa_json() {
  local cluster_name="$1"
  local model_checkpoint="$2"

  local output_dir="$BASE_OUTPUT_DIR/$SAMPLING_STRATEGY/OFA_trained_model_memory/$cluster_name"
  local train_caption_file="/home/akkumar/ALCIE/cluster_facade_training_combined/$cluster_name/train_caption.jsonl"
  local train_image_file="/home/akkumar/ALCIE/cluster_facade_training_combined/$cluster_name/train_image.tsv"

  cp "$TRAIN_ARGS_TEMPLATE" alcie/scripts/trainer/train_ofa.json

  jq --arg output_dir "$output_dir" \
     --arg model_name_or_path "$model_checkpoint" \
     --arg train_caption_file "$train_caption_file" \
     --arg train_image_file "$train_image_file" \
     '.output_dir = $output_dir | 
      .model_name_or_path = $model_name_or_path | 
      .train_caption_file = $train_caption_file | 
      .train_image_file = $train_image_file' \
     alcie/scripts/trainer/train_ofa.json > alcie/scripts/trainer/train_ofa_temp.json

  mv alcie/scripts/trainer/train_ofa_temp.json alcie/scripts/trainer/train_ofa.json
}

# Function to calculate delete percentage for the current cluster
calculate_delete_percent() {
  local cluster_num="$1"
  python - <<END
cluster_num = int(${cluster_num})
delete_percent_per_cluster = 1.0 / cluster_num
print(delete_percent_per_cluster)
END
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

  echo "Starting training for cluster $cluster_name (cluster_num=$cluster_num)..."

  update_train_ofa_json "$cluster_name" "$model_checkpoint"

  if [ "$cluster_num" -gt 1 ]; then
    # Calculate delete percentage and update total capacity
    delete_percent_per_cluster=$(calculate_delete_percent "$cluster_num")
    total_capacity=$(get_memory_buffer_size)
    echo "Updated memory buffer size for cluster $cluster_name: $total_capacity"
  else
    delete_percent_per_cluster=0  # No delete percentage for the first cluster
  fi

  # Construct the training command
  command="python $TRAIN_SCRIPT --train_args_file alcie/scripts/trainer/train_ofa.json \
            --memory_file $MEMORY_FILE \
            --current_cluster $cluster_num \
            --total_capacity $total_capacity \
            --delete_percent $delete_percent_per_cluster \
            --training_mode $SAMPLING_STRATEGY"

  # Conditionally add --use_memory_replay
  if [ "$use_memory_replay" = true ]; then
    command="$command --use_memory_replay"
  fi

  # Execute the command
  echo "Executing command: $command"
  eval $command || exit 1

  echo "Completed training for cluster $cluster_name."

  echo "Waiting for 2 minutes before saving the checkpoint-best"
  sleep 120 

  # Update the model checkpoint for the next cluster
  model_checkpoint="$BASE_OUTPUT_DIR/$SAMPLING_STRATEGY/OFA_trained_model_memory/$cluster_name/checkpoint-best"
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
    model_checkpoint="$INITIAL_BASE_MODEL"  # Use initial model for the first cluster
    use_memory_replay=false
  else
    prev_cluster=$((cluster_num - 1))
    model_checkpoint="$BASE_OUTPUT_DIR/$SAMPLING_STRATEGY/OFA_trained_model_memory/${CLUSTER_DIRS[$prev_cluster]}/checkpoint-best"
    use_memory_replay=true
  fi

  run_cluster "$cluster_name" "$model_checkpoint" "$cluster_num" "$use_memory_replay"

  if [ "$cluster_num" -lt "$END_CLUSTER" ]; then
    echo "Waiting for 2 minutes before starting the next cluster..."
    sleep 120 
  fi
done

echo "Training for all clusters completed."

#!/bin/bash

set -e  # Exit immediately if a command fails

# ========================
# CONFIGURABLE PARAMETERS
# ========================

BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR:-"/home/akkumar/ALCIE/fine_tuned_models/AL"}
TRAIN_ARGS_TEMPLATE="alcie/scripts/trainer/train_ofa_template.json"
MEMORY_DIR="/home/akkumar/ALCIE/memory"
INITIAL_BASE_MODEL="/home/akkumar/ALCIE/OFA-large-caption"
TRAIN_SCRIPT="/home/akkumar/ALCIE/alcie/scripts/trainer/train.py"

START_CLUSTER=1  # Set the starting cluster number
END_CLUSTER=6    # Set the last cluster to train
INITIAL_TOTAL_CAPACITY=10000  # Initial total capacity for the first cluster

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

# Function to update the train_ofa.json file
update_train_ofa_json() {
  local cluster_name="$1"
  local model_checkpoint="$2"

  local output_dir="$BASE_OUTPUT_DIR/$SAMPLING_STRATEGY/OFA_trained_model_memory/$cluster_name"
  local train_caption_file="/home/akkumar/ALCIE/cluster_facade_training_combined/$cluster_name/train_caption.jsonl"
  local train_image_file="/home/akkumar/ALCIE/cluster_facade_training_combined/$cluster_name/train_image.tsv"

  cp "$TRAIN_ARGS_TEMPLATE" alcie/scripts/trainer/train_ofa.json

  jq --arg output_dir "$output_dir" \
     --arg model_name_or_path "$model_checkpoint" \
     --arg train_caption_file "$train_caption_file" \
     --arg train_image_file "$train_image_file" \
     '.output_dir = $output_dir | 
      .model_name_or_path = $model_name_or_path | 
      .train_caption_file = $train_caption_file | 
      .train_image_file = $train_image_file' \
     alcie/scripts/trainer/train_ofa.json > alcie/scripts/trainer/train_ofa_temp.json

  mv alcie/scripts/trainer/train_ofa_temp.json alcie/scripts/trainer/train_ofa.json
}

# Function to calculate delete percentage for the current cluster
calculate_delete_percent() {
  local cluster_num="$1"

  # No deletion for random_no_delete mode
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

  echo "Starting training for cluster $cluster_name (cluster_num=$cluster_num)..."

  update_train_ofa_json "$cluster_name" "$model_checkpoint"

  # Set total_capacity and delete_percent based on the strategy
  if [ "$SAMPLING_STRATEGY" = "random_no_delete" ]; then
    total_capacity=$INITIAL_TOTAL_CAPACITY  # Fixed capacity
    delete_percent_per_cluster=0  # No deletion
  else
    if [ "$cluster_num" -gt 1 ]; then
      delete_percent_per_cluster=$(calculate_delete_percent "$cluster_num")
      total_capacity=$(get_memory_buffer_size)
      echo "Updated memory buffer size for cluster $cluster_name: $total_capacity"
    else
      delete_percent_per_cluster=0  # No deletion for the first cluster
      total_capacity=$INITIAL_TOTAL_CAPACITY
    fi
  fi

  # Construct the training command
  command="python $TRAIN_SCRIPT --train_args_file alcie/scripts/trainer/train_ofa.json \
            --memory_file $MEMORY_FILE \
            --current_cluster $cluster_num \
            --total_capacity $total_capacity \
            --delete_percent $delete_percent_per_cluster \
            --training_mode $SAMPLING_STRATEGY"

  # Conditionally add --use_memory_replay
  if [ "$use_memory_replay" = true ]; then
    command="$command --use_memory_replay"
  fi

  # Execute the command
  echo "Executing command: $command"
  eval $command || exit 1

  echo "Completed training for cluster $cluster_name."

  echo "Waiting for a minute before saving the checkpoint-best"
  sleep 60 

  # Update the model checkpoint for the next cluster
  model_checkpoint="$BASE_OUTPUT_DIR/$SAMPLING_STRATEGY/OFA_trained_model_memory/$cluster_name/checkpoint-best"
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
    model_checkpoint="$INITIAL_BASE_MODEL"  # Use initial model for the first cluster
    use_memory_replay=false
  else
    prev_cluster=$((cluster_num - 1))
    model_checkpoint="$BASE_OUTPUT_DIR/$SAMPLING_STRATEGY/OFA_trained_model_memory/${CLUSTER_DIRS[$prev_cluster]}/checkpoint-best"
    use_memory_replay=true
  fi

  run_cluster "$cluster_name" "$model_checkpoint" "$cluster_num" "$use_memory_replay"

  if [ "$cluster_num" -lt "$END_CLUSTER" ]; then
    echo "Waiting for 30 seconds before starting the next cluster..."
    sleep 30
  fi
done

echo "Training for all clusters completed."
