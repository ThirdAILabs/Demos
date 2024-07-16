#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <path_to_env_file> [username] [machine_ips]"
  exit 1
fi

ENV_FILE_PATH=$1
USERNAME=${2:-}
MACHINE_IPS=${3:-}
LOG_FOLDER_PATH=$(pwd)/logs

echo "Using .env file: $ENV_FILE_PATH"
if [ -z "$MACHINE_IPS" ]; then
  echo "Running locally"
else
  echo "Username: $USERNAME"
  echo "Machine IPs: $MACHINE_IPS"
fi
echo "Log folder path: $LOG_FOLDER_PATH"

# Read machine IPs into an array if provided
if [ -n "$MACHINE_IPS" ]; then
  IFS=',' read -r -a MACHINE_IP_ARRAY <<< "$MACHINE_IPS"
  NUM_MACHINES=${#MACHINE_IP_ARRAY[@]}
  echo "Number of machines: $NUM_MACHINES"
else
  NUM_MACHINES=1
fi

# Ensure the local log directory exists (for later reference if needed)
if [ ! -d "$LOG_FOLDER_PATH" ]; then
  mkdir -p "$LOG_FOLDER_PATH"
fi

# Create a temporary .env file
TEMP_ENV_FILE=$(mktemp)

# Copy the original .env file to the temporary file, removing comments, trimming whitespace, and removing quotes
sed 's/[[:space:]]*#.*//; /^[[:space:]]*$/d; s/[[:space:]]*=[[:space:]]*"/=/g; s/"$//g' "$ENV_FILE_PATH" > "$TEMP_ENV_FILE"

# Function to perform the sed replacement depending on the OS
perform_sed_replacement() {
  local file=$1
  local pattern=$2
  local replacement=$3

  if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s|$pattern|$replacement|" "$file"
  else
    # Linux
    sed -i "s|$pattern|$replacement|" "$file"
  fi
}

# Check if FOLDER_PATH is specified in the .env file
FOLDER_PATH=$(grep -E '^FOLDER_PATH=' "$TEMP_ENV_FILE" | cut -d '=' -f 2)

# Check if PREDICT_LOG_FOLDER and EXTRACT_LOG_FILE_PATH are specified in the .env file
PREDICT_LOG_FOLDER=$(grep -E '^PREDICT_LOG_FOLDER=' "$TEMP_ENV_FILE" | cut -d '=' -f 2)
EXTRACT_LOG_FILE_PATH=$(grep -E '^EXTRACT_LOG_FILE_PATH=' "$TEMP_ENV_FILE" | cut -d '=' -f 2)

# If FOLDER_PATH is specified, update it to the Docker path while keeping the original name
if [ -n "$FOLDER_PATH" ]; then
  FOLDER_NAME=$(basename "$FOLDER_PATH")
  perform_sed_replacement "$TEMP_ENV_FILE" '^FOLDER_PATH=.*' "FOLDER_PATH=/app/data"
fi

# Update the log folder paths to the Docker path while keeping the original names
if [ -n "$PREDICT_LOG_FOLDER" ]; then
  PREDICT_LOG_FOLDER_NAME=$(basename "$PREDICT_LOG_FOLDER")
  perform_sed_replacement "$TEMP_ENV_FILE" '^PREDICT_LOG_FOLDER=.*' "PREDICT_LOG_FOLDER=/app/logs/$PREDICT_LOG_FOLDER_NAME"
else
  printf "\nPREDICT_LOG_FOLDER=/app/logs" >> "$TEMP_ENV_FILE"
fi

if [ -n "$EXTRACT_LOG_FILE_PATH" ]; then
  EXTRACT_LOG_FILE_NAME=$(basename "$EXTRACT_LOG_FILE_PATH")
  perform_sed_replacement "$TEMP_ENV_FILE" '^EXTRACT_LOG_FILE_PATH=.*' "EXTRACT_LOG_FILE_PATH=/app/logs/$EXTRACT_LOG_FILE_NAME"
else
  printf "\nEXTRACT_LOG_FILE_PATH=/app/logs/tika_log_test.txt" >> "$TEMP_ENV_FILE"
fi

# Function to gather NUMA nodes count from all machines
gather_numa_nodes() {
  local TOTAL_NODES=0

  if [ -z "$MACHINE_IPS" ]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
      TOTAL_NODES=1
    else
      TOTAL_NODES=$(numactl --hardware | grep "available:" | awk '{print $2}')
    fi
  else
    for MACHINE_IP in "${MACHINE_IP_ARRAY[@]}"; do
      OS_TYPE=$(ssh -o StrictHostKeyChecking=no $USERNAME@$MACHINE_IP "uname")
      if [[ "$OS_TYPE" == "Darwin" ]]; then
        NODES=1
      else
        NODES=$(ssh -o StrictHostKeyChecking=no $USERNAME@$MACHINE_IP "numactl --hardware | grep 'available:' | awk '{print \$2}'")
      fi
      TOTAL_NODES=$((TOTAL_NODES + NODES))
    done
  fi

  echo $TOTAL_NODES
}

# Function to run Docker locally
run_local_docker() {
  echo "Pulling Docker image..."
  docker pull yashuroyal/ner-pipe:512_latest
  
  echo "FOLDER_PATH: $FOLDER_PATH"
  echo "TEMP_ENV_FILE: $TEMP_ENV_FILE"
  echo "LOG_FOLDER_PATH: $LOG_FOLDER_PATH"

  if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Running on macOS, treating as a single machine."

    docker run --rm \
      -v "$TEMP_ENV_FILE:/app/.env" \
      -e ENV_FILE=/app/.env \
      -e TOTAL_MACHINES=1 \
      -e MACHINE_INDEX=0 \
      -e CPU_COUNT=$(sysctl -n hw.ncpu) \
      -v "$LOG_FOLDER_PATH:/app/logs" \
      $( [ -n "$FOLDER_PATH" ] && echo "-v $FOLDER_PATH:/app/data" ) \
      yashuroyal/ner-pipe:512_latest
  else
    # Find number of NUMA nodes
    NUMA_NODES=$(numactl --hardware | grep "available:" | awk '{print $2}')
    echo "NUMA Nodes: $NUMA_NODES"
    TOTAL_MACHINES=$NUMA_NODES

    for ((i=0; i<NUMA_NODES; i++)); do
      # Get the list of CPUs on this NUMA node
      CPU_LIST=$(numactl --hardware | grep "node $i cpus:" | awk -F': ' '{print $2}' | tr ' ' ',')
      CPU_COUNT=$(echo $CPU_LIST | tr ',' '\n' | wc -l)

      echo "Running Docker on NUMA node $i with $CPU_COUNT CPUs"

      SESSION_NAME="local_docker_session_$i"

      if tmux has-session -t $SESSION_NAME 2>/dev/null; then
        tmux kill-session -t $SESSION_NAME
      fi
      tmux new-session -d -s $SESSION_NAME
      tmux send-keys -t $SESSION_NAME "numactl --cpunodebind=$i --membind=$i docker run --rm \
        -v $TEMP_ENV_FILE:/app/.env \
        -e ENV_FILE=/app/.env \
        -e TOTAL_MACHINES=$TOTAL_MACHINES \
        -e MACHINE_INDEX=$i \
        -e CPU_COUNT=$CPU_COUNT \
        --cpuset-cpus $CPU_LIST \
        -v $LOG_FOLDER_PATH:/app/logs \
        $( [ -n "$FOLDER_PATH" ] && echo "-v $FOLDER_PATH:/app/data" ) \
        yashuroyal/ner-pipe:512_latest 2>&1 | tee $LOG_FOLDER_PATH/docker_run_$i.log; tail -f /dev/null" C-m
    done

    wait
  fi
}

# Function to add user to docker group and ensure the session remains alive on remote machine
setup_remote_machine() {
  local MACHINE_IP=$1
  local SESSION_NAME_PREFIX=$2
  local START_INDEX=$3
  local REMOTE_LOG_FOLDER="/tmp/docker_logs"

  echo "Machine id for $MACHINE_IP : $START_INDEX"

  ssh -o StrictHostKeyChecking=no $USERNAME@$MACHINE_IP << EOF
    sudo usermod -aG docker $USERNAME
    mkdir -p /tmp/docker_env
    mkdir -p $REMOTE_LOG_FOLDER
    echo '$(cat "$TEMP_ENV_FILE")' > /tmp/docker_env/.env
    echo "Pulling Docker image..."
    docker pull yashuroyal/ner-pipe:512_latest

    NUMA_NODES=\$(numactl --hardware | grep "available:" | awk '{print \$2}')
    echo "NUMA Nodes: \$NUMA_NODES"

    for ((i=0; i<\$NUMA_NODES; i++)); do
      # Adjust CPU count for each NUMA node in Linux
      CPU_LIST=\$(numactl --hardware | grep "node \$i cpus:" | awk -F': ' '{print \$2}')
      CPU_COUNT=\$(echo \$CPU_LIST | wc -w)

      echo "Running Docker on NUMA node \$i with \$CPU_COUNT CPUs and \$CPU_LIST"

      SESSION_NAME="${SESSION_NAME_PREFIX}_\$i"

      if tmux has-session -t \$SESSION_NAME 2>/dev/null; then
        tmux kill-session -t \$SESSION_NAME
      fi
      tmux new-session -d -s \$SESSION_NAME
      tmux send-keys -t \$SESSION_NAME "numactl --cpunodebind=\$i --membind=\$i docker run --rm \\
        --env-file /tmp/docker_env/.env \\
        -e ENV_FILE=/app/.env \\
        -e TOTAL_MACHINES=$TOTAL_MACHINES \\
        -e MACHINE_INDEX=\$(($START_INDEX + i)) \\
        -e CPU_COUNT=\$CPU_COUNT \\
        --cpuset-cpus \$(echo \$CPU_LIST | tr ' ' ',') \\
        -v $REMOTE_LOG_FOLDER:/app/logs \\
        $( [ -n "$FOLDER_PATH" ] && echo "-v $FOLDER_PATH:/app/data" ) \\
        yashuroyal/ner-pipe:512_latest 2>&1 | tee $REMOTE_LOG_FOLDER/docker_run_\$i.log; tail -f /dev/null" C-m
    done
EOF
}

if [ -n "$MACHINE_IPS" ]; then
  if [ -z "$USERNAME" ]; then
    echo "Username is required for remote execution."
    exit 1
  fi

  # Gather total NUMA nodes across all machines
  TOTAL_MACHINES=$(gather_numa_nodes)
  echo "Total NUMA nodes across all machines: $TOTAL_MACHINES"

  # Loop through each machine IP to set up the remote environment and run Docker containers
  MACHINE_INDEX=0
  for MACHINE_IP in "${MACHINE_IP_ARRAY[@]}"; do
    SESSION_NAME="docker_session_$MACHINE_INDEX"
    setup_remote_machine "$MACHINE_IP" "$SESSION_NAME" "$MACHINE_INDEX" &
    OS_TYPE=$(ssh -o StrictHostKeyChecking=no $USERNAME@$MACHINE_IP "uname")
    NUMA_NODES=$(ssh -o StrictHostKeyChecking=no $USERNAME@$MACHINE_IP "numactl --hardware | grep 'available:' | awk '{print \$2}'")
    MACHINE_INDEX=$((MACHINE_INDEX + NUMA_NODES))
  done

  # Wait for all background jobs to finish
  wait
else
  # Run Docker locally
  run_local_docker
fi
