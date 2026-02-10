#! /bin/bash

# Script for running the legacypipe code within a Shifter container at NERSC
# Modified to:
# - Accept a single brick name.
# - Find an unused NVIDIA A100 GPU using a lock file mechanism.
# - Wait if all 4 GPUs are in use until one becomes available.
# - Assign the chosen GPU using CUDA_VISIBLE_DEVICES.
# - Create a unique temporary cache directory for each run.

# --- Environment Variables (from original script) ---
export COSMO=/dvs_ro/cfs/cdirs/cosmo

export LEGACY_SURVEY_DIR=$COSMO/work/legacysurvey/dr11
# Default output directory, can be overridden with -outdir
outdir=$SCRATCH/dr11-gpu

export GAIA_CAT_DIR=$COSMO/data/gaia/dr3/healpix
export GAIA_CAT_PREFIX=healpix
export GAIA_CAT_SCHEME=nested
export GAIA_CAT_VER=3

export DUST_DIR=$COSMO/data/dust/v0_1
export UNWISE_COADDS_DIR=$COSMO/data/unwise/neo7/unwise-coadds/fulldepth:$COSMO/data/unwise/allwise/unwise-coadds/fulldepth
export UNWISE_COADDS_TIMERESOLVED_DIR=$COSMO/work/wise/outputs/merge/neo7
export UNWISE_MODEL_SKY_DIR=$COSMO/data/unwise/neo7/unwise-catalog/mod

export TYCHO2_KD_DIR=$COSMO/staging/tycho2
export LARGEGALAXIES_CAT=$COSMO/staging/largegalaxies/v3.0/SGA-ellipse-v3.0.kd.fits
export SKY_TEMPLATE_DIR=$COSMO/work/legacysurvey/dr10/calib/sky_pattern

unset BLOB_MASK_DIR
unset PS1CAT_DIR
unset GALEX_DIR

# Don't add ~/.local/ to Python's sys.path
export PYTHONNOUSERSITE=1
# Force MKL single-threaded
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
# To avoid problems with MPI and Python multiprocessing
export MPICH_GNI_FORK_MODE=FULLCOPY
export KMP_AFFINITY=disabled

# --- Legacypipe Repository Paths ---
# Assuming a local checkout of the tractor and legacypipe repositories
#export TRACTOR_DIR=./tractor-git
#export LEGACYPIPE_DIR=./legacypipe-git/py
#export PYTHONPATH=$TRACTOR_DIR:$LEGACYPIPE_DIR:${PYTHONPATH}

# --- Script Variables ---
brick_name="" # Stores the single brick name
zoom_arg=""
gpumode_arg=""
use_gpu_flag=""
threads_arg=""
subblobs_flag=""
blobid_arg=""
md_suffix="cpu" # default mode suffix for log name
th_suffix=""    # default threads suffix for log name

# --- Global Variables for GPU Management and Cleanup ---
# Array to store detected A100 GPU indices (e.g., 0, 1, 2, 3)
declare -a all_gpu_indices=()
chosen_gpu_id=-1       # The ID of the GPU chosen for this run
current_lock_file=""   # Path to the lock file created for this run
current_tmpcache=""    # Path to the unique temporary cache directory for this run

# Define the shared directory for GPU lock files
# Using SCRATCH is recommended for HPC environments for persistence across jobs
# and visibility to multiple concurrent script invocations.
LOCK_DIR="$SCRATCH/legacypipe_gpu_locks"
mkdir -p "$LOCK_DIR" || { echo "Error: Could not create lock directory $LOCK_DIR. Exiting." >&2; exit 1; }

# --- Function to Release GPU Lock and Clean Up ---
# This function is trapped to run on script exit (normal or abnormal)
release_gpu_lock() {
  if [ -n "$current_lock_file" ] && [ -f "$current_lock_file" ]; then
    echo "$(date): Releasing lock for GPU $chosen_gpu_id by removing: $current_lock_file"
    rm -f "$current_lock_file"
  fi
  # Clean up TMPCACHE directory
  if [ -n "$current_tmpcache" ] && [ -d "$current_tmpcache" ]; then
    echo "$(date): Removing temporary cache directory: $current_tmpcache"
    rm -R "$current_tmpcache"
  fi
}

# Set traps to ensure cleanup on exit or termination signals
trap release_gpu_lock EXIT SIGHUP SIGINT SIGTERM

# --- Function to Acquire a GPU Lock ---
acquire_gpu_lock() {
  local max_wait_seconds=7200 # Wait for up to 2 hours (e.g., for long queues)
  local sleep_interval=60     # Check every 60 seconds for an available GPU

  echo "$(date): Attempting to acquire an available GPU from: ${all_gpu_indices[*]}..."

  start_time=$(date +%s)
  while true; do
    for i in "${!all_gpu_indices[@]}"; do
      local potential_gpu_idx="${all_gpu_indices[$i]}"
      local temp_lock_file="$LOCK_DIR/gpu_${potential_gpu_idx}.lock"

      # Attempt to create an exclusive lock file (atomic operation).
      # If the file doesn't exist, 'set -C' (noclobber) creates it atomically.
      # If it exists, 'set -C' prevents overwriting, and the '2>/dev/null' suppresses error messages.
      if ( set -o noclobber; > "$temp_lock_file" ) 2>/dev/null; then
        echo "$(date): Acquired lock for GPU $potential_gpu_idx. Lock file: $temp_lock_file."
        echo "$$" > "$temp_lock_file" # Write current PID to lock file for debugging
        chosen_gpu_id="$potential_gpu_idx"
        current_lock_file="$temp_lock_file" # Store for global access and cleanup
        return 0 # Success
      fi
    done

    # If we reached here, no GPU was immediately available
    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))

    if [ "$elapsed_time" -ge "$max_wait_seconds" ]; then
      echo "$(date): Error: Max wait time ($max_wait_seconds seconds) exceeded. No GPU became available." >&2
      return 1 # Indicate failure
    fi

    echo "$(date): All managed GPUs appear in use or locked. Waiting for ${sleep_interval} seconds... (Elapsed: ${elapsed_time}s)"
    sleep "$sleep_interval"
  done
}

# --- Argument Parsing ---
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -brick)
      brick_name="$2"
      shift 2 # past argument and value
      ;;
    -zoom)
      zoom_arg="--zoom $2"
      shift 2 # past argument and value
      ;;
    -gpumode)
      gpumode_arg="--gpumode $2"
      md_suffix="gpumode$2"
      shift 2 # past argument and value
      ;;
    -gpu)
      use_gpu_flag="--use-gpu"
      shift # past argument
      ;;
    -sub-blobs)
      subblobs_flag="--sub-blobs"
      shift # past argument
      ;;
    -blobid)
      blobid_arg="--bid $2"
      shift 2 # past argument and value
      ;;
    -outdir)
      outdir="$2"
      shift 2 # past argument and value
      ;;
    -threads)
      threads_arg="--threads $2"
      th_suffix="threads$2"
      shift 2 # past argument and value
      ;;
    *)    # Unknown option
      echo "Error: Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

# --- Validate input brick name ---
if [ -z "$brick_name" ]; then
  echo "Error: No brick name provided. Use -brick <brick_name>" >&2
  exit 1
fi

# --- Discover A100 GPU Indices ---
echo "$(date): Discovering NVIDIA A100 GPUs on this system..."
mapfile -t gpu_detection_output < <(nvidia-smi --query-gpu=index,name --format=csv,noheader,nounits | grep "NVIDIA A100")

if [ ${#gpu_detection_output[@]} -eq 0 ]; then
  echo "Error: No NVIDIA A100 GPUs found. Exiting." >&2
  exit 1
fi

for gi in "${gpu_detection_output[@]}"; do
  # Example output: 0, NVIDIA A100-SXM4-40GB
  gpu_idx=$(echo "$gi" | awk -F', ' '{print $1}')
  all_gpu_indices+=("$gpu_idx")
done

# Limit to the first 4 A100 GPUs if more are detected, as per environment setup
if [ ${#all_gpu_indices[@]} -gt 4 ]; then
  echo "$(date): Warning: More than 4 A100 GPUs (${#all_gpu_indices[@]}) detected. Will manage locks for the first 4: ${all_gpu_indices[@]:0:4}"
  all_gpu_indices=("${all_gpu_indices[@]:0:4}")
fi

if [ ${#all_gpu_indices[@]} -eq 0 ]; then
  echo "Error: Could not identify any A100 GPUs to manage. Exiting." >&2
  exit 1
fi

echo "$(date): Detected A100 GPU indices to manage: ${all_gpu_indices[*]}"

# --- Acquire a GPU Lock ---
if ! acquire_gpu_lock; then
  echo "$(date): Failed to acquire a GPU lock. Exiting." >&2
  exit 1
fi

# --- Setup Unique Temporary Cache Directory ---
current_tmpcache=$(mktemp -d)
if [ ! -d "$current_tmpcache" ]; then
  echo "Error: Could not create temporary cache directory: $current_tmpcache. Exiting." >&2
  exit 1
fi

mkdir -p "$current_tmpcache/cache"
mkdir -p "$current_tmpcache/config"
export XDG_CACHE_HOME="$current_tmpcache/cache"
export XDG_CONFIG_HOME="$current_tmpcache/config"
# Copy existing configs/caches if they exist, suppress errors if they don't
cp -r "$HOME/.astropy/cache" "$XDG_CACHE_HOME/astropy" 2>/dev/null || true
cp -r "$HOME/.astropy/config" "$XDG_CONFIG_HOME/astropy" 2>/dev/null || true
export MPLCONFIGDIR="$current_tmpcache/matplotlib"
mkdir -p "$MPLCONFIGDIR"
cp -r "$HOME/.config/matplotlib" "$MPLCONFIGDIR" 2>/dev/null || true

# --- Prepare Output Directories ---
bri_prefix=${brick_name:0:3} # First three characters for directory naming
mkdir -p "$outdir/logs/$bri_prefix"
mkdir -p "$outdir/metrics/$bri_prefix"
mkdir -p "$outdir/pickles/$bri_prefix"

# Define the log file for this specific brick and GPU
log_file="$outdir/logs/$bri_prefix/$brick_name-$md_suffix-$th_suffix-gpu${chosen_gpu_id}.log"
echo "$(date): Brick $brick_name (GPU $chosen_gpu_id): Logging to: $log_file"

# --- Log Run Details ---
echo -e "\n\n\n" >> "$log_file"
echo "-----------------------------------------------------------------------------------------" >> "$log_file"
echo -e "\n$(date): Starting legacypipe for brick $brick_name on GPU $chosen_gpu_id on $(hostname)\n" >> "$log_file"
echo "-----------------------------------------------------------------------------------------" >> "$log_file"
echo "BRICK = $brick_name" >> "$log_file"
echo "GPUMODE = $gpumode_arg" >> "$log_file"
echo "GPU = $use_gpu_flag" >> "$log_file"
echo "ZOOM = $zoom_arg" >> "$log_file"
echo "THREADS = $threads_arg" >> "$log_file"
echo "SUBBLOBS = $subblobs_flag" >> "$log_file"
echo "BLOBID = $blobid_arg" >> "$log_file"
echo "LOG = $log_file" >> "$log_file"
echo "CUDA_VISIBLE_DEVICES = $chosen_gpu_id" >> "$log_file"
echo "TMPCACHE = $current_tmpcache" >> "$log_file"

# --- Set CUDA_VISIBLE_DEVICES for this process ---
export CUDA_VISIBLE_DEVICES="$chosen_gpu_id"
echo "$(date): Setting CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >> "$log_file"

# --- Execute legacypipe ---
echo "$(date): Executing legacypipe for brick $brick_name..." >> "$log_file"
echo "Command:" >> "$log_file"
echo "python -u \"$LEGACYPIPE_DIR/legacypipe/runbrick.py\" \\" >> "$log_file"
echo "  --brick \"$brick_name\" \\" >> "$log_file"
echo "  $zoom_arg \\" >> "$log_file"
echo "  $use_gpu_flag \\" >> "$log_file"
echo "  $gpumode_arg \\" >> "$log_file"
echo "  $threads_arg \\" >> "$log_file"
echo "  $subblobs_flag \\" >> "$log_file"
echo "  $blobid_arg \\" >> "$log_file"
echo "  --skip-calibs \\" >> "$log_file"
echo "  --bands g,r,i,z \\" >> "$log_file"
echo "  --rgb-stretch 1.5 \\" >> "$log_file"
echo "  --nsatur 2 \\" >> "$log_file"
echo "  --survey-dir \"$LEGACY_SURVEY_DIR\" \\" >> "$log_file"
echo "  --outdir \"$outdir\" \\" >> "$log_file"
echo "  --pickle \"${outdir}/pickles/${bri_prefix}/runbrick-%(brick)s-%%(stage)s.pickle\" \\" >> "$log_file"
echo "  --write-stage srcs \\" >> "$log_file"
echo "  --release 10099 \\" >> "$log_file"
echo "  --no-wise" >> "$log_file"

python -u "$LEGACYPIPE_DIR/legacypipe/runbrick.py" \
  --brick "$brick_name" \
  $zoom_arg \
  $use_gpu_flag \
  $gpumode_arg \
  $threads_arg \
  $subblobs_flag \
  $blobid_arg \
  --skip-calibs \
  --bands g,r,i,z \
  --rgb-stretch 1.5 \
  --nsatur 2 \
  --survey-dir "$LEGACY_SURVEY_DIR" \
  --outdir "$outdir" \
  --pickle "${outdir}/pickles/${bri_prefix}/runbrick-%(brick)s-%%(stage)s.pickle" \
  --write-stage srcs \
  --release 10099 \
  --no-wise \
  >> "$log_file" 2>&1

# Capture the exit status of the python command
status=$?
echo "$(date): legacypipe for brick $brick_name exited with status $status." >> "$log_file"

# The trap will handle cleanup (release_gpu_lock and remove TMPCACHE) automatically on exit
exit $status
