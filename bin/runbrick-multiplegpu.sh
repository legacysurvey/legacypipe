#! /bin/bash

# Script for running the legacypipe code within a Shifter container at NERSC
# Modified to support multiple bricks and GPU assignment with multiprocessing.

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
# https://software.intel.com/en/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
# To avoid problems with MPI and Python multiprocessing
export MPICH_GNI_FORK_MODE=FULLCOPY
export KMP_AFFINITY=disabled

# --- Custom variables for multi-brick/GPU execution ---
bricks=() # Array to store brick names
zoom_arg=""
gpumode_arg=""
use_gpu_flag=""
threads_arg=""
subblobs_flag=""
blobid_arg=""
md_suffix="cpu" # default mode suffix for log name
th_suffix=""    # default threads suffix for log name

# Assuming a local checkout of the tractor and legacypipe repositories
#export TRACTOR_DIR=./tractor-git
#export LEGACYPIPE_DIR=./legacypipe-git/py
#export PYTHONPATH=$TRACTOR_DIR:$LEGACYPIPE_DIR:${PYTHONPATH}

# --- Argument Parsing ---
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -bricks)
      shift # past argument "-bricks"
      # Read multiple brick names until the next flag or end of arguments
      while [[ $# -gt 0 && "$1" != -* ]]; do
        if [ ${#bricks[@]} -ge 4 ]; then
          echo "Warning: Only up to 4 bricks are supported. Ignoring additional bricks: $1" >&2
        else
          bricks+=("$1")
        fi
        shift
      done
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

# --- Validate input bricks ---
if [ ${#bricks[@]} -eq 0 ]; then
  echo "Error: No brick names provided. Use -bricks <brick1> [brick2 ...]" >&2
  exit 1
fi

# --- GPU Availability Check ---
echo "Checking for available GPUs..."
# Get GPU index and memory usage, filtering for A100 (assuming it appears in output)
# Using 'noheader,nounits' for easier parsing. Memory used is in MiB.
mapfile -t gpu_info < <(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader,nounits | grep "NVIDIA A100")

available_gpus=()
min_mem_threshold_mib=100 # Consider GPUs with less than 100 MiB memory used as available

if [ ${#gpu_info[@]} -eq 0 ]; then
  echo "Error: No NVIDIA A100 GPUs found. Exiting." >&2
  exit 1
fi

echo "Detected GPUs:"
for gi in "${gpu_info[@]}"; do
  # Example: 0, NVIDIA A100-SXM4-40GB, 15 MiB
  # We extract index and memory used
  gpu_idx=$(echo "$gi" | awk -F', ' '{print $1}')
  gpu_name=$(echo "$gi" | awk -F', ' '{print $2}')
  mem_used=$(echo "$gi" | awk -F', ' '{print $3}' | awk '{print $1}') # e.g., "15 MiB" -> "15"

  echo "  GPU $gpu_idx ($gpu_name): ${mem_used} MiB used"

  if (( $(echo "$mem_used < $min_mem_threshold_mib" | bc -l) )); then
    available_gpus+=("$gpu_idx")
  fi
done

echo "Available GPUs (memory < ${min_mem_threshold_mib} MiB): ${available_gpus[*]}"

if [ ${#available_gpus[@]} -lt ${#bricks[@]} ]; then
  echo "Error: Not enough available GPUs (${#available_gpus[@]}) for all bricks (${#bricks[@]})." >&2
  exit 1
fi

# --- Run legacypipe for each brick on a dedicated GPU ---
pids=()
tmpcache_dirs=()
brick_status=() # To store status of each brick run

echo "Starting legacypipe processes for ${#bricks[@]} bricks on dedicated GPUs..."

for i in "${!bricks[@]}"; do
  brick="${bricks[$i]}"
  gpu_id="${available_gpus[$i]}" # Assign GPU in order

  bri_prefix=${brick:0:3} # First three characters for directory naming

  # Create unique TMPCACHE for each process
  current_tmpcache=$(mktemp -d)
  tmpcache_dirs+=("$current_tmpcache")

  # Setup Astropy and Matplotlib cache/config directories within unique TMPCACHE
  mkdir -p "$current_tmpcache/cache"
  mkdir -p "$current_tmpcache/config"
  export XDG_CACHE_HOME="$current_tmpcache/cache"
  export XDG_CONFIG_HOME="$current_tmpcache/config"
  mkdir -p "$XDG_CACHE_HOME/astropy"
  cp -r "$HOME/.astropy/cache" "$XDG_CACHE_HOME/astropy" 2>/dev/null || true # Copy if exists
  mkdir -p "$XDG_CONFIG_HOME/astropy"
  cp -r "$HOME/.astropy/config" "$XDG_CONFIG_HOME/astropy" 2>/dev/null || true # Copy if exists
  export MPLCONFIGDIR="$current_tmpcache/matplotlib"
  mkdir -p "$MPLCONFIGDIR"
  cp -r "$HOME/.config/matplotlib" "$MPLCONFIGDIR" 2>/dev/null || true # Copy if exists

  mkdir -p "$outdir/logs/$bri_prefix"
  mkdir -p "$outdir/metrics/$bri_prefix"
  mkdir -p "$outdir/pickles/$bri_prefix"
  
  # Log file for this specific brick and GPU
  log_file="$outdir/logs/$bri_prefix/$brick-$md_suffix-$th_suffix-gpu${gpu_id}.log"
  echo "Brick $brick (GPU $gpu_id): Logging to: $log_file"

  echo -e "\n\n\n" >> "$log_file"
  echo "-----------------------------------------------------------------------------------------" >> "$log_file"
  echo -e "\nStarting legacypipe for brick $brick on GPU $gpu_id on $(hostname)\n" >> "$log_file"
  echo "-----------------------------------------------------------------------------------------" >> "$log_file"
  echo "BRICK = $brick" >> "$log_file"
  echo "GPUMODE = $gpumode_arg" >> "$log_file"
  echo "GPU = $use_gpu_flag" >> "$log_file"
  echo "ZOOM = $zoom_arg" >> "$log_file"
  echo "THREADS = $threads_arg" >> "$log_file"
  echo "SUBBLOBS = $subblobs_flag" >> "$log_file"
  echo "BLOBID = $blobid_arg" >> "$log_file"
  echo "LOG = $log_file" >> "$log_file"
  echo "CUDA_VISIBLE_DEVICES = $gpu_id" >> "$log_file"
  echo "TMPCACHE = $current_tmpcache" >> "$log_file"

  # Launch legacypipe in the background with specific GPU
  # Use a subshell to set CUDA_VISIBLE_DEVICES for each process independently
  (
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    echo "Executing: python -u $LEGACYPIPE_DIR/legacypipe/runbrick.py \\" >> "$log_file"
    echo "    --brick \"$brick\" \\" >> "$log_file"
    echo "    $zoom_arg \\" >> "$log_file"
    echo "    $use_gpu_flag \\" >> "$log_file"
    echo "    $gpumode_arg \\" >> "$log_file"
    echo "    $threads_arg \\" >> "$log_file"
    echo "    $subblobs_flag \\" >> "$log_file"
    echo "    $blobid_arg \\" >> "$log_file"
    echo "    --skip-calibs \\" >> "$log_file"
    echo "    --bands g,r,i,z \\" >> "$log_file"
    echo "    --rgb-stretch 1.5 \\" >> "$log_file"
    echo "    --nsatur 2 \\" >> "$log_file"
    echo "    --survey-dir \"$LEGACY_SURVEY_DIR\" \\" >> "$log_file"
    echo "    --outdir \"$outdir\" \\" >> "$log_file"
    echo "    --pickle \"${outdir}/pickles/${bri_prefix}/runbrick-%(brick)s-%%(stage)s.pickle\" \\" >> "$log_file"
    echo "    --write-stage srcs \\" >> "$log_file"
    echo "    --release 10099 \\" >> "$log_file"
    echo "    --no-wise" >> "$log_file"
    
    python -u "$LEGACYPIPE_DIR/legacypipe/runbrick.py" \
      --brick "$brick" \
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
  ) &
  pids+=($!) # Store the PID of the background process
  brick_status+=("$brick:0") # Initialize status for this brick, format: "brick_name:exit_code"
  echo "Launched process for brick '$brick' on GPU $gpu_id with PID ${pids[-1]}"
done

echo "Waiting for all legacypipe processes to complete..."
overall_status=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  brick_name_from_status=$(echo "${brick_status[$i]}" | cut -d':' -f1) # Extract original brick name

  wait "$pid"
  current_status=$?
  brick_status[$i]="$brick_name_from_status:$current_status" # Update status

  if [ $current_status -ne 0 ]; then
    echo "Error: Process for brick '$brick_name_from_status' (PID $pid) exited with status $current_status." >&2
    overall_status=1
  else
    echo "Process for brick '$brick_name_from_status' (PID $pid) completed successfully."
  fi
done

# --- Cleanup TMPCACHE directories ---
echo "Cleaning up temporary directories..."
for dir in "${tmpcache_dirs[@]}"; do
  if [ -d "$dir" ]; then
    rm -R "$dir"
    echo "Removed $dir"
  fi
done

echo "--- Summary of Runs ---"
for status_entry in "${brick_status[@]}"; do
  echo "$status_entry"
done

exit $overall_status
