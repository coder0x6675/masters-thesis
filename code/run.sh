#!/usr/bin/env bash
# Runs the entire research process.

set -e

readonly DIR_ROOT="${0%/*}"
readonly DIR_LIGHT="${DIR_ROOT}/data_light"
readonly DIR_MESAN="${DIR_ROOT}/data_mesan"
readonly DIR_MODEL="${DIR_ROOT}/model"
readonly DIR_RAW='./raw'
readonly DIR_PARTS='./parts'
readonly FILE_DATASET='./data.csv'

readonly FROM='2016-01-01'
readonly TO='2016-01-01'

mtypes=(
	"dense"
	"rnn"
	"lstm"
	"gru"
)


log-error() {
	local text="${1^}"
	local code=${2:-1}
	echo >&2 "[-] ERROR $code: $text"
	exit $code
}

log-warning() {
	local text="${1^}"
	echo >&2 "[!] $text"
}

log-info() {
	local text="${1^}"
	echo "[+] $text"
}

is-installed() {
	local program="$1"
	command -v "$program" 2>&1 >/dev/null
	return $?
}

function run() {
	"$(realpath "$1")" "${@:2}"
}


# Stage 1: Set up the environment.
log_info 'setting up the environment'
cd "$DIR_ROOT"
chmod -R +x ./*

space_available="$(df --output=avail . | tail --lines=1)"
(( space_available < 1099512000000 )) && log-error 'at least 1 tebibyte of memory is required'

is-installed python || log-error 'python is not installed'
is-installed Rscript || log-error 'rscript is not installed'
is-installed aria2c || log-error 'aria2c is not installed'

Rscript -e 'install.packages("tidyverse")'
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt


# Stage 2: Acquire the lightning dataset.
log_info 'acquiring the lightning dataset'
cd "$DIR_LIGHT"
mkdir "$DIR_RAW" "$DIR_PARTS"
run download.r
run process.r
run analyze.r


# Stage 3: Acquire the mesan dataset.
log_info 'acquireing the mesan dataset'
log_warning 'this may take a some time'
cd "$DIR_MESAN"
mkdir "$DIR_RAW" "$DIR_PARTS"
find "$DIR_LIGHT/$DIR_PARTS" -type f | sort -n > queue.txt
run download.sh
run pool.sh queue.txt python -u ./extract.py '$job' "$DIR_PARTS" "$DIR_RAW"
run process.r
run analyze.r


# Stage 4: Run the ML model.
log_info 'evaluating the ML model'
cd "$DIR_MODEL"
mkdir 'generations' 'evaluations'
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/opt/cuda
run prepare.r "$DIR_LIGHT/$FILE_DATASET" "$DIR_MESAN/$FILE_DATASET"
run evolve.py # Only a single generation will be simulated.
for mtype in "${mtypes[@]}"; do
	run evaluate.py "$mtype" "evaluations/$mtype.txt"
done


echo
echo '---'
echo 'ALL IS DONE'

