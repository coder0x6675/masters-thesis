#!/usr/bin/env bash
# Draws "jobs" (tasks to process) from a "queue" (a shared file).
# Each line in the "queue" corresponds to a "job".

set -e
readonly job_list="$1"
shift
readonly cmd="${@// /\\ }"

[[ ! -f "$job_list" ]] && echo >&2 'Error: Invalid job list specified' && exit 1

exit_handler() {
	echo -e '\nInterrupted. Returning job to queue...'
	echo "$job" >> "$job_list"
	exit
}
trap exit_handler SIGINT

while
	job="$(sed -i -e "1{w /dev/stdout" -e 'd}' "$job_list")"
	[[ -n "$job" ]]
do
	echo -e '\n---'
	echo -e "Starting work on: ${job}\n"
	eval "$cmd"
done

echo -e '\n---'
echo 'No jobs left!'

