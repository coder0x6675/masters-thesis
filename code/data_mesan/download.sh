#!/usr/bin/env bash
# Downloads the raw MESAN data.

readonly BASE_URL='https://opendata-download-grid-archive.smhi.se/data/6'
readonly FROM_DATE='2015-05-20'
readonly TO_DATE='2023-06-01'
readonly OUTPUT_DIR='./raw'

read -r -d '' gen_urls << EOF
import pandas
for ts in pandas.date_range("$FROM_DATE", "$TO_DATE", freq="1h"):
	ym = f"{ts.year:04}{ts.month:02}"
	name = f"MESAN_{ym}{ts.day:02}{ts.hour:02}00+000H00M"
	print(f"$BASE_URL/{ym}/{name}")
EOF

python -c "$gen_urls" | aria2c \
	--input-file=- \
	--dir="$OUTPUT_DIR" \
	--continue \
	--force-sequential \
	--max-concurrent-downloads 1

