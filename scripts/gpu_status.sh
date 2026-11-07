#!/usr/bin/env bash
# Print "<run> {set: (acc, cut)}" for every finished GPU eval on the cluster (always exits 0).
ssh -o LogLevel=ERROR -o ConnectTimeout=20 wslp 'cd ~/sky_workdir/results/gpu 2>/dev/null || exit 0
for d in */; do d=${d%/}; if [ -f "$d/.done" ]; then
  echo "$d $(python3 -c "import json;d=json.load(open(\"$d/scores.json\"));print({k:(round(v[\"acc\"],3),round(v[\"cut_share\"],3),int(v[\"mean_tokens\"])) for k,v in d.items() if not k.startswith(\"_\")})")"
fi; done; exit 0'
