#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/tfc_default.yaml}"
TRIALS="${TFC_TRIALS:-10}"

python -m tfc_benchmark.cli prepare \
  --source-csv "${TFC_SOURCE_CSV:?set TFC_SOURCE_CSV}" \
  --tongue-root "${TFC_TONGUE_ROOT:?set TFC_TONGUE_ROOT}" \
  --face-root "${TFC_FACE_ROOT:?set TFC_FACE_ROOT}" \
  --output-dir artifacts/tfc

python -m tfc_benchmark.cli smoke --config "${CONFIG_PATH}"

for model in drfuse flexmoe healnet m3care shaspec smil aug inforeg; do
  python -m tfc_benchmark.cli tune \
    --config "${CONFIG_PATH}" \
    --model "${model}" \
    --trials "${TRIALS}" \
    --seed 42
done

python -m tfc_benchmark.cli run-all --config "${CONFIG_PATH}"
python -m tfc_benchmark.cli report \
  --output-root experiments_tfc \
  --report-dir reports_tfc

