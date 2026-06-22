#!/usr/bin/env bash
set -euo pipefail

MODEL=/root/BaryIR/checkpoint/model_allweatherBaryNet128__100_1.pth
GPU=0
RESULTS_ROOT=/root/BaryIR/results

RAINDROP_INPUT=/root/autodl-tmp/raindrop_data/test_a/data/
RAINDROP_GT=/root/autodl-tmp/raindrop_data/test_a/gt/

RAIN_INPUT=/root/autodl-tmp/CVPR19RainTrain/test/data/
RAIN_GT=/root/autodl-tmp/CVPR19RainTrain/test/gt/

SNOW_INPUT=/root/autodl-tmp/Snow100K-testset/jdway/GameSSD/overlapping/test/Snow100K-L/synthetic/
SNOW_GT=/root/autodl-tmp/Snow100K-testset/jdway/GameSSD/overlapping/test/Snow100K-L/gt/

run_test () {
  local name=$1
  local deg=$2
  local tar=$3
  local out_dir=${RESULTS_ROOT}/${name}

  echo "=================================================="
  echo "[${name}] starting"
  echo "  degset:  ${deg}"
  echo "  tarset:  ${tar}"
  echo "  out:     ${out_dir}"
  echo "=================================================="

  mkdir -p "${out_dir}"
  python tester_bary.py \
    --model "${MODEL}" \
    --degset "${deg}" \
    --tarset "${tar}" \
    --save "${out_dir}/OUT/" \
    --savetar "${out_dir}/TAR/" \
    --gpus "${GPU}" 2>&1 | tee "${out_dir}/log.txt"

  echo "[${name}] done -> ${out_dir}/log.txt"
}

run_test raindrop "${RAINDROP_INPUT}" "${RAINDROP_GT}"
run_test rain     "${RAIN_INPUT}"     "${RAIN_GT}"
run_test snow     "${SNOW_INPUT}"     "${SNOW_GT}"

# Summary: extract the PSNR/SSIM/FID lines from each log
SUMMARY="${RESULTS_ROOT}/summary.txt"
{
  echo "Model: ${MODEL}"
  echo "Date:  $(date)"
  echo ""
  for name in raindrop rain snow; do
    echo "=== ${name} ==="
    grep -E "PSNR|SSIM|MAE|LPIPS|DISTS|FID|Total parameters|Average:" "${RESULTS_ROOT}/${name}/log.txt" || true
    echo ""
  done
} > "${SUMMARY}"

echo ""
echo "=================================================="
echo "All tests finished. Summary -> ${SUMMARY}"
echo "=================================================="
cat "${SUMMARY}"
