#!/usr/bin/env bash
set -euo pipefail

echo "[1/5] Generate text completions"
python run_llama.py --option generate --use_gpu

echo "[2/5] Zero-shot prompting on SST"
python run_llama.py --option prompt --batch_size 10 \
  --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt \
  --label-names data/sst-label-mapping.json \
  --dev_out sst-dev-prompting-output.txt --test_out sst-test-prompting-output.txt \
  --use_gpu

echo "[3/5] Zero-shot prompting on CFIMDB"
python run_llama.py --option prompt --batch_size 10 \
  --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt \
  --label-names data/cfimdb-label-mapping.json \
  --dev_out cfimdb-dev-prompting-output.txt --test_out cfimdb-test-prompting-output.txt \
  --use_gpu

echo "[4/5] Finetuning on SST"
python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 80 \
  --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt \
  --label-names data/sst-label-mapping.json \
  --dev_out sst-dev-finetuning-output.txt --test_out sst-test-finetuning-output.txt \
  --use_gpu

echo "[5/5] Finetuning on CFIMDB"
python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 10 \
  --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt \
  --label-names data/cfimdb-label-mapping.json \
  --dev_out cfimdb-dev-finetuning-output.txt --test_out cfimdb-test-finetuning-output.txt \
  --use_gpu

echo "All required runs completed."
