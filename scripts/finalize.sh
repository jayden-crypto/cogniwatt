#!/bin/zsh
cd "/Users/jayden/Desktop/rl edge ai"
while pgrep -f "scripts/train.py" > /dev/null || pgrep -f batch2.sh > /dev/null; do sleep 15; done
echo "=== all training complete $(date +%T); evaluating ==="
export OMP_NUM_THREADS=1
python3 scripts/evaluate_policy.py --tags ff_2m capdrop -n 800 > logs/eval_final.log 2>&1
echo "=== figures ==="
python3 scripts/make_figures.py
echo "=== report ==="
python3 scripts/make_report.py > /dev/null
echo "=== done $(date +%T) ==="
