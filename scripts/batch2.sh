#!/bin/zsh
cd "/Users/jayden/Desktop/rl edge ai"
while pgrep -f "scripts/train.py" > /dev/null; do sleep 10; done
echo "batch1 complete, launching batch2 at $(date +%T)"
run () { env OMP_NUM_THREADS=1 python3 -u scripts/train.py --steps 1500000 --no-recurrent "${@:2}" --tag "$1" > "logs/$1.log" 2>&1 & }
run capdrop   --cap-dropout 0.35
run main_s1   --seed 1
run main_s2   --seed 2
run ask6      --lambda-ask 6.0
run ask12     --lambda-ask 12.0
wait
echo "batch2 complete at $(date +%T)"
