#!/bin/zsh
cd "/Users/jayden/Desktop/rl edge ai"
while pgrep -f "scripts/train.py" > /dev/null; do sleep 15; done
echo "launching stability test: n_envs 16 -> 48 (3x batch) at $(date +%T)"
for S in 0 1 2; do
  env OMP_NUM_THREADS=1 python3 -u scripts/train.py --steps 1500000 --no-recurrent \
      --n-envs 48 --seed $S --tag "big_s$S" > "logs/big_s$S.log" 2>&1 &
done
wait
echo "stability test complete $(date +%T)"
