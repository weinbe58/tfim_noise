#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

T=(1 2 4 8 16 32 64 128 256 512)
pids=()

for N in 10 14 18
do
# run processes and store pids in array
for t in ${T[*]}; do
    python tfim_spin_bath_exact.py $N $t 0.1 1.0 data/ &
    pids+=($!)
done
# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done

done
