#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

T=(1 2 4 8 16 32 64 128 256)
pids=()

for N in 4 6 8 10 12
do
# run processes and store pids in array
for t in ${T[*]}; do
    # python tfim_spin_bath_MF.py $N $t 1 1.0 3 ../1d_model_spin_bath_3/ &
    python tfim_spin_bath_local.py $N $t 0.01 1 ../1d_model_spin_bath_4/ &
    pids+=($!)
done
# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done

done
