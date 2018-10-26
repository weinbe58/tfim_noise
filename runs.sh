#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

T1=(1 2 4 8 16 32 64)
T2=(3 6 12 24 48 96)
pids=()

for N in 4 6 8 10 12 14
do
# run processes and store pids in array
for t in ${T1[*]}; do
    # python tfim_spin_bath_local.py $N $t 0.1 1 runfiles/1d_model_spin_bath_3/ &
    python tfim_spin_bath_MF.py $N $t 0.1 1.0 1 runfiles/1d_model_spin_bath_1/ &
    pids+=($!)
done
# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done

# run processes and store pids in array
for t in ${T2[*]}; do
    # python tfim_spin_bath_local.py $N $t 0.1 1 runfiles/1d_model_spin_bath_3/ &
    python tfim_spin_bath_MF.py $N $t 0.1 1.0 1 runfiles/1d_model_spin_bath_1/ &
    pids+=($!)
done
# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done

done
