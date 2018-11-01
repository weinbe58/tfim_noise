#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

T=(3 6 12 24 48 96 192 384 768 1536)
pids=()

mkdir runfiles/1d_model_spin_bath_1/
mkdir runfiles/1d_model_spin_bath_2/ 
mkdir runfiles/1d_model_spin_bath_3/
mkdir runfiles/1d_model_spin_bath_4/
mkdir runfiles/1d_model_spin_bath_5/


omega=1.0
gamma=0.1

for N in 10 12 14 16
do
# run processes and store pids in array
for t in ${T[*]}; do
    # python tfim_spin_bath_local.py $N $t 0.1 1 runfiles/1d_model_local_spin_bath_1/ &
    # python tfim_spin_bath_MF.py $N $N $t 1.0 1.0 4 runfiles/1d_model_spin_bath_5/ &
    # python tfim_spin_bath_MF.py $N $((N)) $t 0.1 0.1 4 runfiles/1d_model_spin_bath_4/ &
    # python tfim_spin_bath_MF.py $N $((4*N)) $t $gamma 0.01 3 runfiles/1d_model_spin_bath_3/ &
    # python tfim_spin_bath_MF.py $N $((4*N)) $t $gamma $omega 3 runfiles/1d_model_spin_bath_3/ &
    python tfim_spin_bath_MF.py $N $((N)) $t 1.0 $omega 2 runfiles/1d_model_spin_bath_2/ &
    # python tfim_spin_bath_MF.py $N $N $t 1.0 1.0 1 runfiles/1d_model_spin_bath_1/ &
    pids+=($!)
done
# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done

done
