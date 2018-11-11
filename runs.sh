#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# T1=(6 24 96 384 1536 )
# T2=(3 12 48 192 768 )
# T1=(0.25 1 4 16 64)
# T2=(0.5 2 8 32 )

pids=()

mkdir runfiles/1d_model_spin_bath_1/
mkdir runfiles/1d_model_spin_bath_2/ 
mkdir runfiles/1d_model_spin_bath_3/
mkdir runfiles/1d_model_spin_bath_4/
mkdir runfiles/1d_model_spin_bath_5/
mkdir runfiles/1d_model_spin_bath_6/


omega=1.0
gamma=0.1

for N in 4 6 8 10 12 14 16 18
do
# run processes and store pids in array
for t in ${T1[*]}; do
    # python tfim_spin_bath_local.py $N $t 10.0 1 runfiles/1d_model_spin_bath_6/ &
    # python tfim_spin_bath_MF.py $N $N $t 1.0 1.0 4 runfiles/1d_model_spin_bath_5/ &
    # python tfim_spin_bath_MF.py $N $((N)) $t 0.1 0.1 4 runfiles/1d_model_spin_bath_4/ &
    python tfim_spin_bath_MF.py $N $N $t 10.0 1.0 5 runfiles/1d_model_spin_bath_5/ &
    # python tfim_spin_bath_MF.py $N $((4*N)) $t $gamma 0.01 3 runfiles/1d_model_spin_bath_3/ &
    # python tfim_spin_bath_MF.py $N $((4*N)) $t $gamma $omega 3 runfiles/1d_model_spin_bath_3/ &
    # python tfim_spin_bath_MF.py $N $((N)) $t 1.0 $omega 2 runfiles/1d_model_spin_bath_2/ &
    # python tfim_spin_bath_MF.py $N $N $t 1.0 1.0 1 runfiles/1d_model_spin_bath_1/ &
    pids+=($!)
done
# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done


for t in ${T2[*]}; do
    # python tfim_spin_bath_local.py $N $t 10.0 1 runfiles/1d_model_spin_bath_6/ &
    # python tfim_spin_bath_MF.py $N $N $t 1.0 1.0 4 runfiles/1d_model_spin_bath_5/ &
    # python tfim_spin_bath_MF.py $N $((N)) $t 0.1 0.1 4 runfiles/1d_model_spin_bath_4/ &
    python tfim_spin_bath_MF.py $N $N $t 10.0 1.0 5 runfiles/1d_model_spin_bath_5/ &
    # python tfim_spin_bath_MF.py $N $((4*N)) $t $gamma 0.01 3 runfiles/1d_model_spin_bath_3/ &
    # python tfim_spin_bath_MF.py $N $((4*N)) $t $gamma $omega 3 runfiles/1d_model_spin_bath_3/ &
    # python tfim_spin_bath_MF.py $N $((N)) $t 1.0 $omega 2 runfiles/1d_model_spin_bath_2/ &
    # python tfim_spin_bath_MF.py $N $N $t 1.0 1.0 1 runfiles/1d_model_spin_bath_1/ &
    pids+=($!)
done
# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done

done
