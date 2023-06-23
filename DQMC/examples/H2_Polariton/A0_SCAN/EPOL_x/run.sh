#!/bin/bash

for Ri in {20..580..40}; do
    for A0i in {0..1000..100}; do
        R=$( echo "$Ri*0.01" | bc )
        A0=$( echo "$A0i*0.001" | bc )
        #echo $R $A0
        sbatch submit.QMC ${R} ${A0}
    done
done


