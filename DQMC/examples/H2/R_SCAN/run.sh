#!/bin/bash

for Ri in {20..600..10}; do
    R=$( echo "$Ri*0.01" | bc )
    #echo $R
    sbatch submit.QMC ${R}
done

sbatch submit.QMC 50.0