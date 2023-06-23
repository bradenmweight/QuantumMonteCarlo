#!/bin/bash

for A0i in {0..1000..10}; do
    A0=$( echo "$A0i*0.001" | bc )
    #echo $A0
    sbatch submit.QMC ${A0}
done
