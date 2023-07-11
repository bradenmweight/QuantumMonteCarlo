#!/bin/bash

for Ri in {40..600..20}; do
    for A0i in {0..500..100}; do
    #for A0i in {0..0}; do
        #for WCi in {500..2000..500}; do
        for WCi in {2000..2000}; do
        #for WCi in {500..1500..500}; do
            R=$( echo "$Ri*0.01" | bc )
            A0=$( echo "$A0i*0.001" | bc )
            WC=$( echo "$WCi*0.01" | bc )
            sbatch submit.QMC ${R} ${A0} ${WC}
            echo ${R} ${A0} ${WC}
        done
    done
done


