#!/bin/sh

START_TIME=$SECONDS

task () {
    local var=$1
    # echo >> $var.text
    printf "Dataset: $var\n"
    nohup julia ./tmp/parallel_exp_1_2_compute_bounds_SAGA_nice.jl $var &> ./nohups/nohup_exp_1_2_$var.out &
}

while read var; 
do
    task "$var" & 
done <$1

wait

#ELAPSED_TIME=$(($SECONDS - $START_TIME))
#printf "\n\nTotal elapsed time: $(($ELAPSED_TIME/3600)) h $(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec\n"
