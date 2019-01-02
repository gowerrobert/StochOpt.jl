#!/bin/sh

START_TIME=$SECONDS

while read var;
do
    printf "Dataset: $var\n"
    julia ./tmp/parallel_exp_1_2_compute_bounds_SAGA_nice.jl $var
done <$1

ELAPSED_TIME=$(($SECONDS - $START_TIME))
printf "\n\nTotal elapsed time: $(($ELAPSED_TIME/3600)) h $(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec\n"    

