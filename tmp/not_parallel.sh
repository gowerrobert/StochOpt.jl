#!/bin/sh

START_TIME=$SECONDS

for var in "$@"
do
  printf "Dataset: $var\n"
  julia ./tmp/parallel_exp_1_2_compute_bounds_SAGA_nice.jl $var
done

ELAPSED_TIME=$(($SECONDS - $START_TIME))
printf "\n\nTotal elapsed time: $(($ELAPSED_TIME/3600)) h $(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec\n"    

