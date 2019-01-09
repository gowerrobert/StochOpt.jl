#!/bin/sh

#START_TIME=$SECONDS

task () {
    local dataset=$1
    local scaling=$2
    local lambda=$3
    # echo >> $dataset.text
    printf "Dataset: $dataset\n"
    nohup julia ./tmp/parallel_exp_1_2_compute_bounds_SAGA_nice_2.jl $dataset $scaling $lambda &> ./nohups/nohup_exp_1_2_${dataset}_${scaling}_${lambda}.out &
    # julia ./tmp/parallel_exp_1_2_compute_bounds_SAGA_nice_2.jl $dataset $scaling $lambda
}

while read dataset;
do
    #task "$dataset" &
    task "$dataset" $2 $3 &
done <$1

wait

#ELAPSED_TIME=$(($SECONDS - $START_TIME))
#printf "\n\nTotal elapsed time: $(($ELAPSED_TIME/3600)) h $(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec\n"
