#!/bin/sh

#START_TIME=$SECONDS

task () {
    local dataset=$1
    local scaling=$2
    local lambda=$3
    local numsimu=$4
    # echo >> $dataset.text
    printf "Dataset: $dataset\n"
    nohup julia ./tmp/exp_3_SAGA_settings_improvement.jl $dataset $scaling $lambda $numsimu &> ./nohups/nohup_exp_3_${dataset}_${scaling}_${lambda}_numsimu_${numsimu}.out &
}

while read dataset;
do
    #task "$dataset" &
    task "$dataset" $2 $3 $4 &
done <$1

wait

#ELAPSED_TIME=$(($SECONDS - $START_TIME))
#printf "\n\nTotal elapsed time: $(($ELAPSED_TIME/3600)) h $(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec\n"
