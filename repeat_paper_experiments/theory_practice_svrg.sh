#!/usr/bin/env bash

# theory_practice_svrg julia_path stochopt_path exp all_problems
julia_path=${1?Error: no path for julia given}
stochopt_path=${2?Error: no path for StochOpt.jl given}
exp=${3?Error: no experiment name given}
all_problems=${4?Error: no problems (boolean) given}
number_processors=$5 # set to 1 by default

case $exp in
    "1a")
        exp_path=$stochopt_path$"repeat_paper_experiments/repeat_theory_practice_SVRG_paper_experiment_1a_without_mini-batching.jl"
        ;;
    "1b")
        exp_path=$stochopt_path$"repeat_paper_experiments/repeat_theory_practice_SVRG_paper_experiment_1b_optimal_mini-batching.jl"
        ;;
    "1c")
        exp_path=$stochopt_path$"repeat_paper_experiments/repeat_theory_practice_SVRG_paper_experiment_1c_optimal_inner_loop.jl"
        ;;
    "2a")
        exp_path=$stochopt_path$"repeat_paper_experiments/repeat_theory_practice_SVRG_paper_experiment_2a_free_minibatch.jl"
        ;;
    "2b")
        exp_path=$stochopt_path$"repeat_paper_experiments/repeat_theory_practice_SVRG_paper_experiment_2b_free_inner_loop.jl"
        ;;
    *)
        printf "ERROR: unkown experiment name. It has to be in {1a, 1b, 1c, 2a, 2b}."
        exit 1 # terminate and indicate error
        ;;
esac

case $all_problems in
    true)
        printf "Running experiment $exp on all eight problems.\n"
        ;;
    false)
        printf "Running experiment $exp only on the first problem (logistic regression on the scaled ijcnn1 dataset with lambda=0.1).\n"
        ;;
    *)
        printf "ERROR: invalid input for all_problems. It has to be either true or false.\n"
        exit 1
        ;;
esac


if [[ -z "$number_processors"  || ($all_problems == false) ]]
then
    number_processors=1
    printf "Number of processors automatically set to 1\n"
elif [[ $number_processors =~ ^[1-8]$ ]]
then
    echo "Number of processors used in parallel: $number_processors"
else
    printf "ERROR: number_processors has to be an integer between 1 and 8\n"
    exit 1
fi


if [ $number_processors == 1 ]
then
    printf "Running the command:\n$julia_path $exp_path $stochopt_path $all_problems\n\n"
    $julia_path $exp_path $stochopt_path $all_problems
else
    printf "Running the command:\n$julia_path -p $number_processors $exp_path $stochopt_path $all_problems\n\n"
    $julia_path -p $number_processors $exp_path $stochopt_path $all_problems
fi


printf "\n"