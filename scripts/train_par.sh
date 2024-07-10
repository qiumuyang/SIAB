#!/bin/bash

file=$1
dataset=$2
device=$3
ratio=$4
config=$5
extra=${@:6}


# check n args
if [ $# -lt 5 ]; then
    echo "Usage: $0 <file> <dataset> <device> <ratio> <config_name> [extra]"
    exit
fi

seed=1339
cfg="configs/$dataset.yaml"  # dataset config
tcfg="configs/$config.yaml"  # training config

# split device by comma
devices=(${device//,/ })

if [[ $seed == 1339 ]]; then
    dir="outputs/$dataset/$ratio/$config"
else
    dir="outputs-$seed/$dataset/$ratio/$config"
fi

echo $dir

case $dataset in
    "prostate")
        domain_ids=(0 1 2 3 4 5)
        domain_names=(A B C D E F)
        ;;
    "fundus")
        domain_ids=(0 1 2 3)
        domain_names=(A B C D)
        ;;
    "mnms")
        domain_ids=(0 1 2 3)
        domain_names=(A B C D)
        ;;
    *)
        echo "Dataset not supported!"
        exit
        ;;
esac

mkdir -p $dir

for i in "${!domain_ids[@]}"; do
    j=$((i % ${#devices[@]}))
    device_id=${devices[$j]}
    domain_id=${domain_ids[$i]}
    domain_name=${domain_names[$i]}
    CUDA_VISIBLE_DEVICES=$device_id python $file \
        --config $cfg \
        --train-config $tcfg \
        --save-path $dir/$domain_name \
        --seed $seed \
        --domain $domain_id \
        --ratio $ratio \
        $extra \
        >> $dir/$domain_name.log 2>&1 &
done
