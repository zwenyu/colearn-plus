#!/bin/bash

cd ..

method_name="zeroshot"
text_templates='ensemble'
classifier_type='prototype'

declare -a NetList=("clip_vit")

for net in ${NetList[@]}; do
    python evaluate_target_zeroshot.py --gpu_id 0 --seed 0 --da uda --output results/target/$net/$method_name --dset office --method_name $method_name --net $net --text_templates $text_templates --classifier_type $classifier_type

    # python evaluate_target_zeroshot.py --gpu_id 0 --seed 0 --da uda --output results/target/$net/$method_name --dset office-home --method_name $method_name --net $net --text_templates $text_templates --classifier_type $classifier_type

    # python evaluate_target_zeroshot.py --gpu_id 0 --seed 0 --da uda --output results/target/$net/$method_name --dset cub --method_name $method_name --net $net --text_templates $text_templates --classifier_type $classifier_type

    # python evaluate_target_zeroshot.py --gpu_id 0 --seed 0 --da uda --output results/target/$net/$method_name --dset VISDA-C --method_name $method_name --net $net --text_templates $text_templates --classifier_type $classifier_type

    # python evaluate_target_zeroshot.py --gpu_id 0 --seed 0 --da uda --output results/target/$net/$method_name --dset domainnet-126 --method_name $method_name --net $net --text_templates $text_templates --classifier_type $classifier_type
done