#!/bin/bash

cd ..

########
# CLIP #
########

results_output=results/target/clip_vit/zeroshot/initialization
final_savename='prototype'

python other_utils/consolidation_utils.py --dset office --results_output $results_output --final_savename $final_savename --consolidate_type target_zeroshot --metric acc --seed_list 0
# python other_utils/consolidation_utils.py --dset office-home --results_output $results_output --final_savename $final_savename --consolidate_type target_zeroshot --metric acc --seed_list 0
# python other_utils/consolidation_utils.py --dset VISDA-C --results_output $results_output --final_savename $final_savename --consolidate_type target_zeroshot --metric accmean --seed_list 0
# python other_utils/consolidation_utils.py --dset domainnet-126 --results_output $results_output --final_savename $final_savename --consolidate_type target_zeroshot --metric acc --seed_list 0
