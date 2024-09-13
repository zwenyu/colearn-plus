#!/bin/bash

cd ..

metric='acc'

declare -a ModelList=("clip_vit")

# colearn++ with weak guidance 
expt='two_branch_colearnplus_weak'

for model in ${ModelList[@]}; do
  results_output=${expt}/results_colearn/pretrained/${model}/vanilla/
  python other_utils/consolidation_utils.py --dset office --results_output $results_output --final_savename MatchOrConf_clfncc_threshold0.5_ent0.0_optsgd_adaptlr0.01_pretrainedlr0.01_bs50 --consolidate_type target --metric $metric --seed_list 0
done

# for model in ${ModelList[@]}; do
#   results_output=${expt}/results_colearn/pretrained/${model}/vanilla/
#   python other_utils/consolidation_utils.py --dset office-home --results_output $results_output --final_savename MatchOrConf_clfncc_threshold0.5_ent0.0_optsgd_adaptlr0.01_pretrainedlr0.01_bs50 --consolidate_type target --metric $metric --seed_list 0
# done

# colearn++ with strong guidance 
# expt='two_branch_colearnplus_strong'

# for model in ${ModelList[@]}; do
#   results_output=${expt}/results_colearn/pretrained/${model}/vanilla/
#   python other_utils/consolidation_utils.py --dset VISDA-C --results_output $results_output --final_savename OtherConf_clfncc_threshold0.1_ent0.0_optsgd_adaptlr0.01_pretrainedlr0.01_bs50 --consolidate_type target --metric acc_classmean --seed_list 0
# done

# for model in ${ModelList[@]}; do
#   results_output=${expt}/results_colearn/pretrained/${model}/vanilla/
#   python other_utils/consolidation_utils.py --dset domainnet-126 --results_output $results_output --final_savename OtherConf_clfncc_threshold0.5_ent0.0_optsgd_adaptlr0.01_pretrainedlr0.01_bs50 --consolidate_type target --metric $metric --seed_list 0
# done