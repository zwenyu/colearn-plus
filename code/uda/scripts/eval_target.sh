#!/bin/bash

cd ..

############
# Proposed #
############

### colearn
for s_idx in 0 1 2; do
    python evaluate_target.py --gpu_id 0 --seed 0 --da uda --output two_branch_colearn/results_colearn/pretrained/clip_vit/vanilla/ --dset office --s $s_idx --method_name proposed --final_savename MatchOrConf_clfncc_threshold0.5_ent0.0_optsgd_adaptlr0.01_pretrainedlr0.01_bs50
done
# for s_idx in 0 1 2 3; do
#     python evaluate_target.py --gpu_id 0 --seed 0 --da uda --output two_branch_colearn/results_colearn/pretrained/clip_vit/vanilla/ --dset office-home --s $s_idx --method_name proposed --final_savename MatchOrConf_clfncc_threshold0.5_ent0.0_optsgd_adaptlr0.01_pretrainedlr0.01_bs50
# done
# for s_idx in 0 1 2 3; do
#     python evaluate_target.py --gpu_id 0 --seed 0 --da uda --output two_branch_colearn/results_colearn/pretrained/clip_vit/vanilla/ --dset domainnet-126 --s $s_idx --method_name proposed --final_savename MatchOrConf_clfncc_threshold0.5_ent0.0_optsgd_adaptlr0.01_pretrainedlr0.01_bs50
# done
# for s_idx in 0; do
#     python evaluate_target.py --gpu_id 0 --seed 0 --da uda --output two_branch_colearn/results_colearn/pretrained/clip_vit/vanilla/ --dset  VISDA-C --s $s_idx --method_name proposed --final_savename MatchOrConf_clfncc_threshold0.1_ent0.0_optsgd_adaptlr0.01_pretrainedlr0.01_bs50
# done


### colearn++
for s_idx in 0 1 2; do
    python evaluate_target.py --gpu_id 0 --seed 0 --da uda --output two_branch_colearnplus_weak/results_colearn/pretrained/clip_vit/vanilla/ --dset office --s $s_idx --method_name proposed --final_savename MatchOrConf_clfncc_threshold0.5_ent0.0_optsgd_adaptlr0.01_pretrainedlr0.01_bs50
done
# for s_idx in 0 1 2 3; do
#     python evaluate_target.py --gpu_id 0 --seed 0 --da uda --output two_branch_colearnplus_weak/results_colearn/pretrained/clip_vit/vanilla/ --dset office-home --s $s_idx --method_name proposed --final_savename MatchOrConf_clfncc_threshold0.5_ent0.0_optsgd_adaptlr0.01_pretrainedlr0.01_bs50
# done
# for s_idx in 0 1 2 3; do
#     python evaluate_target.py --gpu_id 0 --seed 0 --da uda --output two_branch_colearnplus_weak/results_colearn/pretrained/clip_vit/vanilla/ --dset domainnet-126 --s $s_idx --method_name proposed --final_savename MatchOrConf_clfncc_threshold0.5_ent0.0_optsgd_adaptlr0.01_pretrainedlr0.01_bs50
# done
# for s_idx in 0; do
#     python evaluate_target.py --gpu_id 0 --seed 0 --da uda --output two_branch_colearnplus_strong/results_colearn/pretrained/clip_vit/vanilla/ --dset  VISDA-C --s $s_idx --method_name proposed --final_savename OtherConf_clfncc_threshold0.1_ent0.0_optsgd_adaptlr0.01_pretrainedlr0.01_bs50
# done