#!/bin/bash

cd ..

declare -a NetList=("clip_vit")

##########
# Office #
##########

declare -a DataList=("office")
source_net="resnet50"

for dset in ${DataList[@]}; do
    for model in ${NetList[@]}; do
        for s in 0 1 2; do
            for seed in 0; do
                python image_target_two_branch_colearn.py --gpu_id 0 --output results_colearn/pretrained/$model/vanilla/ --dset $dset --s $s --pretrained_net $model --source_net $source_net --pretrained_model_type pretrained --seed $seed --pretrained_ft_comp 'clfncc' --expt_name two_branch_colearn --conf_threshold 0.5 --issave
                python image_target_two_branch_colearnplus_weak.py --gpu_id 0 --output results_colearn/pretrained/$model/vanilla/ --dset $dset --s $s --pretrained_net $model --source_net $source_net --pretrained_model_type pretrained --seed $seed --pretrained_ft_comp 'clfncc' --expt_name two_branch_colearnplus_weak --conf_threshold 0.5 --issave
            done
        done
    done
done

###############
# Office-Home #
###############

# declare -a DataList=("office-home")
# source_net="resnet50"

# for dset in ${DataList[@]}; do
#     for model in ${NetList[@]}; do
#         for s in 0 1 2 3; do
#             for seed in 0; do
#                 python image_target_two_branch_colearn.py --gpu_id 0 --output results_colearn/pretrained/$model/vanilla/ --dset $dset --s $s --pretrained_net $model --source_net $source_net --pretrained_model_type pretrained --seed $seed --pretrained_ft_comp 'clfncc' --expt_name two_branch_colearn --conf_threshold 0.5 --issave
#                 python image_target_two_branch_colearnplus_weak.py --gpu_id 0 --output results_colearn/pretrained/$model/vanilla/ --dset $dset --s $s --pretrained_net $model --source_net $source_net --pretrained_model_type pretrained --seed $seed --pretrained_ft_comp 'clfncc' --expt_name two_branch_colearnplus_weak --conf_threshold 0.5 --issave
#             done
#         done
#     done
# done

#############
# DomainNet #
#############

# declare -a DataList=("domainnet-126")
# source_net="resnet50"

# for dset in ${DataList[@]}; do
#     for model in ${NetList[@]}; do
#         for s in 3; do
#             for seed in 0; do
#                 python image_target_two_branch_colearn.py --gpu_id 0 --output results_colearn/pretrained/$model/vanilla/ --dset $dset --s $s --pretrained_net $model --source_net $source_net --pretrained_model_type pretrained --seed $seed --pretrained_ft_comp 'clfncc' --expt_name two_branch_colearn --conf_threshold 0.5 --issave
#                 python image_target_two_branch_colearnplus_strong.py --gpu_id 0 --output results_colearn/pretrained/$model/vanilla/ --dset $dset --s $s --pretrained_net $model --source_net $source_net --pretrained_model_type pretrained --seed $seed --pretrained_ft_comp 'clfncc' --expt_name two_branch_colearnplus_strong --pseudolabel_strategy OtherConf --conf_threshold 0.5 --issave
#             done
#         done
#     done
# done

#########
# VisDA #
#########

# declare -a DataList=("VISDA-C")
# source_net="resnet101"

# for dset in ${DataList[@]}; do
#     for model in ${NetList[@]}; do
#         for s in 1; do
#             for seed in 0; do
#                 python image_target_two_branch_colearn.py --gpu_id 0 --output results_colearn/pretrained/$model/vanilla/ --dset $dset --s $s --pretrained_net $model --source_net $source_net --pretrained_model_type pretrained --seed $seed --pretrained_ft_comp 'clfncc' --expt_name two_branch_colearn --conf_threshold 0.1 --issave
#                 python image_target_two_branch_colearnplus_strong.py --gpu_id 0 --output results_colearn/pretrained/$model/vanilla/ --dset $dset --s $s --pretrained_net $model --source_net $source_net --pretrained_model_type pretrained --seed $seed --pretrained_ft_comp 'clfncc' --expt_name two_branch_colearnplus_strong --pseudolabel_strategy OtherConf --conf_threshold 0.1 --issave
#             done
#         done
#     done
# done
