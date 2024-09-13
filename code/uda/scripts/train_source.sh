#!/bin/bash

cd ..

### Train model on the source domain

for seed in 0; do
	for s_idx in 0 1 2; do
		python image_source.py --gpu_id 0 --seed $seed --da uda --output results/source/resnet/vanilla/ --dset office --max_epoch 100 --s $s_idx
	done
done

# for seed in 0; do
# 	for s_idx in 0 1 2 3; do
# 		python image_source.py --gpu_id 0 --seed $seed --da uda --output results/source/resnet/vanilla/ --dset office-home --max_epoch 50 --s $s_idx
# 	done
# done

# for seed in 0; do
#     for s_idx in 0 1 2 3; do
#         python image_source.py --gpu_id 0 --seed $seed --da uda --output results/source/resnet/vanilla/ --dset domainnet-126 --max_epoch 50 --s $s_idx
#     done
# done

# for seed in 0; do
#     for s_idx in 0; do
#         python image_source.py --gpu_id 0 --seed $seed --da uda --output results/source/resnet/vanilla/ --dset VISDA-C --net resnet101 --lr 1e-3 --max_epoch 10 --s $s_idx
#     done
# done

# for seed in 0; do
#     for s_idx in 0; do
#         python image_source.py --gpu_id 0 --seed $seed --da uda --output results/source/resnet/vanilla/ --dset cub --max_epoch 50 --s $s_idx
#     done
# done
