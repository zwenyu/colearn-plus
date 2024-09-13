'''
Functions to consolidate results from logs.
Return mean and standard deviation of specified metric on all domain pairs.
'''

import argparse
import os
import re
import json
import numpy as np
import ipdb

def consolidate_so(args, results_dir, domain_names):
    '''
    Consolidate with logs from source-only model.
    '''
    source_test_perf_list = []
    target_test_perf_list = []

    for d in domain_names:
        try:
            for i in domain_names:
                perf_d_i = []
                for s in args.seed_list:
                    results_d_file = '/'.join([results_dir, d, str(s), 'results.jsonl'])
                    with open(results_d_file) as file:
                        lines = file.readlines()
                    lastline = lines[-1]
                    results = json.loads(lastline)
                    perf_d_i.append(float(results[f'{args.metric}_{i}']))
                if d == i:
                    # test performance on source domain
                    source_test_perf_list.append(perf_d_i)
                else:
                    # test performance on target domain
                    target_test_perf_list.append(perf_d_i)
        except:
            print(f'Results for source model of {d} not found.')            
            continue

    round_digit = 1

    # return mean and std strings
    print('=== Test Performance on Source Domain ===')
    source_test_perf_mean = ' & '.join([str(round(np.mean(i), round_digit)) for i in source_test_perf_list])
    source_test_perf_std = ' & '.join([str(round(np.std(i), round_digit)) for i in source_test_perf_list])
    source_test_perf_overall_mean = str(round(np.mean([j for i in source_test_perf_list for j in i]), round_digit))
    source_test_perf_overall_std = str(round(np.std([np.mean([i[s] for i in source_test_perf_list]) for s in range(len(args.seed_list))]), round_digit))
    print('Mean: ', source_test_perf_mean + ' & ' + source_test_perf_overall_mean)
    print('Std: ', source_test_perf_std + ' & ' + source_test_perf_overall_std)

    print('=== Test Performance on Target Domain ===')
    target_test_perf_mean = ' & '.join([str(round(np.mean(i), round_digit)) for i in target_test_perf_list])
    target_test_perf_std = ' & '.join([str(round(np.std(i), round_digit)) for i in target_test_perf_list])
    target_test_perf_overall_mean = str(round(np.mean([j for i in target_test_perf_list for j in i]), round_digit))
    target_test_perf_overall_std = str(round(np.std([np.mean([i[s] for i in target_test_perf_list]) for s in range(len(args.seed_list))]), round_digit))
    print('Mean: ', target_test_perf_mean + ' & ' + target_test_perf_overall_mean)
    print('Std: ', target_test_perf_std + ' & ' + target_test_perf_overall_std)


def consolidate_target(args, results_dir, domain_names):
    '''
    Consolidate with logs from adapted model.
    '''
    target_test_perf_list = []

    for d in domain_names:
        for i in domain_names:
            if d == i:
                continue
            try:
                perf_d_i = []
                for s in args.seed_list:
                    results_d_file = '/'.join([results_dir, f'{d}_{i}', str(s), f'results_{args.final_savename}.jsonl'])
                    with open(results_d_file) as file:
                        lines = file.readlines()
                    lastline = lines[-1]
                    results = json.loads(lastline)
                    perf_d_i.append(float(results[f'{args.metric}_{i}']))

                # test performance on target domain
                target_test_perf_list.append(perf_d_i)
            except:
                print(f'Results for transfer model from {d} to {i} not found.')
                continue

    round_digit = 1

    # return mean and std strings
    print('=== Test Performance on Target Domain ===')
    target_test_perf_mean = ' & '.join([str(round(np.mean(i), round_digit)) for i in target_test_perf_list])
    target_test_perf_std = ' & '.join([str(round(np.std(i), round_digit)) for i in target_test_perf_list])
    target_test_perf_overall_mean = str(round(np.mean([j for i in target_test_perf_list for j in i]), round_digit))
    target_test_perf_overall_std = str(round(np.std([np.mean([i[s] for i in target_test_perf_list]) for s in range(len(args.seed_list))]), round_digit))
    print('Mean: ', target_test_perf_mean + ' & ' + target_test_perf_overall_mean)
    print('Std: ', target_test_perf_std + ' & ' + target_test_perf_overall_std)


def consolidate_target_zeroshot(args, results_dir, domain_names):
    '''
    Consolidate with logs from zeroshot model.
    '''
    target_test_perf_list = []

    for d in domain_names:
        for i in domain_names:
            if d == i:
                continue
            try:
                perf_d_i = []
                for s in args.seed_list:
                    results_d_file = '/'.join([results_dir, f'pretrained_{i}', str(s), f'results_{args.final_savename}.jsonl'])
                    with open(results_d_file) as file:
                        lines = file.readlines()
                    lastline = lines[-1]
                    results = json.loads(lastline)
                    perf_d_i.append(float(results[f'{args.metric}_{i}']))
                # test performance on target domain
                target_test_perf_list.append(perf_d_i)
            except:
                print(f'Results for transfer model from pretrained to {i} not found.')
                continue

    round_digit = 1

    # return mean and std strings
    print('=== Test Performance on Target Domain ===')
    target_test_perf_mean = ' & '.join([str(round(np.mean(i), round_digit)) for i in target_test_perf_list])
    target_test_perf_std = ' & '.join([str(round(np.std(i), round_digit)) for i in target_test_perf_list])
    target_test_perf_overall_mean = str(round(np.mean([j for i in target_test_perf_list for j in i]), round_digit))
    target_test_perf_overall_std = str(round(np.std([np.mean([i[s] for i in target_test_perf_list]) for s in range(len(args.seed_list))]), round_digit))
    print('Mean: ', target_test_perf_mean + ' & ' + target_test_perf_overall_mean)
    print('Std: ', target_test_perf_std + ' & ' + target_test_perf_overall_std)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Result consolidation')
    parser.add_argument('--da', type=str, default='uda')    
    parser.add_argument('--dset', type=str, default='office-home',
        choices=['VISDA-C', 'office', 'office-home', 'domainnet-126'])
    parser.add_argument('--results_output', type=str, default='san')
    parser.add_argument('--final_savename', type=str, default=None)
    parser.add_argument('--consolidate_type', type=str, default=None,
        choices=['source', 'target', 'target_zeroshot'])
    parser.add_argument('--metric', type=str, default='acc')
    parser.add_argument('--seed_list', type=int, nargs='+', default=[0],
        help="list of random seeds")
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['art', 'clipart', 'product', 'realworld']
    elif args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
    elif args.dset == 'VISDA-C':
        names = ['train', 'validation']
    elif args.dset == 'domainnet-126':
        names = ['clipart', 'painting', 'real', 'sketch']

    results_dir = os.path.join(args.results_output, args.da, args.dset)

    if args.consolidate_type == 'source':
        consolidate_so(args, results_dir, names)
    elif args.consolidate_type == 'target':
        consolidate_target(args, results_dir, names)
    elif args.consolidate_type == 'target_zeroshot':
        consolidate_target_zeroshot(args, results_dir, names)