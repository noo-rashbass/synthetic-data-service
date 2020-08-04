## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore")

# 1. TimeGAN model
from timegan import timegan
# 2. Data loading
from data_loading import real_data_loading, sine_data_generation, sine_data_generation_static, sine_data_generation_mix
# 3. Metrics
#from metrics.discriminative_metrics import discriminative_score_metrics
#from metrics.predictive_metrics import predictive_score_metrics
#from metrics.visualization_metrics import visualization


def main (args):
    """Main function for timeGAN experiments.
    
    Args:
        - data_name: sine, stock, or energy
        - seq_len: sequence length
        - Network parameters (should be optimized for different datasets)
        - module: gru, lstm, or lstmLN
        - hidden_dim: hidden dimensions
        - num_layer: number of layers
        - iteration: number of training iterations
        - batch_size: the number of samples in each batch
        - metric_iteration: number of iterations for metric computation
    
    Returns:
        - ori_data: original data
        - generated_data: generated synthetic data
        - metric_results: discriminative and predictive scores
    """
    ## Data loading
    if args.data_name in ['stock', 'energy']:
        ori_data = real_data_loading(args.data_name, args.seq_len)
    elif args.data_name == 'sine':
        # Set number of samples and its dimensions
        no, dim = 10000, 5
        ori_data = sine_data_generation(no, args.seq_len, dim)
        #ori_data_static = sine_data_generation_static(no, args.seq_len, 1)
    elif args.data_name == 'normal':
        no, dim = 10000, 2
        ori_data, ori_data_static, ori_data_s = sine_data_generation_mix(no, args.seq_len, dim)
    
        
    ## Synthetic data generation by TimeGAN
    # Set newtork parameters
    parameters = dict()  
    parameters['module'] = args.module
    parameters['hidden_dim'] = args.hidden_dim
    parameters['num_layer'] = args.num_layer
    parameters['iterations'] = args.iteration
    parameters['batch_size'] = args.batch_size
        
    generated_data = timegan(ori_data, ori_data_s, ori_data_static, parameters)   
    print('Finish Synthetic Data Generation')
    np.save('gen_mix_data_no_seq_2k', generated_data)
    
    """
    ## Performance metrics   
    # Output initialization
    metric_results = dict()
    
    # 1. Discriminative Score
    discriminative_score = list()
    for _ in range(args.metric_iteration):
        temp_disc = discriminative_score_metrics(ori_data, generated_data)
        discriminative_score.append(temp_disc)
        
    metric_results['discriminative'] = np.mean(discriminative_score)
        
    # 2. Predictive score
    predictive_score = list()
    for tt in range(args.metric_iteration):
        temp_pred = predictive_score_metrics(ori_data, generated_data)
        predictive_score.append(temp_pred)   
        
    metric_results['predictive'] = np.mean(predictive_score)     
            
    # 3. Visualization (PCA and tSNE)
    visualization(ori_data, generated_data, 'pca')
    visualization(ori_data, generated_data, 'tsne')
    
    ## Print discriminative and predictive scores
    print(metric_results)
    """

    return ori_data, generated_data #, metric_results


if __name__ == '__main__':  

    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name',
        choices=['sine','stock','energy'],
        default='normal',
        type=str)
    parser.add_argument(
        '--seq_len',
        help='sequence length',
        default=24,
        type=int)
    parser.add_argument(
        '--module',
        choices=['gru','lstm','lstmLN'],
        default='gru',
        type=str)
    parser.add_argument(
        '--hidden_dim',
        help='hidden state dimensions (should be optimized)',
        default=10,
        type=int)
    parser.add_argument(
        '--num_layer',
        help='number of layers (should be optimized)',
        default=5,
        type=int)
    parser.add_argument(
        '--iteration',
        help='Training iterations (should be optimized)',
        default=2000,
        type=int)
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch (should be optimized)',
        default=128,
        type=int)
    # parser.add_argument(
    #     '--metric_iteration',
    #     help='iterations of the metric computation',
    #     default=10,
    #     type=int
    
    args = parser.parse_args() 
    
    # Calls main function  
    ori_data, generated_data = main(args)


# example command to run code, type in terminal
# python main_timegan_v6.py --data_name sine --seq_len 9 --module gru --hidden_dim 3 --num_layer 3 --iteration 5 --batch_size 4
