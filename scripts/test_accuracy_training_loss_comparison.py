"""
MIT License

Copyright (c) 2020 Mahdi S. Hosseini

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

evaluation_directory = ''
datasets = ['CIFAR10', 'CIFAR100']
networks = ['VGG16', 'ResNet34']
color_codes = [(1, 0, 0), (0.8, 0, 0), (0.6, 0, 0), (0.4, 0, 0),
               (0.3, 0, 0), 'steelblue', 'b', 'g', 'c', 'm', 'y', 'orange']
line_style = ['-', '-', '-', '-', '-', '--',
              '--', '--', '--', '--', '--', '--']
evaluating_folders = ['SGD_AdaS_alpha_0.800', 'SGD_AdaS_alpha_0.850',
                      'SGD_AdaS_alpha_0.900', 'SGD_AdaS_alpha_0.950',
                      'SGD_AdaS_alpha_0.975',
                      'SGD_OneCycleLR_epoch_25', 'SGD_OneCycleLR_epoch_50',
                      'SGD_StepLR_StepSize_25_StepDecay_0.5', 'AdaGrad',
                      'AdaM_tuned', 'RMSProp', 'AdaBound']
export_string = ['AdaS_beta_0.800', 'AdaS_beta_0.850', 'AdaS_beta_0.900',
                 'AdaS_beta_0.950', 'AdaS_beta_0.975',
                 'SGD_OneCycleLR-25', 'SGD_OneCycleLR-50', 'SGD_StepLR',
                 'AdaGrad', 'AdaM', 'RMSProp', 'AdaBound']
legend_string = [r'AdaS: $ \beta = 0.800$', r'AdaS: $ \beta = 0.850$',
                 r'AdaS: $ \beta = 0.900$', r'AdaS: $ \beta = 0.950$',
                 r'AdaS: $ \beta = 0.975$',
                 r'SGD-1CycleLR-25', r'SGD-1CycleLR-50', r'SGD-StepLR',
                 'AdaGrad', 'AdaM', 'RMSProp', 'AdaBound']
acc_min = [0.82, 0.85, 0.60, 0.63]
acc_max = [0.945, 0.96, 0.74, 0.78]
loss_min = [5e-4, 5e-4, 1e-3, 1e-3]
loss_max = [10, 10, 10, 10]

for iteration_dataset in range(len(datasets)):
    for iteration_network in range(len(networks)):
        print(networks)
        plt.figure(1, figsize=(5, 5))
        for iteration_folder in range(len(evaluating_folders)):
            print(iteration_folder)
            file_path = evaluation_directory + '/' + \
                datasets[iteration_dataset] + '/' + networks[iteration_network] + \
                '/' + evaluating_folders[iteration_folder]
            file_dir = os.listdir(file_path)
            acc_data = np.empty((250, len(file_dir)))
            acc_data[:] = np.nan
            for iteration_file in range(len(file_dir)):
                file_call = file_path + '/' + file_dir[iteration_file]
                if 'png' in file_call:
                    continue
                print(file_call)
                df = pd.read_excel(file_call)
                df = df.T
                if "AdaS" in evaluating_folders[iteration_folder]:
                    acc_data_vec = np.asarray(df.iloc[12::12, 1])
                else:
                    acc_data_vec = np.asarray(df.iloc[9::9, 1])
                acc_data[0:len(acc_data_vec), iteration_file] = acc_data_vec
            plt.plot(np.array(range(1, acc_data.shape[0] + 1)), acc_data.mean(
                1), line_style[iteration_folder],
                color=color_codes[iteration_folder])
        plt.ylim((acc_min[iteration_dataset * 2 + iteration_network],
                  acc_max[iteration_dataset * 2 + iteration_network]))
        plt.xlim((1, 250))
        plt.ylabel('Test Accuracy', fontsize=16)
        plt.xlabel('Epoch - (t)', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.gca().legend(legend_string, prop={"size": 9}, loc="lower right",
                         bbox_to_anchor=(
            0.98, 0.02), borderaxespad=0., ncol=2)
        plt.grid(True)
        export_name = 'test_accuracy_comparison_' + \
            datasets[iteration_dataset] + '_' + \
            networks[iteration_network] + '.png'
        plt.savefig(export_name, dpi=300, bbox_inches='tight')
        plt.close()

for iteration_dataset in range(len(datasets)):
    for iteration_network in range(len(networks)):
        plt.figure(1, figsize=(5, 5))
        for iteration_folder in range(len(evaluating_folders)):
            file_path = evaluation_directory + '\\' + \
                datasets[iteration_dataset] + '\\' + networks[iteration_network] + \
                '\\' + evaluating_folders[iteration_folder]
            file_dir = os.listdir(file_path)
            error_data = np.empty((250, len(file_dir)))
            error_data[:] = np.nan
            for iteration_file in range(len(file_dir)):
                file_call = file_path + '\\' + file_dir[iteration_file]
                df = pd.read_excel(file_call)
                df = df.T
                if "AdaS" in evaluating_folders[iteration_folder]:
                    error_data_vec = np.asarray(df.iloc[1::12, 1])
                else:
                    error_data_vec = np.asarray(df.iloc[1::9, 1])
                error_data[0:len(error_data_vec),
                           iteration_file] = error_data_vec
            plt.plot(np.array(range(1, error_data.shape[0] + 1)),
                     error_data.mean(
                1), line_style[iteration_folder],
                color=color_codes[iteration_folder])
        plt.ylim((loss_min[iteration_dataset * 2 + iteration_network],
                  loss_max[iteration_dataset * 2 + iteration_network]))
        plt.xlim((1, 250))
        plt.yscale('log', basey=10)
        plt.ylabel('Training Loss', fontsize=16)
        plt.xlabel('Epoch - (t)', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.gca().legend(legend_string, prop={"size": 9}, loc="upper right",
                         bbox_to_anchor=(
            0.98, 0.98), borderaxespad=0., ncol=2)
        plt.grid(True)
        export_name = 'training_loss_comparison_' + \
            datasets[iteration_dataset] + '_' + \
            networks[iteration_network] + '.png'
        plt.savefig(export_name, dpi=300, bbox_inches='tight')
        plt.close()
