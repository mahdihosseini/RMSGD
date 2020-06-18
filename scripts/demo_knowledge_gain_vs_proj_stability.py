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
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import os

datasets = ['CIFAR10', 'CIFAR100']
networks = ['ResNet34', 'VGG16']
datasets = ['CIFAR10']
networks = ['ResNet34']

# color_selections = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
color_codes = [(0.9, 0, 0), (0.7, 0, 0), (0.5, 0, 0),
               (0.3, 0, 0), 'b', 'g', 'c', 'm', 'y']
# clmp_style = ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
clmp_style = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd',
              'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
evaluation_directory = '..'

# evaluating_folders = ['SGD_OneCycleLR_epoch_50', 'SGD_StepLR_StepSize_25_StepDecay_0.5', 'AdaM_tuned', 'RMSProp', 'AdaGrad', 'AdaBound',
#                       'SGD_AdaS_alpha_0.800', 'SGD_AdaS_alpha_0.825', 'SGD_AdaS_alpha_0.850', 'SGD_AdaS_alpha_0.875',
#                       'SGD_AdaS_alpha_0.900', 'SGD_AdaS_alpha_0.925', 'SGD_AdaS_alpha_0.950', 'SGD_AdaS_alpha_0.975']
# export_string = ['SGD_OneCycleLR', 'SGD_StepLR', 'AdaM', 'RMSProp', 'AdaGrad', 'AdaBound',
#                  'AdaS_beta_0.800', 'AdaS_beta_0.825', 'AdaS_beta_0.850', 'AdaS_beta_0.875',
#                  'AdaS_beta_0.900', 'AdaS_beta_0.925', 'AdaS_beta_0.950', 'AdaS_beta_0.975']
evaluating_folders = ['SGD_OneCycleLR_epoch_50', 'SGD_StepLR_StepSize_25_StepDecay_0.5',
                      'AdaM_tuned', 'AdaBound',
                      'SGD_AdaS_alpha_0.800',
                      'SGD_AdaS_alpha_0.900']
export_string = ['SGD -  OneCycleLR', 'SGD - StepLR', 'AdaM',
                 'AdaBound',
                 'AdaS - beta=0.800',
                 'AdaS - beta=0.900']
dpi_resolution = 100

knowledge_gain_data = None
knowledge_gain_vec = None
knowledge_gain_list = None
proj_stab_vec = None
proj_stability_data = None
proj_stability_list = None
calling_blocks = None
EPOCHS = 250
color_vec = np.array(
    range(1, 250 + 1)) / EPOCHS
size_vec = np.ones((250, 1)) * 20 * 0.1


def update(epoch):
    print(epoch)
    for block_index in range(len(calling_blocks)):
        plt.title(f"Epoch: {epoch}")
        plt.scatter([proj_stability_data[epoch, calling_blocks[block_index]]],
                    [knowledge_gain_data[epoch, calling_blocks[block_index]]],
                    c=[epoch],
                    cmap=clmp_style[block_index],
                    norm=matplotlib.colors.Normalize(vmin=0, vmax=EPOCHS),
                    s=[size_vec[epoch]], alpha=0.9)


for iteration_dataset in range(len(datasets)):
    print(datasets[iteration_dataset])
    for iteration_network in range(len(networks)):
        print(networks[iteration_network])
        for iteration_folder in range(0, len(evaluating_folders)):
            file_path = evaluation_directory + '/' + datasets[iteration_dataset] + '/' + networks[
                iteration_network] + '/' + evaluating_folders[iteration_folder]
            file_dir = os.listdir(file_path)
            # figure
            fig, ax1 = plt.subplots()
            fig.set_size_inches(3, 3)
            knowledge_gain_list = []
            proj_stability_list = []
            print(file_dir)
            for iteration_file in range(len(file_dir)):
                file_call = file_path + '/' + file_dir[iteration_file]
                df = pd.read_excel(file_call)
                df = df.T
                if "AdaS" in evaluating_folders[iteration_folder]:
                    input_gain_vec = np.asarray(df.iloc[5::12, :])
                    output_gain_vec = np.asarray(df.iloc[6::12, :])
                    knowledge_gain_vec = (input_gain_vec + output_gain_vec) / 2
                    input_proj_stab_vec = np.asarray(df.iloc[8::12, :])
                    output_proj_stab_vec = np.asarray(df.iloc[9::12, :])
                    proj_stab_vec = (input_proj_stab_vec +
                                     output_proj_stab_vec) / 2
                else:
                    input_gain_vec = np.asarray(df.iloc[4::9, :])
                    output_gain_vec = np.asarray(df.iloc[5::9, :])
                    knowledge_gain_vec = (input_gain_vec + output_gain_vec) / 2
                    input_proj_stab_vec = np.asarray(df.iloc[6::9, :])
                    output_proj_stab_vec = np.asarray(df.iloc[7::9, :])
                    proj_stab_vec = (input_proj_stab_vec +
                                     output_proj_stab_vec) / 2
                knowledge_gain_list.append(knowledge_gain_vec)
                proj_stability_list.append(proj_stab_vec)
            knowledge_gain_data = np.zeros(knowledge_gain_list[0].shape)
            proj_stability_data = np.zeros(proj_stability_list[0].shape)
            for iteration_file in range(len(knowledge_gain_list)):
                knowledge_gain_data = knowledge_gain_data + \
                    knowledge_gain_list[iteration_file]
                proj_stability_data = proj_stability_data + \
                    proj_stability_list[iteration_file]
            knowledge_gain_data = knowledge_gain_data / \
                len(knowledge_gain_list)
            proj_stability_data = proj_stability_data / \
                len(knowledge_gain_list)
            knowledge_gain_data = np.concatenate((knowledge_gain_data, np.tile(
                knowledge_gain_data[-1, :], [250 - knowledge_gain_data.shape[0], 1])), axis=0)
            proj_stability_data = np.concatenate((proj_stability_data, np.tile(
                proj_stability_data[-1, :], [250 - proj_stability_data.shape[0], 1])), axis=0)
            calling_blocks = np.linspace(0, knowledge_gain_data.shape[1]-1, min(
                knowledge_gain_data.shape[1], len(clmp_style)), dtype=int)
            calling_blocks[-1] = min(knowledge_gain_data.shape[1],
                                     len(clmp_style)) - 1
            gif = FuncAnimation(fig, update, frames=EPOCHS)
            # for block_index in range(len(calling_blocks)):
            #     shape = knowledge_gain_data.shape
            #     color_vec = np.array(
            #         range(1, proj_stability_data.shape[0] + 1)) / 250
            #     size_vec = np.ones((proj_stability_data.shape[0], 1)) * 20
            #     plt.scatter(proj_stability_data[:, calling_blocks[block_index]], knowledge_gain_data[:, calling_blocks[block_index]], c=color_vec,
            #                 cmap=clmp_style[block_index], s=size_vec, alpha=0.9)
            plt.ylim((0.0, 0.65))
            plt.xlim((1, 128))
            plt.xlabel(
                f'Mapping Condition - ($\kappa$)\n{export_string[iteration_folder]}', fontsize=8)
            plt.ylabel('Knowledge Gain - (G)', fontsize=8)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            plt.title('Epoch 0', fontsize=9)
            plt.xscale('log', basex=2)
            ax1.set_xticks([1, 2, 4, 8, 16, 32, 64])
            plt.yticks(np.arange(0, 0.65, 0.65/5))
            plt.tight_layout()
            # plt.ylabel('Test Accuracy')
            # plt.xlabel('Epoch')
            # plt.gca().legend(evaluating_folders, prop={"size":11})
            plt.grid(True)
            # export_name = 'knowledge_gain_vs_mapping_condition_' + \
            #     datasets[iteration_dataset] + '_' + networks[iteration_network] + \
            #     '_' + export_string[iteration_folder] + '.png'
            # plt.savefig(export_name, dpi=dpi_resolution, bbox_inches='tight')
            export_name = 'gifs/knowledge_gain_vs_mapping_condition_' + \
                datasets[iteration_dataset] + '_' + networks[iteration_network] + \
                '_' + export_string[iteration_folder] + '.gif'
            print("saving...")
            gif.save(export_name, writer='imagemagick', fps=15)
            print(export_name)
            plt.close()
