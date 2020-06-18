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
from pathlib import Path
from argparse import ArgumentParser

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

clmp_style = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
              'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu',
              'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

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

parser = ArgumentParser()
parser.add_argument('--file', dest='file',
                    help="XLSX file path. If you wish to perform for " +
                    "multiple trials (and average over them)," +
                    "place each trial file in the " +
                    "same directory and pass the directory path")
parser.add_argument('--x-label', dest='x_label',
                    help='Label for x-axis. It will also define the output ' +
                    '.gif filename')
parser.add_argument('--adas', dest='adas', action='store_true',
                    help='Specify whether adas output XLSX or not')
parser.set_defaults(adas=False)
args = parser.parse_args()


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


fig, ax1 = plt.subplots()
fig.set_size_inches(3, 3)
knowledge_gain_list = []
proj_stability_list = []

if 'adas' in str(args.file).lower() and not args.adas:
    print("Warning: This appears to be AdaS data, but the --adas argument " +
          "was not set")
if Path(args.file).is_dir():
    for file_name in Path(args.file).iterdir():
        df = pd.read_excel(file_name)
        df = df.T
        if args.adas:
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
else:
    df = pd.read_excel(args.file)
    df = df.T
    if args.adas:
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
for values in knowledge_gain_list:
    knowledge_gain_data += values
for values in proj_stability_list:
    proj_stability_data += values
knowledge_gain_data /= len(knowledge_gain_list)
proj_stability_data /= len(proj_stability_list)
knowledge_gain_data = np.concatenate((knowledge_gain_data, np.tile(
    knowledge_gain_data[-1, :], [250 - knowledge_gain_data.shape[0], 1])),
    axis=0)
proj_stability_data = np.concatenate((proj_stability_data, np.tile(
    proj_stability_data[-1, :], [250 - proj_stability_data.shape[0], 1])),
    axis=0)
calling_blocks = np.linspace(0, knowledge_gain_data.shape[1]-1, min(
    knowledge_gain_data.shape[1], len(clmp_style)), dtype=int)
calling_blocks[-1] = min(knowledge_gain_data.shape[1],
                         len(clmp_style)) - 1
gif = FuncAnimation(fig, update, frames=EPOCHS)
plt.ylim((0.0, 0.65))
plt.xlim((1, 128))
plt.xlabel(f'Mapping Condition - ($\kappa$)\n{args.x_label}',
           fontsize=8)
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
export_name = f'knowledge_gain_vs_mapping_condition_{args.x_label}.gif'
print(f"Saving to {export_name}")
gif.save(export_name, writer='imagemagick', fps=15)
print(export_name)
plt.close()
