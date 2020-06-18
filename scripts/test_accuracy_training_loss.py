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
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

parser = ArgumentParser()
parser.add_argument('--file', dest='file',
                    help="XLSX file path. If you wish to perform for " +
                    "multiple trials (and average over them)," +
                    "place each trial file in the " +
                    "same directory and pass the directory path")
parser.add_argument('--legend-label', dest='legend_label',
                    help='Label for legend. It will also define the output ' +
                    '.png filename')
parser.add_argument('--adas', dest='adas', action='store_true',
                    help='Specify whether adas output XLSX or not')
args = parser.parse_args()
acc_min = 0.6
acc_max = 1.0
loss_min = 5e-4
loss_max = 10

plt.figure(1, figsize=(5, 5))

if 'adas' in str(args.file).lower() and not args.adas:
    print("Warning: This appears to be AdaS data, but the --adas argument " +
          "was not set")
total_acc_data_vec = list()
if Path(args.file).is_dir():
    for file_name in Path(args.file).iterdir():
        df = pd.read_excel(file_name)
        df = df.T
        if "adas" in file_name.lower():
            acc_data_vec = np.asarray(df.iloc[12::12, 1])
        else:
            acc_data_vec = np.asarray(df.iloc[9::9, 1])
        total_acc_data_vec.append(acc_data_vec)
    acc_data_vec = np.mean(total_acc_data_vec, axis=1)
else:
    df = pd.read_excel(args.file)
    df = df.T
    if "adas" in args.file.lower():
        acc_data_vec = np.asarray(df.iloc[12::12, 1])
    else:
        acc_data_vec = np.asarray(df.iloc[9::9, 1])
        total_acc_data_vec.append(acc_data_vec)
plt.plot(np.array(range(1, len(acc_data_vec) + 1)),
         acc_data_vec, '-',
         color='r')
plt.ylim((acc_min,
          acc_max))

plt.xlim((1, 250))
plt.ylabel('Test Accuracy', fontsize=16)
plt.xlabel('Epoch - (t)', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.gca().legend([args.legend_label], prop={"size": 9}, loc="lower right",
                 bbox_to_anchor=(0.98, 0.02), borderaxespad=0., ncol=2)
plt.grid(True)
export_name = f'test_accuracy_{args.legend_label}.png'
plt.savefig(export_name, dpi=300, bbox_inches='tight')
plt.close()

plt.figure(1, figsize=(5, 5))
total_err_vec = list()
if Path(args.file).is_dir():
    for file_name in Path(args.file).iterdir():
        df = pd.read_excel(file_name)
        df = df.T
        if "adas" in file_name.lower():
            error_data_vec = np.asarray(df.iloc[1::12, 1])
        else:
            error_data_vec = np.asarray(df.iloc[1::9, 1])
        total_err_vec.append(error_data_vec)
    error_data_vec = np.mean(total_err_vec, axis=1)
else:
    df = pd.read_excel(args.file)
    df = df.T
    if "adas" in args.file.lower():
        error_data_vec = np.asarray(df.iloc[1::12, 1])
    else:
        error_data_vec = np.asarray(df.iloc[1::9, 1])

plt.plot(np.array(range(1, len(error_data_vec) + 1)),
         error_data_vec, '-',
         color='r')
plt.ylim((loss_min,
          loss_max))
plt.xlim((1, 250))
plt.yscale('log', basey=10)
plt.ylabel('Training Loss', fontsize=16)
plt.xlabel('Epoch - (t)', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.gca().legend([args.legend_label], prop={"size": 9}, loc="upper right",
                 bbox_to_anchor=(
    0.98, 0.98), borderaxespad=0., ncol=2)
plt.grid(True)
export_name = f'training_loss_{args.legend_label}.png'
plt.savefig(export_name, dpi=300, bbox_inches='tight')
plt.close()
