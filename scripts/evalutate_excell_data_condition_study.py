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

lr_method = 'Slope'
excel_name = '.xlsx'
df = pd.read_excel(excel_name)
df = df.T

loss_data = df.iloc[1::9, 1]
input_S_data = df.iloc[2::9, :]
output_S_data = df.iloc[3::9, :]
input_rank_data = df.iloc[4::9, :]
output_rank_data = df.iloc[5::9, :]
input_condition_data = df.iloc[6::9, :]
output_condition_data = df.iloc[7::9, :]
learning_rate_data = df.iloc[8::9, :]
acc_data = df.iloc[9::9, 1]

# input_rank_data = df.iloc[1::8, :]
# output_rank_data = df.iloc[2::8, :]
# fc_rank_data = df.iloc[3::8, :]
# lr_val = df.iloc[4::8, :]
# slope_conv_data = df.iloc[5::8, :]
# slope_fc_data = df.iloc[6::8, :]
# acc_data = df.iloc[7::8, 1]
# loss_data = df.iloc[8::8, 1]

plt.figure(1, figsize=(20, 8.5))
plt.suptitle('plot title')
for iteration_layer in range(input_rank_data.shape[1]):
    plt.subplot(np.ceil(np.sqrt(input_rank_data.shape[1])), np.ceil(
        np.sqrt(input_rank_data.shape[1])), iteration_layer+1)
    plt.plot(np.array(range(1, output_rank_data.shape[0] + 1)), np.asarray(
        output_rank_data.iloc[:, iteration_layer]), color='b')
    plt.plot(np.array(range(1, input_rank_data.shape[0] + 1)), np.asarray(
        input_rank_data.iloc[:, iteration_layer]), color='k')
    # plt.plot(np.array(range(1, rank_accelerate_data.shape[0] + 1)), np.asarray(rank_accelerate_data.iloc[:, iteration_layer]), color='r')
    # plt.plot(np.array(range(1, rank_velocity_data.shape[0] + 1)), np.asarray(rank_velocity_data.iloc[:, iteration_layer]), color='m')
    plt.plot(np.array(range(1, learning_rate_data.shape[0] + 1)), np.asarray(
        learning_rate_data.iloc[:, iteration_layer]), color='c')
    plt.plot(
        np.array(range(1, acc_data.shape[0] + 1)), np.asarray(acc_data), color='g')
    plt.ylabel('Tensor Rank')
    plt.xlabel('Epoch')
    plt.title('Layer '+str(iteration_layer+1))
    plt.gca().legend(('input Rank', 'Output Rank',
                      'learning rate', 'Test Accuracy'), prop={"size": 5})
    plt.grid(True)
    plt.ylim((-.2, 1))
    plt.xlim((0, 200))

# max_rank = (input_rank_data.values + output_rank_data.values)/2
# max_rank = np.max(max_rank, axis=0)

max_rank_in = np.max(input_rank_data.values, axis=0)
max_rank_out = np.max(output_rank_data.values, axis=0)
max_rank = np.maximum(max_rank_in, max_rank_out)

# conv_arch = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
# np.minimum(np.round(np.multiply(max_rank, conv_arch)*1.5), conv_arch)
# plt.show()
plt.savefig(excel_name+'.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(1, figsize=(20, 8.5))
plt.suptitle('plot title')
for iteration_layer in range(input_rank_data.shape[1]):
    plt.subplot(np.ceil(np.sqrt(input_condition_data.shape[1])), np.ceil(
        np.sqrt(input_condition_data.shape[1])), iteration_layer+1)
    plt.plot(np.array(range(1, input_condition_data.shape[0] + 1)), np.asarray(
        input_condition_data.iloc[:, iteration_layer]), color='b')
    plt.plot(np.array(range(1, output_condition_data.shape[0] + 1)), np.asarray(
        output_condition_data.iloc[:, iteration_layer]), color='k')
    plt.ylabel('Tensor Rank')
    plt.xlabel('Epoch')
    plt.title('Layer '+str(iteration_layer+1))
    plt.gca().legend(('input-condition', 'output-condition'), prop={"size": 5})
    plt.grid(True)
    # plt.ylim((0, 100))
    plt.xlim((0, 200))

plt.savefig(excel_name+'_condition.png', dpi=300, bbox_inches='tight')
plt.close()


plt.figure(4, figsize=(20, 8.5))
plt.suptitle('plot title')
for iteration_layer in range(input_rank_data.shape[1]):
    ax = plt.subplot(np.ceil(np.sqrt(input_rank_data.shape[1])), np.ceil(
        np.sqrt(input_rank_data.shape[1])), iteration_layer+1)
    plt.plot(np.array(
        range(1, input_condition_data.shape[0]+1)), np.asarray(loss_data), color='b')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Layer '+str(iteration_layer+1))
    plt.grid(True)
    plt.ylim((1e-4, 1))
    ax.set_yscale('log')
    plt.xlim((0, 200))

# plt.show()
plt.savefig(excel_name+'_Loss.png', dpi=300, bbox_inches='tight')
plt.close()
