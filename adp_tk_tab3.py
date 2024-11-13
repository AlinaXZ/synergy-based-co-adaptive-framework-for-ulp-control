# Tab3 - graph
import os
import json
import shutil

from myo_input_tk import *
from unity_sim_tk import *
from controller_input_tk import *

from adpframe_tk import *

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as messagebox
import tkinter.filedialog as filedialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
    
    
def show_graph(self, data, figsize, xlim, ylim):
    # data: shape(channel_num, n)
    # figsize =(6,4)
    # xlim, ylim = [0, 40], [-100, 100]
    channel_num = data.shape[0]
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    x = list(range(data.shape[1]))
    line = [None] * channel_num

    for i in range(channel_num):
        line[i], = ax.plot(x, np.zeros(data.shape[1]))
    
    return fig, line, ax


def show_graph_emg(self, data, line, show):
    # data: shape(channel_num, n), type numpy
    channel_num = data.shape[0]
    x = list(range(data.shape[1]))

    for i in range(channel_num):
        if show[i].get():
            line[i].set_data(x, data[i])  # 更新data数据
        else:
            line[i].set_data(x, np.zeros(data.shape[1])) 
    self.graph0.draw()


def show_graph_imu(self, data_que, line):
    # data_que shape(4, n), type numpy
    x = list(range(data_que.shape[1]))

    for i in range(4):
        line[i].set_data(x, data_que[i])  # 更新data数据
    self.graph1.draw()


def show_graph_emg_ft(self, data, line, show_1, var_cb):
    # data: shape(144, n), type numpy
    x = list(range(data.shape[1]))
    num_ft = int(data.shape[0]/8)

    for i in range(8):
        if show_1[i].get():
            line[i].set_data(x, data[var_cb+num_ft*i])  # 更新data数据
        else:
            line[i].set_data(x, np.zeros(data.shape[1])) 
    self.graph2.draw()


# adpframe paras
def adp_paras(self):

    self.adp1.frame = self.paras['window_size']
    # AdpFramework.cluster_range_low = int(raw_data.shape[0]/self.paras['cluster_range_scale'][0])
    # AdpFramework.cluster_range_high = int(raw_data.shape[0]/self.paras['cluster_range_scale'][0])
    self.adp1.c_min_len = self.paras['c_min_len']

    # pre-defined
    self.adp1.w1 = self.paras['w1']
    self.adp1.w2 = self.paras['w2']
    self.adp1.p_th = self.paras['p_threshold']
    self.adp1.eps = self.paras['eps']


def confirm_copy(self):
    name_add = self.e4_1.get()
    txt_print = self.txt4_1.get()
    try:
        restore_path = self.store_path.copy()
        source_folder = self.store_path['folder_path']
        
        for file_obj in ['cluster.json', 'myo_data.json']:
            file_path = os.path.join(source_folder, file_obj)
            if os.path.exists(file_path):
                file_name, file_extend = os.path.splitext(file_obj)
                new_name = file_name + name_add + file_extend
                file_path_new = os.path.join(source_folder, new_name)
                if os.path.exists(file_path_new):
                    result = messagebox.askyesno('Info', new_name + "'\n already exists. Do you want to overwrite it?")
                else:
                    result = True
                if result:
                    shutil.copyfile(file_path, file_path_new)
                    txt_print += "\n" + file_obj + " has been copied as " + new_name + "."
                else:
                    txt_print += "\n" + "You cancelled overwrite "+ new_name + "."
                    break
            else:
                txt_print += "\n" + file_obj + " does not exist."
    except:
        txt_print += "\nPlease comfirm the store path."
    finally:
        self.print_tk(self.txt4_1, self.t4_3, txt_print)
    


def confirm_load(self):
    name_add = self.e4_1.get()
    txt_print = self.txt4_1.get()
    try:
        restore_path = self.store_path.copy()
        source_folder = self.store_path['folder_path']
        
        for file_obj in ['cluster.json', 'myo_data.json']:
            file_name, file_extend = os.path.splitext(file_obj)
            new_name = file_name + name_add + file_extend
            file_path = os.path.join(source_folder, file_obj)
            file_path_new = os.path.join(source_folder, new_name)
            if os.path.exists(file_path_new):
                if os.path.exists(file_path):
                    result = messagebox.askyesno('Info', file_obj + "'\n already exists. Do you want to overwrite it?")
                else:
                    result = True
                if result:
                    shutil.copyfile(file_path_new, file_path)
                    txt_print += "\n" + new_name + " has been load to " + file_obj + "."
                else:
                    txt_print += "\n" + "You cancelled load "+ new_name + "."
                    break
            else:
                txt_print += "\n" + new_name + " does not exist."
    except:
        txt_print += "\nPlease comfirm the store path."
    finally:
        self.print_tk(self.txt4_1, self.t4_3, txt_print)