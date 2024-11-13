import os
import json

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


# Func - Tab1
def import_sets(self):
    import_sets(self)


def import_paras(self):
    result = messagebox.askyesno('Info', "Import parameters now? All inputs will be updated.\nPlease save your inputs first.")
    if result:
        for en, para in [
            (self.e2_1, self.paras['window_size']), 
            (self.e2_2_1, self.paras['cluster_range_scale'][0]),
            (self.e2_2_2, self.paras['cluster_range_scale'][1]),
            (self.e2_3, self.paras['c_min_len']),
            (self.e2_4, self.paras['w1']),
            (self.e2_5, self.paras['w2']),
            (self.e2_6, self.paras['p_threshold']),
            (self.e2_7, self.paras['eps']),
            (self.e2_8, self.paras['time_interval']),
            (self.e2_9, self.paras['modify_scale']),
            (self.e2_10, self.paras['output_scale'])]:
            en.delete(0, tk.END)
            en.insert(0, para)
        messagebox.showinfo("info", "All parameters updated.")
    else:
        messagebox.showinfo("info", "You canceled the import.")


def get_settings(self):
    settings = {
        'folder': self.e0_1.get(),
        'name': self.e0_2.get(),
        'myo_sdk': self.e1_1.get(),
        'unity_port': self.e2_0.get()
    }
    return settings


def import_default_set(self):
    if not os.path.exists(self.default_set_path):
        messagebox.showinfo("info", "'" + self.default_set_path + "'\nnot found. Put the file into the working directory:\n'" + os.getcwd() + "'")
    else:
        try:
            with open(self.default_set_path, 'r') as f:
                self.sets = json.load(f)['settings']
            self.import_sets()
        except:
            messagebox.showinfo("Import Error", "Can't import setup information from\n'" + self.default_set_path + "'\n Please select the proper JSON file.")

def select_file_set(self):
    file_path = filedialog.askopenfilename(filetypes=[("JSON", ".json")])
    if file_path:
        try:
            with open(file_path, 'r') as f:
                self.sets = json.load(f)['settings']
            self.import_sets()
        except:
            messagebox.showinfo("Import Error", "Can't import setup information from\n'" + file_path + "'\n Please select the proper JSON file.")


def save_as_default_set(self):
    settings = self.get_settings()
    default_path = self.default_set_path
    try:
        with open(default_path, 'r+') as f:
            default_load = json.load(f)
            default_load['settings'] = settings
            f.seek(0)
            f.truncate()
            json.dump(default_load, f)
    except:
        default_load = {'settings': settings}
        with open(default_path, 'w') as f:
            json.dump(default_load, f)
    finally:
        messagebox.showinfo("info", "Default settings updated.")


def save_as_set(self):
    settings = self.get_settings()
    file_path = filedialog.asksaveasfilename(title=u'Save settings', filetypes=[("JSON", ".json")])
    if file_path is not None:
        if file_path.endswith('.json')==False:
            file_path += '.json'
        with open(file_path, 'w') as f:
            default_load = {'settings': settings}
            f.seek(0)
            f.truncate()
            json.dump(default_load, f)
        messagebox.showinfo("info", "New settings saved to\n'" + file_path + "'")


def get_paras(self):
    dict_para = {
        'window_size': int(self.e2_1.get()),
        'modify_scale': int(self.e2_9.get()),
        'output_scale': int(self.e2_10.get()),
        'cluster_range_scale': (int(self.e2_2_1.get()), int(self.e2_2_2.get())),
        'c_min_len': int(self.e2_3.get()),
        'w1': float(self.e2_4.get()),
        'w2': float(self.e2_5.get()),
        'p_threshold': float(self.e2_6.get()),
        'eps': float(self.e2_7.get()),
        'time_interval': int(self.e2_8.get()),
        'num_train': int(self.e2_11.get())
    }
    return dict_para


def import_default_para(self):
    default_path = self.default_para_path
    if not os.path.exists(default_path):
        working_dir = os.getcwd()
        messagebox.showinfo("info", "'" + default_path + "'\nnot found. Put the file into the working directory:\n'" + working_dir + "'")
    else:
        try:
            with open(default_path, 'r') as f:
                self.paras = json.load(f)['paras']
            self.import_paras()  
        except:
            messagebox.showinfo("Import Error", "Can't import setup information from\n'" + default_path + "'\n Please select the proper JSON file.")

def select_file_para(self):
    file_path = filedialog.askopenfilename(filetypes=[("JSON", ".json")])
    if file_path:
        try:
            with open(file_path, 'r') as f:
                self.paras = json.load(f)['paras']
            self.import_paras()
        except:
            messagebox.showinfo("Import Error", "Can't import setup information from\n'" + file_path + "'\n Please select the proper JSON file.")


def save_as_default_para(self):
    paras = self.get_paras()
    default_path = self.default_para_path
    try:
        with open(default_path, 'r+') as f:
            default_load = json.load(f)
            default_load['paras'] = paras
            f.seek(0)
            f.truncate()
            json.dump(default_load, f)
    except:
        default_load = {'paras': paras}
        with open(default_path, 'w') as f:
            json.dump(default_load, f)
    finally:
        messagebox.showinfo("info", "Default parameters updated.")


def save_as_para(self):
    paras = self.get_paras()
    file_path = filedialog.asksaveasfilename(title=u'Save parameters', filetypes=[("JSON", ".json")])
    if file_path is not None:
        if file_path.endswith('.json')==False:
            file_path += '.json'
        with open(file_path, 'w') as f:
            default_load = {'paras': paras}
            f.seek(0)  # 这两行代买顺序不能变化，用于清空原文件内容
            f.truncate() # 这两行代买顺序不能变化，用于清空原文件内容
            json.dump(default_load, f)
        messagebox.showinfo("info", "New parameters saved to\n'" + file_path + "'")

# Func - tab2 - frame0
def confirm_storage(self):
    path = self.e0_1.get() + self.e0_2.get() + "/"
    if not os.path.exists(path):
        result = messagebox.askyesno('msg', path + "'\ndoes not exist. Create new folder?")
        if result:
            os.makedirs(path)
            txt_print = "New folder created. Data will be stored in:\n" + "'" + path + "'"
            ready_to_store = True
        else:
            txt_print = path + "'\ndoes not exist. Please confirm again."
            ready_to_store = False
    else:
        txt_print = "Data will be stored in:\n" + "'" + path + "'"
        ready_to_store = True
    self.txt0_1.set(txt_print)

    if ready_to_store:
        self.store_path = {
            "folder_path": path,
            "myo_data_path": path + "myo_data.json",
            "graph_path": path + "graph.json",
            "cluster_path": path + "cluster.json",
            "clf_path": path + "clf.pkl"
        }
        self.t0_3.config(fg="green")

        # plot graph
        if os.path.exists(self.store_path['graph_path']):
            with open(self.store_path['graph_path'], 'r') as f:
                row_col_prob = np.array(json.load(f)['adjacency_mat'])

            # mc = plot_graph_sparse4(row_col_prob)
            # fig4, ax4 = mc.draw(figsize=(5,4))
            # graph4 = FigureCanvasTkAgg(fig4, master=self.tab3_4)
            # canvas4 = graph4.get_tk_widget()
            # canvas4.grid(row=0, column=0)
            # graph4.draw()
    else:
        self.t0_3.config(fg="red")

    # Read default parameters
    if ready_to_store:
        try:
            with open("default_paras.json", 'r') as f:
                self.paras = json.load(f)['paras']
            self.print_tk(self.txt0_2, self.t0_4, "Parameters confirmed.", color="green")
        except:
            self.print_tk(self.txt0_2, self.t0_4, "Cannot read parameters from storage.", color="red")

# Tab2 - frame1-3
def connect_myo(self):
    self.print_tk(self.txt1_1, self.t1_3, "Connecting Myo...\n ")
    self.sdk_path = self.e1_1.get()
    hub, listener, txt_print, connect = initialize_myo_tk(self.sdk_path)
    self.txt1_1.set(txt_print)
    if connect:
        self.t1_3.config(fg="green")
    else:
        self.t1_3.config(fg="red")



def check_vr(self):

    if self.unity_joint == None:
        connect = False
        self.txt3_1.set("Please first connect Unity\n ")
    else:
        txt_print, connect = check_controller_tk(self.unity_joint)
        self.txt3_1.set(txt_print)
    if connect:
        self.t3_3.config(fg="green")
    else:
        self.t3_3.config(fg="red")

def confirm_para(self):
    try:
        self.paras = self.get_paras()
        self.adp_paras()
        self.print_tk(self.txt2_9, self.t2_9, "Parameters confirmed.", color="green")
    except:
        self.print_tk(self.txt2_9, self.t2_9, "Please fill in all parameters.", color="red")