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


def new_train(self):

    try:
        # with open(self.store_path['myo_data_path'], 'r', encoding="utf8") as f:
        #     data_df = pd.read_json(f, precise_float=True)
        #     last_synergy  = np.array(data_df.iloc[-1]['s'])

        with open(self.store_path['graph_path'], 'r') as f:
            adja_mat = np.array(json.load(f)['adjacency_mat'])

        txt_print = "Graph data was read."
        self.print_tk(self.txt4_1, self.t4_3, txt_print)


        with open(self.store_path['cluster_path'], 'r', encoding="utf8") as f:
            cluster_read = pd.read_json(f, precise_float=True)
            emg_ft_mean = np.array(cluster_read['emg_ft_mean'].tolist())
            synergy_mean = np.array(cluster_read['s_mean'].tolist())

        txt_print += "\nCluster data was read."
        self.print_tk(self.txt4_1, self.t4_3, txt_print)

        data_ready = True

    except:
        txt_print = "Path Error/Data Error. Cannot read data."
        self.print_tk(self.txt4_1, self.t4_3, txt_print)

        data_ready = False
    
    if data_ready:

        self.new_train_start(cluster_read, adja_mat, emg_ft_mean, synergy_mean, txt_print)


def new_train_start(self, cluster_read, adja_mat, emg_ft_mean, synergy_mean, txt_print):
    myo.init(sdk_path=self.sdk_path)
    hub = myo.Hub()
    # 队列最长40
    listener = DataCollector(n=40)

    # 初始user_input = True
    d_pre_l, d_pre_r = True, True

    # 用于存储最终数据
    data_df2 = pd.DataFrame()

    # only for plot
    imu_queue = deque(maxlen=self.paras['window_size'])
    emg_ft_que = deque(maxlen=self.paras['window_size'])
    emg_ft_que.extend(np.zeros((40, 144)).tolist())

    # 初始synergy
    if hub.run(listener.on_event, self.paras['time_interval']):
        pass

    try:
        # 监听键盘键入
        with keyboard.Events() as k_events:

            txt_print += "\nStart collecting data..."
            self.print_tk(self.txt4_1, self.t4_3, txt_print)
            self.print_unity(text1="Start collecting data...", text2="Please wait.", color="black")

            ready = False
            start_pressed = False

            # 暂停50ms(sample_rate=20)
            while hub.run(listener.on_event, self.paras['time_interval']):
                listener.update_que()
                emg_que = np.array(listener.emg_data_queue)
                imu = np.array([x for x in listener.imu_data_curr])

                # only for plot
                imu_queue.append(imu.tolist())
                
                # using keyboard input
                key_input, break_loop = listen_keyboard(k_events)

                # use VR Controller input
                if self.unity_joint != None:
                    controller = self.unity_joint.ReturnInput()
                else:
                    controller = 0

                # if controller!=0:
                #     print(controller)
                    # 1-Menu, 2-Trig, 3-Pad, 4-PadLeft, 5-PadRight

                if break_loop or controller==1:
                    break
                
                if key_input == "s" or controller==3:
                    start_pressed = True
                    self.print_tk(self.txt4_1, self.t4_3, "---Start---")
                    self.print_unity(text1="Start.", text2="Press 'Menu' to stop.", color="black")

                user_input = True
                if key_input == "h" or controller==2:
                    user_input = False
                

                # 使用输入
                # d_pre_l, val_input_l = detect_input(vrsystem, d_pre_l, left_id)
                # d_pre_r, val_input_r = detect_input(vrsystem, d_pre_r, right_id)
                # if val_input_l != 0 or val_input_r != 0:
                #     user_input = False
                # else:
                #     user_input = True


                if emg_que.shape[0] < self.adp1.frame:
                    # initial training start
                    # s_curr = s0
                    node_last = None ## ----- different from initial training

                
                if emg_que.shape[0] == self.adp1.frame-1:
                    ready = True
                    self.print_tk(self.txt4_1, self.t4_3, "Ready to start", "green")
                    self.t4_3.config(fg="black")
                    self.print_unity(text1="Ready to start.", text2="Press 'Pad(Center)' to confirm start.", color="green")
                
                # 当emg长度足够时
                if start_pressed and emg_que.shape[0] == self.adp1.frame:

                    # emg_fl = self.adp1.filter_signal(emg_que)

                    # # plot emg_fl
                    # self.show_graph_emg(emg_fl.T, line0, show)
                    
                    # self.show_graph_imu(np.array(imu_queue).T, line1)

                    emg_ft = self.adp1.extract_features_dynamic(emg_que, self.adp1.eps, axis=0)
                    emg_ft_norm = normalize(emg_ft.reshape(-1, 1), axis=0, norm='max').T[0]

                    # only for plot
                    emg_ft_que.append(emg_ft_norm.tolist())
                    # self.show_graph_emg_ft(np.array(emg_ft_que).T, line2, show2_1, var_cb.get())
                    
                    # ----- different from initial training -----
                    
                    #pred_id, pred_prob = pred_next_node(emg_ft_norm, cluster_read, node_last, graph_read, w, eps)
                    pred_id, pred_prob = self.adp1.predict_next_node(emg_ft_norm, cluster_read, node_last, adja_mat)
                    s_curr = synergy_mean[pred_id]
                    
                    # m_curr, s_curr, imu_last, m_store, motion_angle = training_loop3(imu, imu_last, s_curr, m_store, motion_angle, 
                    # user_input=user_input, scale1=self.paras['modify_scale'], scale2=self.paras['output_scale'])
                
                    
                    # unity_joint.SendRotate2(trans_v = MyVector3(0, m_curr[0], m_curr[1]))

                    node_last = pred_id
                    if user_input==False:
                        # "wrong"
                        # print(s_curr)
                        # print(dict_data['reach'])
                        # print(last_synergy)
                        s_curr[:,1] = -s_curr[:,1]
                        # s_curr = synergy_modify(s_curr, dict_data['reach'], last_synergy, scale=self.paras['modify_scale'], round=3) #这个应该有所改变
                    # ------------

                    m_curr = motion_ctrl(imu, s_curr) * self.paras['output_scale']

                    if self.unity_joint != None:
                        self.unity_joint.SendRotate2(trans_v = MyVector3(0, 0, m_curr[1]))

                    # # store data as dataframe
                    data2 = pd.DataFrame([{"emg_ft":np.round(emg_ft_norm, 6), "s":np.round(s_curr, 6), "user":user_input, "pred":pred_id}])
                    data_df2 = pd.concat([data_df2, data2], axis=0, ignore_index=True)

                    # # initial training
                    # m_curr, s_curr = training_loop(imu, s_curr, user_input=user_input)

                    # print("user_input:",user_input ,"pred_id:", pred_id, "prob:", np.round(pred_prob, 2) , "---output motion: ", np.round(m_curr, 2))
                    txt_print = "user_input: " + str(user_input)
                    txt_print += "\npred_id: " + str(pred_id)
                    txt_print += "\nprob: " + str(np.round(pred_prob, 2))
                    txt_print += "\nimu: " + str(np.round(imu,3))
                    txt_print += "\nsynergy: " + str(np.round(s_curr[0],2))
                    txt_print += "\noutput motion: " + str(np.round(m_curr, 2))
                    
                    self.print_tk(self.txt4_1, self.t4_3, txt_print)
        
        new_train_done = True

    except:
        txt_print += "KeyboardInterrupt ..."
        self.print_tk(self.txt4_1, self.t4_3, txt_print)
        new_train_done = False

    finally:
        hub.stop()

        txt_print = "Training end. Data is being stored..."
        self.print_tk(self.txt4_1, self.t4_3, txt_print)
        self.print_unity(text1="Training end. Data is being stored...", text2="Please wait...", color="black")

        if new_train_done:
            self.new_train_offline(txt_print, data_df2)

def new_train_offline(self, txt_print, data_df2):
    try:
        # new_path = self.store_path['folder_path']+ "myo_data1.json"
        data_df2.to_json(self.store_path['myo_data_path'], orient='columns')
        txt_print += "\nTraining data stored in " + self.store_path['myo_data_path']
        self.print_tk(self.txt4_1, self.t4_3, txt_print)
        self.print_unity(text1="Training data stored.", text2="Please wait...", color="black")
        
        try:
            # clustering
            emg_ft = np.array(data_df2['emg_ft'].tolist())
            synergy = np.array(data_df2['s'].tolist())
            user_input_list  = data_df2['user'].tolist()

            # 这里要改，这个变量不该作为属性
            self.adp1.cluster_range_low = int(data_df2.shape[0]/self.paras['cluster_range_scale'][0])
            self.adp1.cluster_range_high = int(data_df2.shape[0]/self.paras['cluster_range_scale'][1])+1

            df_c, pos_list, pos_list_converted, appear_list, appear_list_full = self.adp1.ft_clustering(emg_ft)
            c_mean_df, c_marker_df = self.adp1.cal_c_mean_marker(emg_ft, synergy, df_c, appear_list_full, user_input_list)

        
            #组合dataframe
            node_df2 = pd.concat([df_c, c_mean_df, c_marker_df], axis=1, ignore_index=False)
        
            row_col_prob = self.adp1.build_graph(pos_list_converted, plot=False)
    

            self.testdic = [cluster_read, node_df2, adja_mat, row_col_prob]
        

            graph_final, c_final = self.adp1.modify_KB(cluster_read, node_df2, adja_mat, row_col_prob)

    
            transform_prob = transform_graph(graph_final)
        

            # 写入json
            c_final.to_json(self.store_path['cluster_path'], orient='columns')

            txt_print += "\nCluster data stored in " + self.store_path['cluster_path']
            self.print_tk(self.txt4_1, self.t4_3, txt_print)
            self.print_unity(text1="Cluster data stored.", text2="Please wait...", color="black")

            # 写入json
            dict_graph = {
                'adjacency_mat': graph_final.tolist(),
                'transition_prob': transform_prob
            }
            with open(self.store_path['graph_path'], 'w') as f:
                json.dump(dict_graph, f)

            txt_print += "\nGraph data stored in " + self.store_path['graph_path']

            txt_print += "\nNew Training Finish."
            self.print_tk(self.txt4_1, self.t4_3, txt_print)
            self.print_unity(text1="Graph data stored. New Training Finish.", text2="Please select the mode.", color="black", id_init=3)

        # except Exception as e:
        except:
            # print(e)
            txt_print += "\nClustering Error."
            self.print_tk(self.txt4_1, self.t4_3, txt_print)

        # try:    
        #     # plot graph
        #     mc = plot_graph_sparse4(graph_final)
        #     fig4, ax4 = mc.draw(figsize=(5,4))
        #     graph4 = FigureCanvasTkAgg(fig4, master=tab3_4)
        #     canvas4 = graph4.get_tk_widget()
        #     canvas4.grid(row=0, column=0)
        #     graph4.draw()

        # except:
        #     txt_print += "\nPlotting Graph Error."
        #     self.print_tk(self.txt4_1, self.t4_3, txt_print)

    except:
        txt_print += "\nPath Error. Please check if the storage path has been set."
        self.print_tk(self.txt4_1, self.t4_3, txt_print)        