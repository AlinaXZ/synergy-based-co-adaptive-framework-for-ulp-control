import os
import json
import pickle

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


def deployment(self):

    # # unity
    # unity_joint = dict_data['unity_joint']

    # try:
        # with open(self.store_path['graph_path'], 'r') as f:
        #     graph_read = json.load(f)
        #     adja_mat = np.array(graph_read['adjacency_mat'])
        #     # trans_prob = np.array(graph_read['transition_prob'])

        # txt_print = "Graph data was read."
        # self.print_tk(self.txt4_1, self.t4_3, txt_print)

    with open(self.store_path['cluster_path'], 'r', encoding="utf8") as f:
        cluster_read = pd.read_json(f, precise_float=True)
        # emg_ft_mean = np.array(cluster_read['emg_ft_mean'].tolist())
        # synergy_mean = np.array(cluster_read['s_mean'].tolist())
    # with open(self.store_path['clf_path'], 'rb') as f:
    #     clf_read = pickle.load(f)
    txt_print = "Learning data was read."
    self.print_tk(self.txt4_1, self.t4_3, txt_print)

    data_ready = True

    # except:
    #     txt_print = "Path Error/Data Error. Cannot read data."
    #     self.print_tk(self.txt4_1, self.t4_3, txt_print)

    #     data_ready = False
    
    if data_ready:
        myo.init(sdk_path=self.sdk_path)
        hub = myo.Hub()
        # 队列最长 self.paras['window_size']
        listener = DataCollector(n=self.paras['window_size'])


        # 用于存储最终数据
        data_df3 = pd.DataFrame()
        motion_angle = np.array([0, 0])

        # only for plot
        imu_queue = deque(maxlen=self.paras['window_size'])
        emg_ft_que = deque(maxlen=self.paras['window_size'])
        emg_ft_que.extend(np.zeros((self.paras['window_size'], 80)).tolist())

        imu_last = np.zeros(4)
        emg_ft_last = np.zeros(80)

        # 初始synergy
        if hub.run(listener.on_event, self.paras['time_interval']):
            pass

        try:
            # 监听键盘键入
            with keyboard.Events() as k_events:
                txt_print += "Start collecting data..."
                self.print_tk(self.txt4_1, self.t4_3, txt_print)
                self.print_unity(text1="Start collecting data...", text2="Please wait.", color="black")

                ready = False
                start_pressed = False

                # 暂停50ms(sample_rate=20)
                while hub.run(listener.on_event, self.paras['time_interval']):
                    listener.update_que()
                    emg_que = np.array(listener.emg_data_queue)
                    imu = np.array([x for x in listener.imu_data_curr])
                    imu_velocity = imu - imu_last
                    imu_last = imu.copy()

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


                    # #使用输入
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

                        # plot emg, imu
                        self.show_graph_emg(emg_que.T, self.line0, self.show)
                        self.show_graph_imu(np.array(imu_queue).T, self.line1)

                        emg_ft = self.adp1.extract_features_dynamic_selected(emg_que, self.adp1.eps, axis=0)
                        emg_ft_add_v = np.hstack((emg_ft, emg_ft-emg_ft_last))
                        emg_ft_last = emg_ft.copy()
                        emg_ft_add_v_norm = normalize(emg_ft_add_v.reshape(-1, 1), axis=0, norm='max').T[0]


                        # only for plot
                        emg_ft_norm = normalize(emg_ft.reshape(-1, 1), axis=0, norm='max').T[0]
                        emg_ft_que.append(emg_ft_norm.tolist())
                        self.show_graph_emg_ft(np.array(emg_ft_que).T, self.line2, self.show2_1, self.var_cb.get())
                        
                        
                        # emg_ft = self.adp1.extract_features_dynamic(emg_que, self.adp1.eps, axis=0)
                        # emg_ft_norm = normalize(emg_ft.reshape(-1, 1), axis=0, norm='max').T[0]
                        # emg_ft_norm_velocity = emg_ft_norm - emg_ft_norm_last
                        # emg_ft_norm_last = emg_ft_norm.copy()

                        # # only for plot
                        # emg_ft_que.append(emg_ft_norm.tolist())
                        # self.show_graph_emg_ft(np.array(emg_ft_que).T, self.line2, self.show2_1, self.var_cb.get())
                        
                        # ----- different from initial training -----
                        
                        # pred_id, pred_prob = pred_next_node(emg_ft_norm, cluster_read, node_last, graph_read, w, eps)
                        # pred_id, pred_prob = pred_next_node_in(emg_ft_norm, cluster_read)
                        # knn_id, pred_id, pred_prob = self.adp1.pred_knn(emg_ft_norm, clf_read, cluster_read)
                        # knn_id, pred_id, pred_prob = self.adp1.pred_knn(emg_ft_add_v_norm, clf_read, cluster_read)
                        pred_id, pred_prob = pred_next_node_in(emg_ft_add_v_norm, cluster_read)
                        # pred_id, pred_prob = self.adp1.predict_next_node(emg_ft_norm, cluster_read, node_last, adja_mat)
                        # s_curr = synergy_mean[pred_id]
                        # m_curr = motion_ctrl(imu, s_curr) * self.paras['output_scale']
                        # unity_joint.SendRotate2(trans_v = MyVector3(0, m_curr[0], m_curr[1]))
                        #  
                        s_pred = np.array(cluster_read.loc[pred_id,'synergy'])
                        imu_v_with_s = np.hstack((imu_velocity, np.ones(1)))
                        m_pred = np.dot(s_pred, imu_v_with_s)

                        # if knn_id > 10 and knn_id < 20:
                        #     m_pred = np.array([0,2])
                        # elif knn_id > 20:
                        #     m_pred = np.array([0,-2])
                        # else:
                        #     m_pred = np.array([0,0])

                        motion_angle = motion_angle + m_pred
                        if np.all(np.vstack((motion_angle >= np.array([-90,0]), motion_angle <= np.array([90,120])))):
                            if self.unity_joint != None:
                                self.unity_joint.SendRotate2(trans_v = MyVector3(0, 0, m_pred[1]))
                                self.unity_joint.SendRotate3(trans_v = MyVector3(0, m_pred[0], 0))
                        else:
                            motion_angle = motion_angle - m_pred

                        # node_last = pred_id
                        # ------------

                        # # store data as dataframe
                        # data3 = pd.DataFrame([{"emg_ft":np.round(emg_ft_norm, 6), "s":np.round(s_curr, 6), "pred":pred_id}])
                        # data_df3 = pd.concat([data_df3, data3], axis=0, ignore_index=True)

                        # # initial training
                        # m_curr, s_curr = training_loop(imu, s_curr, user_input=user_input)

                        # print("user_input:",user_input ,"pred_id:", pred_id, "prob:", np.round(pred_prob, 2) , "---output motion: ", np.round(m_curr, 2))
                        # txt_print = "user_input: " + str(user_input)
                        txt_print = "pred_id: " + str(pred_id)
                        txt_print += " prob: " + str(np.round(pred_prob, 2))
                        txt_print += "\nimu: " + str(np.round(imu,3))
                        # txt_print += "\nimu velocity: " + str(np.round(imu_velocity, 4))
                        txt_print += "\nsynergy: " + str(np.round(s_pred[0,:3],2)) + "..."
                        txt_print += "\noutput motion: " + str(np.round(m_pred, 2))
                        txt_print += "\nangle_sum: " + str(np.round(motion_angle, 2))
                        
                        self.print_tk(self.txt4_1, self.t4_3, txt_print)
                        if self.unity_joint != None:
                            self.unity_joint.SetText3(text3=txt_print)

        except KeyboardInterrupt:
            txt_print += "KeyboardInterrupt ..."
            self.print_tk(self.txt4_1, self.t4_3, txt_print)

        finally:
            hub.stop()

            self.print_tk(self.txt4_1, self.t4_3, "Stop.")
            self.print_unity(text1="Stop.", text2="Please select the mode.", color="black", id_init=4)