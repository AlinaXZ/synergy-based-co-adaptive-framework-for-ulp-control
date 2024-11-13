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


def initial_train(self):
    try:
        myo.init(sdk_path=self.sdk_path)
        hub = myo.Hub()
        listener = DataCollector(n=self.paras['window_size'])
        myo_connect = True
    except:
        self.print_tk(self.txt4_1, self.t4_3, "Please first check the Myo connection.", color="red.")
        myo_connect = False
    finally:
        if myo_connect:
            self.initial_train_start(hub, listener)


def initial_train_start(self, hub, listener):

    # init
    motion_angle = np.array([0, 0])
    data_df = pd.DataFrame()
    # emg_list = np.array([])


    # # only for plot
    imu_queue = deque(maxlen=self.paras['window_size'])
    emg_ft_que = deque(maxlen=self.paras['window_size'])
    emg_ft_que.extend(np.zeros((self.paras['window_size'], 80)).tolist())


    train_list = [
        ([0],30,0,0),
        ([11],30,0,2), ([0],60,0,0), ([11],30,0,2),
        # ([11],15,0,2), ([12],15,0,2), ([13],15,0,2), ([14],15,0,2),
        ([0],60,0,0),
        ([31],20,4,0), 
        ([0],60,0,0),
        ([41],20,-4,0), 
        ([0],60,0,0), 
        ([21],30,0,-2), ([0],60,0,0), ([21],30,0,-2),
        # ([21],15,0,-2), ([22],15,0,-2), ([23],15,0,-2), ([24],15,0,-2), 
        ([0],30,0,0), 
        
        # ([33],10,4,0), ([34],10,4,0),
        # ([0],20,0,0),
        # ([14],15,0,2), ([15],15,0,2), ([16],15,0,2), 
        # ([5],20,0,0), 
        # ([35],10,4,0), ([36],10,4,0),
        # ([6],20,0,0), 
        # ([45],10,-4,0), ([46],10,-4,0), ([47],10,-4,0), ([48],10,-4,0),
        # ([7],20,0,0), 
        # ([37],10,4,0), ([38],10,4,0),
        # ([5],20,0,0),
        # ([25],15,0,-2), ([26],15,0,-2), ([27],15,0,-2), 
        # ([0],20,0,0), 
        
    ]
    train_list = train_list*self.paras['num_train']
    m_list, m_label = generate_training_data(train_list)


    id_m = 0
    init_train_done = False
    imu_last = np.zeros(4)
    emg_ft_last = np.zeros(80)

    try:


        with keyboard.Events() as k_events:
            self.print_tk(self.txt4_1, self.t4_3, "Start collecting data...")
            self.print_unity(text1="Start collecting data...", text2="Please wait.", color="black")
            
            ready = False
            start_pressed, start_store = False, False

        
            # 暂停50ms(sample_rate=20)
            while hub.run(listener.on_event, self.paras['time_interval']):


                listener.update_que()
                emg_que = np.array(listener.emg_data_queue)

                imu = np.array([x for x in listener.imu_data_curr])
                imu_velocity = imu - imu_last
                imu_last = imu.copy()
                # emg_curr = np.array([x for x in listener.emg_data_curr])
                # emg_list = np.append(emg_list, emg_curr)
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


                if break_loop or controller==1 or id_m==len(m_list)-1:
                    break
                
                if key_input == "s" or controller==3:
                    if ready:
                        start_pressed = True
                        self.print_tk(self.txt4_1, self.t4_3, "---Start---")
                        self.print_unity(text1="Start.", text2="Press 'Menu' to stop.", color="black")

                user_input = True
                if key_input == "h" or controller==2:
                    user_input = False
                

                if emg_que.shape[0] < self.adp1.frame:
                    # initial training start
                    # s_curr = s0
                    # imu_last = imu
                    # m_store = self.adp1.m0
                    pass


                if emg_que.shape[0] == self.adp1.frame-1:
                    ready = True
                    id_m = 0
                    self.print_tk(self.txt4_1, self.t4_3, "Ready to start", "green")
                    self.t4_3.config(fg="black")
                    self.print_unity(text1="Ready to start.", text2="Press 'Pad(Center)' to confirm start.", color="green")
                

                # 当emg长度足够时
                if start_pressed and emg_que.shape[0] == self.adp1.frame:

                    # emg_fl = self.adp1.filter_signal(emg_que)

                    # plot emg_fl
                    self.show_graph_emg(emg_que.T, self.line0, self.show)
                    self.show_graph_imu(np.array(imu_queue).T, self.line1)

                    if self.unity_joint != None:
                        # 不能同时输入elbow运动，unity的规则是先y后z。
                        self.unity_joint.SendRotate2(trans_v = MyVector3(0, 0, m_list[id_m][1]))
                        self.unity_joint.SendRotate3(trans_v = MyVector3(0, m_list[id_m][0], 0))
                    id_m += 1
                    
                    motion_angle = motion_angle + np.array(m_list[id_m])
                    

                    # emg_ft = self.adp1.extract_features_dynamic(emg_que, self.adp1.eps, axis=0)
                    # emg_ft_norm = normalize(emg_ft.reshape(-1, 1), axis=0, norm='max').T[0]

                    
                    emg_ft = self.adp1.extract_features_dynamic_selected(emg_que, self.adp1.eps, axis=0)
                    emg_ft_add_v = np.hstack((emg_ft, emg_ft-emg_ft_last))
                    emg_ft_last = emg_ft.copy()
                    emg_ft_add_v_norm = normalize(emg_ft_add_v.reshape(-1, 1), axis=0, norm='max').T[0]


                    # only for plot
                    emg_ft_norm = normalize(emg_ft.reshape(-1, 1), axis=0, norm='max').T[0]
                    emg_ft_que.append(emg_ft_norm.tolist())
                    self.show_graph_emg_ft(np.array(emg_ft_que).T, self.line2, self.show2_1, self.var_cb.get())


                    # store data as dataframe
                    # data = pd.DataFrame([{"emg_ft": np.round(emg_ft_norm, 6), "m": m_list[id_m], "imu_velocity": imu_velocity}])
                    data = pd.DataFrame([{"emg_ft": np.round(emg_ft_add_v_norm, 6), "m": m_list[id_m], "imu_velocity": imu_velocity}])
                    data_df = pd.concat([data_df, data], axis=0, ignore_index=True)

                    # initial training
                    # m_curr, s_curr, imu_last, m_store = training_loop(imu, s_curr, imu_last, m_store, user_input=user_input, scale=self.paras['modify_scale'])
                    # m_curr = m_curr * self.paras['output_scale']
                    # m_curr, s_curr, imu_last, m_store, motion_angle = training_loop2(imu, imu_last, s_curr, m_store, motion_angle, 
                    #     user_input=user_input, scale1=self.paras['modify_scale'], scale2=self.paras['output_scale'])
                    # m_curr, s_curr, imu_last, m_store, motion_angle = training_loop3(imu, imu_last, s_curr, m_store, motion_angle, 
                    #     user_input=user_input, scale1=self.paras['modify_scale'], scale2=self.paras['output_scale'])
                    
                    # use VR Controller input
                    # if self.unity_joint != None:
                    #     self.unity_joint.SendRotate2(trans_v = MyVector3(0, 0, m_curr[1]))
                        # unity_joint.SendRotate2(trans_v = MyVector3(0, m_curr[0], m_curr[1]))
                        # unity_joint.SendRotate2(trans_v = MyVector3(0, m_curr[0], 0))
                    

                    # print("imu:", np.round(imu,3), "output motion: ", np.round(m_curr, 2), "synergy:", np.round(s_curr[0],2))
                    # txt_print = "user_input: " + str(user_input)
                    txt_print = "imu: " + str(np.round(imu,3))
                    # txt_print += "\nsynergy: " + str(np.round(s_curr[0],2))
                    txt_print += "\nimu velocity: " + str(np.round(imu_velocity, 4))
                    txt_print += "\noutput motion: " + str(np.round(m_list[id_m], 2))
                    txt_print += "\nangle_sum: " + str(np.round(motion_angle, 2))
                    
                    self.print_tk(self.txt4_1, self.t4_3, txt_print)
                    if self.unity_joint != None:
                        self.unity_joint.SetText3(text3=txt_print)

        init_train_done = True


    except:
        txt_print = "\nInitial Training stop."
        self.print_tk(self.txt4_1, self.t4_3, txt_print, color="red")
        
    finally:
        hub.stop()

        txt_print = "Training end. Data is being stored..."
        self.print_tk(self.txt4_1, self.t4_3, txt_print)
        self.print_unity(text1="Training end. Data is being stored...", text2="Please wait...", color="black")

        if init_train_done:
            self.initial_train_offline(data_df, m_label, txt_print)
        
        
def initial_train_offline(self, data_df, m_label, txt_print):


    try:
        data_df.to_json(self.store_path['myo_data_path'], orient='columns')
        txt_print += "\nTraining data stored in " + self.store_path['myo_data_path']
        self.print_tk(self.txt4_1, self.t4_3, txt_print)
        self.print_unity(text1="Training data stored.", text2="Please wait...", color="black")
        
        try:
            data_df_shift = shift_emg_ft(data_df, int(self.paras['window_size']/2))
            # data_df_shift = shift_emg_ft(data_df, self.paras['window_size'])
            # clf, mean_synergy_df = self.adp1.training_offline2(data_df_shift, m_label[:data_df_shift.shape[0]], nn=10)
            mean_synergy_df = self.adp1.training_offline3(data_df_shift, m_label[:data_df_shift.shape[0]])

            # 写入json
            mean_synergy_df.to_json(self.store_path['cluster_path'], orient='columns')

            # # 写入piclke
            # with open(self.store_path['clf_path'], 'wb') as f:
            #     pickle.dump(clf, f)

            txt_print += "\nLearning data stored in " + self.store_path['cluster_path']
            self.print_tk(self.txt4_1, self.t4_3, txt_print)
            self.print_unity(text1="Learning data stored.", text2="Please select the mode.", color="black", id_init=4)

            

        except:
            txt_print += "\nLearning Error."
            self.print_tk(self.txt4_1, self.t4_3, txt_print)

        # try:    
        #     # build graph
        #     row_col_prob = self.adp1.build_graph(pos_list_converted, plot=False)
        #     transform_prob = transform_graph(row_col_prob)
        #     dict_graph = {
        #         'adjacency_mat': row_col_prob.tolist(),
        #         'transition_prob': transform_prob
        #     }
        #     with open(self.store_path['graph_path'], 'w') as f:
        #         json.dump(dict_graph, f)
            
        #     txt_print += "\nGraph data stored in " + self.store_path['graph_path']
        #     txt_print += "\nInitial Training Finish."
        #     self.print_tk(self.txt4_1, self.t4_3, txt_print)
        #     self.print_unity(text1="Graph data stored. Initial Training Finish.", text2="Please select the mode.", color="black", id_init=3)

        #     try:
        #         pass
        #         # # plot graph
        #         # mc = plot_graph_sparse4(row_col_prob)
        #         # fig4, ax4 = mc.draw(figsize=(5,4))
        #         # graph4 = FigureCanvasTkAgg(fig4, master=tab3_4)
        #         # canvas4 = graph4.get_tk_widget()
        #         # canvas4.grid(row=0, column=0)
        #         # graph4.draw()
        #     except:
        #         txt_print += "\nPlot Graph Error."
        #         self.print_tk(self.txt4_1, self.t4_3, txt_print)

        # except:
        #     txt_print += "\nBuild Graph Error."
        #     self.print_tk(self.txt4_1, self.t4_3, txt_print)

    except:
        txt_print += "\nPath Error. Please check if the storage path has been set."
        self.print_tk(self.txt4_1, self.t4_3, txt_print)