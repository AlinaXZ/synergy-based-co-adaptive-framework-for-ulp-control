import numpy as np
import pandas as pd
from signal_process import *
from feature_extraction_new import *
from training import *
from graph import *
import json


class AdpFramework():
    '''
    整个Framework
    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of centroids to generate.

    Attributes
    ----------
    See More
    --------
    https://github.com/scikit-learn/scikit-learn/blob/8a2d5ffa0512873a75da608eb14832253979ec44/sklearn/cluster/_kmeans.py#L1196
    '''
    
    # 私有属性
    __class_version = 1.0

    # 类属性
    def __init__(self, samp_freq=2e3):
        
        # Filter and Feature extractor
        self.samp_freq = samp_freq
        self.frame = 40
        self.step = 1

        # 添加一个很小值防止特征提取inf出现
        self.eps = 3e-7

        # training
        #self.m0 = np.array([1, 1, 0.1, 0.1, 0.01, 0.01])
        self.m0 = np.array([0, 0, 0, 0, 0, 0])

        # clustering
        self.cluster_range_low = 10
        self.cluster_range_high = 30
        self.c_min_len = 10  #最小cluster长度

        # 其他 pre-define
        self.w1 = 0.1   # cluster predict 给 Markov_trans的w
        self.w2 = 0.05  # 每次modify 的比重
        self.p_th = 0.5 # 超过这个比重才归于已有的cluster

    
    # 类方法,将需要访问类属性的方法定义为类方法
    @classmethod
    def cls_ver_get(cls):
        return cls.__class_version

    @staticmethod
    # create default label list
    def num_sequence(n):
        seq = []
        for i in range(n):
            seq.append(str(i+1))
        return seq

    # Functions for calculations
    @staticmethod
    def synergy_cal(imu, motion, round=3):
        """
        imu: shape(10,)
        motion: shape(6,)
        return synergy: shape(10,6)
        """
        
        imu_growth_inv = np.linalg.pinv([imu])
        synergy = np.dot(imu_growth_inv, [motion])
        return np.round(synergy,round)

    @staticmethod
    def motion_ctrl(imu, synergy):
        """
        imu: shape(10,)
        synergy: shape(10,6)
        return motion: shape(6,)
        """
        return np.dot([imu], synergy).flatten()


    # filter emg signal
    def filter_signal(self, sig_raw, bp_low_freq=10, bp_high_freq=500, plot=False):
        '''
        sig_raw.shape: multi-channel(signal_length, channel_num), single-channel(signal_length, )
        plot = False: donnot plot
        plot = True: plot with default row and col
        plot = (row, col)
        '''
    
        if sig_raw.ndim == 1:
            # single-channel signal
            sig_fl_1 = notch_filter(sig_raw, self.samp_freq)
            sig_filtered = bp_filter(sig_fl_1, bp_low_freq, bp_high_freq, self.samp_freq)

            if plot:
                plot_signal([sig_raw, sig_fl_1, sig_filtered], self.samp_freq, labels=["raw signal","notch filter", "band pass filter"], fig_size=plot)


        elif sig_raw.ndim == 2:
            # multi-channel signal
            for i in range(sig_raw.shape[1]):
                sig_each = sig_raw[:,i]

                sig_fl_1 = notch_filter(sig_each, self.samp_freq)
                sig_fl_1_reshape = sig_fl_1.reshape(1,sig_raw.shape[0])

                sig_fl_2 = bp_filter(sig_fl_1, bp_low_freq, bp_high_freq, self.samp_freq)
                sig_fl_2_reshape = sig_fl_2.reshape(1,sig_raw.shape[0])

                if i == 0:
                    sig_ft_multi_1 = sig_fl_1_reshape
                    sig_ft_multi_2 = sig_fl_2_reshape
                else:
                    sig_ft_multi_1 = np.append(sig_ft_multi_1, sig_fl_1_reshape, axis=0)
                    sig_ft_multi_2 = np.append(sig_ft_multi_2, sig_fl_2_reshape, axis=0)
                
                sig_filtered_1 = sig_ft_multi_1.T
                sig_filtered = sig_ft_multi_2.T

            if plot:
                if type(plot) == tuple:
                    row, col = plot
                else:
                    row = 1
                    col = sig_raw.shape[1]

                plot_signal_multi([sig_raw, sig_filtered_1, sig_filtered], (row, col), labels=["raw signal","notch filter", "band pass filter"])
        else:
            print("Signal Format Error")
            sig_filtered = np.array([])
        
        

        return sig_filtered

    def extract_features_dynamic(self, signal, eps, channel_labels=None, axis=0):
        """
        Compute time, frequency and time-frequency features from signal.
        :param signal: numpy array signal(multichannel)
        :param channel_name: string variable with the EMG channel name in analysis.
        :param frame: sliding window size
        :param step: sliding window step size
        :axis: 0:纵向拼接, 1:横向拼接
        :param plot: bolean variable to plot estimated features.

        :return feature_extracted: Array

        """

        channels = signal.shape[1]
        feature_extracted = []

        if channel_labels==None:
            channel_labels = self.num_sequence(channels)

        for i in range(channels):

            time_matrix = time_features(signal[:,i], self.frame, eps)
            frequency_matrix = frequency_features(signal[:,i], self.samp_freq, self.frame, eps)
            time_frequency_matrix = time_frequency_features(signal[:,i], self.frame, eps)

            feature_single_channel = np.column_stack((time_matrix, frequency_matrix, time_frequency_matrix)).T
            feature_extracted.append(feature_single_channel)

            # print('EMG features were from channel', channel_labels[i] ,'extracted successfully')


        if axis==0:
            feature_extracted = np.vstack(tuple(feature_extracted))
        elif axis==1:
            feature_extracted = np.hstack(tuple(feature_extracted))


        return feature_extracted.T[0]
    

    def extract_features_dynamic_selected(self, signal, eps, channel_labels=None, axis=0):

        channels = signal.shape[1]
        feature_extracted = []

        if channel_labels==None:
            channel_labels = self.num_sequence(channels)

        for i in range(channels):

            features_matrix = selected_features(signal[:,i], self.samp_freq, self.frame, eps).T
            feature_extracted.append(features_matrix)

        if axis==0:
            feature_extracted = np.vstack(tuple(feature_extracted))
        elif axis==1:
            feature_extracted = np.hstack(tuple(feature_extracted))


        return feature_extracted.T[0]




    def ft_clustering(self, emg_ft):
        '''
        # emg_ft.shape: (feature_length, features*channels)
        # 对行归一化
        emg_ft已经归一化
        return: 分类dataframe, label列表, 出现的列表
        '''
        #emg_ft_norm = normalize(emg_ft, axis=0, norm='max')

        #best_c_num, _ = best_cluster_num(self.cluster_range_low, self.cluster_range_high, emg_ft_norm)
        #kmeans = KMeans(n_clusters=best_c_num, random_state=0).fit(emg_ft_norm)
        
        # print(self.cluster_range_low, self.cluster_range_high)
        best_c_num, _ = best_cluster_num(self.cluster_range_low, self.cluster_range_high, emg_ft)
        print("best_c_num", best_c_num)
        kmeans = KMeans(n_clusters=best_c_num, random_state=0).fit(emg_ft)

        # 输出df
        label_list = [int(x) for x in kmeans.labels_]
        print("label_list", label_list)
        
        df_c, pos_list, pos_list_converted, appear_list = cluster_process(label_list, self.c_min_len)
        # print("ppooss:", df_c, pos_list, pos_list_converted, appear_list)

        appear_list_full = appear_list.copy()
        appear_list_full.append(len(kmeans.labels_))


        return df_c, pos_list, pos_list_converted, appear_list, appear_list_full



    @staticmethod
    def build_graph(pos_list, plot=False, edge_labels=False):

        row_col_prob = np.array([[],[],[]], dtype=int)

        for i in range(len(pos_list)-1):
            
            isin, in_pos = vec_in_mat([pos_list[i],pos_list[i+1]], row_col_prob[0:2])
            if isin:
                p = row_col_prob[2,in_pos]
                #row_col_prob[2,in_pos]= p/([p+1])
                #稀疏矩阵只能用整形做data
                row_col_prob[2,in_pos]= p+1
                        
            else:
                new_node = np.array([[pos_list[i]],[pos_list[i+1]],[1]])
                row_col_prob = np.hstack((row_col_prob, new_node))
        
        if plot:
            #plot_graph_sparse(row_col_prob, pos_list, edge_labels=edge_labels)
            plot_graph_sparse2(row_col_prob, edge_labels=edge_labels)
                
        return row_col_prob


    @staticmethod
    def cal_c_mean_marker(emg_ft, synergy, df_c, appear_list_full, user_input):
        c_mean_df = calculate_mean(emg_ft, synergy, df_c, appear_list_full)
        c_marker_df = mark_cluster(appear_list_full, user_input, df_c)
        return c_mean_df, c_marker_df


    def predict_next_node(self, emg_curr, cluster, node_last, graph):
    
        # 如果下一个id同上一个id，不考虑cluster间的transition
        pred_id, pred_prob = pred_next_node_in(emg_curr, cluster)

        # 如果下一个id 不等于 上一个id（或上一个是None)，考虑cluster间的transition
        if pred_id != node_last:
            pred_id, pred_prob = pred_next_node_inter(emg_curr, cluster, node_last, graph, self.w1, self.eps)
       
        return pred_id, pred_prob

    # modify knowledge base
    def modify_KB(self, c_pre, c_new, graph_pre, graph_new):
        print("---modify_KB---")

        # 判断哪些 cluster 合并，哪些 新增
        combine_idx_l, new_idx_l = enlarge_cluster(c_pre, c_new, self.p_th)
        print("+combine_idx_l, new_idx_l", combine_idx_l, new_idx_l)

        # modify sparse graph matrix
        graph_final = modify_graph(graph_pre, graph_new, new_idx_l, combine_idx_l)
        print("+graph_final", graph_final)
        # modify cluster dataframe
        c_final = modify_cluster(c_pre, c_new, combine_idx_l, self.w2)
        print("+c_final", c_final)

        return graph_final, c_final

    
    def training_offline(self, data_df, cluster_range=(5,20)):
        emg_ft = np.array(data_df['emg_ft'].tolist())
        df_c, _, _, appear_list_full = ft_clustering2(emg_ft, cluster_range=cluster_range)
        mean_synergy_df = calculate_mean_synergy(data_df, df_c, appear_list_full)
        return mean_synergy_df


    def training_offline2(self, data_df, m_label, nn=10):
        emg_ft = np.array(data_df['emg_ft'].tolist())
        clf, df_c, _, _, appear_list_full = ft_clustering3(emg_ft, m_label, nn=nn)
        mean_synergy_df = calculate_mean_synergy(data_df, df_c, appear_list_full)
        return clf, mean_synergy_df

    
    def pred_knn(self, emg_ft_norm, clf, df_c):
        knn_id = clf.predict([emg_ft_norm])[0]
        id_list = list(set(df_c['id'].tolist()))
        pred_prob = clf.predict_proba([emg_ft_norm])[0, id_list.index(knn_id)]
        pred_id = df_c[(df_c.id==knn_id)].index[0]

        return knn_id, pred_id, pred_prob

    def training_offline3(self, data_df, m_label):
        emg_ft = np.array(data_df['emg_ft'].tolist())
        df_c, _, _, appear_list_full = cluster_process2(m_label)
        mean_synergy_df = calculate_mean_synergy(data_df, df_c, appear_list_full)
        return mean_synergy_df