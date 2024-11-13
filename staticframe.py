import numpy as np
import json
import pandas as pd

from adpframe_tk import AdpFramework
from training import *
from feature_extraction_new import *

# 子类继承父类
class StaticFramework(AdpFramework):
    def __init__(self, samp_freq=2e3):
        super().__init__(samp_freq) # 子类重写了父类的属性或方法，如果想调用父类的action方法

    @staticmethod
    def load_sig_data(path):
        '''
        path = "C:\\...\\data.json"
        data example:
        {'timestamp': '26/06/20/22:28:12', 'duration': 2000, 'emg': {'frequency': 200, 'data': [[...],...,[...]]}, 
        'imu': {'frequency': 200, 'data': [{'gyroscope': [...], 'acceleration': [...], 'acceleration': [...]}, ...
        {'gyroscope': [...], 'acceleration': [...], 'orientation': [...]}]}}
        '''
        with open(path, 'r') as raw_data:
            raw_data = json.load(raw_data)

        # Process raw data

        # Process EMG data
        emg_mat = np.array(raw_data['emg']['data'])

        # Process IMU data
        imu_raw = np.array(raw_data['imu']['data'])
        imu_mat = []
        len_data = len(raw_data['imu']['data'])
        for j in range(len_data):
            feature = np.hstack(( imu_raw[j]['gyroscope'], imu_raw[j]['acceleration'], imu_raw[j]['orientation']))
            imu_mat.append(feature)
        imu_mat = np.array(imu_mat)

        return emg_mat, imu_mat

    def remove_imu_delay(self, imu_mat):
        deley = self.frame - 1
        imu_mat = np.delete(imu_mat, list(range(deley)), axis=0)
        return imu_mat

    
    def extract_features(self, signal, eps, channel_labels=None, axis=0):
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

        if signal.ndim == 1:
            signal = np.array([signal]).T
        
        channels = signal.shape[1]
        feature_extracted = []

        if channel_labels==None:
            channel_labels = self.num_sequence(channels)

        for i in range(channels):

            time_matrix = time_features_estimation(signal[:,i], self.frame, self.step, eps)
            frequency_matrix = frequency_features_estimation(signal[:,i], self.samp_freq, self.frame, self.step, eps)
            time_frequency_matrix = time_frequency_features_estimation(signal[:,i], self.frame, self.step, eps)
            # time_matrix = time_features_estimation(signal[:,i], self.frame, self.step)
            # frequency_matrix = frequency_features_estimation(signal[:,i], self.samp_freq, self.frame, self.step)
            # time_frequency_matrix = time_frequency_features_estimation(signal[:,i], self.frame, self.step)

            feature_single_channel = np.column_stack((time_matrix, frequency_matrix, time_frequency_matrix)).T
            feature_extracted.append(feature_single_channel)

            print('EMG features were from channel', channel_labels[i] ,'extracted successfully')


        if axis==0:
            feature_extracted = np.vstack(tuple(feature_extracted))
        elif axis==1:
            feature_extracted = np.hstack(tuple(feature_extracted))


        return feature_extracted.T



    # 整合所有步骤
    def process_inputs(self, emg_mat, imu_mat):
        # filter
        emg_fl = self.filter_signal(emg_mat, plot=False)
        print("EMG signal filtered.")

        # EMG feature extraction
        channel_labels = ['CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8']
        emg_features = self.extract_features(emg_fl, self.eps, channel_labels, axis = 0)
        
        # Remove IMU signal delay
        imu_proc = self.remove_imu_delay(imu_mat)
        print("EMG and IMU signals synchronized.")
        return emg_features, imu_proc


    def init_training(self, imu_mat, user_input):
        # create an empty synergy dataframe
        synergy_df = pd.DataFrame()

        imu0 = imu_mat[0]
        s0 = synergy_cal(imu0, self.m0, round=3)

        s_curr = s0

        for i, imu_curr in enumerate(imu_mat):
            input_last = user_input[i]
            m_curr, s_curr = training_loop(imu_curr, s_curr, user_input=user_input)

            # output the current motion

            # store the syenergy list
            synergy_data = pd.DataFrame([{"s":s_curr}])
            synergy_df = pd.concat([synergy_df, synergy_data], axis=0, ignore_index=True)

        return synergy_df
