import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from func import *

# Functions for calculations
def synergy_cal(imu, motion, round=3):
    """
    imu-dim: 4
    motion-dim: 6
    
    imu: shape(4,)
    motion: shape(6,)
    return synergy: shape(4,6)
    """
    
    imu_growth_inv = np.linalg.pinv([imu])
    synergy = np.dot(imu_growth_inv, [motion])
    return np.round(synergy,round)

def motion_ctrl(imu, synergy):
    """
    imu: shape(4,)
    synergy: shape(4,6)
    return motion: shape(6,)
    """
    return np.dot([imu], synergy).flatten()


def compensation_mechanism(target_synergy):
    pass



# modify synergy when user inputs "wrong!"
def synergy_modify(synergy, reach, target, scale=1, round=3):
    # gauss_noise(mu, sigma, size)
    gauss_noise = np.random.normal(0, 0.05, size=(4,6))
    if reach:
        gauss_noise = np.absolute(gauss_noise)
        synergy_vec = target - synergy
        gauss_noise = np.multiply(gauss_noise, synergy_vec)
        
    synergy_modified = (synergy + gauss_noise) * scale

    return  np.round(synergy_modified, round)


# one iteration
def training_loop(imu_curr, s_curr, imu_last, m_store, user_input=False, scale=1):
    # 手臂未移动(小数点后3位)
    if np.round(imu_curr-imu_last, 3).any()==0:
        s_next = np.zeros((4, 6))
        m_curr = np.zeros(6)

    # 手臂移动了
    else:
        # 若上一个s是0，则保持归0前的运动
        if s_curr.any()==0:
            # 存储的m=0，则给个随机值    
            if m_store.any()==0:
                noise1 = np.random.normal(0, 0.5, size=2)
                noise2 = np.random.normal(0, 0.1, size=2)
                noise3 = np.random.normal(0, 0.05, size=2)
                m_store = np.hstack((noise1,noise2,noise3))
            
            s_curr = synergy_cal(imu_curr, m_store, round=3)

        m_curr = motion_ctrl(imu_curr, s_curr)
        m_store = m_curr
        if user_input==False:
            # "wrong"
            s_next = synergy_modify(s_curr, False, None, scale=scale, round=3)
        else:
            s_next = s_curr

    return m_curr, s_next, imu_curr, m_store


def training_loop2(imu_curr, imu_last, s_last, m_store, motion_angle_last, user_input=False, scale1=1, scale2=1):
    # 手臂未移动(小数点后3位)
    if np.round(imu_curr-imu_last, 3).any()==0:
        s_curr = np.zeros((4, 6))
        m_curr = np.zeros(6)

    # 手臂移动了
    else:
        # 若上一个s是0，则保持归0前的运动
        if s_last.any()==0:
            # 存储的m=0，则给个随机值    
            if m_store.any()==0:
                noise1 = np.random.normal(0, 0.5, size=2)
                noise2 = np.random.normal(0, 0.1, size=2)
                noise3 = np.random.normal(0, 0.05, size=2)
                m_curr = np.hstack((noise1,noise2,noise3))
            else:
                m_curr = m_store.copy()
            
            s_curr = synergy_cal(imu_curr, m_curr, round=3)
        elif user_input==False:
            # "wrong"
            s_curr = synergy_modify(s_last, False, None, scale=scale1, round=3)
        else:
            s_curr = s_last
    
    m_curr = motion_ctrl(imu_curr, s_curr) * scale2

    motion_angle_curr = motion_angle_last + m_curr[0:2]    
        
    if motion_angle_last[0]>=-90 and motion_angle_last[0]<=90 and (motion_angle_curr[0]<=-90 or motion_angle_curr[0]>=90):
        s_curr[:,0] = -s_curr[:,0]
    
    if motion_angle_last[0]>=0 and motion_angle_last[0]<=150 and (motion_angle_curr[1]<=0 or motion_angle_curr[1]>=150):
        s_curr[:,1] = -s_curr[:,1]
    
    m_curr = motion_ctrl(imu_curr, s_curr) * scale2
    motion_angle_curr = motion_angle_last + m_curr[0:2]

    return m_curr, s_curr, imu_curr, m_curr, motion_angle_curr


def training_loop3(imu_curr, imu_last, s_last, m_store, motion_angle_last, user_input=False, scale1=1, scale2=1):
    # 手臂未移动(小数点后3位)
    if np.round(imu_curr-imu_last, 3).any()==0:
        s_curr = np.zeros((4, 6))
        m_curr = np.zeros(6)

    # 手臂移动了
    else:
        # 若上一个s是0，则保持归0前的运动
        if s_last.any()==0:
            # 存储的m=0，则给个随机值    
            if m_store.any()==0:
                noise1 = np.random.normal(0, 0.5, size=2)
                noise2 = np.random.normal(0, 0.1, size=2)
                noise3 = np.random.normal(0, 0.05, size=2)
                m_curr = np.hstack((noise1,noise2,noise3))
            else:
                m_curr = m_store.copy()
            
            s_curr = synergy_cal(imu_curr, m_curr, round=3)
        else:
            s_curr = s_last
            if user_input==False:
                # "wrong"
                s_curr[:,1] = -s_last[:,1]

    
    m_curr = motion_ctrl(imu_curr, s_curr) * scale2

    motion_angle_curr = motion_angle_last + m_curr[0:2]    
        
    # if motion_angle_last[0]>=-90 and motion_angle_last[0]<=90 and (motion_angle_curr[0]<=-90 or motion_angle_curr[0]>=90):
    #     s_curr[:,0] = -s_curr[:,0]
    
    if (motion_angle_curr[1]<0 and m_curr[1]<0) or (motion_angle_curr[1]>150 and m_curr[1]>0):
        s_curr[:,1] = -s_curr[:,1]

    # if motion_angle_last[0]>=0 and motion_angle_last[0]<=150 and (motion_angle_curr[1]<0 or motion_angle_curr[1]>150):
    #     s_curr[:,1] = -s_curr[:,1]
    
    m_curr = motion_ctrl(imu_curr, s_curr) * scale2
    motion_angle_curr = motion_angle_last + m_curr[0:2]

    return m_curr, s_curr, imu_curr, m_curr, motion_angle_curr


# 多少cluster最合适
def best_cluster_num(low, high, X):
    best_cluster_num = 0
    best_score = 0
    for i in range(low, high):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
        # evaluate clustering quality
        labels_pred = kmeans.predict(X)
        score = silhouette_score(X,labels_pred)
        if score > best_score:
            best_cluster_num = i
            best_score = score
        #print(i, 'clusters, score:', "%.3f" % score)

    return best_cluster_num, best_score



# 找到一个label的cluster的所有位置
def cluster_appear(list, find_value, c_min_len):
    '''
    list: label列表, 必须是list类型
    find_value: node/label的编号
    c_min_len: 最小cluster长度
    return: 所有find_value的聚类的位置
    '''

    all_pos = unique_index(list, find_value)


    v_last = -c_min_len-1
    v_appear = []

    # for i in all_pos:

        # v_curr = i
        # if (v_curr-v_last) > c_min_len:
        #     if list[i: i+c_min_len].count(find_value) > c_min_len/2:
        #         v_appear.append(i)
        # v_last  = i

    for i, pos in enumerate(all_pos):
        if i == 0:
            v_appear.append(pos)
        else:
            if all_pos[i-1] != all_pos[i]-1:
                v_appear.append(pos)


    return v_appear



def del_same_neighbor3(list1, list2, list3):
    # 删除列表相邻相同的元素
    # list2按照list删除对应index元素
    del_list =[]
    for i in range(len(list1)-1):
        if list1[i]==list1[i+1]:
            del_list.append(i+1)

    list1_copy = list1.copy()
    list2_copy = list2.copy()
    list3_copy = list3.copy()

    del_df_list = []

    for i, x in enumerate(del_list):
        list1_copy.pop(x-i)
        list2_copy.pop(x-i)
        del_df_list.append(list3_copy[x-i])
        list3_copy.pop(x-i)

    del_id = [list2[i] for i in del_list]

    return list1_copy, list2_copy, list3_copy, del_id, del_df_list



def cluster_process(label_list, c_min_len):
    '''
    list: label列表, 必须是list类型
    c_min_len: 最小cluster长度
    return-appear_df: 包含所有cluster位置的dataframe
    return-c_sort: cluster出现的顺序
    '''

    
    # 创建空df，list
    appear_df = pd.DataFrame(columns=['id', 'c_pos'])
    appear_list = []

    label_id = remove_duplicates(label_list)
    label_id_valid = label_id.copy()

    
    for i in label_id:
        c_pos = cluster_appear(label_list, i, c_min_len)
        
        if c_pos !=[]:
            ser_appear = pd.Series({'id':i, 'c_pos': c_pos})
            appear_df = pd.concat([appear_df, ser_appear.to_frame().T], ignore_index=True)
            appear_list = appear_list + c_pos
        else:
            label_id_valid.remove(i)
    
    
    appear_list.sort()
    c_sort = [label_list[i] for i in appear_list]
    pos_list = [int(x) for x in c_sort]
    
    convert_list = appear_df['id'].tolist()
    pos_list_converted = [convert_list.index(i) for i in pos_list]
 

    # 删除临近相同元素（合并相邻的同一cluster)
    # 处理 pos_list
    pos_list2, pos_list_converted2, appear_list2, del_id, del_df_list = del_same_neighbor3(pos_list, pos_list_converted, appear_list)

    # 处理appear_df

    for idx in del_id:
        list_pos = appear_df.loc[idx, "c_pos"].copy()
        for x in del_df_list:
            if x in list_pos:
                list_pos.remove(x)
        appear_df.loc[idx, "c_pos"]=list_pos

    return appear_df, pos_list2, pos_list_converted2, appear_list2





def empty_arr(arr_shape):
    # 创建空array shape (0,arr_len)
    empty = []
    if isinstance(arr_shape, tuple):
        arr_len = arr_shape[0] * arr_shape[1]
        
        for x in range(arr_len):
            empty.append([])
        empty_arr = np.array(empty, dtype=float).reshape(0, arr_shape[0], arr_shape[1])

    else:
        arr_len = arr_shape

        for x in range(arr_len):
            empty.append([])
        empty_arr = np.array(empty, dtype=float).reshape(0, arr_len)

    return empty_arr



# Calculate feature_mean and synergy_mean
def calculate_mean(emg_ft, synergy, df_c, appear_list_full):
    # 创建空df
    c_mean_df = pd.DataFrame(columns=['emg_ft_mean','s_mean'])
    
    
    for i in range(df_c.shape[0]):
        # 创建空df
        synergy_c_df = pd.DataFrame(columns=['s'])

        # 创建空matrix，shape为(0,原shape)
        emg_c = empty_arr(emg_ft.shape[1])
        synergy_c = empty_arr(synergy.shape[1:3])


        for j in df_c.loc[i, 'c_pos']:
            # 对于df_c中的第i个cluster，其所有emg_ft的位置为j~appear_list_full[idx_j+1]
            idx_j = appear_list_full.index(j)
            
            synergy_c_this = synergy[j:appear_list_full[idx_j+1]]
            emg_c_this = emg_ft[j:appear_list_full[idx_j+1]]

            #print(i, j, "~", appear_list_full[idx_j+1], synergy_c_this.shape, emg_c_this.shape)

            # concat
            emg_c = np.vstack((emg_c,emg_c_this))
            synergy_c = np.vstack((synergy_c,synergy_c_this))

        # calculate mean
        emg_c_mean = np.mean(emg_c, axis=0)
        synergy_c_mean = np.mean(synergy_c, axis=0)

        #print("--- sum:", i, emg_c.shape, synergy_c.shape, "mean:", emg_c_mean.shape, synergy_c_mean.shape)

        mean_this = pd.DataFrame({'emg_ft_mean': [emg_c_mean], 's_mean': [synergy_c_mean]})
        c_mean_df = pd.concat([c_mean_df, mean_this], ignore_index=True)


    return c_mean_df



def mark_cluster(appear_pos, user_input, df_c):

    wrong_pos = unique_index(user_input, False)

    cluster_state = []

    for i in range(len(wrong_pos)):
        for j in range(len(appear_pos)):
            if wrong_pos[i]<appear_pos[j]:
                break
            #输入"wrong"后，设置当前cluster状态为"wrong"
        
        wrong_cluster_id = appear_pos[j]
        cluster_state.append(wrong_cluster_id)

    #消去重复id
    cluster_state = set(cluster_state)

    c_input = []

    for i in range(df_c.shape[0]):
        set_i = set(df_c['c_pos'][i])
        if set_i.intersection(cluster_state):
            c_input.append(False)
        else:
            c_input.append(True)
    
    c_input_df = pd.DataFrame(pd.Series(c_input), columns=['c_marker'])

    return c_input_df




# ------new task training-------

def calculate_prob(dist_list, eps=3e-7):
    d_min = min(dist_list)
    d_max = max(dist_list)
    prob = [(d_max-x)/(d_max-d_min + eps) for x in dist_list]
    return [p/sum(prob) for p in prob]


# predict based on clusters
# 检查ft与哪个mean更近
def predict_match(vec, mat, index, eps):
    id_closest = 0
    dist0 = np.sqrt(np.sum(np.square(vec - mat[0])))
    dist_list = []
    for i in range(0, mat.shape[0]):
        dist = np.sqrt(np.sum(np.square(vec - mat[i])))
        #dist_list.append((i,dist))
        dist_list.append(dist)
        if dist < dist0:
            dist0 = dist
            id_closest = i

    #dist_l_value = [1/(x+eps) for x in dist_list]
    #dist_l_norm = [x/sum(dist_l_value) for x in dist_l_value]
    
    return [index, calculate_prob(dist_list, eps)]



# predict based on graph transition
def predict_trans(node_last, transition):

    pred_trans, new_node = None, None

    if node_last != None:
        pos_last = unique_index(transition[0], node_last)
        if pos_last == []:
            # 上一个节点不在row里
            new_node = True
        else:
            new_node = False
            pred_trans = [[transition[1][i] for i in pos_last], [transition[2][i] for i in pos_last]]
    return pred_trans, new_node


# predict based on both
def pred_both(pred_match, pred_trans, new_node, w):
    if pred_trans == None:
        pred_final = pred_match.copy()

    else:
        prob = pred_match[1].copy()
        for i, idx in enumerate(pred_trans[0]):
            pos_idx = unique_index(pred_match[0], idx)
            if pos_idx != []:
                prob[pos_idx[0]] = pred_match[1][pos_idx[0]]* (1-w) + pred_trans[1][i] * w

        pred_final = [pred_match[0], [p/sum(prob) for p in prob]]
        
    return pred_final


def transform_graph(graph):
    if type(graph) is np.ndarray:
        graph = graph.tolist()
    return [x/sum(graph[2]) for x in graph[2]]


# 依据pred_final返回最佳id和概率
def predict_result(pred_final):
    pred_index = pred_final[1].index(max(pred_final[1]))
    pred_id = pred_final[0][pred_index]
    pred_prob = pred_final[1][pred_index]

    return pred_id, pred_prob


# 考虑cluster间的transition
def pred_next_node_inter(emg_curr, cluster, node_last, graph, w, eps=3e-7):
    # predict based on clusters
    emg_ft_mean = np.array(cluster['emg_ft_mean'].tolist())
    index = list(cluster.index)
    pred_match = predict_match(emg_curr, emg_ft_mean, index, eps)

    #predict based on graph transition
    #transition = transform_graph(graph)
    transition = [graph[0].tolist(), graph[1].tolist(), transform_graph(graph)]
    #transition = [graph[0].tolist(), graph[1].tolist(), [x/sum(graph[2]) for x in graph[2]]]
    pred_trans, new_node = predict_trans(node_last, transition)


    pred_final = pred_both(pred_match, pred_trans, new_node, w)
    
    # pred_index = pred_final[1].index(max(pred_final[1]))
    # pred_id = pred_final[0][pred_index]
    # pred_prob = pred_final[1][pred_index]

    # return pred_id, pred_prob
    return predict_result(pred_final)


# 同一个cluster内不考虑cluster间的transition
def pred_next_node_in(emg_curr, cluster, eps=3e-7):
    emg_ft_mean = np.array(cluster['emg_ft_mean'].tolist())
    index = list(cluster.index)
    pred_match = predict_match(emg_curr, emg_ft_mean, index, eps)

    # pred_index = pred_match[1].index(max(pred_match[1]))
    # pred_id = pred_match[0][pred_index]
    # pred_prob = pred_match[1][pred_index]

    # return pred_id, pred_prob
    return predict_result(pred_match)



# modify ---------
def enlarge_cluster(c_pre, c_new, p_threshold):
    print("---enlarge_cluster")
    pred_id_l, pred_prob_l = [], []
    emg_ft_mean2 = np.array(c_new['emg_ft_mean'].tolist())
    for m in emg_ft_mean2:
        pred_id, pred_prob = pred_next_node_in(m, c_pre)
        pred_id_l.append(pred_id)
        pred_prob_l.append(pred_prob)
    
    print("pred_prob_l", pred_prob_l)

    combine_idx_l = []
    index = list(c_new.index)
    new_idx_l = index.copy()
    for p in pred_prob_l:
        if p > p_threshold:
            pred_idx = pred_prob_l.index(p)
            combine_idx_l.append(pred_idx)
            new_idx_l.remove(pred_idx)

    # combine_idx_l 的 p 超过阈值，归于已存在的 cluster
    # new_idx_l 的 p < 阈值，建立新的 cluster
    
    return combine_idx_l, new_idx_l




# 返回扩大后的 sparse graph
def modify_graph(graph_pre, graph_new, new_idx_l, combine_idx_l):
    
    # 处理新增的列
    add1 = list(range(len(new_idx_l)))
    add = len(set(list(graph_pre[0])+list(graph_pre[1])))

    graph_modified = graph_new.copy()
    
    for i in range(graph_new.shape[1]):
        for j in [0,1]:
            if graph_new[j, i] in new_idx_l:
                graph_modified[j, i] = add1[new_idx_l.index(graph_new[j, i])] + add
    
    graph_final = np.hstack((graph_pre, graph_modified))

    
    # 处理合并的列
    check = graph_final[0:2].T.tolist()
    repeat = []

    for x in combine_idx_l:
        for i in unique_index(graph_final[0], x):
            idx_x = unique_index(check, check[i])
            if len(idx_x) > 1 and not idx_x in repeat:
                repeat.append(idx_x)

    if repeat!=[]:
        for i in repeat:
            graph_final[2,i[0]] += graph_final[2,i[1]]
        
        
        graph_final = np.delete(graph_final, np.array(repeat).T[1], axis = 1)
    
    return graph_final


def modify_cluster(c_pre, c_new, combine_idx_l, w):
    c_drop = c_new.iloc[combine_idx_l]
    print("+++c_drop", type(c_drop), c_drop)
    c_new2 = c_new.drop(index=combine_idx_l)
    c_final = pd.concat([c_pre, c_new2], ignore_index=True)
    for i in combine_idx_l:
        c_final.at[i, 'emg_ft_mean'] = np.array(c_final.loc[i,'emg_ft_mean']) * (1-w) + c_drop.loc[i,'emg_ft_mean'] * w
        if c_final.loc[i,'c_marker'] == False and c_drop.loc[i,'c_marker'] == True:
            c_final.loc[i,'c_marker'] = True
            c_final.loc[i,'s_mean'] = c_drop.loc[i,'s_mean']
        elif c_final.loc[i,'c_marker'] == True and c_drop.loc[i,'c_marker'] == True:
            c_final.loc[i,'s_mean'] = np.mean(c_final.loc[i,'s_mean'], c_drop.loc[i,'s_mean'])
    return c_final



def calculate_mean_synergy(data_df, df_c, appear_list_full):
    # 创建空df
    mean_synergy_df = pd.DataFrame(columns=['emg_ft_mean','synergy'])
    # 提取data_df信息
    emg_ft = np.array(data_df['emg_ft'].tolist())
    motion = np.array(data_df['m'].tolist())
    imu_v = np.array(data_df['imu_velocity'].tolist())
    imu_v_with_s = np.hstack((imu_v, np.ones((len(imu_v),1))))
    
    # calculate emg_mean
    for i in range(df_c.shape[0]):

        # 创建空matrix，shape为(0,原shape)
        emg_c = empty_arr(emg_ft.shape[1])
        motion_c = empty_arr(motion.shape[1])
        imu_v_c = empty_arr(imu_v_with_s.shape[1])

        for j in df_c.loc[i, 'c_pos']:
            # 对于df_c中的第i个cluster，其所有emg_ft的位置为j~appear_list_full[idx_j+1]
            idx_j = appear_list_full.index(j)
            emg_c_this = emg_ft[j:appear_list_full[idx_j+1]]
            motion_c_this = motion[j:appear_list_full[idx_j+1]]
            imu_v_c_this = imu_v_with_s[j:appear_list_full[idx_j+1]]
            # concat
            emg_c = np.vstack((emg_c,emg_c_this))
            motion_c = np.vstack((motion_c,motion_c_this))
            imu_v_c = np.vstack((imu_v_c,imu_v_c_this))

        # calculate emg_c_mean
        emg_c_mean = np.mean(emg_c, axis=0)
        #synergy_c = (np.linalg.pinv(imu_v_c)@motion_c).T
        synergy_c = motion_c.T @ np.linalg.pinv(imu_v_c.T)
        
        mean_this = pd.DataFrame({'id': df_c.loc[i, 'id'],'emg_ft_mean': [emg_c_mean], 'synergy': [synergy_c]})
        mean_synergy_df = pd.concat([mean_synergy_df, mean_this], ignore_index=True)

        # calculate synergy

    return mean_synergy_df



def ft_clustering2(emg_ft, cluster_range=(5,20)):
    '''
    # emg_ft.shape: (feature_length, features*channels)
    # 对行归一化
    emg_ft已经归一化
    return: 分类dataframe, label列表, 出现的列表
    df_c: 包含所有cluster位置的dataframe
    appear_list: 将c_pos按序号排列
    pos_list: appear_list对应的id (自定义的id, kmeans labels id)
    pos_list_converted: appear_list对应的行id(dataframe id)
    '''
    
    best_c_num, _ = best_cluster_num(cluster_range[0], cluster_range[1], emg_ft)
    kmeans = KMeans(n_clusters=best_c_num, random_state=0).fit(emg_ft)

    # 输出df
    label_list = [int(x) for x in kmeans.labels_]
    print("label_list", label_list)

    c_min_len = 3 # 暂时没有意义
    df_c, pos_list, pos_list_converted, appear_list = cluster_process(label_list, c_min_len)
    # print("ppooss:", df_c, pos_list, pos_list_converted, appear_list)

    appear_list_full = appear_list.copy()
    appear_list_full.append(len(kmeans.labels_))


    return df_c, pos_list, pos_list_converted, appear_list, appear_list_full



from sklearn.model_selection import train_test_split
from collections import Counter

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def knn_learning(emg_ft, m_label, nn=10):
    
    X = emg_ft.copy()
    y = m_label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=0)

    clf = Pipeline(
        steps=[("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=nn))]
    )
    clf.set_params(knn__weights="distance").fit(X_train, y_train)
    pred_label = [clf.predict([x])[0] for x in X_test]
    result = [1 if pred_label[i]==y_test[i] else 0 for i in range(len(y_test))]
    test_accuracy = Counter(result)[1]/len(y_test)
    
    return clf, test_accuracy


def ft_clustering3(emg_ft, m_label, nn=10):
    
    clf, test_accuracy = knn_learning(emg_ft, m_label, nn=nn)
    
    # 输出df
    label_list = [clf.predict([x])[0] for x in emg_ft]
    # print("label_list", label_list)
    
    df_c, pos_list, pos_list_converted, appear_list_full = cluster_process2(label_list)

    return clf, df_c, pos_list, pos_list_converted, appear_list_full



# 找到一个label的cluster的所有位置
def cluster_appear2(list, find_value):
    '''
    list: label列表, 必须是list类型
    find_value: node/label的编号
    c_min_len: 最小cluster长度
    return: 所有find_value的聚类的位置
    '''

    all_pos = unique_index(list, find_value)
    v_appear = []

    for i, pos in enumerate(all_pos):
        if i == 0:
            v_appear.append(pos)
        else:
            if all_pos[i-1] != all_pos[i]-1:
                v_appear.append(pos)

    return v_appear


def cluster_process2(label_list):
    '''
    list: label列表, 必须是list类型
    c_min_len: 最小cluster长度
    return-appear_df: 包含所有cluster位置的dataframe
    return-c_sort: cluster出现的顺序
    '''
    
    # 创建空df，list
    appear_df = pd.DataFrame(columns=['id', 'c_pos'])
    appear_list = []

    label_id = remove_duplicates(label_list)
    label_id_valid = label_id.copy()

    
    for i in label_id:
        c_pos = cluster_appear2(label_list, i)
        
        if c_pos !=[]:
            ser_appear = pd.Series({'id':i, 'c_pos': c_pos})
            appear_df = pd.concat([appear_df, ser_appear.to_frame().T], ignore_index=True)
            appear_list = appear_list + c_pos
        else:
            label_id_valid.remove(i)
    
    
    appear_list.sort()
    c_sort = [label_list[i] for i in appear_list]
    pos_list = [int(x) for x in c_sort]
    
    convert_list = appear_df['id'].tolist()
    pos_list_converted = [convert_list.index(i) for i in pos_list]

    appear_list_full = appear_list.copy()
    appear_list_full.append(len(label_list))

    return appear_df, pos_list, pos_list_converted, appear_list_full


def shift_emg_ft(data_df, shift_len):
    data_df2 = data_df.copy()
    data_df2['emg_ft'] = data_df['emg_ft'].shift(-shift_len)
    data_df3 = data_df2.drop(np.arange(data_df.shape[0]-shift_len,data_df.shape[0]), axis=0)
    return data_df3

