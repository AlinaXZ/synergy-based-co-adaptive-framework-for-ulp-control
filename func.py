# 检测所有 element 出现在 list 或 1维 ndarray 中的位置
import numpy as np

def unique_index(list,element):
    return [i for (i,j) in enumerate(list) if j == element]


# 使用辅助集合保持顺序地去重
def remove_duplicates(lst):
    seen = set()
    unique_list = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            unique_list.append(item)
    return unique_list


def generate_motion(len_m, m1, m2):
    return np.hstack((np.random.normal(m1, 0.01, size=(len_m,1)), np.random.normal(m2, 0.01, size=(len_m,1)))).tolist()

def generate_training_data(train_list):
    '''
    默认label=0是静止
    example: train_list = [([0], 60, 0, 0), ([1], 120, 0, 1), ([2], 120, 0, -1)]
    m_between_len = 30
    '''
    m_list = []
    m_label = []
    for x in train_list:
        m = generate_motion(x[1], x[2], x[3])
        m_list = m_list + m
        m_label = m_label + x[0] * x[1]
    
    return m_list, m_label