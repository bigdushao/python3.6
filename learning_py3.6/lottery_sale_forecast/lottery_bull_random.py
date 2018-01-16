# coding:utf-8

'''
随机生成中奖号码
'''

import random


def random_index(bull_num):
    '''
    根据球的数量随机产生球的下标
    根据下标选择一个新的号码
    :param bull_num: 
    :return: 
    '''
    if bull_num is int:
        return random.randint(0, bull_num)
    else:
        return random.randint(0, int(bull_num))
red_bull_num = list(range(1, 33))
blue_bull_num = list(range(1, 16))


def update_bull_num(bull_num, selected_bull_num):
    '''
    更新球的数组，将已经挑选过的数据剔除掉
    :param bull_num: 
    :param selected_bull_num: 
    :return: 
    '''
    new_bull = list((bull for bull in bull_num if bull != selected_bull_num))
    return new_bull


def get_random_bull(bull_num_list, random_index_func=random_index):
    '''
    随机选取一个号码
    :param bull_num_list: 
    :param random_index_func: 
    :return: 
    '''
    return bull_num_list[random_index_func(len(bull_num_list) - 1)]


def select_bull(red_bull_list, get_bull, select_num, bull_type):
    '''
    随机生成红球，根据红球的初始号码，选择红球的个数，获取号码的方法
    :param red_bull_list: 
    :param get_bull: 
    :param select_num: 
    :return: 
    '''
    for num in range(0, select_num):
        bull = get_bull(red_bull_list)
        red_bull_list = update_bull_num(red_bull_list, bull)
        print('第 %d 个%s是 %d' % (num + 1, bull_type, bull))

select_bull(red_bull_num, get_random_bull, 5, "红球")
select_bull(blue_bull_num, get_random_bull, 2, '蓝球')

