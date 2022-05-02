# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 06:27:59 2022

@author: hyc
"""

#应该扫描各个puzzle的csv文件，与benchmark1.csv作比较
#复合（多条件）筛选出对应puzzle_number、tag的某一行，然后将该行的ares换成比如
#new_rna_puzzle_10.csv里的pred

"""
import pandas as pd
import os

file_path="/fsa/home/ww_duyy/hyc/data/Townshend/augmented_puzzles/decoys/pred_output"
file_arr = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", 
            "14b", "14f", "15", "17", "18", "19", "20", "21"]
natives="new_rna_puzzle_natives.csv" #特殊情况，单独处理
puzzle_number=file_arr.copy()
file_num=len(file_arr)
for i in range(file_num):
    file_arr[i] = "new_rna_puzzle_" + file_arr[i] +".csv" 

df_b1=pd.read_csv("/fsa/home/ww_duyy/hyc/data/notebooks/benchmark1/benchmark1.csv")

for i in range(file_num):
    df_t=pd.read_csv(os.path.join(file_path, file_arr[i]))
    for index,row in df_t.iterrows():
        tag=str(row['id'])[:-4]   #去掉.pdb后缀
        df_b1[(df_b1['tag'] == tag) & (df_b1['puzzle_number']==puzzle_number[i])]['ares'] = row['pred']
df_t=pd.read_csv(os.path.join(file_path, natives))
for index,row in df_t.iterrows():
    tag=str(row['id'])[:-4]   #去掉.pdb后缀
    df_b1[(df_b1['tag'] == tag) & (df_b1['source']== "native")]['ares'] = row['pred']


df_b1.to_csv(os.path.join(file_path, "new_benchmark1.csv"))
"""

#改进版
#换一种思路，先将benchmark1.csv按puzzle_number分组，然后对应组的puzzle文件仅需比较tag
#并引入多进程，效率会提高不少

import pandas as pd
import os
import multiprocessing as mp
from multiprocessing import Manager, Lock

file_path="/fsa/home/ww_duyy/hyc/data/Townshend/augmented_puzzles/decoys/pred_output"
natives="new_rna_puzzle_natives.csv" #特殊情况，单独处理

df_b1=pd.read_csv("/fsa/home/ww_duyy/hyc/data/notebooks/benchmark1/benchmark1.csv" , dtype={'puzzle_number': object})

new_df_b1 = pd.DataFrame()

mgr = Manager()
ns = mgr.Namespace()
ns.df = new_df_b1

def decoys(ns, puzzle_number, group, lock):
    df_t=pd.read_csv(os.path.join(file_path, "new_rna_puzzle_" + str(puzzle_number) +".csv"))
    for index,row in df_t.iterrows():
        tag=str(row['id'])[:-4]   #去掉.pdb后缀
        # if index % 1000 == 0:
        #     print(tag)
        #     print(group.loc[group['tag'] == tag), 'ares'])
        group.loc[(group['tag'] == tag) , 'ares'] = row['pred']
    #对于共享变量ns.df进行写操作时需加锁
    with lock:
        ns.df= pd.concat([ns.df, group])  
    #最后的new_df_b1没有decoys的部分，找不出bug在哪里
    # 会不会是多进程的原因，查了一下说是：
    #主进程与子进程是并发执行的，进程之间默认是不能共享全局变量的(子进程不能改变主进程中全局变量的值)。
    #建议查一下python如何在多进程间共享pandas.DataFrame
    #https://stackoverflow.com/questions/22487296/multiprocessing-in-python-sharing-large-object-e-g-pandas-dataframe-between
    #实在不行可以用一个列表存储各个group，最后再concat起来，但只是个无奈的解决方法
#FutureWarning: The frame.append method is deprecated and will be removed from 
#pandas in a future version. Use pandas.concat instead.

process_list=[]
lock=Lock()
for puzzle_number, group in df_b1.groupby('puzzle_number'):
    p = mp.Process(target=decoys, args=(ns, puzzle_number, group, lock,))
    p.start()
    process_list.append(p)
for p in process_list:
    p.join()

new_df_b1 = ns.df
new_df_b1.drop(new_df_b1[new_df_b1['source'] == 'native'].index, inplace=True)

for source, group in df_b1.groupby('source'):
    #这里不再需要global声明否则报错（因为这里作用域与其定义同级）：
    #SyntaxError: name 'new_df_b1' is assigned to before global declaration
    if str(source) == 'native':
        # print("native!!!")
        df_t=pd.read_csv(os.path.join(file_path, natives))
        for index,row in df_t.iterrows():
            tag=str(row['id'])[:-4]   #去掉.pdb后缀
            group.loc[group['tag'] == tag, 'ares'] = row['pred']
        new_df_b1= pd.concat([new_df_b1, group])
    else:
        continue

new_df_b1.to_csv(os.path.join(file_path, "new_benchmark1.csv"),index=False)



#test
#结论：修改groupby的group的值不会影响原DataFrame
"""
import numpy as np
import pandas as pd

df = pd.DataFrame({'str':['a', 'a', 'b', 'b', 'a'],
'no':['one', 'two', 'one', 'two', 'one'],
'data1':np.random.randn(5),
'data2':np.random.randn(5)})
print(df)

for no, group in df.groupby('no'):
    print(no,"\n",group)
    print(group.loc[(group['str']=='a'),'data1'])
    group.loc[(group['str']=='a'),'data1']=666
    print(group)
print(df)
"""
#output:
"""
str   no     data1     data2
0   a  one  0.778860 -1.978310
1   a  two  0.902356  1.545739
2   b  one -1.045822  1.206926
3   b  two  0.727399  0.391220
4   a  one  0.509008 -0.874452
one 
   str   no     data1     data2
0   a  one  0.778860 -1.978310
2   b  one -1.045822  1.206926
4   a  one  0.509008 -0.874452
0    0.778860
4    0.509008
Name: data1, dtype: float64
  str   no       data1     data2
0   a  one  666.000000 -1.978310
2   b  one   -1.045822  1.206926
4   a  one  666.000000 -0.874452
two 
   str   no     data1     data2
1   a  two  0.902356  1.545739
3   b  two  0.727399  0.391220
1    0.902356
Name: data1, dtype: float64
  str   no       data1     data2
1   a  two  666.000000  1.545739
3   b  two    0.727399  0.391220
  str   no     data1     data2
0   a  one  0.778860 -1.978310
1   a  two  0.902356  1.545739
2   b  one -1.045822  1.206926
3   b  two  0.727399  0.391220
4   a  one  0.509008 -0.874452
"""
