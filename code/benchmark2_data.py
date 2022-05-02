import pandas as pd
import os
import multiprocessing as mp
from multiprocessing import Manager, Lock

file_path="/fsa/home/ww_duyy/hyc/data/Watkins/pred_output"
'''
csv_list=['new_cdigmp.csv', 'new_cgamp.csv', 'new_cobalamin.csv',
'new_fmn.csv', 'new_gln2_monomer.csv', 'new_guanine.csv',  'new_mn.csv', 'new_nada.csv',
'new_neomycin.csv', 'new_nico.csv', 'new_preq1.csv', 'new_prpp.csv', 'new_sam3.csv',
'new_thf.csv', 'new_thim.csv', 'new_ykoy.csv'] # 16RNAs
'''
# 'rna_name'字段用于指示某一rna分子，'structure_name'表示去除.pdb后缀的文件名，'pred'即ares分数
df_b2=pd.read_csv("/fsa/home/ww_duyy/hyc/data/notebooks/benchmark2/all_data_scores.csv")
#benchmark2_data.py:14: DtypeWarning: Columns (1,10,11,12,13,48,49,53,62,320,321) have mixed types. Specify dtype option on import or set low_memory=False.
df_b2.drop(df_b2.columns[0],axis=1,inplace=True)

new_df_b2 = pd.DataFrame()
mgr = Manager()
ns = mgr.Namespace()
ns.df = new_df_b2

def decoys(ns, rna_name, group, lock):
    df_t=pd.read_csv(os.path.join(file_path, "new_" + str(rna_name) +".csv"))
    for index, row in df_t.iterrows():
        tag=str(row['id'])[:-4]   #去掉.pdb后缀
        group.loc[(group['structure_name'] == tag) , 'pred'] = row['pred']
    with lock:
        ns.df= pd.concat([ns.df, group]) 

process_list=[]
lock=Lock()
for rna_name, group in df_b2.groupby('rna_name'):
    p = mp.Process(target=decoys, args=(ns, rna_name, group, lock,))
    p.start()
    process_list.append(p)
for p in process_list:
    p.join()

new_df_b2 = ns.df

new_df_b2.to_csv(os.path.join(file_path, "new_all_data_scores.csv"),index=True)