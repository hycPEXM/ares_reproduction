import pandas as pd
import os

file_path="/fsa/home/ww_duyy/hyc/data/Townshend/translation"

df_transl=pd.read_csv("/fsa/home/ww_duyy/hyc/data/Townshend/translation/translation.csv")
df_new=pd.read_csv("/fsa/home/ww_duyy/hyc/data/Townshend/translation/new_translation.csv")

for index,row in df_new.iterrows():
    tag=str(row['id'])[:-4]   #去掉.pdb后缀
    df_transl.loc[(df_transl['tag'] == tag), 'ares'] = row['pred']

df_transl.to_csv(os.path.join(file_path, "new_translation_new.csv"),index=False)