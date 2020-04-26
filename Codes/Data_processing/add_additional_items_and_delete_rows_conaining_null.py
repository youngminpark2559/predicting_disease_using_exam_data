# conda activate py36gputorch100 && \
# cd /mnt/external_disk/Companies/B_Link/Project_code/DeepLearning_predicting_disease/Codes/Data_processing && \
# rm e.l && python add_additional_items_and_delete_rows_conaining_null.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
import pandas as pd
import numpy as np
import os
import sys 
sys.path.append('..')

import utils as utils

# ================================================================================
def add_bmi(csv_df):
  bmi_list=np.round(np.array(csv_df["WEIGHT"])/(np.array(csv_df["HEIGHT"])/100)**2,1)
  # print("bmi_list",bmi_list)

  csv_df["BMI"]=bmi_list
  return csv_df

def create_directory(target_dir):
  if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# ================================================================================
loaded_csv=utils.load_csv('../../Data/NHID2013.csv')
# print("loaded_csv",loaded_csv)

remove_0_or_null_from_height=utils.remove_rows_containing_0_or_null(loaded_csv,"HEIGHT")
remove_0_or_null_from_weight=utils.remove_rows_containing_0_or_null(remove_0_or_null_from_height,"WEIGHT")
# print("remove_0_or_null_from_weight",remove_0_or_null_from_weight.shape)
remove_0_or_null_from_ldl=utils.remove_rows_containing_0_or_null(remove_0_or_null_from_weight,"LDL_CHOLE")
# print("remove_0_or_null_from_ldl",remove_0_or_null_from_ldl.shape)

df_with_bmi=add_bmi(remove_0_or_null_from_weight)

create_directory('../../Data/WithBMI/')

fn="../../Data/WithBMI/NHID2013.csv"
df_with_bmi.to_csv(fn,sep=',',encoding='utf-8',index=False)

