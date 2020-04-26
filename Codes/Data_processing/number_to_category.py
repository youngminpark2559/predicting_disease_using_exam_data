# conda activate py36gputorch100 && \
# cd /mnt/external_disk/Companies/B_Link/Project_code/DeepLearning_predicting_disease/Codes/Data_processing && \
# rm e.l && python number_to_category.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
import pandas as pd
import numpy as np
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns',None)
import sys 
sys.path.append('..')
from collections import Counter

import utils as utils

# ================================================================================
def categorize_systolic_bp(loaded_csv):
  # 정상
  loaded_csv['BP_HIGH']=np.where(
    (loaded_csv['BP_HIGH']>=100)&(loaded_csv['BP_HIGH']<=119),0,loaded_csv['BP_HIGH'])
  # 경계
  loaded_csv['BP_HIGH']=np.where(
    ((120<=loaded_csv['BP_HIGH'])&(loaded_csv['BP_HIGH']<=129))|
    ((90<=loaded_csv['BP_HIGH'])&(loaded_csv['BP_HIGH']<=99)),
    1,loaded_csv['BP_HIGH'])
  # 주의
  loaded_csv['BP_HIGH']=np.where(
    ((130<=loaded_csv['BP_HIGH'])&(loaded_csv['BP_HIGH']<=139))|
    ((80<=loaded_csv['BP_HIGH'])&(loaded_csv['BP_HIGH']<=89)),
    2,loaded_csv['BP_HIGH'])
  # 위험
  loaded_csv['BP_HIGH']=np.where(
    ((140<=loaded_csv['BP_HIGH'])&(loaded_csv['BP_HIGH']<=149))|
    ((70<=loaded_csv['BP_HIGH'])&(loaded_csv['BP_HIGH']<=79)),
    3,loaded_csv['BP_HIGH'])
  # 고위험
  loaded_csv['BP_HIGH']=np.where(
    (loaded_csv['BP_HIGH']>=150)|
    ((10<=loaded_csv['BP_HIGH'])&(loaded_csv['BP_HIGH']<=69)),
    4,loaded_csv['BP_HIGH'])

  return loaded_csv

def categorize_diastolic_bp(loaded_csv):
  # 정상
  loaded_csv['BP_LWST']=np.where(
    (loaded_csv['BP_LWST']>=70)&(loaded_csv['BP_LWST']<=79),0,loaded_csv['BP_LWST'])
  # 경계
  loaded_csv['BP_LWST']=np.where(
    ((80<=loaded_csv['BP_LWST'])&(loaded_csv['BP_LWST']<=89))|
    ((60<=loaded_csv['BP_LWST'])&(loaded_csv['BP_LWST']<=69)),
    1,loaded_csv['BP_LWST'])
  # 주의
  loaded_csv['BP_LWST']=np.where(
    ((90<=loaded_csv['BP_LWST'])&(loaded_csv['BP_LWST']<=99))|
    ((50<=loaded_csv['BP_LWST'])&(loaded_csv['BP_LWST']<=59)),
    2,loaded_csv['BP_LWST'])
  # 위험
  loaded_csv['BP_LWST']=np.where(
    ((100<=loaded_csv['BP_LWST'])&(loaded_csv['BP_LWST']<=109))|
    ((40<=loaded_csv['BP_LWST'])&(loaded_csv['BP_LWST']<=49)),
    3,loaded_csv['BP_LWST'])
  # 고위험
  loaded_csv['BP_LWST']=np.where(
    (loaded_csv['BP_LWST']>=110)|
    ((10<=loaded_csv['BP_LWST'])&(loaded_csv['BP_LWST']<=39)),
    4,loaded_csv['BP_LWST'])

  return loaded_csv

def categorize_bmi(loaded_csv):
  # 정상
  loaded_csv['BMI']=np.where(
    (loaded_csv['BMI']>=18.5)&(loaded_csv['BMI']<=24.9),
    0,loaded_csv['BMI'])
  # 경계
  loaded_csv['BMI']=np.where(
    ((5<=loaded_csv['BMI'])&(loaded_csv['BMI']<=18.4))|
    ((25<=loaded_csv['BMI'])&(loaded_csv['BMI']<=29.9)),
    1,loaded_csv['BMI'])
  # 주의
  loaded_csv['BMI']=np.where(
    ((30<=loaded_csv['BMI'])&(loaded_csv['BMI']<=34.9)),
    2,loaded_csv['BMI'])
  # 위험
  loaded_csv['BMI']=np.where(
    ((35<=loaded_csv['BMI'])&(loaded_csv['BMI']<=39.9)),
    3,loaded_csv['BMI'])
  # 고위험
  loaded_csv['BMI']=np.where(
    (loaded_csv['BMI']>=40),
    4,loaded_csv['BMI'])

  return loaded_csv

def categorize_ldl(loaded_csv):
  # 정상
  loaded_csv['LDL_CHOLE']=np.where(
    (loaded_csv['LDL_CHOLE']>=60)&(loaded_csv['LDL_CHOLE']<=119),
    0,loaded_csv['LDL_CHOLE'])
  # 경계
  loaded_csv['LDL_CHOLE']=np.where(
    ((40<=loaded_csv['LDL_CHOLE'])&(loaded_csv['LDL_CHOLE']<=59))|
    ((120<=loaded_csv['LDL_CHOLE'])&(loaded_csv['LDL_CHOLE']<=129)),
    1,loaded_csv['LDL_CHOLE'])
  # 주의
  loaded_csv['LDL_CHOLE']=np.where(
    ((30<=loaded_csv['LDL_CHOLE'])&(loaded_csv['LDL_CHOLE']<=39))|
    ((130<=loaded_csv['LDL_CHOLE'])&(loaded_csv['LDL_CHOLE']<=139)),
    2,loaded_csv['LDL_CHOLE'])
  # 위험
  loaded_csv['LDL_CHOLE']=np.where(
    ((20<=loaded_csv['LDL_CHOLE'])&(loaded_csv['LDL_CHOLE']<=29))|
    ((140<=loaded_csv['LDL_CHOLE'])&(loaded_csv['LDL_CHOLE']<=149)),
    3,loaded_csv['LDL_CHOLE'])
  # 고위험
  loaded_csv['LDL_CHOLE']=np.where(
    (loaded_csv['LDL_CHOLE']>=150)|
    ((5<=loaded_csv['LDL_CHOLE'])&(loaded_csv['LDL_CHOLE']<=19)),
    4,loaded_csv['LDL_CHOLE'])

  return loaded_csv

def create_data_X(loaded_csv):
  # print("loaded_csv",loaded_csv)
  d={'systolic_bp':loaded_csv["BP_HIGH"],
     'diastolic_bp':loaded_csv["BP_LWST"],
     'bmi':loaded_csv["BMI"],
     'ldl':loaded_csv["LDL_CHOLE"]
  }
  data_X=pd.DataFrame(d)
  return data_X

def high_bp_label(loaded_csv):
  # print("loaded_csv",loaded_csv.shape)
  disease_class_1=(np.array(loaded_csv["systolic_bp"])==0)&(np.array(loaded_csv["diastolic_bp"])==0)
  disease_class_1=np.array(disease_class_1+0)
  # print("disease_class_0",list(disease_class_0))

  disease_class_2=((np.array(loaded_csv["systolic_bp"])==0)&(np.array(loaded_csv["diastolic_bp"])==1))|((np.array(loaded_csv["systolic_bp"])==1)&(np.array(loaded_csv["diastolic_bp"])==0))
  disease_class_2=np.array(disease_class_2+0)
  disease_class_2=np.where(disease_class_2==1,2,0)

  disease_class_3=((np.array(loaded_csv["systolic_bp"])==1)&(np.array(loaded_csv["diastolic_bp"])==1))
  disease_class_3=np.array(disease_class_3+0)
  disease_class_3=np.where(disease_class_3==1,3,0)

  disease_class_4=((np.array(loaded_csv["systolic_bp"])==0)&(np.array(loaded_csv["diastolic_bp"])==2))|((np.array(loaded_csv["systolic_bp"])==2)&(np.array(loaded_csv["diastolic_bp"])==0))
  disease_class_4=np.array(disease_class_4+0)
  disease_class_4=np.where(disease_class_4==1,4,0)

  disease_class_5=((np.array(loaded_csv["systolic_bp"])==2)&(np.array(loaded_csv["diastolic_bp"])==1))|((np.array(loaded_csv["systolic_bp"])==1)&(np.array(loaded_csv["diastolic_bp"])==2))
  disease_class_5=np.array(disease_class_5+0)
  disease_class_5=np.where(disease_class_5==1,5,0)

  disease_class_6=((np.array(loaded_csv["systolic_bp"])==0)&(np.array(loaded_csv["diastolic_bp"])==3))|((np.array(loaded_csv["systolic_bp"])==3)&(np.array(loaded_csv["diastolic_bp"])==0))
  disease_class_6=np.array(disease_class_6+0)
  disease_class_6=np.where(disease_class_6==1,6,0)

  disease_class_7=((np.array(loaded_csv["systolic_bp"])==2)&(np.array(loaded_csv["diastolic_bp"])==2))
  disease_class_7=np.array(disease_class_7+0)
  disease_class_7=np.where(disease_class_7==1,7,0)

  disease_class_8=((np.array(loaded_csv["systolic_bp"])==1)&(np.array(loaded_csv["diastolic_bp"])==3))|((np.array(loaded_csv["systolic_bp"])==3)&(np.array(loaded_csv["diastolic_bp"])==1))
  disease_class_8=np.array(disease_class_8+0)
  disease_class_8=np.where(disease_class_8==1,8,0)

  disease_class_9=((np.array(loaded_csv["systolic_bp"])==3)&(np.array(loaded_csv["diastolic_bp"])==2))|((np.array(loaded_csv["systolic_bp"])==2)&(np.array(loaded_csv["diastolic_bp"])==3))
  disease_class_9=np.array(disease_class_9+0)
  disease_class_9=np.where(disease_class_9==1,9,0)

  disease_class_10=((np.array(loaded_csv["systolic_bp"])==4)&(np.array(loaded_csv["diastolic_bp"])==0))|((np.array(loaded_csv["systolic_bp"])==0)&(np.array(loaded_csv["diastolic_bp"])==4))
  disease_class_10=np.array(disease_class_10+0)
  disease_class_10=np.where(disease_class_10==1,10,0)

  disease_class_11=((np.array(loaded_csv["systolic_bp"])==4)&(np.array(loaded_csv["diastolic_bp"])==1))|((np.array(loaded_csv["systolic_bp"])==1)&(np.array(loaded_csv["diastolic_bp"])==4))
  disease_class_11=np.array(disease_class_11+0)
  disease_class_11=np.where(disease_class_11==1,11,0)

  disease_class_12=((np.array(loaded_csv["systolic_bp"])==3)&(np.array(loaded_csv["diastolic_bp"])==3))
  disease_class_12=np.array(disease_class_12+0)
  disease_class_12=np.where(disease_class_12==1,12,0)

  disease_class_13=((np.array(loaded_csv["systolic_bp"])==4)&(np.array(loaded_csv["diastolic_bp"])==2))|((np.array(loaded_csv["systolic_bp"])==2)&(np.array(loaded_csv["diastolic_bp"])==4))
  disease_class_13=np.array(disease_class_13+0)
  disease_class_13=np.where(disease_class_13==1,13,0)

  disease_class_14=((np.array(loaded_csv["systolic_bp"])==4)&(np.array(loaded_csv["diastolic_bp"])==3))|((np.array(loaded_csv["systolic_bp"])==3)&(np.array(loaded_csv["diastolic_bp"])==4))
  disease_class_14=np.array(disease_class_14+0)
  disease_class_14=np.where(disease_class_14==1,14,0)

  disease_class_15=((np.array(loaded_csv["systolic_bp"])==4)&(np.array(loaded_csv["diastolic_bp"])==4))
  disease_class_15=np.array(disease_class_15+0)
  disease_class_15=np.where(disease_class_15==1,15,0)

  data_x=list(np.array(disease_class_1)+np.array(disease_class_2)+np.array(disease_class_3)+np.array(disease_class_4)+np.array(disease_class_5)+\
              np.array(disease_class_6)+np.array(disease_class_7)+np.array(disease_class_8)+np.array(disease_class_9)+np.array(disease_class_10)+\
              np.array(disease_class_11)+np.array(disease_class_12)+np.array(disease_class_13)+np.array(disease_class_14)+np.array(disease_class_15))
  return data_x

def wrong_fat_label(loaded_csv):
  disease_class_1=(np.array(loaded_csv["bmi"])==0)&(np.array(loaded_csv["ldl"])==0)
  disease_class_1=np.array(disease_class_1+0)
  # print("disease_class_0",list(disease_class_0))

  disease_class_2=((np.array(loaded_csv["bmi"])==0)&(np.array(loaded_csv["ldl"])==1))|((np.array(loaded_csv["bmi"])==1)&(np.array(loaded_csv["ldl"])==0))
  disease_class_2=np.array(disease_class_2+0)
  disease_class_2=np.where(disease_class_2==1,2,0)

  disease_class_3=((np.array(loaded_csv["bmi"])==1)&(np.array(loaded_csv["ldl"])==1))
  disease_class_3=np.array(disease_class_3+0)
  disease_class_3=np.where(disease_class_3==1,3,0)

  disease_class_4=((np.array(loaded_csv["bmi"])==0)&(np.array(loaded_csv["ldl"])==2))|((np.array(loaded_csv["bmi"])==2)&(np.array(loaded_csv["ldl"])==0))
  disease_class_4=np.array(disease_class_4+0)
  disease_class_4=np.where(disease_class_4==1,4,0)

  disease_class_5=((np.array(loaded_csv["bmi"])==2)&(np.array(loaded_csv["ldl"])==1))|((np.array(loaded_csv["bmi"])==1)&(np.array(loaded_csv["ldl"])==2))
  disease_class_5=np.array(disease_class_5+0)
  disease_class_5=np.where(disease_class_5==1,5,0)

  disease_class_6=((np.array(loaded_csv["bmi"])==0)&(np.array(loaded_csv["ldl"])==3))|((np.array(loaded_csv["bmi"])==3)&(np.array(loaded_csv["ldl"])==0))
  disease_class_6=np.array(disease_class_6+0)
  disease_class_6=np.where(disease_class_6==1,6,0)

  disease_class_7=((np.array(loaded_csv["bmi"])==2)&(np.array(loaded_csv["ldl"])==2))
  disease_class_7=np.array(disease_class_7+0)
  disease_class_7=np.where(disease_class_7==1,7,0)

  disease_class_8=((np.array(loaded_csv["bmi"])==1)&(np.array(loaded_csv["ldl"])==3))|((np.array(loaded_csv["bmi"])==3)&(np.array(loaded_csv["ldl"])==1))
  disease_class_8=np.array(disease_class_8+0)
  disease_class_8=np.where(disease_class_8==1,8,0)

  disease_class_9=((np.array(loaded_csv["bmi"])==3)&(np.array(loaded_csv["ldl"])==2))|((np.array(loaded_csv["bmi"])==2)&(np.array(loaded_csv["ldl"])==3))
  disease_class_9=np.array(disease_class_9+0)
  disease_class_9=np.where(disease_class_9==1,9,0)

  disease_class_10=((np.array(loaded_csv["bmi"])==4)&(np.array(loaded_csv["ldl"])==0))|((np.array(loaded_csv["bmi"])==0)&(np.array(loaded_csv["ldl"])==4))
  disease_class_10=np.array(disease_class_10+0)
  disease_class_10=np.where(disease_class_10==1,10,0)

  disease_class_11=((np.array(loaded_csv["bmi"])==4)&(np.array(loaded_csv["ldl"])==1))|((np.array(loaded_csv["bmi"])==1)&(np.array(loaded_csv["ldl"])==4))
  disease_class_11=np.array(disease_class_11+0)
  disease_class_11=np.where(disease_class_11==1,11,0)

  disease_class_12=((np.array(loaded_csv["bmi"])==3)&(np.array(loaded_csv["ldl"])==3))
  disease_class_12=np.array(disease_class_12+0)
  disease_class_12=np.where(disease_class_12==1,12,0)

  disease_class_13=((np.array(loaded_csv["bmi"])==4)&(np.array(loaded_csv["ldl"])==2))|((np.array(loaded_csv["bmi"])==2)&(np.array(loaded_csv["ldl"])==4))
  disease_class_13=np.array(disease_class_13+0)
  disease_class_13=np.where(disease_class_13==1,13,0)

  disease_class_14=((np.array(loaded_csv["bmi"])==4)&(np.array(loaded_csv["ldl"])==3))|((np.array(loaded_csv["bmi"])==3)&(np.array(loaded_csv["ldl"])==4))
  disease_class_14=np.array(disease_class_14+0)
  disease_class_14=np.where(disease_class_14==1,14,0)

  disease_class_15=((np.array(loaded_csv["bmi"])==4)&(np.array(loaded_csv["ldl"])==4))
  disease_class_15=np.array(disease_class_15+0)
  disease_class_15=np.where(disease_class_15==1,15,0)

  data_x=list(np.array(disease_class_1)+np.array(disease_class_2)+np.array(disease_class_3)+np.array(disease_class_4)+np.array(disease_class_5)+\
              np.array(disease_class_6)+np.array(disease_class_7)+np.array(disease_class_8)+np.array(disease_class_9)+np.array(disease_class_10)+\
              np.array(disease_class_11)+np.array(disease_class_12)+np.array(disease_class_13)+np.array(disease_class_14)+np.array(disease_class_15))
  
  return data_x

def upsample(train_data,column_name):
  
  key_of_max_value=max(dict(Counter(train_data[column_name])), key=dict(Counter(train_data[column_name])).get)
  # print("key_of_max_value",key_of_max_value)
  key_list=list(range(1,16))
  key_list.remove(key_of_max_value)
  # print("key_list",key_list)

  num_of_data_of_one_class=train_data[train_data[column_name]==key_of_max_value].shape[0]
  # print("num_of_data_of_one_class",num_of_data_of_one_class)
  # 84860

  df_list=[]
  for i in key_list:
    # replicated=[train_data[train_data["wrong_fat"]==i]]*int(num_of_data_of_one_class/train_data[train_data["wrong_fat"]==i].shape[0])
    replicated=pd.concat([train_data[train_data[column_name]==i]]*int(num_of_data_of_one_class/train_data[train_data[column_name]==i].shape[0]),ignore_index=False)
    # print("replicated",replicated)
    # train_data2.append(replicated,ignore_index=True)
    # train_data2.concat(replicated,ignore_index=True)
    df_list.append(replicated)

  merged_df=pd.concat(df_list,axis=0)
  # print("merged_df",merged_df)
  # print("aff",Counter(merged_df[column_name]))
  
  return merged_df

# ================================================================================
loaded_csv=utils.load_csv('../../Data/WithBMI/NHID2013.csv')
# print("loaded_csv",loaded_csv)

categorized_systolic_bp=categorize_systolic_bp(loaded_csv)
print(Counter(list(categorized_systolic_bp['BP_HIGH'])))

categorized_diastolic_bp=categorize_diastolic_bp(categorized_systolic_bp)
print(Counter(list(categorized_diastolic_bp['BP_LWST'])))

categorized_bmi=categorize_bmi(categorized_diastolic_bp)
categorized_bmi.BMI=categorized_bmi.BMI.astype(int)
print(Counter(list(categorized_bmi['BMI'])))

categorized_ldl=categorize_ldl(categorized_bmi)
print(Counter(list(categorized_ldl['LDL_CHOLE'])))

data_X=create_data_X(categorized_ldl)

high_bp_label_data=high_bp_label(data_X)
# print("high_bp_label_data",high_bp_label_data)
print(Counter(high_bp_label_data))

wrong_fat_label_data=wrong_fat_label(data_X)
# print("wrong_fat_label_data",wrong_fat_label_data)
print(Counter(wrong_fat_label_data))

# Full train data X and y
data_X["high_bp"]=high_bp_label_data
data_X["wrong_fat"]=wrong_fat_label_data
# print("data_X",data_X)

upsampled_df_by_wrong_fat=upsample(data_X,"wrong_fat")
upsampled_df_by_high_bp=upsample(upsampled_df_by_wrong_fat,"high_bp")
# merged_df=pd.concat([upsampled_df_by_wrong_fat,upsampled_df_by_high_bp],axis=0)
# print("merged_df",merged_df)

print(Counter(upsampled_df_by_high_bp["high_bp"]))
print(Counter(upsampled_df_by_high_bp["wrong_fat"]))


fn='../../Data/Train/NHID2013.csv'
upsampled_df_by_high_bp.to_csv(fn,sep=',',encoding='utf-8',index=False)

# print("categorized_bmi",categorized_bmi)
afaf
