#ファインチューニングしたモデルを使って性能を確認する




import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from tqdm import tqdm

print("import完了")
"""
path1 = "/content/drive/My Drive/sotuken/tweet_data2send/tweet_data_zengo.csv"
path2 = "/content/drive/My Drive/sotuken/tweet_data2send/tweet_data_zengo_nakahigasi.csv"
path3 = "/content/drive/My Drive/sotuken/tweet_data2send/tweet_data_rep_inyou.csv"
path4 = "/content/drive/My Drive/sotuken/tweet_data2send/tweet_data_zengo_ALLnakahigasi.csv"
"""

path1 = "~/tweetdata/tweet_data_rep_inyou.csv"
path2 = "~/tweetdata/tweet_data_zengo.csv"
path3 = "~/tweetdata/tweet_data_zengo_nakahigasi.csv"
path4 = "~/tweetdata/tweet_data_zengo_ALLnakahigasi.csv"
path5 = "~/tweetdata/tweet_data_zengo_Psion_Nnakahigasi.csv"
path6 = "~/tweetdata/tweet_data_zengo_Psion_Nnakahigasi_more_clean.csv"
path7 = "~/tweetdata/tweet_data_zengo_exist.csv"
path8 = "~/tweetdata/tweet_data_hankyousi_seirei.csv"
path9 = "~/tweetdata/tweet_data_hankyousi_random.csv"
path10 = "~/tweetdata/tweet_data_nakahigasi_seirei.csv"
path11 = "~/tweetdata//tweet_data_nakahigasi_hurei.csv"
path12 = "~/tweetdata/tweet_data_hankyousi_str_hiniku.csv"
path13 = "~/tweetdata/tweet_data_hankyousi_kakko_hiniku.csv"


#more_clean データ
path1MC = "~/tweetdata/tweet_data_zengo_more_clean.csv"
path3MC = "~/tweetdata/tweet_data_rep_inyou.csv"
path5MC = "~/tweetdata/tweet_data_zengo_Psion_Nnakahigasi_more_clean.csv"
path6MC = "~/tweetdata/tweet_data_rep_inyou_exist.csv"
path7MC = "~/tweetdata/tweet_data_zengo_exist_more_clean.csv"
path8MC = "~/tweetdata/tweet_data_hankyousi_seirei_more_clean.csv"
path9MC = "~/tweetdata/tweet_data_hankyousi_random_more_clean.csv"
#path10MC = "~/tweetdata/tweet_data_nakahigasi_seirei_more_clean.csv"                                                                                                                                                                                                           
#path10MC = "~/tweetdata/tweet_data_seirei_thouhuku_URL_RT_OK.csv"
path10MC = "~/tweetdata/nakahigasi_seirei_NakaiLabeling_1.csv"
#path11MC = "~/tweetdata/tweet_data_nakahigasi_hurei_more_clean.csv"                                                                                                                                                                                                            
path11MC = "~/tweetdata/tweet_data_hurei_thouhuku_URL_RT_OK.csv"

path12MC = "~/tweetdata/tweet_data_hankyousi_str_hiniku_more_clean.csv"
path13MC = "~/tweetdata/tweet_data_hankyousi_kakko_hiniku_more_clean.csv"


#「皮肉」の単語をそのまま削除したデータ
path8MC_1 = "~/tweetdata/tweet_data_hankyousi_seirei_more_clean_remove_str_hiniku.csv"
path9MC_1 = "~/tweetdata/tweet_data_hankyousi_random_more_clean_remove_str_hiniku.csv"
path12MC_1 = "~/tweetdata/tweet_data_hankyousi_str_hiniku_more_clean_remove_str_hiniku.csv"
path13MC_1 = "~/tweetdata/tweet_data_hankyousi_kakko_hiniku_more_clean_remove_str_hiniku.csv"


#句読点で文を分割し、「皮肉」が含まれる箇所を削除したデータ                                                                                                                                                                                                                      
path8MC_2 = "~/tweetdata/tweet_data_hankyousi_seirei_more_clean_split_kutouten.csv"
path9MC_2 = "~/tweetdata/tweet_data_hankyousi_random_more_clean_split_kutouten.csv"
path12MC_2 = "~/tweetdata/tweet_data_hankyousi_str_hiniku_more_clean_split_kutouten.csv"
path13MC_2 = "~/tweetdata/tweet_data_hankyousi_kakko_hiniku_more_clean_split_kutouten.csv"


#「皮肉」にマスク、「皮肉」を含まない文に関しては名詞を一つランダムに選択しマスクしたデータ                                                                                                                                                                                     
path8MC_3 = "~/tweetdata/tweet_data_hankyousi_seirei_more_clean_mask.csv"
path9MC_3 = "~/tweetdata/tweet_data_hankyousi_random_more_clean_mask.csv"
path12MC_3 = "~/tweetdata/tweet_data_hankyousi_str_hiniku_more_clean_mask.csv"
path13MC_3 = "~/tweetdata/tweet_data_hankyousi_kakko_hiniku_more_clean_mask.csv"


#特定のフレーズ（という皮肉、等）を削除したデータ
path12MC_4 = "~/tweetdata/tweet_data_hankyousi_str_hiniku_more_clean_remove_specific_character.csv"




#1月23日追加　データを増やしたもの                                                                                                                                                                                                                                                

#more_clean データ                                                                                                                                                                                                                                                              
#path8_BIG_MC = "~/tweetdata/tweet_data_hankyousi_seirei_BIG_more_clean.csv"
path8_BIG_MC = "~/tweetdata/seirei_NakaiLabeling_1.csv"
path9_BIG_MC = "~/tweetdata/tweet_data_hankyousi_random_BIG_more_clean.csv"
path12_BIG_MC = "~/tweetdata/tweet_data_hankyousi_str_hiniku_BIG_more_clean.csv"
path13_BIG_MC = "~/tweetdata/tweet_data_hankyousi_kakko_hiniku_BIG_more_clean.csv"


#「皮肉」にマスク、「皮肉」を含まない文に関しては名詞を一つランダムに選択しマスクしたデータ                                                                                                                                                                                     
path8_BIG_MC_3 = "~/tweetdata/tweet_data_hankyousi_seirei_BIG_more_clean_mask.csv"
path9_BIG_MC_3 = "~/tweetdata/tweet_data_hankyousi_random_BIG_more_clean_mask.csv"
path12_BIG_MC_3 = "~/tweetdata/tweet_data_hankyousi_str_hiniku_BIG_more_clean_mask.csv"
path13_BIG_MC_3 = "~/tweetdata/tweet_data_hankyousi_kakko_hiniku_BIG_more_clean_mask.csv"


#特定のフレーズ（という皮肉、等）を削除したデータ                                                                                                                                                                                                                              
path12_BIG_MC_4 = "~/tweetdata/tweet_data_hankyousi_str_hiniku_BIG_more_clean_remove_specific_character.csv"















#テストデータは中東先輩のデータ
df_test_seirei = pd.read_csv(path10MC,encoding="utf_8",names=('honbun','rep','inyou','label'),dtype=object)
df_test_hurei = pd.read_csv(path11MC,encoding="utf_8",names=('honbun','rep','inyou','label'),dtype=object)

df_test_seirei = df_test_seirei.applymap(str)
df_test_hurei = df_test_hurei.applymap(str)

df_test_seirei['label'] = df_test_seirei['label'].astype('int')
df_test_hurei['label'] = df_test_hurei['label'].astype('int')


test_seirei_num = len(df_test_seirei)
df_test_hurei = df_test_hurei.iloc[:test_seirei_num]

print("テストデータ正例：",len(df_test_seirei))
print("テストデータ負例：",len(df_test_hurei))

df_test = pd.concat([df_test_seirei,df_test_hurei],ignore_index=True)
test_honbun_list = df_test["honbun"].tolist()
test_rep_list = df_test["rep"].tolist()
test_inyou_list = df_test["inyou"].tolist()
test_label_list = df_test["label"].tolist()



#bertで分類開始


Accuracy_list = []
Precision_list = []
Recall_list = []
F1_score_list = []

test = test_honbun_list
test2 = test_rep_list
test3 = test_inyou_list
seikai = test_label_list

for i in range(5):
    num = i+1


    model_name = "../model/hozon_sarcasm_detection_baceline" + str(num) + "/model_epoch2.model" 
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to("cuda:0")
    tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking") 
    model.eval()
    r_encorded =  tokenizer(test, padding=True, return_tensors='pt')                                                                                                                                                                                                               
    #r_encorded =  tokenizer(test,test2,test3,padding=True, return_tensors='pt')
    r_ds = TensorDataset(r_encorded['input_ids'].to("cuda:0"), r_encorded['attention_mask'].to("cuda:0"),r_encorded['token_type_ids'].to("cuda:0"))#####                                                                                                                            
    r_dl = train_dl = DataLoader(r_ds, batch_size=1) # バッチサイズを指定                                                                                                                                                                                                           
    sf = torch.nn.Softmax(dim=1)
    #result = []
    exec("result_{}=[]".format(num))
    for input_ids, attention_mask,token_type_ids in r_dl:
        r_classification = model(input_ids, attention_mask = attention_mask,token_type_ids=token_type_ids)
        #print("r_classification",r_classification)                                                                                                                                                                                                                                   
        r_soft = sf(r_classification.logits)
        #print("r_soft",r_soft)                                                                                                                                                                                                                                                       
        r_zero_one = torch.argmax(r_soft)
        #print("r_zero_one",r_zero_one)                                                                                                                                                                                                                                               
        #result.append(r_zero_one)
        exec("result_{}.append(r_zero_one)".format(num))
    #print(result)
    mozi = str(i) + "個目のモデルで評価終了"
    print("----------------------------------------")

print("result1〜5が正しく格納できているか確認")

print("result_1")
print(result_1[:10])
print("result_2")
print(result_2[:10])
print("result_3")
print(result_3[:10])
print("result_4")
print(result_4[:10])
print("result_5")
print(result_5[:10])

print("result の要素数確認")

print("result_1")
print(len(result_1))
print("result_2")
print(len(result_2))
print("result_3")
print(len(result_3))
print("result_4")
print(len(result_4))
print("result_5")
print(len(result_5))

result_ensemble=[]
#多数決により、アンサンブル学習
for i in range(len(result_1)):
    if (result_1[i] + result_2[i] + result_3[i] + result_4[i] + result_5[i] ) >= 3:
        result_ensemble.append(1) 
    else:
        result_ensemble.append(0)

print("アンサンブル学習のラベル確認")
print(result_ensemble)

TP = 0
FP = 0
TN = 0
FN = 0
seikaisu = 0
huseikaisu = 0

for i in range(len(result_ensemble)):
    if result_ensemble[i] == 0 and seikai[i]==0:
        TN += 1
    elif result_ensemble[i] ==0 and seikai[i]==1:
        FN += 1
    elif result_ensemble[i] ==1 and seikai[i]==0:
        FP += 1
    elif result_ensemble[i] == 1 and seikai[i]==1:
        TP += 1
    else:
        print("nanika_okasii")

seikaisu = TP + TN
huseikaisu = FP + FN

Accuracy = (TP+TN)/(TP+TN+FP+FN)
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1_score = (2 * Precision * Recall)/(Precision + Recall)

Accuracy_list.append(Accuracy)
Precision_list.append(Precision)
Recall_list.append(Recall)
F1_score_list.append(F1_score)

print("ベースラインのアンサンブル結果")
print("真陽性：",TP,"擬陽性：",FP,"真陰性：",TN,"偽陰性：",FN)
print("正解数：",seikaisu,"不正解数：",huseikaisu)
print("正解率：",Accuracy,"適合率：",Precision,"再現率：",Recall,"F値：",F1_score)
moziretu = str(num) + "回目の学習結果"
print(moziretu)
print("==============================================")



#####################################################################################################################################################################

Accuracy_list = []
Precision_list = []
Recall_list = []
F1_score_list = []

test = test_honbun_list
test2 = test_rep_list
test3 = test_inyou_list
seikai = test_label_list

for i in range(5):
    num = i+1


    model_name = "../model/hozon_sarcasm_detection_PU" + str(num) + "/model_epoch2.model" 
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to("cuda:0")
    tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking") 
    model.eval()
    r_encorded =  tokenizer(test, padding=True, return_tensors='pt')                                                                                                                                                                                                               
    #r_encorded =  tokenizer(test,test2,test3,padding=True, return_tensors='pt')
    r_ds = TensorDataset(r_encorded['input_ids'].to("cuda:0"), r_encorded['attention_mask'].to("cuda:0"),r_encorded['token_type_ids'].to("cuda:0"))#####                                                                                                                            
    r_dl = train_dl = DataLoader(r_ds, batch_size=1) # バッチサイズを指定                                                                                                                                                                                                           
    sf = torch.nn.Softmax(dim=1)
    #result = []
    exec("result_{}=[]".format(num))
    for input_ids, attention_mask,token_type_ids in r_dl:
        r_classification = model(input_ids, attention_mask = attention_mask,token_type_ids=token_type_ids)
        #print("r_classification",r_classification)                                                                                                                                                                                                                                   
        r_soft = sf(r_classification.logits)
        #print("r_soft",r_soft)                                                                                                                                                                                                                                                       
        r_zero_one = torch.argmax(r_soft)
        #print("r_zero_one",r_zero_one)                                                                                                                                                                                                                                               
        #result.append(r_zero_one)
        exec("result_{}.append(r_zero_one)".format(num))
    #print(result)
    mozi = str(i) + "個目のモデルで評価終了"
    print("----------------------------------------")

print("result1〜5が正しく格納できているか確認")

print("result_1")
print(result_1[:10])
print("result_2")
print(result_2[:10])
print("result_3")
print(result_3[:10])
print("result_4")
print(result_4[:10])
print("result_5")
print(result_5[:10])

print("result の要素数確認")

print("result_1")
print(len(result_1))
print("result_2")
print(len(result_2))
print("result_3")
print(len(result_3))
print("result_4")
print(len(result_4))
print("result_5")
print(len(result_5))

result_ensemble=[]
#多数決により、アンサンブル学習
for i in range(len(result_1)):
    if (result_1[i] + result_2[i] + result_3[i] + result_4[i] + result_5[i] ) >= 3:
        result_ensemble.append(1) 
    else:
        result_ensemble.append(0)

print("アンサンブル学習のラベル確認")
print(result_ensemble)

TP = 0
FP = 0
TN = 0
FN = 0
seikaisu = 0
huseikaisu = 0

for i in range(len(result_ensemble)):
    if result_ensemble[i] == 0 and seikai[i]==0:
        TN += 1
    elif result_ensemble[i] ==0 and seikai[i]==1:
        FN += 1
    elif result_ensemble[i] ==1 and seikai[i]==0:
        FP += 1
    elif result_ensemble[i] == 1 and seikai[i]==1:
        TP += 1
    else:
        print("nanika_okasii")

seikaisu = TP + TN
huseikaisu = FP + FN

Accuracy = (TP+TN)/(TP+TN+FP+FN)
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1_score = (2 * Precision * Recall)/(Precision + Recall)

Accuracy_list.append(Accuracy)
Precision_list.append(Precision)
Recall_list.append(Recall)
F1_score_list.append(F1_score)

print("PU学習のアンサンブル結果")
print("真陽性：",TP,"擬陽性：",FP,"真陰性：",TN,"偽陰性：",FN)
print("正解数：",seikaisu,"不正解数：",huseikaisu)
print("正解率：",Accuracy,"適合率：",Precision,"再現率：",Recall,"F値：",F1_score)
moziretu = str(num) + "回目の学習結果"
print(moziretu)
print("==============================================")


#####################################################################################################################################################################



#bertで分類開始


Accuracy_list = []
Precision_list = []
Recall_list = []
F1_score_list = []

test = test_honbun_list
test2 = test_rep_list
test3 = test_inyou_list
seikai = test_label_list

for i in range(5):
    num = i+1


    model_name = "../model/hozon_sarcasm_detection_PU_NU" + str(num) + "/model_epoch2.model" 
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to("cuda:0")
    tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking") 
    model.eval()
    r_encorded =  tokenizer(test, padding=True, return_tensors='pt')                                                                                                                                                                                                               
    #r_encorded =  tokenizer(test,test2,test3,padding=True, return_tensors='pt')
    r_ds = TensorDataset(r_encorded['input_ids'].to("cuda:0"), r_encorded['attention_mask'].to("cuda:0"),r_encorded['token_type_ids'].to("cuda:0"))#####                                                                                                                            
    r_dl = train_dl = DataLoader(r_ds, batch_size=1) # バッチサイズを指定                                                                                                                                                                                                           
    sf = torch.nn.Softmax(dim=1)
    #result = []
    exec("result_{}=[]".format(num))
    for input_ids, attention_mask,token_type_ids in r_dl:
        r_classification = model(input_ids, attention_mask = attention_mask,token_type_ids=token_type_ids)
        #print("r_classification",r_classification)                                                                                                                                                                                                                                   
        r_soft = sf(r_classification.logits)
        #print("r_soft",r_soft)                                                                                                                                                                                                                                                       
        r_zero_one = torch.argmax(r_soft)
        #print("r_zero_one",r_zero_one)                                                                                                                                                                                                                                               
        #result.append(r_zero_one)
        exec("result_{}.append(r_zero_one)".format(num))
    #print(result)
    mozi = str(i) + "個目のモデルで評価終了"
    print("----------------------------------------")

print("result1〜5が正しく格納できているか確認")

print("result_1")
print(result_1[:10])
print("result_2")
print(result_2[:10])
print("result_3")
print(result_3[:10])
print("result_4")
print(result_4[:10])
print("result_5")
print(result_5[:10])

print("result の要素数確認")

print("result_1")
print(len(result_1))
print("result_2")
print(len(result_2))
print("result_3")
print(len(result_3))
print("result_4")
print(len(result_4))
print("result_5")
print(len(result_5))

result_ensemble=[]
#多数決により、アンサンブル学習
for i in range(len(result_1)):
    if (result_1[i] + result_2[i] + result_3[i] + result_4[i] + result_5[i] ) >= 3:
        result_ensemble.append(1) 
    else:
        result_ensemble.append(0)

print("アンサンブル学習のラベル確認")
print(result_ensemble)

TP = 0
FP = 0
TN = 0
FN = 0
seikaisu = 0
huseikaisu = 0

for i in range(len(result_ensemble)):
    if result_ensemble[i] == 0 and seikai[i]==0:
        TN += 1
    elif result_ensemble[i] ==0 and seikai[i]==1:
        FN += 1
    elif result_ensemble[i] ==1 and seikai[i]==0:
        FP += 1
    elif result_ensemble[i] == 1 and seikai[i]==1:
        TP += 1
    else:
        print("nanika_okasii")

seikaisu = TP + TN
huseikaisu = FP + FN

Accuracy = (TP+TN)/(TP+TN+FP+FN)
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1_score = (2 * Precision * Recall)/(Precision + Recall)

Accuracy_list.append(Accuracy)
Precision_list.append(Precision)
Recall_list.append(Recall)
F1_score_list.append(F1_score)

print("PU_NU学習のアンサンブル結果")
print("真陽性：",TP,"擬陽性：",FP,"真陰性：",TN,"偽陰性：",FN)
print("正解数：",seikaisu,"不正解数：",huseikaisu)
print("正解率：",Accuracy,"適合率：",Precision,"再現率：",Recall,"F値：",F1_score)
moziretu = str(num) + "回目の学習結果"
print(moziretu)
print("==============================================")

print("学習完了")