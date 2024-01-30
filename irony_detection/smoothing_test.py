# bert_hannkyousi_bunnruiki_2_RP_ver6.py に対応


#hannkyousi_bunnruiki_2で作成したモデルを使って、しきい値を定める
#文中に「皮肉」を含むツイートのうち、皮肉である確率がしきい値を上回るデータをＲＰとして抽出
#＃皮肉がタグ付けされたデータと信頼できる正例ＲＰを正例、ＲＮを負例として学習して新たな分類器２を作成する

#ver0：スパイを一度選出し、信頼できる正例を抽出して利用する
#ver1：スパイを複数回選出し、それぞれのパターン全てで信頼できる正例とみなされたものを最終的な信頼できる正例として利用する機能追加
#ver3: 信頼できる正例の数が信頼できる負例の数を超えている時、正しく同じ数だけ利用するできるよう変更
#ver4: 最終的な分類器の性能を出す際に、５つの平均ではなく、２０個の平均とする
#ver5: 閾値b％以下のものを確かな正例データとする操作とは別に、各スパイデータで信頼できる正例である確率が高いと判断されたデータを集め、信頼できる正例データとするコードの追加 
#ver6: 純正例数＋信頼できる正例数が信頼できる負例数よりも大きい時、データ数を揃える際に必ず純正例データはすべて用いるように改善

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
path11 = "~/tweetdata/tweet_data_nakahigasi_hurei.csv"
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





#path_spy = "~/soturon/tweetdata_str_hiniku_include_spy.csv"


#ｋ分割する

def slice_df(df: pd.DataFrame, size: int) -> list:
    """pandas.DataFrameを行数sizeずつにスライスしてリストに入れて返す"""
    previous_index = list(df.index)
    df = df.reset_index(drop=True)
    n = df.shape[0]
    list_indices = [(i, i+size) for i in range(0, n, size)]
    df_indices = [(i, i+size-1) for i in range(0, n, size)]
    sliced_dfs = []
    for i in range(len(df_indices)):
        begin_i, end_i = df_indices[i][0], df_indices[i][1]
        begin_l, end_l = list_indices[i][0], list_indices[i][1]
        df_i = df.loc[begin_i:end_i, :]
        df_i.index = previous_index[begin_l:end_l]
        sliced_dfs += [df_i]
    return sliced_dfs





#作成した信頼できる正例データを保存
#df_hurei_reliable.to_csv("~/soturon/tweetdata_seirei_reliable.csv",header=False,index=False,encoding="utf-8")
#df_hurei_reliable.to_csv("/home/aquamarine/sion/shuusi/data/step2/tweetdata_seirei_reliable.csv",header=False,index=False,encoding="utf-8")
#df_hurei_reliable.to_csv("../data/step2/tweetdata_seirei_reliable_smoothing.csv",header=False,index=False,encoding="utf-8")

#作成したとても信頼できる正例データを保存                                                                                                                                                                                                                                         
#df_hurei_more_reliable.to_csv("~/soturon/tweetdata_seirei_more_reliable.csv",header=False,index=False,encoding="utf-8")
#df_hurei_more_reliable.to_csv("/home/aquamarine/sion/shuusi/data/step2/tweetdata_seirei_more_reliable.csv",header=False,index=False,encoding="utf-8")
#df_hurei_more_reliable.to_csv("../data/step2/tweetdata_seirei_more_reliable_smoothing.csv",header=False,index=False,encoding="utf-8")


df_hurei_reliable = pd.read_csv("../data/step2/tweetdata_seirei_reliable_smoothing.csv",encoding="utf_8",names=('honbun','rep','inyou','label'),dtype=object)

df_hurei_more_reliable = pd.read_csv("../data/step2/tweetdata_seirei_more_reliable_smoothing.csv",encoding="utf_8",names=('honbun','rep','inyou','label'),dtype=object)

tensor_key_num = 0.0038032096344977617



#ここから交差検証 


#信頼できる正例データのラベルを１に変更
#df_hurei_reliable.loc[0:len(df_hurei_reliable),"label"] = 1
#df_hurei_more_reliable.loc[0:len(df_hurei_more_reliable),"label"] = 1

#信頼できる正例データのラベルを,正例である確率に変更
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
RPtensor_list=[]
spy_tensor_list=[]
df = df_hurei_more_reliable

df = df.applymap(str)
df['label'] = df['label'].astype('int')




honbun_lst = df["honbun"].tolist()
rep_lst = df["rep"].tolist()
inyou_lst = df["inyou"].tolist()
label_lst = df["label"].tolist()


print("ＢＥＲＴ開始、RPの確率計算")

Accuracy_list = []
Precision_list = []
Recall_list = []
F1_score_list = []



#評価

test = honbun_lst
test2 = rep_lst
test3 = inyou_lst
seikai = label_lst
    
Ptensor_list = []
count = 0


model_name = "../model/hozon_spy_RP"  + str(1) + "/model_epoch2.model"
    
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to("cuda:0")
tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking") 
model.eval()
r_encorded =  tokenizer(test,test2,test3,padding=True, return_tensors='pt')

r_ds = TensorDataset(r_encorded['input_ids'].to("cuda:0"), r_encorded['attention_mask'].to("cuda:0"),r_encorded['token_type_ids'].to("cuda:0"))#####
r_dl = DataLoader(r_ds, batch_size=1) # バッチサイズを指定
sf = torch.nn.Softmax(dim=1)
result = []
for input_ids, attention_mask,token_type_ids in r_dl:
    r_classification = model(input_ids, attention_mask = attention_mask,token_type_ids=token_type_ids)
    #print("r_classification",r_classification)
    r_soft = sf(r_classification.logits)
    #print("r_soft",r_soft)
    r_zero_one = torch.argmax(r_soft)
    #print("r_zero_one",r_zero_one)
    result.append(r_zero_one)
    RPtensor_list.append(r_soft[count,1].item())

for i in range(len(RPtensor_list)):
    RPtensor_list[i] = round(RPtensor_list[i],7)

print("RPtensor_list")
print(RPtensor_list)

new_label_lst = []
for i in range(len(RPtensor_list)):
   x = RPtensor_list[i]
   a = ((tensor_key_num - x)/tensor_key_num)*0.5 + 0.5
   a = round(a,7)
   new_label_lst.append(a)



print("しきい値：",tensor_key_num)
print("確率のラベル確認")
print(new_label_lst)


#df_hurei_reliable.loc[0:len(df_hurei_reliable),"label"] = 1
#df_hurei_more_reliable.loc[0:len(df_hurei_more_reliable),"label"] = 1

df_hurei_reliable = pd.DataFrame((zip(honbun_lst,rep_lst,inyou_lst ,new_label_lst)),columns = ["honbun","rep","inyou","label"])



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


print(df_hurei_reliable)
print(df_hurei_more_reliable)
print("↑ラベルが２じゃなくて確率になってるか確認")
 
#信頼できる正例データと純正例データを結合
#ここで再び正例と負例の意味するところが逆転、通常通りになる
df_seirei = pd.read_csv(path8_BIG_MC,encoding="utf_8",names=('honbun','rep','inyou','label'),dtype=object)#####################################################################################################################

#ver6で追加
df_jun_seirei_ver6 = df_seirei
df_seirei_reliable_ver6 = df_hurei_reliable
df_seirei_more_reliable_ver6 = df_hurei_more_reliable

print("SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS")
print(df_seirei_reliable_ver6['label'])


df_seirei_more_reliable = pd.concat([df_seirei,df_hurei_more_reliable],ignore_index=True)
df_seirei = pd.concat([df_seirei,df_hurei_reliable],ignore_index=True)
print("信頼できる正例＋純正例数確認:",len(df_seirei))
print("とても信頼できる正例＋純正例数確認:",len(df_seirei_more_reliable))


#それぞれランダムに並び替える
df_seirei = df_seirei.sample(frac=1)
df_seirei_more_reliable = df_seirei_more_reliable.sample(frac=1)
#df_hurei_reliable = pd.read_csv("~/soturon/tweetdata_hurei_reliable.csv",encoding="utf_8",names=('honbun','rep','inyou','label'),dtype=object)
df_hurei_reliable = pd.read_csv("../data/step1/tweetdata_hurei_reliable.csv",encoding="utf_8",names=('honbun','rep','inyou','label'),dtype=object)

df_hurei_reliable = df_hurei_reliable.sample(frac=1)


print("信頼できる負例データ数：",len(df_hurei_reliable))
print("純正例データ＋信頼できる正例データ数：",len(df_seirei))
print("純正例データ＋とても信頼できる正例データ数：",len(df_seirei_more_reliable))

num_seirei = len(df_seirei)
num_seirei_more_reliable = len(df_seirei_more_reliable)
num_hurei_reliable = len(df_hurei_reliable)

#ver3で追加
#ver6で改善

num_df_jun_seirei_ver6 = len(df_jun_seirei_ver6)
num_df_seirei_reliable_ver6 = len(df_seirei_reliable_ver6)
num_df_seirei_more_reliable_ver6 = len(df_seirei_more_reliable_ver6)

if len(df_seirei)<=len(df_hurei_reliable):
    #信頼できる負例データから、正例データ数と同じ数だけランダムにデータを取ってくる操作を20回行い、20個のデータを作成する
    for i in range(20):
        num = i+1
        df_hurei_reliable = df_hurei_reliable.sample(frac=1)
        exec("df_hurei_{} = df_hurei_reliable[:num_seirei]".format(num))
else:
    #seireiとhureiを逆転　ＢＥＲＴに渡す際には逆転してても問題ないため
    for i in range(20):
        key_num = num_hurei_reliable - num_df_jun_seirei_ver6
        num = i+1

        df_seirei_reliable_ver6 = df_seirei_reliable_ver6.sample(frac=1)
        #df_seirei = df_seirei.sample(frac=1)

        df_hurei = df_seirei_reliable_ver6[:key_num]
        df_hurei = pd.concat([df_jun_seirei_ver6,df_hurei],ignore_index=True)
        exec("df_hurei_{} = df_hurei".format(num))
        
    df_seirei = df_hurei_reliable


    
#とても信頼できる正例データに関しても同様に
if len(df_seirei_more_reliable)<=len(df_hurei_reliable):
    #信頼できる負例データから、正例データ数と同じ数だけランダムにデータを取ってくる操作を20回行い、20個のデータを作成する                                                                                                                                                         
    for i in range(20):
        num = i+1
        df_hurei_reliable = df_hurei_reliable.sample(frac=1)
        exec("df_more_hurei_{} = df_hurei_reliable[:num_seirei_more_reliable]".format(num))
else:
    #seireiとhureiを逆転　ＢＥＲＴに渡す際には逆転してても問題ないため                                                                                                                                                                                                            
    for i in range(20):
        key_num = num_hurei_reliable - num_df_jun_seirei_ver6
        num = i+1
        
        df_seirei_more_reliable_ver6 = df_seirei_more_reliable_ver6.sample(frac=1)
        #df_seirei_more_reliable = df_seirei_more_reliable.sample(frac=1)

        df_more_hurei = df_seirei_more_reliable_ver6[:key_num]
        df_more_hurei = pd.concat([df_jun_seirei_ver6,df_more_hurei],ignore_index=True)
        exec("df_more_hurei_{} = df_more_hurei".format(num))
    df_seirei_more_reliable = df_hurei_reliable


    
print("seireiの数確認",len(df_seirei))
print("hurei1の数確認",len(df_hurei_1))
print("hurei2の数確認",len(df_hurei_2))
print("hurei3の数確認",len(df_hurei_3))
print("hurei4の数確認",len(df_hurei_4))
print("hurei5の数確認",len(df_hurei_5))

print("more_hurei1の数確認",len(df_more_hurei_1))
print("more_hurei2の数確認",len(df_more_hurei_2))
print("more_hurei3の数確認",len(df_more_hurei_3))
print("more_hurei4の数確認",len(df_more_hurei_4))
print("more_hurei5の数確認",len(df_more_hurei_5))


print("信頼できる負例データの中から、異なるデータを抽出してるか確認")
print("hurei1のインデックス",df_hurei_1.index)
print("hurei2のインデックス",df_hurei_2.index)
print("hurei3のインデックス",df_hurei_3.index)
print("hurei4のインデックス",df_hurei_4.index)
print("hurei5のインデックス",df_hurei_5.index)

print("とても信頼できる負例データの中から、異なるデータを抽出してるか確認")
print("more_hurei1のインデックス",df_more_hurei_1.index)
print("more_hurei2のインデックス",df_more_hurei_2.index)
print("more_hurei3のインデックス",df_more_hurei_3.index)
print("more_hurei4のインデックス",df_more_hurei_4.index)
print("more_hurei5のインデックス",df_more_hurei_5.index)


df_seirei = df_seirei.applymap(str)
df_hurei_1 = df_hurei_1.applymap(str)
df_hurei_2 = df_hurei_2.applymap(str)
df_hurei_3 = df_hurei_3.applymap(str)
df_hurei_4 = df_hurei_4.applymap(str)
df_hurei_5 = df_hurei_5.applymap(str)
df_hurei_6 = df_hurei_6.applymap(str)
df_hurei_7 = df_hurei_7.applymap(str)
df_hurei_8 = df_hurei_8.applymap(str)
df_hurei_9 = df_hurei_9.applymap(str)
df_hurei_10 = df_hurei_10.applymap(str)
df_hurei_11 = df_hurei_11.applymap(str)
df_hurei_12 = df_hurei_12.applymap(str)
df_hurei_13 = df_hurei_13.applymap(str)
df_hurei_14 = df_hurei_14.applymap(str)
df_hurei_15 = df_hurei_15.applymap(str)
df_hurei_16 = df_hurei_16.applymap(str)
df_hurei_17 = df_hurei_17.applymap(str)
df_hurei_18 = df_hurei_18.applymap(str)
df_hurei_19 = df_hurei_19.applymap(str)
df_hurei_20 = df_hurei_20.applymap(str)


df_seirei_more_reliable = df_seirei_more_reliable.applymap(str)
df_more_hurei_1 = df_more_hurei_1.applymap(str)
df_more_hurei_2 = df_more_hurei_2.applymap(str)
df_more_hurei_3 = df_more_hurei_3.applymap(str)
df_more_hurei_4 = df_more_hurei_4.applymap(str)
df_more_hurei_5 = df_more_hurei_5.applymap(str)
df_more_hurei_6 = df_more_hurei_6.applymap(str)
df_more_hurei_7 = df_more_hurei_7.applymap(str)
df_more_hurei_8 = df_more_hurei_8.applymap(str)
df_more_hurei_9 = df_more_hurei_9.applymap(str)
df_more_hurei_10 = df_more_hurei_10.applymap(str)
df_more_hurei_11 = df_more_hurei_11.applymap(str)
df_more_hurei_12 = df_more_hurei_12.applymap(str)
df_more_hurei_13 = df_more_hurei_13.applymap(str)
df_more_hurei_14 = df_more_hurei_14.applymap(str)
df_more_hurei_15 = df_more_hurei_15.applymap(str)
df_more_hurei_16 = df_more_hurei_16.applymap(str)
df_more_hurei_17 = df_more_hurei_17.applymap(str)
df_more_hurei_18 = df_more_hurei_18.applymap(str)
df_more_hurei_19 = df_more_hurei_19.applymap(str)
df_more_hurei_20 = df_more_hurei_20.applymap(str)







df_seirei['label'] = df_seirei['label'].astype('float16')
df_hurei_1['label'] = df_hurei_1['label'].astype('float16')
df_hurei_2['label'] = df_hurei_2['label'].astype('float16')
df_hurei_3['label'] = df_hurei_3['label'].astype('float16')
df_hurei_4['label'] = df_hurei_4['label'].astype('float16')
df_hurei_5['label'] = df_hurei_5['label'].astype('float16')
df_hurei_6['label'] = df_hurei_6['label'].astype('float16')
df_hurei_7['label'] = df_hurei_7['label'].astype('float16')
df_hurei_8['label'] = df_hurei_8['label'].astype('float16')
df_hurei_9['label'] = df_hurei_9['label'].astype('float16')
df_hurei_10['label'] = df_hurei_10['label'].astype('float16')
df_hurei_11['label'] = df_hurei_11['label'].astype('float16')
df_hurei_12['label'] = df_hurei_12['label'].astype('float16')
df_hurei_13['label'] = df_hurei_13['label'].astype('float16')
df_hurei_14['label'] = df_hurei_14['label'].astype('float16')
df_hurei_15['label'] = df_hurei_15['label'].astype('float16')
df_hurei_16['label'] = df_hurei_16['label'].astype('float16')
df_hurei_17['label'] = df_hurei_17['label'].astype('float16')
df_hurei_18['label'] = df_hurei_18['label'].astype('float16')
df_hurei_19['label'] = df_hurei_19['label'].astype('float16')
df_hurei_20['label'] = df_hurei_20['label'].astype('float16')

df_seirei_more_reliable['label'] = df_seirei_more_reliable['label'].astype('int')
df_more_hurei_1['label'] = df_more_hurei_1['label'].astype('int')
df_more_hurei_2['label'] = df_more_hurei_2['label'].astype('int')
df_more_hurei_3['label'] = df_more_hurei_3['label'].astype('int')
df_more_hurei_4['label'] = df_more_hurei_4['label'].astype('int')
df_more_hurei_5['label'] = df_more_hurei_5['label'].astype('int')
df_more_hurei_6['label'] = df_more_hurei_6['label'].astype('int')
df_more_hurei_7['label'] = df_more_hurei_7['label'].astype('int')
df_more_hurei_8['label'] = df_more_hurei_8['label'].astype('int')
df_more_hurei_9['label'] = df_more_hurei_9['label'].astype('int')
df_more_hurei_10['label'] = df_more_hurei_10['label'].astype('int')
df_more_hurei_11['label'] = df_more_hurei_11['label'].astype('int')
df_more_hurei_12['label'] = df_more_hurei_12['label'].astype('int')
df_more_hurei_13['label'] = df_more_hurei_13['label'].astype('int')
df_more_hurei_14['label'] = df_more_hurei_14['label'].astype('int')
df_more_hurei_15['label'] = df_more_hurei_15['label'].astype('int')
df_more_hurei_16['label'] = df_more_hurei_16['label'].astype('int')
df_more_hurei_17['label'] = df_more_hurei_17['label'].astype('int')
df_more_hurei_18['label'] = df_more_hurei_18['label'].astype('int')
df_more_hurei_19['label'] = df_more_hurei_19['label'].astype('int')
df_more_hurei_20['label'] = df_more_hurei_20['label'].astype('int')

#ここでせっかくデータフレームにしたデータをもう一度リスト化する    



seirei_honbun_list = df_seirei['honbun'].tolist()
seirei_rep_list = df_seirei['rep'].tolist()
seirei_inyou_list = df_seirei['inyou'].tolist()
seirei_label_list = df_seirei['label'].tolist()

print("AAAAAA")
print(seirei_label_list)


seirei_honbun_list_M = df_seirei_more_reliable['honbun'].tolist()
seirei_rep_list_M = df_seirei_more_reliable['rep'].tolist()
seirei_inyou_list_M = df_seirei_more_reliable['inyou'].tolist()
seirei_label_list_M = df_seirei_more_reliable['label'].tolist()




for i in range(20):
    num = i+1
    exec("hurei_honbun_list_{} = df_hurei_{}['honbun'].tolist()".format(num,num))
    exec("hurei_rep_list_{} = df_hurei_{}['rep'].tolist()".format(num,num))
    exec("hurei_inyou_list_{} = df_hurei_{}['inyou'].tolist()".format(num,num))
    exec("hurei_label_list_{} = df_hurei_{}['label'].tolist()".format(num,num))

#とても信頼できる負例データにも同様に                                                                                                                                                                                                                                            
for i in range(20):
    num = i+1
    exec("hurei_honbun_list_M{} = df_more_hurei_{}['honbun'].tolist()".format(num,num))
    exec("hurei_rep_list_M{} = df_more_hurei_{}['rep'].tolist()".format(num,num))
    exec("hurei_inyou_list_M{} = df_more_hurei_{}['inyou'].tolist()".format(num,num))
    exec("hurei_label_list_M{} = df_more_hurei_{}['label'].tolist()".format(num,num))


#hurei_honbun_list_1 = df_hurei_1['honbun'].tolist()
#hurei_rep_list_1 = df_hurei_1['rep'].tolist()
#hurei_inyou_list_1 = df_hurei_1['inyou'].tolist()
#hurei_label_list_1 = df_hurei_1['label'].tolist()

########################################################

#hurei_honbun_list_2 = df_hurei_2['honbun'].tolist()
#hurei_rep_list_2 = df_hurei_2['rep'].tolist()
#hurei_inyou_list_2 = df_hurei_2['inyou'].tolist()
#hurei_label_list_2 = df_hurei_2['label'].tolist()

########################################################

#hurei_honbun_list_3 = df_hurei_3['honbun'].tolist()
#hurei_rep_list_3 = df_hurei_3['rep'].tolist()
#hurei_inyou_list_3 = df_hurei_3['inyou'].tolist()
#hurei_label_list_3 = df_hurei_3['label'].tolist()

########################################################

#hurei_honbun_list_4 = df_hurei_4['honbun'].tolist()
#hurei_rep_list_4 = df_hurei_4['rep'].tolist()
#hurei_inyou_list_4 = df_hurei_4['inyou'].tolist()
#hurei_label_list_4 = df_hurei_4['label'].tolist()

########################################################                                                                                                                                                                                                                           
#hurei_honbun_list_5 = df_hurei_5['honbun'].tolist()
#hurei_rep_list_5 = df_hurei_5['rep'].tolist()
#hurei_inyou_list_5 = df_hurei_5['inyou'].tolist()
#hurei_label_list_5 = df_hurei_5['label'].tolist()

########################################################

#hurei_honbun_list_6 = df_hurei_6['honbun'].tolist()
#hurei_rep_list_6 = df_hurei_6['rep'].tolist()
#hurei_inyou_list_6 = df_hurei_6['inyou'].tolist()
#hurei_label_list_6 = df_hurei_6['label'].tolist()

########################################################                                                                                                                                                                                                                           






#徹夜作業中に手順間違えちゃったから、最後にリストの形のまま結合

for i in range(20):
    num = i+1
    exec("train_honbun_list_{} = seirei_honbun_list + hurei_honbun_list_{}".format(num,num))
    exec("train_rep_list_{} = seirei_rep_list + hurei_rep_list_{}".format(num,num))
    exec("train_inyou_list_{} = seirei_inyou_list + hurei_inyou_list_{}".format(num,num))
    exec("train_label_list_{} = seirei_label_list + hurei_label_list_{}".format(num,num))

#とても信頼できる負例データにも同様に                                                                                                                                                                                                                                           
for i in range(20):
    num = i+1
    exec("train_honbun_list_M{} = seirei_honbun_list_M + hurei_honbun_list_M{}".format(num,num))
    exec("train_rep_list_M{} = seirei_rep_list_M + hurei_rep_list_M{}".format(num,num))
    exec("train_inyou_list_M{} = seirei_inyou_list_M + hurei_inyou_list_M{}".format(num,num))
    exec("train_label_list_M{} = seirei_label_list_M + hurei_label_list_M{}".format(num,num))


#train_honbun_list_1 = seirei_honbun_list + hurei_honbun_list_1
#train_rep_list_1 = seirei_rep_list + hurei_rep_list_1
#train_inyou_list_1 = seirei_inyou_list + hurei_inyou_list_1
#train_label_list_1 = seirei_label_list + hurei_label_list_1

#######################################################

#train_honbun_list_2 = seirei_honbun_list + hurei_honbun_list_2
#train_rep_list_2 = seirei_rep_list + hurei_rep_list_2
#train_inyou_list_2 = seirei_inyou_list + hurei_inyou_list_2
#train_label_list_2 = seirei_label_list + hurei_label_list_2

#######################################################

#train_honbun_list_3 = seirei_honbun_list + hurei_honbun_list_3
#train_rep_list_3 = seirei_rep_list + hurei_rep_list_3
#train_inyou_list_3 = seirei_inyou_list + hurei_inyou_list_3
#train_label_list_3 = seirei_label_list + hurei_label_list_3

#######################################################

#train_honbun_list_4 = seirei_honbun_list + hurei_honbun_list_4
#train_rep_list_4 = seirei_rep_list + hurei_rep_list_4
#train_inyou_list_4 = seirei_inyou_list + hurei_inyou_list_4
#train_label_list_4 = seirei_label_list + hurei_label_list_4

#######################################################

#train_honbun_list_5 = seirei_honbun_list + hurei_honbun_list_5
#train_rep_list_5 = seirei_rep_list + hurei_rep_list_5
#train_inyou_list_5 = seirei_inyou_list + hurei_inyou_list_5
#train_label_list_5 = seirei_label_list + hurei_label_list_5

#######################################################

#train_honbun_list_6 = seirei_honbun_list + hurei_honbun_list_6
#train_rep_list_6 = seirei_rep_list + hurei_rep_list_6
#train_inyou_list_6 = seirei_inyou_list + hurei_inyou_list_6
#train_label_list_6 = seirei_label_list + hurei_label_list_6

#######################################################  



print("bertに渡すリストの数確認、下の数字が全部同じならＯＫ")

print("train_honbun_list_1:",len(train_honbun_list_1))
print("train_rep_list_1:",len(train_rep_list_1))
print("train_inyou_list_1:",len(train_inyou_list_1))
print("train_label_list_1:",train_label_list_1)


#print("train_honbun_list_M1:",len(train_honbun_list_M1))
#print("train_rep_list_M1:",len(train_rep_list_M1))
#print("train_inyou_list_M1:",len(train_inyou_list_M1))
#print("train_label_list_M1:",len(train_label_list_M1))


#テストデータは中東先輩のデータ
df_test_seirei = pd.read_csv(path10MC,encoding="utf_8",names=('honbun','rep','inyou','label'),dtype=object)
df_test_hurei = pd.read_csv(path11MC,encoding="utf_8",names=('honbun','rep','inyou','label'),dtype=object)

df_test_seirei = df_test_seirei.applymap(str)
df_test_hurei = df_test_hurei.applymap(str)

df_test_seirei['label'] = df_test_seirei['label'].astype('float16')
df_test_hurei['label'] = df_test_hurei['label'].astype('float16')




test_seirei_num = len(df_test_seirei)
df_test_hurei = df_test_hurei.iloc[:test_seirei_num]

print("テストデータ正例：",len(df_test_seirei))
print("テストデータ負例：",len(df_test_hurei))

df_test = pd.concat([df_test_seirei,df_test_hurei],ignore_index=True)
test_honbun_list = df_test["honbun"].tolist()
test_rep_list = df_test["rep"].tolist()
test_inyou_list = df_test["inyou"].tolist()
test_label_list = df_test["label"].tolist()



#BERTで学習
print("BERTに渡すデータ作成完了")

print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("信頼できる正例データを用いた交差検証開始")
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")


Accuracy_list = []
Precision_list = []
Recall_list = []
F1_score_list = []


for i in range(5):
  num = i+1
  exec("x_train = train_honbun_list_{}".format(num))
  exec("x_train2 = train_rep_list_{}".format(num))
  exec("x_train3 = train_inyou_list_{}".format(num))
  exec("y = train_label_list_{}".format(num))
  print("QQQQQQQQQQQQQ")
  print(y)
  y_train = torch.tensor(y, dtype=torch.float16)
  print(y_train)

  #exec("test = test_honbun_list_{}".format(num))
  #exec("test2 = test_rep_list_{}".format(num))
  #exec("test3 = test_inyou_list_{}".format(num))
  #exec("seikai = test_label_list_{}".format(num))
  test = test_honbun_list
  test2 = test_rep_list
  test3 = test_inyou_list
  seikai = test_label_list


  #save_path = "hozon_RN_RP" + str(num)
  save_path = "../model/hozon_sarcasm_detection_PU_NU" + str(num)
  """

  # 学習                                                                                                                                                                                                                                                                           
  model = BertForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking", num_labels=2).to("cuda:0")
  tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
  #x_train_encorded = tokenizer(x_train, padding=True, return_tensors='pt')#ここにリスト型で学習データを入力する
  x_train_encorded = tokenizer(x_train,x_train2,x_train3, padding=True, return_tensors='pt')#ここにリスト型で学習データを入力する
  #train_ds = TensorDataset(x_train_encorded['input_ids'].to("cuda:0"), x_train_encorded['attention_mask'].to("cuda:0"), y_train.to("cuda:0"))#トークンタイプIDSで1文か2文か差別化                                                                                                
  train_ds = TensorDataset(x_train_encorded['input_ids'].to("cuda:0"), x_train_encorded['attention_mask'].to("cuda:0"), x_train_encorded['token_type_ids'].to("cuda:0"),y_train.to("cuda:0"))#トークンタイプIDSで1文か2文か差別化                                                  

  from tqdm import tqdm
  train_dl = DataLoader(train_ds, batch_size=16, shuffle=True) # バッチサイズを指定
  optimizer = AdamW(model.parameters(), lr=2e-5) # 最適化手法を指定、正則化の指定も可能
  epochs = 3  # エポック数を指定                                                                                                                                                                                                                                                   
  for epoch in tqdm(range(epochs)):
    model.train()
    #for input_ids, attention_mask, y in train_dl:
    for input_ids, attention_mask, token_type_ids,zzz in train_dl:
      optimizer.zero_grad()
      #outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=None, labels=y)
      outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=zzz)
      loss = outputs.loss
      loss.backward()
      optimizer.step()
      
    out_path = save_path + "/model_epoch{}.model".format(epoch)
  
    model.save_pretrained(out_path)                                                                                                                                                                                                                                             
"""

#################################################################################################################################################
#実験環境
  num = i+1
  num_classes = 2
  exec("x_train = train_honbun_list_{}".format(num))
  exec("x_train2 = train_rep_list_{}".format(num))
  exec("x_train3 = train_inyou_list_{}".format(num))
  exec("y = train_label_list_{}".format(num))
  print("QQQQQQQQQQQQQ")
  print(y)
  y_train = torch.tensor(y, dtype=torch.int64)
  print(y_train)
  #exec("test = test_honbun_list_{}".format(num))
  #exec("test2 = test_rep_list_{}".format(num))
  #exec("test3 = test_inyou_list_{}".format(num))
  #exec("seikai = test_label_list_{}".format(num))
  test = test_honbun_list
  test2 = test_rep_list
  test3 = test_inyou_list
  seikai = test_label_list


  #save_path = "hozon_RN_RP" + str(num)
  save_path = "../model/hozon_sarcasm_detection_PU_NU" + str(num)


  model = BertForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking", num_labels=2).to("cuda:0")
  tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")

  # ダミーのデータ生成
  # ここではランダムに仮のデータを生成しているので、実際のデータに置き換えてください
  # x_train, y_train = ...

  # トークナイズとエンコード
  x_train_encorded = tokenizer(x_train, padding=True, return_tensors='pt')
  train_ds = TensorDataset(x_train_encorded['input_ids'].to("cuda:0"), x_train_encorded['attention_mask'].to("cuda:0"), y_train.to("cuda:0"))

  # DataLoaderの作成
  train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)

  # オプティマイザの設定
  optimizer = AdamW(model.parameters(), lr=2e-5)

  # エポック数
  epochs = 3

  # トレーニングループ
  for epoch in tqdm(range(epochs)):
      model.train()
      
      for input_ids, attention_mask, y in train_dl:
          optimizer.zero_grad()
          print("input:",input_ids)
          print("y:",y)
          outputs = model(input_ids, attention_mask=attention_mask, labels=y)
          loss = outputs.loss
          loss.backward()
          optimizer.step()
        
      out_path = save_path + "/model_epoch{}.model".format(epoch)
      model.save_pretrained(out_path)

###############################################################################################################################################



  print("信頼できる正例データを用いて学習完了")



  #評価                                                                                                                                                                                                            
  model.eval()
  #r_encorded =  tokenizer(test, padding=True, return_tensors='pt')                                                                                                                                                                                                               
  r_encorded =  tokenizer(test,test2,test3,padding=True, return_tensors='pt')
  r_ds = TensorDataset(r_encorded['input_ids'].to("cuda:0"), r_encorded['attention_mask'].to("cuda:0"),r_encorded['token_type_ids'].to("cuda:0"))#####                                                                                                                            
  r_dl = train_dl = DataLoader(r_ds, batch_size=1) # バッチサイズを指定                                                                                                                                                                                                           
  sf = torch.nn.Softmax(dim=1)
  result = []
  for input_ids, attention_mask,token_type_ids in r_dl:
    r_classification = model(input_ids, attention_mask = attention_mask,token_type_ids=token_type_ids)
    #print("r_classification",r_classification)                                                                                                                                                                                                                                   
    r_soft = sf(r_classification.logits)
    #print("r_soft",r_soft)                                                                                                                                                                                                                                                       
    r_zero_one = torch.argmax(r_soft)
    #print("r_zero_one",r_zero_one)                                                                                                                                                                                                                                               
    result.append(r_zero_one)
  #print(result)
  print("----------------------------------------")
  TP = 0
  FP = 0
  TN = 0
  FN = 0
  seikaisu = 0
  huseikaisu = 0


  
  #ここでは皮肉文であることをpositiveとする                                                                                                                                                                                                                                        
  for i in range(len(result)):
    if result[i] == 0 and seikai[i]==0:
      TN += 1
    elif result[i] ==0 and seikai[i]==1:
      FN += 1
    elif result[i] ==1 and seikai[i]==0:
      FP += 1
    elif result[i] == 1 and seikai[i]==1:
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


  print("真陽性：",TP,"擬陽性：",FP,"真陰性：",TN,"偽陰性：",FN)
  print("正解数：",seikaisu,"不正解数：",huseikaisu)
  print("正解率：",Accuracy,"適合率：",Precision,"再現率：",Recall,"F値：",F1_score)
  moziretu = str(num) + "回目の学習結果"
  print(moziretu)
  print("==============================================")




print("Accuracy_list")
print(Accuracy_list)
print("Precision_list")
print(Precision_list)
print("Recall_list")
print(Recall_list)
print("F1_score_list")
print(F1_score_list)

av_Accuracy = sum(Accuracy_list)/5
av_Precision = sum(Precision_list)/5
av_Recall = sum(Recall_list)/5
av_F1_score = sum(F1_score_list)/5

print("信頼できる正例を用いた半教師あり学習最終結果")
print("正解率：",av_Accuracy,"適合率：",av_Precision,"再現率：",av_Recall,"F値：",av_F1_score)

print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("ここからとても信頼できる正例データを用いた交差検証開始")
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")


Accuracy_list = []
Precision_list = []
Recall_list = []
F1_score_list = []
