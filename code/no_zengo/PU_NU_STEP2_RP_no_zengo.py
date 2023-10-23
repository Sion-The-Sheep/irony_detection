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





#ここから下が半教師のために追加したコード
#確率をtensor_listに格納

#除外すべきデータのインデックスをjogai_listに格納する
jogai_list=[]
jogai_list1=[]
jogai_list2=[]
jogai_list3=[]
jogai_list4=[]
jogai_list5=[]


#ver5で追加 bunnruiki_1_RN_ver3からコピーしてるから名前がver3となっている。
#新たに、要素として（確率、インデックス）というタプルを要素に持つようなリストver3_listを作成                                                                                                                                                                                      
ver3_list1 = []
ver3_list2 = []
ver3_list3 = []
ver3_list4 = []
ver3_list5 = []


for q in range(5):
    spy_pattern = q+1

    print("=========================")
    print("=========================")
    words = str(spy_pattern) + "種類目のスパイでの操作開始"
    print(words)
    print("=========================")
    print("=========================")

    #path_spy = "~/soturon/tweetdata_str_hiniku_include_spy" + str(spy_pattern) + ".csv"
    path_spy = "~/shuusi/data/step2/tweetdata_str_hiniku_include_spy_no_zengo" + str(spy_pattern) + ".csv"
    #path_spy = "~/soturon/tweetdata_str_hiniku_include_spy_mask" + str(spy_pattern) + ".csv"###############################################################################################################maskの際はこっち
    df = pd.read_csv(path_spy,encoding="utf_8",names=('honbun','rep','inyou','label','new_label'),dtype=object)
    print("スパイ込みのランダムデータ数→",len(df))



    tensor_list=[]
    spy_tensor_list=[]


    df = df.sample(frac=1)
    df = df.applymap(str)
    df['label'] = df['label'].astype('int')
    df['new_label'] = df['new_label'].astype('int')



    honbun_lst = df["honbun"].tolist()
    rep_lst = df["rep"].tolist()
    inyou_lst = df["inyou"].tolist()
    label_lst = df["label"].tolist()
    new_label_lst = df["new_label"].tolist()

    print("↓データ数確認")
    print(len(honbun_lst))
    print(len(rep_lst))
    print(len(inyou_lst))
    print(len(label_lst))
    print(len(new_label_lst))
    print("1なら正例、0なら負例")
    print(label_lst)
    print(new_label_lst)
    print("↑データ数確認")



    print("ＢＥＲＴ開始、スパイデータの確率計算")

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

    #model_name = "/home/aquamarine/sion/soturon/hozon_rp" + str(spy_pattern) + "/model_epoch2.model"
    model_name = "/home/aquamarine/sion/shuusi/model/hozon_spy_RP_no_zengo"  + str(spy_pattern) + "/model_epoch2.model"
    #model_name = "/home/aquamarine/sion/soturon/hozon_rp_mask" + str(spy_pattern) + "/model_epoch2.model"########################################################################################MASK処理したデータ作成したモデルはこっち
    
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to("cuda:0")
    tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking") 
    model.eval()
    r_encorded =  tokenizer(test, padding=True, return_tensors='pt')
    #r_encorded =  tokenizer(test,test2,test3,padding=True, return_tensors='pt')

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
        tensor_list.append(r_soft[count,1].item())




    for i in range(len(df)):
        if label_lst[i] == 0 and new_label_lst[i] == 0:
            print("honbunn:",honbun_lst[i])
            spy_tensor_list.append(tensor_list[i])


    #昇順に並び替え
    spy_tensorted_list = sorted(spy_tensor_list,key=None,reverse=False)

    #確率が低い方からn番目の確率をしきい値とする
    #ここは手元の計算機でb=15%にするなりなんなり、自由にどうぞ
    tensor_key_num = spy_tensorted_list[0]################################################################################################################################################################################################################


    #print("spy_tensored_list:",spy_tensorted_list)
    #print("spyの数：",len(spy_tensor_list))
    print("しきい値：",tensor_key_num)





    #ランダムに集めたデータに関して、tensor_key_num以下の値を持つデータを除外
    #負例のtensor値をNtensor_list = []に格納                                                                                                                                                                                                                                          

    df = pd.read_csv(path12_BIG_MC_4,encoding="utf_8",names=('honbun','rep','inyou','label'),dtype=object)##############################################################################################################################################################

    df = df.applymap(str)
    df['label'] = df['label'].astype('int')


    honbun_lst = df["honbun"].tolist()
    rep_lst = df["rep"].tolist()
    inyou_lst = df["inyou"].tolist()
    label_lst = df["label"].tolist()


    print("↓データ数確認")
    print(len(honbun_lst))
    print(len(rep_lst))
    print(len(inyou_lst))
    print(len(label_lst))
    print("0なら正例、2なら負例")
    print(label_lst)
    print("↑データ数確認")



    test = honbun_lst
    test2 = rep_lst
    test3 = inyou_lst
    seikai = label_lst




    Ntensor_list = []

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to("cuda:0")
    tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    model.eval()
    r_encorded =  tokenizer(test, padding=True, return_tensors='pt')
    #r_encorded =  tokenizer(test,test2,test3,padding=True, return_tensors='pt')
    r_ds = TensorDataset(r_encorded['input_ids'].to("cuda:0"), r_encorded['attention_mask'].to("cuda:0"),r_encorded['token_type_ids'].to("cuda:0"))#####
    r_dl = DataLoader(r_ds, batch_size=1) # バッチサイズを指定
    sf = torch.nn.Softmax(dim=1)
    result = []


    Count = 0 #ver5新規            
    ver3_list = []


    
    for input_ids, attention_mask,token_type_ids in r_dl:
        r_classification = model(input_ids, attention_mask = attention_mask,token_type_ids=token_type_ids)
        #print("r_classification",r_classification)
        r_soft = sf(r_classification.logits)
        #print("r_soft",r_soft)####################
        r_zero_one = torch.argmax(r_soft)
        #print("r_zero_one",r_zero_one)
        result.append(r_zero_one)
        Ntensor_list.append(r_soft[count,1].item())

        ver3_list.append((Count,r_soft[count,1].item()))#ver5新規    
        Count += 1

    ver3_list.sort(key = lambda x: x[1])
    exec("ver3_list{} = ver3_list".format(spy_pattern))
    
        
    #tensor値が一定以上のデータのインデックスを記録
    jogai_list=[]
    for i in range(len(Ntensor_list)):
        if Ntensor_list[i]>tensor_key_num:
        #if Ntensor_list[i]<tensor_key_num:
            #if Ntensor_list[i]<0.51:
            #print(Ntensor_list[i])
            exec("jogai_list{}.append(i)".format(spy_pattern))
    words = str(spy_pattern) + "種類目のスパイでの除外データ選択終了"
    print(words)

print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("5パターンのスパイで実行終了")
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("1回目でで除外したインデックス")
print(jogai_list1)
print("除外した個数：",len(jogai_list1))
print("=========================")
print("2回目でで除外したインデックス")
print(jogai_list2)
print("除外した個数：",len(jogai_list2))
print("=========================")
print("3回目でで除外したインデックス")
print(jogai_list3)
print("除外した個数：",len(jogai_list3))
print("=========================")
print("4回目でで除外したインデックス")
print(jogai_list4)
print("除外した個数：",len(jogai_list4))
print("=========================")
print("5回目でで除外したインデックス")
print(jogai_list5)
print("除外した個数：",len(jogai_list5))
print("=========================")


print("##################################################################")
high_tensor_list = []
for i in range(100):
    high_tensor_list.append(ver3_list1[i][0])
    high_tensor_list.append(ver3_list2[i][0])
    high_tensor_list.append(ver3_list3[i][0])
    high_tensor_list.append(ver3_list4[i][0])
    high_tensor_list.append(ver3_list5[i][0])

data_set = set(high_tensor_list)
high_list = list(data_set)

print("確率が高いデータのインデックスが並んだリスト")
print(high_list)
print("その数：",len(high_list))


print("##################################################################")


        
for j in range(5):
    count = 0
    num = j + 1
    exec("jogai = jogai_list{}".format(num))
    for k in jogai:
        if k in jogai_list:
            count+=1
        else:
            jogai_list.append(k)

print("最終的なjogai_list")
print(jogai_list)
print("最終的なjogai_listの数：",len(jogai_list))
print("除外する前のデータ数：",len(df))







#ここから２つ目の分類器作成
print("ここからふたつ目の分類器作成")
df = pd.read_csv(path12_BIG_MC_4,encoding="utf_8",names=('honbun','rep','inyou','label'))#################################################################################################################################################################3



#しょうがないからＤＦ→リスト→ＤＦ

#信頼できない負例データを格納するリスト
hurei_not_reliable_honbun = []
hurei_not_reliable_rep = []
hurei_not_reliable_inyou = []
hurei_not_reliable_label = []


#信頼できる負例データを格納するリスト
ittan_list_honbun = df['honbun'].tolist()
ittan_list_rep = df['rep'].tolist()
ittan_list_inyou = df['inyou'].tolist()
ittan_list_label = df['label'].tolist()

"""
print(len(ittan_list_honbun))
print(len(ittan_list_rep))

print(len(ittan_list_inyou))
print(len(ittan_list_label))
"""

Ittan_honbun = []
Ittan_rep = []
Ittan_inyou = []
Ittan_label = []

high_Ittan_honbun = []
high_Ittan_rep = []
high_Ittan_inyou = []
high_Ittan_label = []


for i in range(len(ittan_list_honbun)):
    if i in jogai_list:
        #print(i)
        hurei_not_reliable_honbun.append(ittan_list_honbun[i])
        hurei_not_reliable_rep.append(ittan_list_rep[i])
        hurei_not_reliable_inyou.append(ittan_list_inyou[i])
        hurei_not_reliable_label.append(ittan_list_label[i])
    else:
        Ittan_honbun.append(ittan_list_honbun[i])
        Ittan_rep.append(ittan_list_rep[i])
        Ittan_inyou.append(ittan_list_inyou[i])
        Ittan_label.append(ittan_list_label[i])
        if i in high_list:
            high_Ittan_honbun.append(ittan_list_honbun[i])
            high_Ittan_rep.append(ittan_list_rep[i])
            high_Ittan_inyou.append(ittan_list_inyou[i])
            high_Ittan_label.append(ittan_list_label[i])


df_hurei_not_reliable = pd.DataFrame((zip(hurei_not_reliable_honbun,hurei_not_reliable_rep,hurei_not_reliable_inyou,hurei_not_reliable_label)),columns = ["honbun","rep","inyou","label"])
df_hurei_reliable = pd.DataFrame((zip(Ittan_honbun,Ittan_rep,Ittan_inyou,Ittan_label)),columns = ["honbun","rep","inyou","label"])
df_hurei_more_reliable = pd.DataFrame((zip(high_Ittan_honbun,high_Ittan_rep,high_Ittan_inyou,high_Ittan_label)),columns = ["honbun","rep","inyou","label"])

print("信頼できる正例数確認：",len(df_hurei_reliable))
print("とても信頼できる正例数：",len(df_hurei_more_reliable))

print("信頼できる正例作成完了")


#作成した信頼できる正例データを保存
#df_hurei_reliable.to_csv("~/soturon/tweetdata_seirei_reliable.csv",header=False,index=False,encoding="utf-8")
#df_hurei_reliable.to_csv("/home/aquamarine/sion/shuusi/data/step2/tweetdata_seirei_reliable.csv",header=False,index=False,encoding="utf-8")
df_hurei_reliable.to_csv("../data/step2/tweetdata_seirei_reliable_no_zengo.csv",header=False,index=False,encoding="utf-8")

#作成したとても信頼できる正例データを保存                                                                                                                                                                                                                                         
#df_hurei_more_reliable.to_csv("~/soturon/tweetdata_seirei_more_reliable.csv",header=False,index=False,encoding="utf-8")
#df_hurei_more_reliable.to_csv("/home/aquamarine/sion/shuusi/data/step2/tweetdata_seirei_more_reliable.csv",header=False,index=False,encoding="utf-8")
df_hurei_more_reliable.to_csv("../data/step2/tweetdata_seirei_more_reliable_no_zengo.csv",header=False,index=False,encoding="utf-8")







#ここから交差検証 


#信頼できる正例データのラベルを１に変更
df_hurei_reliable.loc[0:len(df_hurei_reliable),"label"] = 1
df_hurei_more_reliable.loc[0:len(df_hurei_more_reliable),"label"] = 1


print(df_hurei_reliable)
print(df_hurei_more_reliable)
print("↑ラベルが２じゃなくて１になってるか確認")
 
#信頼できる正例データと純正例データを結合
#ここで再び正例と負例の意味するところが逆転、通常通りになる
df_seirei = pd.read_csv(path8_BIG_MC,encoding="utf_8",names=('honbun','rep','inyou','label'),dtype=object)#####################################################################################################################

#ver6で追加
df_jun_seirei_ver6 = df_seirei
df_seirei_reliable_ver6 = df_hurei_reliable
df_seirei_more_reliable_ver6 = df_hurei_more_reliable

df_seirei_more_reliable = pd.concat([df_seirei,df_hurei_more_reliable],ignore_index=True)
df_seirei = pd.concat([df_seirei,df_hurei_reliable],ignore_index=True)
print("信頼できる正例＋純正例数確認:",len(df_seirei))
print("とても信頼できる正例＋純正例数確認:",len(df_seirei_more_reliable))


#それぞれランダムに並び替える
df_seirei = df_seirei.sample(frac=1)
df_seirei_more_reliable = df_seirei_more_reliable.sample(frac=1)
#df_hurei_reliable = pd.read_csv("~/soturon/tweetdata_hurei_reliable.csv",encoding="utf_8",names=('honbun','rep','inyou','label'),dtype=object)
df_hurei_reliable = pd.read_csv("/home/aquamarine/sion/shuusi/data/step1/tweetdata_hurei_reliable_no_zengo.csv",encoding="utf_8",names=('honbun','rep','inyou','label'),dtype=object)

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







df_seirei['label'] = df_seirei['label'].astype('int')
df_hurei_1['label'] = df_hurei_1['label'].astype('int')
df_hurei_2['label'] = df_hurei_2['label'].astype('int')
df_hurei_3['label'] = df_hurei_3['label'].astype('int')
df_hurei_4['label'] = df_hurei_4['label'].astype('int')
df_hurei_5['label'] = df_hurei_5['label'].astype('int')
df_hurei_6['label'] = df_hurei_6['label'].astype('int')
df_hurei_7['label'] = df_hurei_7['label'].astype('int')
df_hurei_8['label'] = df_hurei_8['label'].astype('int')
df_hurei_9['label'] = df_hurei_9['label'].astype('int')
df_hurei_10['label'] = df_hurei_10['label'].astype('int')
df_hurei_11['label'] = df_hurei_11['label'].astype('int')
df_hurei_12['label'] = df_hurei_12['label'].astype('int')
df_hurei_13['label'] = df_hurei_13['label'].astype('int')
df_hurei_14['label'] = df_hurei_14['label'].astype('int')
df_hurei_15['label'] = df_hurei_15['label'].astype('int')
df_hurei_16['label'] = df_hurei_16['label'].astype('int')
df_hurei_17['label'] = df_hurei_17['label'].astype('int')
df_hurei_18['label'] = df_hurei_18['label'].astype('int')
df_hurei_19['label'] = df_hurei_19['label'].astype('int')
df_hurei_20['label'] = df_hurei_20['label'].astype('int')

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
print("train_label_list_1:",len(train_label_list_1))

print("train_honbun_list_M1:",len(train_honbun_list_M1))
print("train_rep_list_M1:",len(train_rep_list_M1))
print("train_inyou_list_M1:",len(train_inyou_list_M1))
print("train_label_list_M1:",len(train_label_list_M1))


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
  y_train = torch.tensor(y, dtype=torch.int64)
  
  #exec("test = test_honbun_list_{}".format(num))
  #exec("test2 = test_rep_list_{}".format(num))
  #exec("test3 = test_inyou_list_{}".format(num))
  #exec("seikai = test_label_list_{}".format(num))
  test = test_honbun_list
  test2 = test_rep_list
  test3 = test_inyou_list
  seikai = test_label_list


  #save_path = "hozon_RN_RP" + str(num)
  save_path = "../model/hozon_sarcasm_detection_PU_NU_no_zengo" + str(num)

  # 学習                                                                                                                                                                                                                                                                           
  model = BertForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking", num_labels=2).to("cuda:0")
  tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
  x_train_encorded = tokenizer(x_train, padding=True, return_tensors='pt')#ここにリスト型で学習データを入力する
  #x_train_encorded = tokenizer(x_train,x_train2,x_train3, padding=True, return_tensors='pt')#ここにリスト型で学習データを入力する
  #train_ds = TensorDataset(x_train_encorded['input_ids'].to("cuda:0"), x_train_encorded['attention_mask'].to("cuda:0"), y_train.to("cuda:0"))#トークンタイプIDSで1文か2文か差別化                                                                                                
  train_ds = TensorDataset(x_train_encorded['input_ids'].to("cuda:0"), x_train_encorded['attention_mask'].to("cuda:0"), x_train_encorded['token_type_ids'].to("cuda:0"),y_train.to("cuda:0"))#トークンタイプIDSで1文か2文か差別化                                                  

  from tqdm import tqdm
  train_dl = DataLoader(train_ds, batch_size=16, shuffle=True) # バッチサイズを指定
  optimizer = AdamW(model.parameters(), lr=2e-5) # 最適化手法を指定、正則化の指定も可能
  epochs = 3  # エポック数を指定                                                                                                                                                                                                                                                   
  for epoch in tqdm(range(epochs)):
    model.train()
    #for input_ids, attention_mask, y in train_dl:
    for input_ids, attention_mask, token_type_ids,y in train_dl:
      optimizer.zero_grad()
      #outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=None, labels=y)
      outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=y)
      loss = outputs.loss
      loss.backward()
      optimizer.step()
      
    out_path = save_path + "/model_epoch{}.model".format(epoch)
    model.save_pretrained(out_path)                                                                                                                                                                                                                                             

  print("信頼できる正例データを用いて学習完了")


  #評価                                                                                                                                                                                                            
  model.eval()
  r_encorded =  tokenizer(test, padding=True, return_tensors='pt')                                                                                                                                                                                                               
  #r_encorded =  tokenizer(test,test2,test3,padding=True, return_tensors='pt')
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

