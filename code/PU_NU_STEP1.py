# bert_hannkyousi_bunnruiki_1_ver1.py に対応


#半教師あり学習
#正例データとランダムに集めてきたデータを用いてファインチューニングを行う
#正例データの15%をランダムデータに混ぜて学習した分類器を作成する
#作成したモデルは"hozon"ディレクトリ内に保存


import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from tqdm import tqdm
import json
import sys

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




#################################################### config 


# config_path = sys.argv[1]

# with open(config_path, mode="r") as f:
#     config = json.load(f)

# path8_BIG_MC = config["path8_BIG_MC"]
# path9_BIG_MC = config["path9_BIG_MC"]
# print(sys.argv)
# print(config["path8_BIG_MC"])


##################################################


#df = pd.read_csv(path7MC,encoding="utf_8",names=('honbun','rep','inyou','label'))
df_seirei = pd.read_csv(path8_BIG_MC,encoding="utf_8",names=('honbun','rep','inyou','label'),dtype=object)#####################################################################
df_random = pd.read_csv(path9_BIG_MC,encoding="utf_8",names=('honbun','rep','inyou','label'),dtype=object)#####################################################################

#print("df_random")
#print(df_seirei["label"])
#print(df_random["label"])


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



#スパイデータとして、正例データのa%をスライスし、ランダムデータに加える

num_a = 15
spy_num = int(((len(df_seirei))//(100/num_a))+1) 
print("スパイの数：",spy_num)


#2022/12/01
#5種類のスパイを用意し、それぞれを用いた5パターンのモデルを作成するようにする


print("AAA")
df_seirei = df_seirei.sample(frac=1)
df_seirei_spy1 = df_seirei[:spy_num]
df_seirei_notspy1 = df_seirei[spy_num:]

df_seirei = df_seirei.sample(frac=1)
df_seirei_spy2 = df_seirei[:spy_num]
df_seirei_notspy2 = df_seirei[spy_num:]

df_seirei = df_seirei.sample(frac=1)
df_seirei_spy3 = df_seirei[:spy_num]
df_seirei_notspy3 = df_seirei[spy_num:]

df_seirei = df_seirei.sample(frac=1)
df_seirei_spy4 = df_seirei[:spy_num]
df_seirei_notspy4 = df_seirei[spy_num:]

df_seirei = df_seirei.sample(frac=1)
df_seirei_spy5 = df_seirei[:spy_num]
df_seirei_notspy5 = df_seirei[spy_num:]


print("異なるスパイが選ばれてるかチェック")
#print(df_seirei_spy1[:3])
print(df_seirei_spy1.index)
print("=============================")
#print(df_seirei_spy2[:3])
print(df_seirei_spy2.index)
print("=============================")
#print(df_seirei_spy3[:3])
print(df_seirei_spy3.index)
print("=============================")
#print(df_seirei_spy4[:3])
print(df_seirei_spy4.index)
print("=============================")
#print(df_seirei_spy5[:3])
print(df_seirei_spy5.index)
print("=============================")
print("上のデータが異なっていればＯＫ")


print("BBB")

#execを使って、5パターンのモデルを作成
#2022/12/01
for q in range(5):
    spy_pattern = q+1
    exec("df_seirei_spy = df_seirei_spy{}".format(spy_pattern))
    exec("df_seirei_notspy = df_seirei_notspy{}".format(spy_pattern))

    #スパイデータをランダムデータに加え、新しいカラム（new_label）に０を代入
    df_hurei = pd.concat([df_seirei_spy,df_random],ignore_index=True)
    df_seirei_notspy["new_label"] = 1
    df_hurei["new_label"] = 0


    #df とlistの行き来で発生するデータ形式のズレを矯正
    print("CCC")
    df_seirei_notspy = df_seirei_notspy.applymap(str)
    df_hurei = df_hurei.applymap(str)

    print("DDD")
    print("len(df_seirei)",len(df_seirei))
    print("len(df_random)",len(df_random))

    print("len(seirei_notspy)",len(df_seirei_notspy))
    print("len(df_hurei)",len(df_hurei))


    #作成したスパイ入りランダムデータを保存
    #2022/12/01
    #df_hurei.to_csv("~/soturon/tweetdata_random_include_spy.csv",header=False,index=False,encoding="utf-8")
    #save_path = "~/soturon/tweetdata_random_include_spy" + str(spy_pattern) + ".csv"
    #save_path = "~/shuusi/data/step1/tweetdata_random_include_spy" + str(spy_pattern) + ".csv"
    #save_path = "~/irony_detection/data/step1/tweetdata_random_include_spy" + str(spy_pattern) + ".csv"    
    save_path = "../data/step1/tweetdata_random_include_spy" + str(spy_pattern) + ".csv"    
    
    df_hurei.to_csv(save_path,header=False,index=False,encoding="utf-8")



    #正例データとスパイを忍ばせたランダムデータを結合
    df=pd.concat([df_seirei_notspy, df_hurei],ignore_index=True)



    #ここから下が半教師のために追加したコード


    df = df.sample(frac=1)
    df = df.applymap(str)
    df['label'] = df['label'].astype('int')
    df['new_label'] = df['new_label'].astype('int')


    honbun_lst = df["honbun"].tolist()
    rep_lst = df["rep"].tolist()
    inyou_lst = df["inyou"].tolist()
    label_lst = df["new_label"].tolist()
    past_label_lst = df["label"].tolist()


    print(" data kakunin")
    print(len(honbun_lst))
    print(len(rep_lst))
    print(len(inyou_lst))
    print(len(label_lst))
    print(past_label_lst)
    print(label_lst)
    print("↑データ数確認")






    #BERTで学習
    print("BERTに渡すデータ作成完了")

    Accuracy_list = []
    Precision_list = []
    Recall_list = []
    F1_score_list = []


    
    #データを代入

    x_train= honbun_lst#本文
    x_train2 = rep_lst
    x_train3 = inyou_lst

    y = label_lst#らべる
    y_train = torch.tensor(y, dtype=torch.int64)

    """
    test = test_honbun_list
    test2 = test_rep_list
    test3 = test_inyou_list
    seikai = test_label_list
    """
    #2022/12/01
    save_path = "../model/hozon_spy_RN" + str(spy_pattern)

    # 学習
    model = BertForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking", num_labels=2).to("cuda:0")
    tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")


    #x_train_encorded = tokenizer(x_train, padding=True, return_tensors='pt')#ここにリスト型で学習データを入力する
    x_train_encorded = tokenizer(x_train,x_train2,x_train3, padding=True, return_tensors='pt')#ここにリスト型で学習データを入力する



    #train_ds = TensorDataset(x_train_encorded['input_ids'].to("cuda:0"), x_train_encorded['attention_mask'].to("cuda:0"), y_train.to("cuda:0"))#トークンタイプIDSで1文か2文か差別化
    train_ds = TensorDataset(x_train_encorded['input_ids'].to("cuda:0"), x_train_encorded['attention_mask'].to("cuda:0"), x_train_encorded['token_type_ids'].to("cuda:0"),y_train.to("cuda:0"))#トークンタイプIDSで1文か2文か差別化



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

        mozi = "{}エポック目のデータを保存".format(epoch)

        print(mozi)
        out_path = save_path + "/model_epoch{}.model".format(epoch)
        model.save_pretrained(out_path)
    
    words = str(spy_pattern)+"種類目のスパイで学習終了"
    print(words)
