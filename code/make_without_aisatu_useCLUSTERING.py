#"~/soturon/tweetdata_hurei_reliable.csv"に存在する、挨拶を含む文を除外し、"~/soturon/tweetdata_hurei_reliable_without_aisatu_useCLUSTERING.csv"を作成する

#"/home/aquamarine/sion/shuusi/data/step1/tweetdata_hurei_reliable.csv"に存在する、挨拶を含む文を除外し、"../data/step1/tweetdata_hurei_reliable_without_aisatu_useCLUSTERING.csv"


import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from tqdm import tqdm

#新規追加                                                                                                                                                                                                                                                                          
import MeCab
import unidic
import random
mecab = MeCab.Tagger()

import csv
import pprint
import re

##新規追加
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


###新規追加




model = SentenceTransformer('all-MiniLM-L6-v2')
def calculate_similarity(text_a, text_b):
    # SentenceTransformerモデルの初期化
   # model = SentenceTransformer('all-MiniLM-L6-v2')

    # 文章Aと文章Bをベクトル化
    vec_a = model.encode([text_a])[0]
    vec_b = model.encode([text_b])[0]

    # ベクトル同士の類似度を計算
    similarity_score = cosine_similarity([vec_a], [vec_b])[0][0]

    return similarity_score

def filter_similar_tweets(tweet):
    n = len(tweet)
    sakujo_lst = []
    print("類似度計算開始")
    print("データ数は",len(tweet))
    for i in range(n):
        for j in range(n):
            if i != j and i not in sakujo_lst:
                similarity = calculate_similarity(tweet[i], tweet[j])
                print(i,j,similarity)
                if similarity > 0.5:
                    sakujo_lst.append(j)
    return sakujo_lst


print("import完了")



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





nnew_csv_inf = []
ccount = 0
count = 0
all_num = 0
#open_path = "tweetdata_hurei_reliable.csv"
open_path = "/home/aquamarine/sion/shuusi/data/step1/tweetdata_hurei_reliable.csv"


main_tweet_text_lst=[]
with open(open_path,'r',encoding="utf-8") as f:
  test=f.readlines()
  for moziretu in test:
    tweet_text = moziretu.split(",")
    main_tweet_text = tweet_text[0]
    main_tweet_text_lst.append(main_tweet_text)

#for i in range(len(main_tweet_text_lst)):
#  print(main_tweet_text_lst[i])

#リスト化完了

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

docs = np.array(main_tweet_text_lst)
vectorizer = TfidfVectorizer(use_idf=True, token_pattern=u'(?u)\\b\\w+\\b')
vecs = vectorizer.fit_transform(docs)

count = 0
#clusters = KMeans(n_clusters=4, random_state=0).fit_predict(vecs)
clusters = KMeans(n_clusters=2, random_state=0).fit_predict(vecs)

label_lst = []
for doc, cls in zip(docs, clusters):
  label_lst.append(cls)
#print(label_lst)
jogai=[]
nnnew_csv_inf = []
nnnew_csv_inf_0 = []
nnnew_csv_inf_1 = []
nnnew_csv_inf_2 = []
nnnew_csv_inf_3 = []


with open(open_path,'r',encoding="utf-8") as f:
  test=f.readlines()
  for i in range(len(test)):
    if label_lst[i] == 0:
      nnnew_csv_inf_0.append(test[i])
    elif label_lst[i] == 1:
      nnnew_csv_inf_1.append(test[i])
      jogai.append(test[i])
      #print("LABEL1",test[i])
    elif label_lst[i] == 2:
      print("LABEL2",test[i])
      #nnnew_csv_inf_2.append(test[i])
    elif label_lst[i] == 3:
      print("LABEL3",test[i])
      #nnnew_csv_inf_3 = []
"""
if len(nnnew_csv_inf_0) >= len(nnnew_csv_inf_1) and len(nnnew_csv_inf_0) >= len(nnnew_csv_inf_2):
  nnnew_csv_inf = nnnew_csv_inf_0
elif len(nnnew_csv_inf_1) >= len(nnnew_csv_inf_2):
   nnnew_csv_inf = nnnew_csv_inf_1
else:
  nnnew_csv_inf = nnnew_csv_inf_2
"""

nnnew_csv_inf = nnnew_csv_inf_0

with open("../data/step1/CLUSTERING_check.csv"  ,'w',encoding="utf-8") as f: 
    for i in nnnew_csv_inf_0:
       # print(i)
        f.write(i)
    f.write("/n")
    f.write("#####################################################################")
    f.write("#####################################################################")
    for i in nnnew_csv_inf_1:
       # print(i)
        f.write(i)
    f.write("/n")
    f.write("#####################################################################")
    f.write("#####################################################################")
    for i in nnnew_csv_inf_2:
       # print(i)
        f.write(i)


print("データ作成完了")



#with open("tweetdata_hurei_reliable_without_aisatu_useCLUSTERING3.csv",'w',encoding="utf-8") as f:
with open("../data/step1/tweetdata_hurei_reliable_without_aisatu_useCLUSTERING.csv"  ,'w',encoding="utf-8") as f: 
    for i in nnnew_csv_inf:
       # print(i)
        f.write(i)
print("データ作成完了")

print("ラベル0の数：",len(nnnew_csv_inf_0))
print("ラベル1の数：",len(nnnew_csv_inf_1))
print("ラベル2の数：",len(nnnew_csv_inf_2))
print("ラベル3の数：",len(nnnew_csv_inf_3))



#ここからは再起処理
for j in range(20):
  print("再起処理開始")
  nnew_csv_inf = []
  ccount = 0
  count = 0
  all_num = 0
  #open_path = "tweetdata_hurei_reliable.csv"
  #open_path = "/home/aquamarine/sion/shuusi/data/step1/tweetdata_hurei_reliable.csv"
  open_path = "../data/step1/tweetdata_hurei_reliable_without_aisatu_useCLUSTERING.csv"

  main_tweet_text_lst=[]
  with open(open_path,'r',encoding="utf-8") as f:
    test=f.readlines()
    for moziretu in test:
      tweet_text = moziretu.split(",")
      main_tweet_text = tweet_text[0]
      main_tweet_text_lst.append(main_tweet_text)

  #for i in range(len(main_tweet_text_lst)):
  #  print(main_tweet_text_lst[i])

  #リスト化完了


  docs = np.array(main_tweet_text_lst)
  vectorizer = TfidfVectorizer(use_idf=True, token_pattern=u'(?u)\\b\\w+\\b')
  vecs = vectorizer.fit_transform(docs)

  count = 0
  #clusters = KMeans(n_clusters=4, random_state=0).fit_predict(vecs)
  clusters = KMeans(n_clusters=2, random_state=0).fit_predict(vecs)

  label_lst = []
  for doc, cls in zip(docs, clusters):
    label_lst.append(cls)
  #print(label_lst)

  nnnew_csv_inf = []
  nnnew_csv_inf_0 = []
  nnnew_csv_inf_1 = []
  nnnew_csv_inf_2 = []
  nnnew_csv_inf_3 = []


  with open(open_path,'r',encoding="utf-8") as f:
    test=f.readlines()
    for i in range(len(test)):
      if label_lst[i] == 0:
        nnnew_csv_inf_0.append(test[i])
      elif label_lst[i] == 1:
        nnnew_csv_inf_1.append(test[i])
        #jogai.append(test[i])
        #print("LABEL1",test[i])
      elif label_lst[i] == 2:
        print("LABEL2",test[i])
        #nnnew_csv_inf_2.append(test[i])
      elif label_lst[i] == 3:
        print("LABEL3",test[i])
        #nnnew_csv_inf_3 = []

  if len(nnnew_csv_inf_0)<len(nnnew_csv_inf_1):
     print("n回目の再起で終了",j)
     break
  else:
    for i in range(len(nnnew_csv_inf_1)):
       jogai.append(nnnew_csv_inf_1[i])
    """
    if len(nnnew_csv_inf_0) >= len(nnnew_csv_inf_1) and len(nnnew_csv_inf_0) >= len(nnnew_csv_inf_2):
      nnnew_csv_inf = nnnew_csv_inf_0
    elif len(nnnew_csv_inf_1) >= len(nnnew_csv_inf_2):
      nnnew_csv_inf = nnnew_csv_inf_1
    else:
      nnnew_csv_inf = nnnew_csv_inf_2
    """
    nnnew_csv_inf = nnnew_csv_inf_0

    with open("../data/step1/CLUSTERING_check.csv"  ,'w',encoding="utf-8") as f: 
        for i in nnnew_csv_inf_0:
          # print(i)
            f.write(i)
        f.write("/n")
        f.write("#####################################################################")
        f.write("#####################################################################")
        for i in nnnew_csv_inf_1:
          # print(i)
            f.write(i)
        f.write("/n")
        f.write("#####################################################################")
        f.write("#####################################################################")
        for i in nnnew_csv_inf_2:
          # print(i)
            f.write(i)


    print("データ作成完了")



    #with open("tweetdata_hurei_reliable_without_aisatu_useCLUSTERING3.csv",'w',encoding="utf-8") as f:
    with open("../data/step1/tweetdata_hurei_reliable_without_aisatu_useCLUSTERING.csv"  ,'w',encoding="utf-8") as f: 
        for i in nnnew_csv_inf:
          # print(i)
            f.write(i)
    print("n回目のデータ作成完了",j)

    print("ラベル0の数：",len(nnnew_csv_inf_0))
    print("ラベル1の数：",len(nnnew_csv_inf_1))
    print("ラベル2の数：",len(nnnew_csv_inf_2))
    print("ラベル3の数：",len(nnnew_csv_inf_3))

with open("../data/step1/CLUSTERING_check_jogai.csv"  ,'w',encoding="utf-8") as f: 
      for i in jogai:
        # print(i)
          f.write(i)

#########################################################################################################################################################

# print("ここからコンテキストなしに関するデータ成形")


# nnew_csv_inf = []
# ccount = 0
# count = 0
# all_num = 0
# #open_path = "tweetdata_hurei_reliable.csv"
# open_path = "/home/aquamarine/sion/shuusi/data/step1/tweetdata_hurei_reliable_no_zengo.csv"


# main_tweet_text_lst=[]
# with open(open_path,'r',encoding="utf-8") as f:
#   test=f.readlines()
#   for moziretu in test:
#     tweet_text = moziretu.split(",")
#     main_tweet_text = tweet_text[0]
#     main_tweet_text_lst.append(main_tweet_text)

# #for i in range(len(main_tweet_text_lst)):
# #  print(main_tweet_text_lst[i])

# #リスト化完了


# docs = np.array(main_tweet_text_lst)
# vectorizer = TfidfVectorizer(use_idf=True, token_pattern=u'(?u)\\b\\w+\\b')
# vecs = vectorizer.fit_transform(docs)

# count = 0
# #clusters = KMeans(n_clusters=4, random_state=0).fit_predict(vecs)
# clusters = KMeans(n_clusters=3, random_state=0).fit_predict(vecs)

# label_lst = []
# for doc, cls in zip(docs, clusters):
#   label_lst.append(cls)
# print(label_lst)

# nnnew_csv_inf = []

# with open(open_path,'r',encoding="utf-8") as f:
#   test=f.readlines()
#   for i in range(len(test)):
#     if label_lst[i] == 0:
#       nnnew_csv_inf.append(test[i])
#     elif label_lst[i] == 1:
#       print("LABEL1",test[i])
#     elif label_lst[i] == 2:
#       print("LABEL2",test[i])
#     elif label_lst[i] == 3:
#       print("LABEL3",test[i])



# #with open("tweetdata_hurei_reliable_without_aisatu_useCLUSTERING3.csv",'w',encoding="utf-8") as f:
# with open("../data/step1/tweetdata_hurei_reliable_without_aisatu_useCLUSTERING_no_zengo.csv"  ,'w',encoding="utf-8") as f: 
#     for i in nnnew_csv_inf:
#        # print(i)
#         f.write(i)
# print("データ作成完了")
