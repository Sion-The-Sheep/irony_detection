import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from tqdm import tqdm
import MeCab
import unidic
import random
mecab = MeCab.Tagger()

import csv
import pprint
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


#"../data/step1/tweetdata_hurei_reliable.csv"に存在する、挨拶を含む文を除外し
#"../data/step1/tweetdata_hurei_reliable_without_aisatu_useCLUSTERING.csv" を作成する関数を作成

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
    for i in range(n):
        for j in range(n):
            if i != j and i not in sakujo_lst:
                similarity = calculate_similarity(tweet[i], tweet[j])
                print(i,j,similarity)
                if similarity > 0.5:
                    sakujo_lst.append(j)
    return sakujo_lst

def exclude_greeting(openpass: str, savepass: str):
    # 挨拶を除外するCSVファイルのパスと、保存先のパスを渡す
    # 挨拶を除外したファイルをsavepathに保存
    
    nnew_csv_inf = []
    ccount = 0
    count = 0
    all_num = 0
    open_path = openpass
    save_path = savepass

    main_tweet_text_lst=[]
    with open(open_path,'r',encoding="utf-8") as f:
        test = f.readlines()
        for moziretu in test:
            tweet_text = moziretu.split(",")
            main_tweet_text = tweet_text[0]
            main_tweet_text_lst.append(main_tweet_text)

    #リスト化完了

    docs = np.array(main_tweet_text_lst)
    vectorizer = TfidfVectorizer(use_idf=True, token_pattern=u'(?u)\\b\\w+\\b')
    vecs = vectorizer.fit_transform(docs)

    count = 0
    clusters = KMeans(n_clusters=2, random_state=0).fit_predict(vecs)

    label_lst = []
    for doc, cls in zip(docs, clusters):
        label_lst.append(cls)
    jogai=[]
    nnnew_csv_inf = []
    nnnew_csv_inf_0 = []
    nnnew_csv_inf_1 = []
    nnnew_csv_inf_2 = []
    nnnew_csv_inf_3 = []


    with open(open_path,'r',encoding="utf-8") as f:
        test = f.readlines()
        for i in range(len(test)):
            if label_lst[i] == 0:
                nnnew_csv_inf_0.append(test[i])
            elif label_lst[i] == 1:
                nnnew_csv_inf_1.append(test[i])
                jogai.append(test[i])

    nnnew_csv_inf = nnnew_csv_inf_0

    with open(save_path, 'w', encoding="utf-8") as f: 
        for i in nnnew_csv_inf:
            f.write(i)
    print("ラベル0の数：",len(nnnew_csv_inf_0))
    print("ラベル1の数：",len(nnnew_csv_inf_1))
    print("ラベル2の数：",len(nnnew_csv_inf_2))
    print("ラベル3の数：",len(nnnew_csv_inf_3))

    #ここから再起処理
    for j in range(999):
        nnew_csv_inf = []
        ccount = 0
        count = 0
        all_num = 0
        main_tweet_text_lst=[]

        with open(save_path,'r',encoding="utf-8") as f:
            test=f.readlines()
            for moziretu in test:
                tweet_text = moziretu.split(",")
                main_tweet_text = tweet_text[0]
                main_tweet_text_lst.append(main_tweet_text)

        docs = np.array(main_tweet_text_lst)
        vectorizer = TfidfVectorizer(use_idf=True, token_pattern=u'(?u)\\b\\w+\\b')
        vecs = vectorizer.fit_transform(docs)

        count = 0
        clusters = KMeans(n_clusters=2, random_state=0).fit_predict(vecs)

        label_lst = []
        for doc, cls in zip(docs, clusters):
            label_lst.append(cls)

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
                elif label_lst[i] == 2:
                    print("LABEL2",test[i])
                elif label_lst[i] == 3:
                    print("LABEL3",test[i])

        if len(nnnew_csv_inf_0)<len(nnnew_csv_inf_1):
            print("n回目の再起で終了→",j)
            break
        else:
            for i in range(len(nnnew_csv_inf_1)):
                jogai.append(nnnew_csv_inf_1[i])

            nnnew_csv_inf = nnnew_csv_inf_0

            with open("../data/step1/CLUSTERING_check.csv", 'w', encoding="utf-8") as f: 
                for i in nnnew_csv_inf_0:
                    f.write(i)
                f.write("/n")
                f.write("#####################################################################")
                f.write("#####################################################################")
                for i in nnnew_csv_inf_1:
                    f.write(i)
                f.write("/n")
                f.write("#####################################################################")
                f.write("#####################################################################")
                for i in nnnew_csv_inf_2:
                    f.write(i)

            with open(save_path,'w',encoding="utf-8") as f: 
                for i in nnnew_csv_inf:
                    f.write(i)

    with open("../data/exclud_greeting_check_jogai.csv"  ,'w',encoding="utf-8") as f: 
        for i in jogai:
            f.write(i)
    
