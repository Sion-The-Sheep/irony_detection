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
import os
from .libs import util
from .libs import parameter

class LeaningClassificationModel:
    def __init__(self, positive_data: pd.DataFrame, negative_data: pd.DataFrame, parameter: parameter.Parameter):
        self.positive_data = positive_data
        self.negative_data = negative_data
        self.parameter = parameter
        self.posi_spy_df_list = []
        self.nega_spy_df_list = []

    def format_spy_data(self, include_spydata_savepath: str): #STEP1なら"/step1/tweetdata_randam_include_spy"
        spy_num = int(((len(self.positive_data))//(100/self.parameter.spy_hyperparameter_a))+1) #スパイデータの数
        df_seirei = self.positive_data
        for i in range(self.parameter.spy_pattern_num):
            df_seirei = df_seirei.sample(frac=1)
            df_seirei_spy = df_seirei[:spy_num]
            df_seirei_notspy = df_seirei[spy_num:]
            df_hurei = pd.concat([self.negative_data,df_seirei_spy],ignore_index=True)

            df_seirei_notspy["new_label"] = 1
            df_hurei["new_label"] = 0

            df_seirei_notspy = df_seirei_notspy.applymap(str)
            df_hurei = df_hurei.applymap(str)

            file_name  = "data" + include_spydata_savepath + str(i+1) + ".csv"
            path = os.path.join(util.BASE_DIR, file_name)
            df_hurei.to_csv(path, header=False, index=False, encoding="utf-8")

            self.posi_spy_df_list.append(df_seirei_notspy)
            self.nega_spy_df_list.append(df_hurei)

            txt = str(i+1) + "個目のスパイ入りデータ作成完了"
            print(txt)
        print(self.posi_spy_df_list[0])

    
    def leaning_bert(self, model_savepath: str):
        for i in range(self.parameter.spy_pattern_num):
            posi = self.posi_spy_df_list[i]
            nega = self.nega_spy_df_list[i]
            df = pd.concat([posi, nega],ignore_index=True)
            df = df.sample(frac=1)
            df = df.applymap(str)
            df['label'] = df['label'].astype('int')
            df['new_label'] = df['new_label'].astype('int')

            honbun_lst = df["honbun"].tolist()
            rep_lst = df["rep"].tolist()
            inyou_lst = df["inyou"].tolist()
            label_lst = df["new_label"].tolist()
            past_label_lst = df["label"].tolist()

            Accuracy_list = []
            Precision_list = []
            Recall_list = []
            F1_score_list = []

            x_train= honbun_lst #本文
            x_train2 = rep_lst #前
            x_train3 = inyou_lst #後
            y = label_lst #らべる
            y_train = torch.tensor(y, dtype=torch.int64)

            #save_path = "../model/hozon_spy_RN" + str(i)
            save_path = "../model/" + model_savepath + str(i)
            
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
                for input_ids, attention_mask, token_type_ids, y in train_dl:
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
        
class PULeaningClassificationModel(LeaningClassificationModel):
    def __init__(self, model_to_spytechnique, positive_data: pd.DataFrame, negative_data: pd.DataFrame, parameter: parameter.Parameter):
        super().__init__()
        self.model_to_spytechnique = model_to_spytechnique #スパイテクニックを行うための要素を新たに追加
        self.threshold_lst = [] # アンサンブルの結果、最終的に除外するべきインデックスが格納された１次元配列

        self.df_test_seirei = pd.read_csv("../data/tweetdata_nakahigasi_seirei_NakaiLabeling_1.csv",encoding="utf_8",names=('honbun','rep','inyou','label'),dtype=object)
        self.df_test_hurei = pd.read_csv("../data/tweet_data_hurei_thouhuku_URL_RT_OK.csv",encoding="utf_8",names=('honbun','rep','inyou','label'),dtype=object)
        
        self.label_lst_num = 1 #PU学習とNU学習で付与するラベルに差異があるためここで定義　オーバーライド必須
        self.num_b = self.parameter.spy_hyperparameter_b_step1

    def calculate_spy_threshold(self, include_spydata_loadpath: str):#スパイテクニックを行い、信頼できる負例データをdf型で返す
        exclition_list = []# 各パターンで除外するべきインデックスが格納されたリストが「self.parameter.spy_pattern_num」個、格納されている２次元配列
        threshold_list = []#要素として（皮肉である確率、インデックス）というタプルを要素に持つようなリスト

        for q in range(self.parameter.spy_pattern_num):
            spy_pattern = q + 1
            tensor_list=[]
            spy_tensor_list=[]
            Ptensor_list = []
            spy_count = 0 #閾値を決めるためにスパイデータの数が必要、カウントする
            count = 0

            Accuracy_list = []
            Precision_list = []
            Recall_list = []
            F1_score_list = []

            path_spy = include_spydata_loadpath + str(spy_pattern) + ".csv"

            df = pd.read_csv(path_spy, encoding="utf_8", names=('honbun','rep','inyou','label','new_label'), dtype=object)
            df = df.sample(frac=1)
            df = df.applymap(str)
            df['label'] = df['label'].astype('int')
            df['new_label'] = df['new_label'].astype('int')

            honbun_lst = df["honbun"].tolist()
            rep_lst = df["rep"].tolist()
            inyou_lst = df["inyou"].tolist()
            label_lst = df["label"].tolist()
            new_label_lst = df["new_label"].tolist()

            test = honbun_lst
            test2 = rep_lst
            test3 = inyou_lst
            seikai = label_lst

            model = BertForSequenceClassification.from_pretrained(self.model_to_spytechnique, num_labels=2).to("cuda:0")
            tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking") 
            model.eval()

            #r_encorded =  tokenizer(test, padding=True, return_tensors='pt')
            r_encorded =  tokenizer(test,test2,test3,padding=True, return_tensors='pt')

            r_ds = TensorDataset(r_encorded['input_ids'].to("cuda:0"), r_encorded['attention_mask'].to("cuda:0"),r_encorded['token_type_ids'].to("cuda:0"))#####
            r_dl = DataLoader(r_ds, batch_size=1) # バッチサイズを指定
            sf = torch.nn.Softmax(dim=1)
            result = []
            for input_ids, attention_mask,token_type_ids in r_dl:
                r_classification = model(input_ids, attention_mask = attention_mask,token_type_ids=token_type_ids)
                r_soft = sf(r_classification.logits)
                r_zero_one = torch.argmax(r_soft)
                result.append(r_zero_one)
                tensor_list.append(r_soft[count,1].item())            

            for i in range(len(df)):
                if label_lst[i] == self.label_lst_num and new_label_lst[i] == 0:
                    #print("honbunn:",honbun_lst[i])
                    spy_tensor_list.append(tensor_list[i])
                    spy_count += 1

            #昇順に並び替え
            spy_tensorted_list = sorted(spy_tensor_list,key=None,reverse=False)
            #確率が低い方からn番目の確率をしきい値とする
            threshold_index = int(spy_count*(self.num_b/100)) #閾値となるスパイデータのインデックス
            #tensor_key_num = spy_tensorted_list[3]
            tensor_key_num = spy_tensorted_list[threshold_index]
            print("スパイで一番低い確率：",tensor_key_num)


            #ランダムに集めたデータに関して、tensor_key_num以上の値を持つデータを除外
            #負例のtensor値をNtensor_list = []に格納                                                                                                                                                                                                                                          
            df = pd.read_csv(self.negative_data, encoding="utf_8", names=('honbun','rep','inyou','label'), dtype=object)
            df = df.applymap(str)
            df['label'] = df['label'].astype('int')


            honbun_lst = df["honbun"].tolist()
            rep_lst = df["rep"].tolist()
            inyou_lst = df["inyou"].tolist()
            label_lst = df["label"].tolist()
                    
            test = honbun_lst
            test2 = rep_lst
            test3 = inyou_lst
            seikai = label_lst

            Ntensor_list = []           
                    
            model = BertForSequenceClassification.from_pretrained(self.model_to_spytechnique, num_labels=2).to("cuda:0")
            tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
            model.eval()
            #r_encorded =  tokenizer(test, padding=True, return_tensors='pt')
            r_encorded =  tokenizer(test,test2,test3,padding=True, return_tensors='pt')
            r_ds = TensorDataset(r_encorded['input_ids'].to("cuda:0"), r_encorded['attention_mask'].to("cuda:0"),r_encorded['token_type_ids'].to("cuda:0"))#####     
            r_dl = DataLoader(r_ds, batch_size=1) # バッチサイズを指定
            sf = torch.nn.Softmax(dim=1)
            result = []

            Count = 0
            list_ikichi = [] # list_for_thresholdにこのリストを格納　５パターンの結果が格納されることになる
            
            for input_ids, attention_mask,token_type_ids in r_dl:
                r_classification = model(input_ids, attention_mask = attention_mask,token_type_ids=token_type_ids)
                r_soft = sf(r_classification.logits)
                r_zero_one = torch.argmax(r_soft)
                result.append(r_zero_one)
                Ntensor_list.append(r_soft[count,1].item())
                
                list_ikichi.append((Count,r_soft[count,1].item()))
                Count += 1
                
            list_ikichi.sort(key = lambda x: x[1]) # ここで皮肉（非皮肉）である確率が高い順に並び替える。
            threshold_list.append(list_ikichi)

            #tensor値が一定以上のデータのインデックスを記録
            jogai_list=[]
            for i in range(len(Ntensor_list)):
                if Ntensor_list[i]>tensor_key_num:
                    jogai_list.append(i)
            exclition_list.append(jogai_list)

            words = str(spy_pattern) + "種類目のスパイでの除外データ選択終了"
            print(words)

        for j in range(self.parameter.spy_pattern_num):
            for k in exclition_list[j]:
                if k not in self.threshold_lst:
                    self.threshold_lst.append(k)

        print("最終的なthreshold_lst")
        print(self.threshold_lst)
        print("最終的なjogai_listの数：",len(jogai_list))
        print("除外する前のデータ数：",len(df))
        
        #信頼できる負例データを作成　保存

        df = self.negative_data
        
        #信頼できる負例データを格納するリスト
        ittan_list_honbun = df['honbun'].tolist()
        ittan_list_rep = df['rep'].tolist()
        ittan_list_inyou = df['inyou'].tolist()
        ittan_list_label = df['label'].tolist()

        Ittan_honbun = []
        Ittan_rep = []
        Ittan_inyou = []
        Ittan_label = []

        for i in range(len(ittan_list_honbun)):
            if i not in jogai_list:
                Ittan_honbun.append(ittan_list_honbun[i])
                Ittan_rep.append(ittan_list_rep[i])
                Ittan_inyou.append(ittan_list_inyou[i])
                Ittan_label.append(ittan_list_label[i])

        df_hurei_reliable = pd.DataFrame((zip(Ittan_honbun,Ittan_rep,Ittan_inyou,Ittan_label)),columns = ["honbun","rep","inyou","label"])
        
        if self.label_lst_num == 0:
            df_hurei_reliable.loc[0:len(df_hurei_reliable),"label"] = 1#PU学習とNU学習ではラベルの意味合いが異なるため、ここで修正
        
        return df_hurei_reliable

    def format_learnig_datas(self, df_seirei_datas: pd.DataFrame, df_hurei_datas: pd.DataFrame):
        #信頼できる負例データから、正例データ数と同じ数だけランダムにデータを取ってくる操作をn回行い,n個の学習データを作成する

        df_seirei = df_seirei_datas
        df_hurei = df_hurei_datas

        train_honbun_list = [] # n個の「学習テキストデータを要素に持つリスト」が格納された２次元配列
        train_rep_list = [] #上に同じ
        train_inyou_list = [] #上に同じ
        train_label_list = [] #上に同じ

        num_seirei = len(df_seirei)
        for i in range(self.parameter.number_of_leaning_data_pattern):
            num = i + 1
            df_hurei_reliable = df_hurei_reliable.sample(frac=1)
            exec("df_hurei_{} = df_hurei_reliable[:num_seirei]".format(num))

            df_seirei = df_seirei.applymap(str)
            exec("df_hurei_{} = df_hurei_{}.applymap(str)".format(num,num))

            df_seirei['label'] = df_seirei['label'].astype('int')
            exec("df_hurei_{}['label']  = df_hurei_{}['label'].astype('int')".format(num,num))
            
            seirei_honbun_list = df_seirei['honbun'].tolist()
            seirei_rep_list = df_seirei['rep'].tolist()
            seirei_inyou_list = df_seirei['inyou'].tolist()
            seirei_label_list = df_seirei['label'].tolist()

            exec("hurei_honbun_list_{} = df_hurei_{}['honbun'].tolist()".format(num,num))
            exec("hurei_rep_list_{} = df_hurei_{}['rep'].tolist()".format(num,num))
            exec("hurei_inyou_list_{} = df_hurei_{}['inyou'].tolist()".format(num,num))
            exec("hurei_label_list_{} = df_hurei_{}['label'].tolist()".format(num,num))

            exec("train_honbun_list_{} = seirei_honbun_list + hurei_honbun_list_{}".format(num,num))
            exec("train_rep_list_{} = seirei_rep_list + hurei_rep_list_{}".format(num,num))
            exec("train_inyou_list_{} = seirei_inyou_list + hurei_inyou_list_{}".format(num,num))
            exec("train_label_list_{} = seirei_label_list + hurei_label_list_{}".format(num,num))

            exec("train_honbun_list.append(train_honbun_list_{})".format(num))
            exec("train_rep_list.append(train_rep_list_{})".format(num))
            exec("train_inyou_list.append(train_inyou_list_{})".format(num))
            exec("train_label_list.append(train_label_list_{})".format(num))

            #返り値は４つのリスト　評価のときに使いやすくするためDF型にはしない
            return(train_honbun_list, train_rep_list, train_inyou_list,train_label_list)
        
    def evaluation(self, train_honbun_list: list, train_rep_list: list, train_inyou_list: list,train_label_list: list, savepath_name: str):
        #選択したデータで分類器を学習した際の性能をテストデータを用いて検証する
        #モデルの名前　例
        # ../model/hozon_sarcasm_detection_baceline
        # ../model/hozon_sarcasm_detection_PU
        # ../model/hozon_sarcasm_detection_PU_NU

        Accuracy_list = []
        Precision_list = []
        Recall_list = []
        F1_score_list = []
        list_for_ensemble = [] #アンサンブル学習のために分類器ごとの予測を格納

        df_test_seirei = self.df_test_seirei.applymap(str)
        df_test_hurei = self.df_test_hurei.applymap(str)
        df_test_seirei['label'] = df_test_seirei['label'].astype('int')
        df_test_hurei['label'] = df_test_hurei['label'].astype('int')

        test_seirei_num = len(df_test_seirei)
        df_test_hurei = df_test_hurei.iloc[:test_seirei_num]

        df_test = pd.concat([df_test_seirei,df_test_hurei],ignore_index=True)
        test_honbun_list = df_test["honbun"].tolist()
        test_rep_list = df_test["rep"].tolist()
        test_inyou_list = df_test["inyou"].tolist()
        test_label_list = df_test["label"].tolist()
        
        for i in range(self.parameter.number_of_leaning_data_pattern):
            #与えたデータで分類機を学習
            savepath = savepath_name + str(i+1)
            x_train = train_honbun_list[i]
            x_train2 = train_rep_list[i]
            x_train3 = train_inyou_list[i]
            y = train_label_list[i]
            y_train = torch.tensor(y, dtype=torch.int64)

            test = test_honbun_list
            test2 = test_rep_list
            test3 = test_inyou_list
            seikai = test_label_list

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
                for input_ids, attention_mask, token_type_ids,y in train_dl:
                    optimizer.zero_grad()
                    #outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=None, labels=y)
                    outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=y)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                
                out_path = savepath + "/model_epoch{}.model".format(epoch)
                model.save_pretrained(out_path)


            #学習したモデルの性能をテストデータで確認            
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

            list_for_ensemble.append(result)
                        
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            seikaisu = 0
            huseikaisu = 0

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
                    print("error")


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
            moziretu = str(i+1) + "回目の学習結果"
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

        av_Accuracy = sum(Accuracy_list)/self.parameter.number_of_leaning_data_pattern
        av_Precision = sum(Precision_list)/self.parameter.number_of_leaning_data_pattern
        av_Recall = sum(Recall_list)/self.parameter.number_of_leaning_data_pattern
        av_F1_score = sum(F1_score_list)/self.parameter.number_of_leaning_data_pattern

        result_text = str(self.parameter.number_of_leaning_data_pattern) + "個の分類器の単純平均結果"
        print("正解率：",av_Accuracy,"適合率：",av_Precision,"再現率：",av_Recall,"F値：",av_F1_score)

        print("アンサンブル学習結果")

        result_ensemble=[]
        #多数決により、アンサンブル学習
        for i in range(len(result_ensemble[0])):
            label_sum = 0
            for j in range(self.parameter.number_of_leaning_data_pattern):
                label_sum += int(result_ensemble[j][i])
            if label_sum >= ((self.parameter.number_of_leaning_data_pattern)//2) + 1:
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

        print("アンサンブル結果")
        print("真陽性：",TP,"擬陽性：",FP,"真陰性：",TN,"偽陰性：",FN)
        print("正解数：",seikaisu,"不正解数：",huseikaisu)
        print("正解率：",Accuracy,"適合率：",Precision,"再現率：",Recall,"F値：",F1_score)
        print(moziretu)


class NULeaningClassificationModel(PULeaningClassificationModel):
    def __init__(self, model_to_spytechnique, positive_data: pd.DataFrame, negative_data: pd.DataFrame, parameter: parameter.Parameter):
        super().__init__()
        self.label_lst_num = 0 
        self.num_b = self.parameter.spy_hyperparameter_b_step2
    
    def conect_p_rp(self, p: pd.DataFrame, rp: pd.DataFrame, rn: pd.DataFrame):#calculate_spy_thresholdで求めた信頼できるデータを第２引数として受け取る
        df_seirei = pd.concat([p, rp], ignore_index=True)
        print("信頼できる正例＋純正例数確認:",len(df_seirei))
        print("信頼できる負例データ数：",len(rn))

        if len(df_seirei)<=len(rn):
            return(df_seirei)
        else:
            print("エラー：P＋RPがRNよりも多い")






