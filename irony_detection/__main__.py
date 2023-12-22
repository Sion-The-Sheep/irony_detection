import sys
import json
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

from . import learning_base as LB
from .libs import exclude_greeting
from .libs import parameter
from .libs import util



def main():
    #学習データ
    print(util.BASE_DIR)
    path_train_posi = os.path.join(util.BASE_DIR, "data/train/train_data_posi.csv")
    path_train_nega = os.path.join(util.BASE_DIR, "data/train/train_data_randam.csv")
    #テストデータ
    # path_test_posi = os.path.join(util.BASE_DIR, "data/test/tweet_data_nakahigasi_seirei_thouhuku_URL_RT_OK.csv")
    # path_test_nega = os.path.join(util.BASE_DIR, "data/test/tweet_data_nakahigasi_hurei_thouhuku_URL_RT_OK.csv")

    df_train_posi = pd.read_csv(path_train_posi, encoding="utf_8", names=('honbun','rep','inyou','label'), dtype=object)
    df_train_nega = pd.read_csv(path_train_nega, encoding="utf_8", names=('honbun','rep','inyou','label'), dtype=object)


    #提案手法ステップ１
    parampath = sys.argv[1]
    pram = parameter.init(parampath)
    step1 = LB.LeaningClassificationModel(df_train_posi, df_train_nega, pram)
    step1.format_spy_data(os.path.join(util.BASE_DIR, "/step1/tweetdata_random_include_spy"))


if __name__ == '__main__':
    main()