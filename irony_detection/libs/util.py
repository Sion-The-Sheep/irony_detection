import os
import datetime
from pathlib import Path
import shutil

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SOURCE_DIR = Path(__file__).resolve().parent.parent
LIBS_DIR = Path(__file__).resolve().parent

dt_now = datetime.datetime.now()

logDir = f"{BASE_DIR}/logs/{dt_now.strftime('%Y-%m-%d-%H-%M-%S')}/"
modelDir  = f"{SOURCE_DIR}/model"



def exConfigSave(configPath):
    shutil.copy(configPath, logDir)