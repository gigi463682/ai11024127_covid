# COVID-19：使用深度學習的醫學診斷
>11024127蘇家弘 期末報告
---
**介紹**
>正在進行的名為 COVID-19 的全球大流行是由 SARS-COV-2 引起的，該病毒傳播迅速並發生變異，引發了幾波疫情，主要影響第三世界和發展中國家。隨著世界各國政府試圖控制傳播，受影響的人數正穩定上升。
![image]()
>本文將使用 CoronaHack-Chest X 射線資料集。它包含胸部 X 光影像，我們必須找到受冠狀病毒影響的影像。

>我們之前談到的 SARS-COV-2 是主要影響呼吸系統的病毒類型，因此胸部 X 光是我們可以用來識別受影響肺部的重要影像方法之一。這是一個並排比較：
![image]()

>如你所見，COVID-19 肺炎如何吞噬整個肺部，並且比細菌和病毒類型的肺炎更危險。

>本文，將使用深度學習和遷移學習對受 Covid-19 影響的肺部的 X 光影像進行分類和識別。
__導入庫和載入數據__
'''
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import numpy as np
import pandas as pd
sns.set()
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  *
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.applications import DenseNet121, VGG19, ResNet50

import PIL.Image
import matplotlib.pyplot as mpimg
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.preprocessing import image

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from sklearn.utils import shuffle
train_df = pd.read_csv('../input/coronahack-chest-xraydataset/Chest_xray_Corona_Metadata.csv')
train_df.shape
'''
> (5910, 6)

'''train_df.head(5)'''
![image]()

__處理缺失值__
'''
missing_vals = train_df.isnull().sum()
missing_vals.plot(kind = 'bar')
'''
![image]()

'''
train_df.dropna(how = 'all')
train_df.isnull().sum()
'''
