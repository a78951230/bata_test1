# 原作者
- https://github.com/RockyXu66/Faster_RCNN_for_Open_Images_Dataset_Keras
# LabelImg
- https://tzutalin.github.io/labelImg/
<p float="left">
   <img src="/img/l1.jpg" width="536"/>
</p>

# vgg-16
- https://github.com/fchollet/deep-learning-models/releases 

# 檔案架構

frcnn_test_vgg.ipynb  
frcnn_train_vgg.ipynb  
xml_to_csv_txt.ipynb  
bata_test1  
-----| README.md  
-----| img  
-----| xml_to_csv_train.py  
-----| xml_to_csv.py  
-----| model  
----------|  model_frcnn_vgg.hdf5  
----------|  record.csv  
----------| vgg16_weights_tf_dim_ordering_tf_kernels.h5  
-----| dataset  
----------| test  
--------------|  1.jpg  
--------------|  2.jpg  
----------|  train  
--------------|  1.jpg  
--------------|  2.jpg  
-----| test  
----------|  1.xml  
----------|  2.xml  
-----| train  
----------|  1.xml  
----------|  2.xml  


# requirements.txt
環境安裝

    conda create -n your_env_name python=3.6 anaconda
切換環境

    activate your_env_name
tensorflow-gpu 安裝

    pip install tensorflow-gpu==2.0.0beta1
    conda install cudnn=7.6.0
    conda install cudatoolkit=10.0.130
測試
```python
import tensorflow as tf
import keras
```
安裝錯誤部分:

    ImportError: DLL load failed: 找不到指定的模块
請查詢: https://www.tensorflow.org/install/source_windows <br />
如果沒出現報錯，應該就表示已經成功安裝。我們可以來看一下現在的版本。

```python
tf.__version__
```
然後確定一下是否有使用到 GPU。
```python
tf.test.is_gpu_available()
```
若顯示 True 則表示 GPU 版本是沒有問題的。

套件安裝

    pip install numpy
    pip install opencv-python
    pip install scikit-metrics
    pip install matplotlib
    pip install pandas
查看套件

    conda list
    or
    pip list
<p float="left">
   <img src="/img/l2.jpg" width="536"/>
</p>

# 圖片標註
<p float="left">
   <img src="/img/l3.jpg" width="536"/>
</p>

    Save->bata_test1/test or bata_test1/train
XML-->TXT:

    cd /bata_test1
    python xml_to_csv.py and xml_to_csv_train.py
<p float="left">
   <img src="/img/l4.png" width="746"/>
</p>
     
    activate your_env_name
    jupyter notebook
    open xml_to_csv_txt.ipynb
    Cell-->Run All
    
 <p float="left">
   <img src="/img/l5.png" width="755"/>
</p>

# train

    activate your_env_name
    jupyter notebook
    open frcnn_train_vgg.ipynb
路徑修改:
```python
base_path = 'bata_test1'#路徑

train_path =  'bata_test1/annotation.txt' #訓練檔案路徑

num_rois = 4

# Augmentation flag
horizontal_flips = True #訓練中水平翻轉增強
vertical_flips = True   #訓練中垂直翻轉增強
rot_90 = True           #訓練中旋轉90度增強

output_weight_path = os.path.join(base_path, 'model/model_frcnn_vgg.hdf5')#權重檔
record_path = os.path.join(base_path, 'model/record.csv')#記錄數據（用於節省損失，分類精度和平均平均精度）
base_weight_path = os.path.join(base_path, 'model/vgg16_weights_tf_dim_ordering_tf_kernels.h5')#vgg16預訓練模型
config_output_filename = os.path.join(base_path, 'model_vgg_config.pickle')#設定檔
```
訓練次數及一次性丟入張數:
```python
epoch_length = 464  #一次張數
num_epochs = 35 #次數
```
訓練時出現:

    RuntimeError: CUDA out of memory. Tried to allocate 146.88 MiB (GPU 0; 2.00 GiB total capacity; 374.63 MiB already allocated; 0 bytes free; 1015.00 KiB cached)
    
開頭請補上:
```python    
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
sess = tf.compat.v1.InteractiveSession(config=config)
```
# test

    activate your_env_name
    jupyter notebook
    open frcnn_test_vgg.ipynb
