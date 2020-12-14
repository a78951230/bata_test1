# 原作者
- https://github.com/RockyXu66/Faster_RCNN_for_Open_Images_Dataset_Keras
# LabelImg
- https://tzutalin.github.io/labelImg/
<p float="left">
   <img src="/img/l1.jpg" width="536"/>
</p>

# vgg-16
- https://github.com/fchollet/deep-learning-models/releases 

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
