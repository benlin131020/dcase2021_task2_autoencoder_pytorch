# dcase2021_task2_autoencoder_pytorch
## Usage

### 1. Clone repository
```
$ git clone https://github.com/benlin131020/dcase2021_task2_autoencoder_pytorch.git
```

### 2. Python 環境
建立 Python 環境
```
$ conda create -n myenv python=3.6
```

進入 Python 環境
```
$ conda activate myenv
```
離開 Python 環境
```
$ conda deactivate
```
如果 conda 版本較舊
```
$ source activate myenv
$ source deactivate
```

### 3. 安裝 Python 套件
進入 Python 環境後
```
$ pip install -r requirements.txt
```

### 4. Run
`CUDA_VISIBLE_DEVICES=1` 指定使用第幾顆 GPU
```
$ CUDA_VISIBLE_DEVICES=1 python 00_train -d
$ CUDA_VISIBLE_DEVICES=1 python 01_test -d
```

### 5. yaml 設定
在 `baseline.yaml` 中:
- `linear: 0`
    - `0` 為使用 mel-scale, `1` 為使用 linear-scale
