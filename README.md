# Sequential mnist
>pytorch 프레임워크의 공부목적으로 sequential mnist dataset을 학습

</br>

## 2. 사용 기술
- **language**: python
- **framework**: pytorch
- **model**: LSTM, GRU, TCN
- **optimize hyperparameter**: Random Search, Bayesian optimize

</br>

## 3. file 설명
### 'train.py' 'train_cv.py' 
- 모델 학습, k-fold를 통한 학습
### 'model.py' 
- LSTM, GRU, TCN
### 'dataset.py' 
- custom dataset
### 'RandomSearch.py', 'Optimzer.py'
- optimize hyper parameter.
