



> ### Model

dataset : cifal10

train_data.shape(50000, 3072)

test_data.shape(10000, 3072)

-------------
파라미터 값들에 대한 결과값 비교
Numpy와 수학적 수식만 사용해 딥러닝 학습 코드 구현

Sigmoid, LeakyReLU, Dropout, SoftmaxWithLoss

-------------

데이터셋 설정
1. 데이터셋 단색화
2. 기존 데이터로 학습

>1 은 결과값이 낮음
-------------


### HyperParameter
weight_init_std|number_of_nodes|batch_size|test_accuracy|use_Dropout
---|---|---|---|---
1|128, 64, 32|10|X|O
0.1|128, 64, 32|100|43|O
0.01|64, 64, 32|100|47|X
0.01|128, 64, 32|100|51.37|O(dropout_ratio=.2)
0.01|128, 64, 32|100|51.14|O(dropout_ratio=.3)
0.01|128,128,128|52.14|52.46|O(dropout_ratio=.2)


## How to Run

**config.yaml 파일에 원하는 파라미터 값을 입력 후**
주요 파라미터 설명:
```python
optimizer #train 시에 다른 optimizer는 사용하지 않아 Adam만 사용할 수 있음.
verbose : 학습 과정 log file 기록 및 출력
save_weight : 학습 후 완료된 가중치와 바이어스를 weights 폴더에 저장합니다.
use_weights : weights 폴더에 저장되어 있는 가중치와 바이어스를 불러와 사용합니다.
```


**Typing in terminal**
```python
$python train.py
```
**Test in Jupyter Notebook**
```python
check Jupyter notebook cell
```
 학습결과는 코드실행 창 또는 result 폴더에서 확인할 수 있습니다.
 
 
 > How to test
 **Typing in terminal**
```python
$python evaluate.py
```
학습된 가중치와 바이어스에 대한 cifar10 데이터로 test auccracy 가 바로 나옵니다.
(해당 과정은 log 생성 하지 않음.)
 
