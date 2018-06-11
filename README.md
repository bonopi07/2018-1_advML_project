# 2018-1_advML_project
- This repository is related to the class, which name is "Advanced Machine Learning Skills" in Kookmin Univ at 2018-1.

## Contributor
- Department: Computer Science in Kookmin University
- Student name: Seongmin Jeong (English name: Allen)
- Laboratory: Information Security Lab

## Project Environment
- Python v3.6.x
- Tensorflow v1.8

## Goals
- 호텔 리뷰가 주어지면 평점을 예측할 수 있는 텐서플로우 기반 RNN(or LSTM) model을 학습한다.
- 자연어 처리 기반의 many-to-one 방식 RNN을 학습한다.
  - RNN 실습을 통해 동적 입력 크기에 대한 처리 방법을 익힌다.
  - char-level, word-level 의 다양한 자연어 처리 방법을 공부한다.
  - deep, wide한 방법(e.g.stacked RNN)과 다양한 딥러닝 테크닉(e.g.dropout, word embedding)을 적용해본다.
  - RNN model의 뒤에 softmax나 FC layer를 붙혀본다.

## Version Update Log
- 2018.05.07: create repository. update README
- 2018.06.03: version 1.0 (LSTM 3 Layer, char-level one-hot encoding vector, no dropout, loss: cross entropy, opt: Adam)
- 2018.06.10: version 1.1 (LSTM 3 Layer, word-level embedding-lookup vector, no dropout, loss: cross entropy, opt: Adam)

* * *

# Algorithms
- 본 프로젝트는 리뷰 데이터(data)와 평점 데이터(label)를 이용하여 LSTM 기반의 예측모델을 만들 것이다.
## RNN
- RNN은 인공 신경망의 한 종류로, 유닛 간의 연결이 순환적 구조를 갖는 특징을 갖고 있다. 이러한 구조는 순차적 동적 특징을 모델링 할 수 있도록 신경망 내부에 상태를 저장할 수 있게 해준다. Feedforward 신경망과 달리, recurrent 인공 신경망은 내부의 메모리를 이용해 시퀀스 형태의 입력을 처리할 수 있다. 따라서 recurrent 인공 신경망은 필기체 인식이나 음성 인식과 같이 순차적 특징을 가지는 데이터를 처리할 수 있다.

- 하지만 RNN은 관련 정보와 그 정보를 사용하는 지점 사이 거리가 멀 경우 back propagation시 gradient가 점점 줄어 학습 능력이 크게 저하되는 현상을 가진다. 이를 vanishing gradient problem이라고 한다. 이 문제를 보완하기 위해 RNN의 hidden state와 cell state를 추가로 가지고 있는 LSTM 모델을 사용할 것이다.

- LSTM은 cell state와 hidden state를 통해 추가적인 연산을 하게 되는데, 이를 통한 3가지의 gate가 생성된다. Forget gate, input gate, output gate인데, 이 gate들은 이전 값을 얼마나 기억할지, 그리고 현재 값을 얼마나 기억할지 등의 값을 제어하기 위한 목적으로 사용된다. 본 프로젝트는 간단한 LSTM model을 설계함으로써 호텔 리뷰와 같은 다대일(many-to-one) 문제를 해결한다.

## Model Architecture
- LSTM Layer #
    - Single (1 Layer) --> Stacked (3 layer)

- Data Dimension
    - char-level : 44차원
    - word-level : 300차원

- Loss Function
    - Softmax cross entropy (with Adam Optimizer)

- 데이터셋 검증 및 모델 파라미터 검증 방법
    - K-fold Cross Validation (K = 5)