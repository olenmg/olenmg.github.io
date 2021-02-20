---
layout: post
title: "Day17. NLP with RNN"
subtitle: "RNN, LSTM, GRU"
date: 2021-02-16 15:16:12+0900
background: '/img/posts/bg-posts.png'
use_math: true
---

## 개요 <!-- omit in toc -->
> 이전에 다루었던 RNN과 이에 기반한 모델들을 복습하는 시간을 가졌다.     
  
아래와 같은 순서로 작성하였다.  
- [Basic of RNN](#basic-of-rnn)
    - [RNN의 동작 과정](#rnn의-동작-과정)
    - [Types of RNNs](#types-of-rnns)
    - [Character-level Language Model(예시)](#character-level-language-model예시)
    - [Hidden state vector의 의미](#hidden-state-vector의-의미)
    - [Backpropagation에서의 문제](#backpropagation에서의-문제)
- [LSTM, GRU](#lstm-gru)
    - [LSTM](#lstm)
    - [GRU](#gru)
    - [Backpropagation in LSTM/GRU](#backpropagation-in-lstmgru)
- [Reference](#reference)


<br/>

<span class="link_button">[이전 포스트](/2021/02/04/boostcamp-day14.html)</span>와 내용이 많이 겹치므로 몇몇 내용은 제외하고 담았다.    

<br/>

## Basic of RNN
![RNN](/img/posts/14-1.png){: width="90%" height="90%"}{: .center}   
RNN은 기본적으로 동일한 모듈을 중심에 두고 input이 시간에 따라 변화하면서 들어가게 된다. 
모듈 내에서 곱해지는 가중치가 시간에 따라 변하지 않는다는 점을 기억하도록 하자.
내부에서 출력과 직접적인 연관이 있는 hidden state의 값은 시간에 따라 변화한다.
hidden state는 **과거의 정보를 기억**하는 역할을 한다. 물론 과거는 바로 이전 뿐만이 아니라 먼 과거까지도 포함이다.  
  
수업을 들으면서 다시 배웠던 것은, hidden state 자체가 당연히 그대로 output으로 나오지는 않는다는 것이다.
output이 필요한 경우(필요하지 않은 시점도 존재한다), hidden state $h\_t$는 output layer를 거친 output $y\_t$로 나오게 된다.

<br />

#### RNN의 동작 과정
![RNN_process](/img/posts/17-1.png){: width="90%" height="90%"}{: .center}   
hidden state vector가 가지는 차원 수는 hyper parameter로 우리가 직접 정해야한다. 

앞서 말했듯이 output이 필요한 경우 hidden state는 output layer의 가중치 $W\_{hy}$와 곱해져나온 결과값이 된다.
**원하는 것이 무엇이느냐에 따라 $y$의 차원 역시 달라진다.** 
만약 binary classification을 하고자 한다면 $y$는 scalar로 나올 것이고 multinomial classification을 하고자한다면 class 갯수 만큼의 dimension을 가진 vector가 출력으로 나올 것이다.
이후 softmax를 통해 결과 값을 얻고, softmax loss 등을 사용하여 학습을 할 수 있을 것이다.  
  

![RNN_process2](/img/posts/17-2.jpeg){: width="70%" height="70%"}{: .center}   
실제 연산에서는 위와 같이 $x$와 $h$가 concatenation 된 상태로 하나의 가중치 $W$에 곱해지는 형태로 연산이 된다. 
다만 분리하고 보면 어차피 똑같은 이야기이다. 전체적으로 shape이 어떻게 되는지를 잘 보도록 하자.
$x$와 $h$의 차원수는 사전에 결정된 값이고, 이에 따라 $W$의 shape도 자동으로 결정된다.

<br />

#### Types of RNNs
![RNN_types](/img/posts/17-3.jpeg){: width="90%" height="90%"}{: .center}   
RNN은 크게 보면 입출력이 어떤식으로 이루어지느냐에 따라 위와 같이 종류를 나눠볼 수 있다.
- One to One  
    + 사실 이건 시계열 데이터가 들어오지 않기 때문에 엄밀하게는 RNN 구조는 아니다.
    + 일반적인 신경망 구조이다.
- One to Many  
    + input 하나로 sequence output을 내놓는 구조이다.
    + i.e. image captioning에 사용될 수 있다. 이미지라는 단일 데이터에서 캡션 문장의 단어들을 순차적으로 생성한다.
    + 다만 일반적인 경우 첫번째 input 이후부터는 input으로 같은 사이즈의 영벡터가 들어갈 것으로 추측할 수 있다.
- Many to One  
    + sequence input으로 output 하나가 나오는 구조이다.
    + i.e. sentiment classification에 사용될 수 있다. 결국 마지막에 나온 벡터로 하나의 감정 상태를 추측하기 때문에 최종 output은 하나이다. 
      이 예시에서는 input으로 문장(sequence of embedding vector), output으로 softmax를 통과할(multinomial이라면) 벡터가 나올 것이다.
- Many to Many (Sequence to Sequence) (1)
    + 가장 상상하기 쉬운 RNN 모델이다. input sequence에 대하여 output sequence가 나온다.
    + i.e. 번역에 사용할 수 있다. input으로 번역하고자 하는 문장, output으로 번역된 문장이 나올 것이다.
    + 여기서는 input이 모두 들어올때까지 기다렸다가 다 읽고 output을 내놓는 형태이다.
- Many to Many (Sequence to Sequence) (2)
    + 이전과의 차이점은 똑같이 sequence to sequence 모델이지만, 실시간 처리를 수행한다.
      이전에는 input이 모두 들어온 이후 output이 출력되기 시작했다면 여기서는 input이 들어올때마다 output이 바로바로 도출된다. 
    + i.e. 동영상의 매 scene의 내용을 맞히는 video classification(on frame level), POS tagging(품사 태깅) 등에 사용될 수 있다.

<br />

#### Character-level Language Model(예시)
RNN 모델의 예시로 어떤 글자가 들어왔을 때 다음 글자를 맞힐 수 있는 모델을 떠올려보자.  
  
![RNN_character_level](/img/posts/17-4.jpeg){: width="90%" height="90%"}{: .center}   
여기서는 hidden state의 dimension을 3으로 설정했고, 딕셔너리의 크기가 4이므로 hidden layer 가중치 행렬의 shape을 3 x 7로 추측할 수 있다.
첫번째 hidden layer의 input으로 들어가는 hidden state $h\_0$는 3차원 영벡터로 주게 된다. (default는 이렇지만, 상황에 따라 다를 수 있을 것이다)
그리고 fully connected layer처럼, 실제로는 아래와 같이 bias term($b$)이 각 layer에 들어간다.

<center>

$$
h_t = \tanh \left( W_{hh} h_{t-1} + W_{xh} x_t + b \right)
$$
$$
\text{Logit} = W_{hy} h_{t} + b
$$

</center>

output으로 나온 벡터는 softmax에 통과되어 확률으로 변환된 후 이에 맞추어 예측 및 학습을 진행하게 된다. 
단순하게 보면 아까 본 이미지에서 예측결과는 output vector에서 가장 큰 값을 가진 index가 된다.
예를 들어 첫번째 output의 예측결과는 [0,0,0,1]로 인코딩되는 'o'이다. 
원래는 e가 와야하므로 틀린 값이며, softmax loss를 통해 학습을 진행한다.   
   
여기서 logit이라는 단어가 쓰였는데, 간단히 짚고 넘어가자면 sigmoid를 통해 확률로 변환될 수 있는 값을 말한다.
즉 $(-\infty, \infty) \rightarrow (0, 1)$ mapping이 가능한 값으로, sigmoid 함수와 역함수 관계이다.

logit에 대해 더 자세한 내용은 <span class="link_button">[이 블로그 글](https://bit.ly/2N7RLo4)</span>을 참고하도록 하자.

![RNN_result](/img/posts/17-5.png){: width="40%" height="40%"}{: .center} 
실제 추론(inference)시에는 위와 같이 이전의 output을 다음 input으로 넣어주게 된다.
  

지금까지 배운 character-level language model은 간단하게는 긴 글부터, 코드, 논문 등에까지도 적용해볼 수 있다. 
다만 글에서의 띄어쓰기, 쉼표, 줄바꿈 등을 모두 반영할 수 있도록 이 글자들도 딕셔너리에 추가되어야 한다. 
마찬가지로 이를 코드에 적용한다고 하면 indentation이나 괄호 등에 대한 규칙이 모두 학습되어야 한다.

<br />

#### Hidden state vector의 의미
RNN의 특성을 분석해볼때, hidden state vector의 각 성분이 무엇을 의미하는지를 생각해볼 수 있다. 
이를 분석하기 위해 hidden state vector의 한 dimension만을 지속적으로 관찰해볼 수 있다.

![RNN_search_result](/img/posts/17-6.png){: width="100%" height="100%"}{: .center} 
그 결과로 위와 같은 재미있는 결과를 얻을 수 있다. 
이것은 위에서 말한대로 hidden state vector의 특정 한 cell의 값만을 추적한 결과인데
따옴표를 기준으로 값이 바뀌는 특정 cell이 존재한다는 점을 확인할 수 있다.
참고로 빨간색이 짙을수록 양수, 파란색이 짙을수록 음수를 의미한다.

<br />

#### Backpropagation에서의 문제
이전 글에서도 보았지만 sequence가 너무 길어지면 역전파가 vanishing/exploding될 우려가 있다. 
이는 같은 값의 가중치 $W\_{hh}$가 매 레이어마다 계속 곱해지면서 전달되기 때문에 발생하는 현상이다.   
  
이를 해결하기 위해 truncation을 적용하여 제한된 길이의 sequence만 잘라서 학습하는 방법을 적용하였다.
그러나 이후 LSTM, GRU 등의 모델이 등장하면서 현재에 이르러서는 Vanilla RNN의 경우 거의 사용되지 않는다.

<br />

## LSTM, GRU
RNN의 역전파에서 vanishing/exploding 등의 현상을 해결한 LSTM, GRU 모델에 대해 알아보도록 하자.

<br />

#### LSTM
![LSTM](/img/posts/14-4.png){: width="90%" height="90%"}{: .center}   
LSTM은 Long Short-Term Memory의 약자로, 단기 기억을 길게 가져간다는 의미이다. 
**LSTM 모델의 궁극적 아이디어는 cell state 정보를 어떠한 변환 없이 그대로 계속 가져가는데에 있다.** 
이 모델에서 핵심적으로 사용되는 cell state는 과거의 기억들을 최대한 보존하면서 가져갈 수 있으며 이를 통해 Long Short-Term Memory라는 이름에 걸맞는 역할을 할 수 있게 된다.  
  
**그래서 사실 cell state는 과거 대부분의 정보를 기억하고 있는 벡터, hidden state는 현재 시점에서 필요한 정보만을 cell state에서 filtering한 벡터라고 보면 된다.**

![LSTM_calculation](/img/posts/17-7.png){: width="70%" height="70%"}{: .center}  
i, f, o, g는 각각 Input, Forget, Output, Gate gate를 의미하며 
각 게이트는 셀에 정보를 쓸지, 셀의 정보를 지울지, 얼마나 셀의 정보를 드러낼지, 얼마나 셀에 정보를 쓸지를 결정한다. 
$x$와 $h$의 dimension을 $\text{h}$라고 하면 첫번째 가중치 행렬에는 $x$와 $h$가 concatenation 된 벡터가 곱해지므로 그 가중치 행렬의 가로 길이는 $2\text{h}$가 된다. 
또한 LSTM의 구조 자체가 4개의 sigmoid/tanh를 통과해야하기 때문에 그 세로 길이는 $4\text{h}$가 된다. (사실 이건 이전 글에서도 언급하였다)
나머지(게이트의 동작 등)는 이전과 같으므로 이전 글을 참고하도록 하자. 
  
아까 말했던 cell state와 hidden state의 역할에 대해 다시 한 번 생각해보기 위해 아래와 같은 예시를 생각해보자.  
  
"Hello"라는 단어를 예측하고 싶을 때(따옴표 포함) 현재 "He 까지 예측이 완료되었다고 가정하자. 
따옴표가 열려있기 때문에 언젠간 저 따옴표를 닫아야한다. 하지만 그 정보는 현재에는 중요하지 않을 것이다. 지금은 단어 l을 예측해야 하며 따옴표에 대한 정보는 중요하지 않다. 
따라서 이러한 경우 cell state가 따옴표에 대한 정보를 기억하고 있지만 hidden state는 현재 시점에서 필요한 정보인 l에 대한 예측 정보만 기억하려고 할 것이다.  

<br/>

#### GRU
![GRU](/img/posts/14-10.png){: width="90%" height="90%"}{: .center}  
LSTM의 경량화 모델이다. hidden state와 cell state를 hidden state 하나로 통합하여 사용한다. 
따라서 여기서의 hidden state는 LSTM의 cell state에 가까운 역할을 하게 된다.  
  
아래와 같이 GRU의 hidden state의 점화식과 LSTM의 cell state의 점화식을 보게 되면 그 형태가 매우 유사하다.

<center>

$$
\begin{aligned}
\text{(GRU)} \;\; h_t = (1-z_t) \circ h_{t-1} + z_t \circ \tilde{h_t} \\
\text{(LSTM)} \;\; c_t = f_t \circ c_{t-1} + i_t \circ \tilde{c _t}
\end{aligned}
$$

</center>

GRU에서는 LSTM과 달리 forget gate 없이 input gate만을 사용하였으며 이를 통해 도출된 값만으로($z$ 및 $1 - z$를 이용해 가중 평균의 형태로 계산) 다음 state 값을 도출해내었다. 
두 개의 게이트가 하던 일을 한 개의 게이트가 혼자 하도록 그 규모를 축소하였음에도 성능 및 동작이 비슷하다는 데에 의의가 있다.

<br />

#### Backpropagation in LSTM/GRU
Vanilla RNN은 같은 $W\_{hh}$를 계속 곱해지는 것에서 문제가 있었다.  
  

![LSTM_GRU_backpropagation](/img/posts/17-8.png){: width="100%" height="100%"}{: .center}  
하지만 LSTM에서는 매번 똑같은 값을 곱하는게 아니라 그때그때 서로 다른 값으로 나오는 forget gate의 값을 곱한다. 
또한 RNN의 hidden state 역할(과거 정보 기억)을 LSTM에서는 cell state가 수행하였는데, cell state는 그 값이 **덧셈으로부터 도출되므로 gradient가 소실되지 않는다.**
덧셈은 미분할 때 gradient를 복사해주는 역할을 하기 때문에 gradient flow가 잘 이루어지게 해준다.   
  
위 설명에서는 LSTM을 예시로 들었으나 GRU 역시 같은 구조로 동작하기 때문에 동일한 이유로 역전파 손실이 발생하지 않는다.

<br />

## Reference  
[logit, sigmoid, softmax의 관계](https://bit.ly/2N7RLo4)    
[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)  
[The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)  