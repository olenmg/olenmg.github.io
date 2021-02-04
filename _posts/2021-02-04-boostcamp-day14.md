---
layout: post
title: "Day14. RNN, Transformer"
subtitle: "RNN(LSTM, GRU)과 Transformer"
date: 2021-02-04 23:40:12+0900
background: '/img/posts/bg-posts.png'
use_math: true
---

## 개요 <!-- omit in toc -->
> 오늘은 고전 RNN과 현대의 RNN, 그리고 이들을 모두 대체할 수 있는 별개의 구조 Transformer에 대해 다루어보았다.
  

오늘은 아래 내용을 다루었다.
- [RNN(Recurrent Neural Network)](#rnnrecurrent-neural-network)
    - [시퀀스 데이터와 RNN](#시퀀스-데이터와-rnn)
    - [Backpropagation Through Time(BPTT)](#backpropagation-through-timebptt)
- [LSTM과 GRU](#lstm과-gru)
    - [LSTM(Long Short Term Memory)](#lstmlong-short-term-memory)
    - [GRU(Gated Recurrent Unit)](#grugated-recurrent-unit)
    - [그 외](#그-외)
- [Transformer](#transformer)
    - [Transformer의 구조](#transformer의-구조)
- [Reference](#reference)

<br/>

## RNN(Recurrent Neural Network)
시계열 데이터, 시퀀스 데이터(음성, 비디오, 문자열, 주가 등)를 다루기 위해 고안된 모델이다. 
지금부터는 시계열 데이터가 어떤 식으로 처리될 수 있는지와 이를 위해 RNN이 어떻게 동작하는지 살펴보도록 하자.  

<br />

#### 시퀀스 데이터와 RNN
시퀀스 데이터는 <strong>독립동등분포(i.i.d.)</strong> 가정을 잘 위배하기 때문에 순서를 바꾸거나 과거 정보에 손실이 발생하면 데이터의 확률 분포도 바뀌게 된다.  

시퀀스 데이터의 확률 분포를 가장 간단하게 나타내보면 아래와 같다.

<center>

$$
P(X_1, \cdots, X_t) = P(X_t \vert X_1, \cdots, X_{t-1})P(X_1, \cdots, X_{t-1})
$$
$$
P(X_1, \cdots, X_t) = \prod\limits _{s=1} ^t P(X_s \vert X_{s-1}, \cdots, X_1)
$$

</center>

첫번째 식에서 좌변은 결합확률분포이고, 우변은 $X\_t$에 대한 조건부확률분포에 결합확률분포를 곱한 형태이다.
그리고 이를 일반화하면 두번째 식이 나오게 된다.   
  

근데 실제로는 이렇게 과거의 모든 정보를 사용하진 않는다. 현재의 주가 예측을 위해 10년전 주가 정보를 사용하는 일은 드물 것이다. 
또한 시간이 지날수록 데이터가 쌓이기 때문에 시점에 따라 데이터의 길이가 달라지게 된다. 이에 따라 우리는 길이가 가변적인 데이터를 다룰 수 있는 모델이 필요하게 된다.
  
  
하지만 가변 길이를 다루는 일은 쉽지 않다. 그래서 우리는 고정된 길이를 다루는 모델을 찾아야하는데, 실제로 고정된 길이 $\tau$만큼의 데이터만 다루는 아래와 같은 모델이 제안된다.

<center>

$$
X_{t} \sim P(X_{t} \vert X_{t-1}, \cdots, X_{t-\tau})
$$
$$
X_{t + 1} \sim P(X_{t + 1} \vert X_{t+1}, X_{t}, \cdots, X_{t - \tau + 1})
$$

</center>

위와 같은 모델을 AR(Autoregressive Model) 자기회귀모델이라고 부른다.   
이 모델에서는 미래 예측 시 과거 $\tau$ 이전 시점의 데이터는 모두 고려하지 않는다.  

  

아래와 같이 바로 이전의 과거만 고려하는 모델도 존재한다.(Markov model)

<center>

$$
P(X_1, \cdots, X_t) = P(X_{t} \vert X_{t-1})P(X_{t-1} \vert X_{t-2})\cdots P(X_{2} \vert X_{1})P(X_1) = \prod\limits _{t=1} ^T p(X_t \vert X_{t-1})
$$

</center>

표현이 쉽다는 장점은 있지만 과거의 많은 정보를 버리는 위와 같은 모델은 예측할 수 있는 것이 얼마 없을 것이다. 
다만, 이 구조는 추후 <strong>generative 모델에서 많이 쓴다고 하니</strong> 기억해두자.  


아무튼 위에서 소개한 두 모델은 딱봐도 한계점이 존재하고, 현실적으로 잘 쓰이지는 않는다.  

  

그래서 다시 제안된 것이 잠재 AR모델이다.

<center>
$$
H_t = \text{Net} _\theta (H_{t-1}, X_{t-1}) \text{일 때,}
$$
$$
X_{t} \sim P(X_{t} \vert X_{t-1}, H_t)
$$
$$
X_{t + 1} \sim P(X_{t + 1} \vert X_t, H_{t + 1})
$$

</center>

그리고 이렇게 <strong>잠재변수 $H_t$를 신경망을 통해 반복해서 사용하여 시퀀스 데이터의 패턴을 학습하는 모델이 RNN(Recurrent Neural Network)이다.</strong>  
잠재변수라는 이름이 붙은 이유는, 이 변수는 시간이 갈수록 변화하지만 직접적인 output으로 출력되지는 않기 때문이다.   
  
![RNN](/img/posts/14-1.png){: width="90%" height="90%"}{: .center}   
왼쪽 그림이 이 모델이 recurrent로 보이게 된 이유다. 그리고 실제로 왼쪽 그림에서 가운데 부분은 <strong>내부에서 자기 자신을 이용해 반복되어 갱신될 뿐 그 자체가 출력이 되지는 않는다.</strong>  

그럼 모델이 구체적으로 어떻게 달라지는지 살펴보도록 하자.   
원래의 MLP 모델이 층을 거칠 때 아래와 같이 output이 나왔다면, 

<center>

$$
\mathrm{H} _t = \sigma(\mathrm{X} _t \mathrm{W} ^{(1)} + \mathrm{b} ^{(1)})
$$
$$
\mathrm O_t = \mathrm H _t \mathrm W ^{(2)} + \mathrm b ^{(2)}
$$

</center>

RNN 모델은 MLP와 기본적인 구조는 비슷하지만 아래와 같이 <strong>입력에 이전 시점의 잠재변수가 추가된다.</strong>

<center>

$$
\mathrm{H} _t = \sigma(\mathrm{X} _t \mathrm{W} _X ^{(1)} + \mathrm{H} _{t - 1} \mathrm{W} _H ^{(1)} + \mathrm{b} ^{(1)})
$$
$$
\mathrm O_t = \mathrm H _t \mathrm W ^{(2)} + \mathrm b ^{(2)}
$$

</center>

여기서 모든 가중치 $\mathrm{W_X}$, $\mathrm{W_H}$, $\mathrm{W}$는 $t$에 따라 변하지 않으며, 각 시점에서의 입력 $\mathrm{X}_t$와 잠재벡터 $\mathrm{H} _ t$ 값이 시간에 따라 변한다는 것을 기억하자.

<br/>

#### Backpropagation Through Time(BPTT)
RNN에서도 역전파를 사용하는데, 이를 BPTT라고 부른다.   
근데 방법은 결국 똑같다. 잠재변수의 연결그래프에 따라 앞부분의 역전파를 순차적으로 계산할 수 있다.   
  
먼저, 각 시간 $t$에서의 잠재변수 $h_t$와 출력 $o_t$는 아래와 같다.

<center>

$$
h_t = f(x_t, h_{t-1}, w_h)
$$
$$
o_t = g(h_t, w_o)
$$

</center>

이를 기반으로 역전파를 계산해보자.     


output이 모든 시간 $t$에서 하나씩 나오므로 손실함수에 는 각 시점에서의 예측값과 기댓값이 parameter로 들어가게된다.  

<center>

$$
L(x, y, w_h, w_o) = \sum\limits _{t=1} ^T \mathit{l}(y_t, o_t)
$$
$$
\partial _{w_h} L(x, y, w_h, w_o) = \sum\limits _{t=1} ^T \partial w_h \mathit{l}(y_t, o_t)
$$

</center>

역전파는 chain rule에 의해 $\dfrac{\partial L}{\partial w\_h} = \dfrac{\partial L}{\partial o\_t} \dfrac{\partial o\_t}{\partial h\_t} \dfrac{\partial h\_t}{\partial w\_h}$이다.
따라서, 

<center>

$$
\partial _{w_h} L(x, y, w_h, w_o) = \sum\limits _{t=1} ^T \partial _{o_t} \mathit{l}(y_t, o_t) \cdot \partial _{h_t} g(h_t, w_h) \cdot [\partial _{w_h} h_t]
$$
</center>

나머지는 그냥 계산하면 되는데, 중요한건 맨 뒤에 대괄호로 씌워져 붙어있는 $\dfrac{\partial h\_t}{\partial w\_h}$ $\left(=\partial _{w_h} h_t \right)$이다.   
  
$h_t$에는 $t$시점 이전의 모든 과거 데이터의 값이 쌓여있다. 
그래서 사실 <strong>RNN의 각 시점을 input이 수없이 많은 FC layer(서로 parameter를 share하는)로 바꾸어 나타낼수도 있긴 하다.</strong>   
  
아무튼 하고자 하는 말은, 각 $h_t$는 $t$이전 모든 시점의 잠재변수의 값에 영향을 받고 있고, 심지어 $w_h$에도 영향을 받고 있다. 그래서 $h_t$를 미분하려고 하면 아래와 같게된다.  

<center>

$$
\frac{\partial h_t}{\partial w_h}= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h} +\frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_h}
$$
$$
\frac{\partial h_{t-1}}{\partial w_h}= \frac{\partial f(x_{t-1},h_{t-2},w_h)}{\partial w_h} +\frac{\partial f(x_{t-1},h_{t-2},w_h)}{\partial h_{t-2}} \frac{\partial h_{t-2}}{\partial w_h}
$$
$\cdots$

</center>

따라서 정리하면 아래와 같다.  

<center>

$$
\partial _{w_h} h_t = \partial _{w_h} f(x_t, h_{t-1}, w_h) + \sum\limits _{i=1} ^{t-1} \left( \prod\limits _{j=i+1} ^{t} \partial _{h _{j-1}} f(x_j, h_{j-1}, w_h) \right)
\partial _{w_h} f(x_i, h_{i-1}, w_h)
$$

</center>

이 부분은 $\dfrac{\partial L}{\partial w\_h}$의 값에 치명적 영향을 준다. 수없이 많은 곱이 붙어있기 때문이다.   
  
특히, $t$가 커질수록(즉, 시퀀스의 길이가 길어지수록) 곱해지는 값이 많아져 만약 1보다 작은 값이 많으면 vanishing 현상이, 1보다 큰 값이 많으면 exploding 현상이 나타나게 된다. 
즉, 이 항은 굉장히 불안정하다. 그래서 Vanila RNN은 Long term sequence를 제대로 처리할 수 없다는 문제점이 있다. (이를 long-term dependencies라고 부른다)  
  
  
이를 해결하기 위해 BPTT 계산시 중간중간 $H_t$간 연결을 끊어주는 기법도 제시되었다. (Truncated BPTT)  
이렇게 하면 가중치 하나는 주변 몇 개의 $H_t$에만 영향을 받으므로 곱해지는 값이 줄어들게 되어 비교적 안정적이게 된다.  
  
<strong>그리고 이후 LSTM, GRU 등의 advanced RNN을 도입하여 이 문제를 해결하기도 하였다. </strong> 

![RNN_vanishing_exploding](/img/posts/14-2.png){: width="90%" height="90%"}{: .center}  
> 위와 같이 activation function 때문에 h_t 자체에서도 exploding/vanishing이 나타나기도 한다.  
   
<br />

## LSTM과 GRU
기존 RNN 모델은 아래와 같이 나타낼 수 있다.
![RNN_pre](/img/posts/14-3.png){: width="90%" height="90%"}{: .center}   
지금부터는 이를 개선한 LSTM과 GRU 모델에 대해 알아보자.  

<br />

#### LSTM(Long Short Term Memory)
LSTM은 아래와 같은 구조를 가졌다.
![LSTM](/img/posts/14-4.png){: width="90%" height="90%"}{: .center}   
여기서 각 component가 어떻게 동작하는지를 이해하는 것이 역시 중요할 것이다. 
기존 RNN과 다르게 <strong>input이 하나 더 들어온다는 점(cell state)</strong>을 짚고 넘어가자. 
  

더 세부적으로 내부 구조를 보면 아래와 같이 파트를 나눌 수 있다.
![LSTM_deep](/img/posts/14-5.png){: width="90%" height="90%"}{: .center}   

- State(input/output)
    + Input: 이번 시점에 들어오는 데이터 input이다. i.e. NLP의 단어 등 
    + Previous cell state: 이전 cell state가 input으로 들어온다. <strong>cell state는 output에 영향을 줄 뿐, 직접적으로 출력되지는 않는다.</strong>
    + Previous hidden state: 이전 레이어의 hidden state($h_{t-1}$)이 input으로 들어온다.
    + Next cell state: 게이트를 거쳐 업데이트된 cell state가 다음 레이어의 input으로 들어간다.
    + Next hidden state: 업데이트 된 hidden state($h_t$)가 다음 레이어의 input으로 들어간다.
    + Output (hidden state): 들어온 이전 hidden state는 업데이트된 후 다음 레이어의 input으로 들어가는 한편 output으로도 나간다.

- Gate/Cell 
    + Forget gate
        ![LSTM_forget](/img/posts/14-6.png){: width="50%" height="50%"}{: .center}
        - 과거 정보를 잊는다(forget)
        <center>
        
        $f_t=\sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
        
        </center>
        - 새로 들어온 정보에 기반하여 이전까지 기억하고 있던 정보들 중 어떤 정보를 버릴지 sigmoid를 통해 판단한다.
    + Input gate
        ![LSTM_input](/img/posts/14-7.png){: width="50%" height="50%"}{: .center}
        - 정보를 기억한다(input)
        <center>
        

        $$
        \tilde{C_t} = \tanh{(W_C \cdot [h_{t-1}, x_t] + b_C)}
        $$
        
        </center>
        - 새로 들어온 정보 $x_t$에 대하여 어떤 정보를 그대로 유지할지 sigmoid / tanh 함수로 판단하여 후보 $\tilde{C_t}$를 정한다.
        - 
    + Update cell 
        ![LSTM_update](/img/posts/14-8.png){: width="50%" height="50%"}{: .center}
        - input gate와 forget gate 각각에서 나온 output을 더하여 다음 cell state를 최종적으로 업데이트(결정)한다.
        <center>
        
        $$
        i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
        $$
        $$
        C_t = f_t * C_{t-1} + i_t * {\tilde{C_t}}
        $$
        
        </center>
        - forget gate쪽의 경우 $f_t$를 이전 cell state $C_{t-1}$과 곱하여 잊을 정보를 진짜 잊어버린다.
        - input gate쪽의 경우 구한 $\tilde{C_t}$에 $i_t$를 곱한다. $i_t$는 기억할 값을 얼마나 강하게 기억할지 scaling 하는 역할을 한다.

    + Output gate
        ![LSTM_output](/img/posts/14-9.png){: width="50%" height="50%"}{: .center}
        - 어떤 것을 output으로 내보낼지 결정한다.
        <center>
        
        $$
        o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)
        $$
        $$
        h_t = o_t * \tanh{(C_t)}
        $$
        
        </center>
        - 밖으로 나가는 output(위쪽)은 cell state를 바탕으로 필터링된 값이 나간다.
            + 기존 hidden state($h_{t-1}$)를 sigmoid에 통과시켜 어느 부분을 output으로 내보낼지 결정한다.
            + update cell에서 업데이트된 cell state($C_t$)를 tanh에 통과시켜 얼마나 output으로 내보낼지 결정한다. 
            + 둘을 곱해주어 둘 모두의 수치가 반영된 값을 이번 단계의 output과 다음 단계의 input으로 내보낸다.  
  
  
그래서 Vanishing/Exploding 문제는 어떻게 해결될까?  
LSTM의 역전파는 $C_t$가 업데이트되어야하는데, $C_t$의 경로에는 $f_t$가 곱해지는 부분이 있다.  
즉, 역전파의 경로가 훨씬 단순해졌으며, forget gate의 parameter를 적절히 업데이트해주면 해당 시점 $t$에서의 vanishing 문제를 해결할 수 있다.  
또한 $f_t$는 sigmoid를 통과한 값이므로 exploding 현상도 일어나지 않는다.   

<br />

#### GRU(Gated Recurrent Unit)
![GRU](/img/posts/14-10.png){: width="90%" height="90%"}{: .center}  
GRU 구조에서는 hidden state와 cell state가 hidden state 하나로 통합된다.  
또한 게이트의 갯수가 reset gate와 update gate 총 2개로, gate 갯수 역시 간소화되었다.  

- Reset gate
    ![GRU_reset](/img/posts/14-11.png){: width="50%" height="50%"}{: .center}  
    + 과거의 정보를 적당히 리셋(reset)시키는 것이 목적으로, 시그모이드 함수를 통과한다. ($r_t)

- Update gate
    ![GRU_update](/img/posts/14-12.png){: width="50%" height="50%"}{: .center}  
    + LSTM의 forget gate와 input gate의 역할을 합쳐놓은 게이트이다.
    + 과거의 정보와 최신의 정보에 각각 가중치를 얼마나 줄지 결정한다. $z_t$ 위에 '$1-$'라고 적힌 것은 $1-z_t$를 의미한다.

- Candidate
    ![GRU_candidate](/img/posts/14-13.png){: width="50%" height="50%"}{: .center}  
    + 리셋 게이트를 통과하여 리셋된 정보를 tanh에 통과시켜 가져갈 정보의 후보를 정한다.

- Hidden layer 값 갱신
    ![GRU_hidden_layer](/img/posts/14-14.png){: width="50%" height="50%"}{: .center}  
    + update gate의 결과와 candidate의 결과를 합하여 hidden state를 갱신한다.
  

GRU는 LSTM과 구조적으로 큰 차이가 없고 성능도 비슷하지만, <strong>gate가 적어 학습할 parameter의 갯수가 적기 때문에 연산량에서의 이점이 있다.</strong>  


<br />

#### 그 외
PyTorch로 LSTM을 구현하다가 발견한 몇 가지를 여기 기술한다.  
- <code>nn.LSTM</code>의 parameter로 들어가는 n_layer는 말그대로 layer의 갯수이다. 우리가 위에서 배운건 레이어 1개짜리 LSTM인데, multi-layer LSTM의 구조는 아래와 같다.
![multi_layer_LSTM](/img/posts/14-15.png){: width="80%" height="80%"}{: .center}  
- LSTM의 cell state dimension과 hidden state dimension은 서로 같아야 한다
- LSTM에는 생각보다 parameter 갯수가 많다.
    + 게이트의 갯수는 3개지만 사실 update cell에서도 update 스케일을 정하기 위한 가중치가 있어, 총 가중치는 4개가 존재한다.
        - 각각의 가중치는 input과 한번, hidden과 한번 곱해진다고도 이해할 수 있다.(실제로는 concatenation해서 곱한다)
        - 따라서 input에 대한 가중치 갯수는 (input dim) * (hidden dim(=output dim)) * 4개이다.
        - 따라서 hidden에 대한 가중치 갯수는 (hidden dim) * (hidden dim(=output dim)) * 4개이다.  
        - 이에 대한 이야기는 <span class="link_button">[PyTorch 공식 문서](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM)</span>에 잘 나와있다.

<br/>

## Transformer
지금까지 살펴보았던 RNN 기반 모델들은 문장에서 몇 개의 단어가 빠지거나 순서가 조금이라도 바뀌면 다음 단어의 예측이 어려워진다. 
실제로 언어라는게, 단어를 생략할수도 있고 같은 문장에서 단어 간의 순서가 바뀌어도 사람은 모두 이해할 수 있다. 
그래서 RNN과는 완전히 다른, 별개의 모델이 제시되는데 그것이 바로 <strong> self attention 구조를 활용한 Transformer라는 모델이다.</strong>

<br />

#### Transformer의 구조
![transformer](/img/posts/14-16.jpg){: width="70%" height="70%"}{: .center}  

전체적인 구조는 위와 같다.
- RNN의 재귀적인 구조를 사용하지 않는다.
- <strong>attention</strong>이라는 구조에 기반을 둔 모델이다.
- <strong>입력과 출력의 길이가 다를 수 있다.</strong>
    + 원래 CNN/RNN 등의 모델에서는 input 길이가 고정이면 output 길이가 고정이었다.
    + transformer는 input의 길이가 똑같아도 그 순서나 단어 종류에 따라 output 길이가 가변적이다.
- 입력과 출력의 domain이 다를 수 있다. (입력의 차원이 다를 수 있음) 

![encoding_decoding](/img/posts/14-17.png){: width="70%" height="70%"}{: .center}  
- transformer는 크게 보면 같은 갯수로 stack된 Encoding파트와 Decoding파트로 나눌 수 있다.
    + 왼쪽 Encoding 파트에서는 들어온 단어(embedded word) sequence에서 특징을 추출한다.
    + 오른쪽 Decoding 파트에서는 왼쪽에서 추출한 것들로 새로운 sequence를 표현한다.
    + 결과적으로 sequence to sequence 모델이다.
    + Stack되어있는 각 인코더와 디코더는 <strong>동일한 구조를 가지지만, 파라미터는 다르게 학습된다. (즉, 별개의 모델)</strong>
- RNN 모델에서는 단어를 시간에 따라 하나씩 순서대로 넣어줬지만 <strong>transformer는 단어 sequence를 한번에 입력받는다.</strong> (encoder 부분으로 입력받음)
    
<br/> 

지금부터는 다음과 같은 3가지 의문점을 해결할 것이다.
1. N개의 단어가 encoder에서 어떻게 한 번에 처리되는가?
2. Encoder-Decoder간 어떤 정보가 오가는가?
3. Decoder가 단어를 어떻게 생성하는가?


>> 작성중
<br />

## Reference  
[Long Short Term Memory(LSTM)](https://dgkim5360.tistory.com/entry/understanding-long-short-term-memory-lstm-kr)  
[Gated Recurrent Unit(GRU)](https://yjjo.tistory.com/18)  