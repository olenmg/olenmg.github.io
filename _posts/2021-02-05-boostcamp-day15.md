---
layout: post
title: "Day15. Generator"
subtitle: "Generative model, VAE, GAN"
date: 2021-02-05 23:59:12+0900
background: '/img/posts/bg-posts.png'
use_math: true
---

## 개요 <!-- omit in toc -->
> Generative Model에 대해 알아보고, 대표적인 generative model인 VAE, GAN에 대해 자세히 알아보았다.  
  

오늘은 아래 내용을 다루었다.
- [Generator](#generator)
    - [Discrete Distributions](#discrete-distributions)
    - [필요한 parameter의 수](#필요한-parameter의-수)
    - [Conditonal Independence](#conditonal-independence)
    - [Auto regressive Model (AR model)](#auto-regressive-model-ar-model)
- [작성중 쿠쿠쿠](#작성중-쿠쿠쿠)
- [Reference](#reference)

<br/>

## Generator 
Genarative model이란 학습 데이터의 분포를 따르는 유사한 데이터를 생성해내는 모델이다. 
근데 사실 실제로는 생성뿐만 아니라 더 많은 기능을 수행할 수 있다.
  
여기서는 학습을 통해 학습 데이터의 분포 $p(x)$를 찾는 것이 주요 목적이다.  
그리고 이 확률분포 $p(x)$를 이용해 아래와 같은 task를 수행할 수 있다.  
  

1. Generation
    - 만약 우리가 $x\_{\text{new}} \sim p(x)$ 인 $x\_{\text{new}}$를 샘플링한다고하면, $x\_{\text{new}}$는 우리가 원하는 것이어야 한다.
    - i.e. 만약 우리가 개에 대한 이미지를 학습했다면, $x\_{\text{new}}$는 개처럼 생겨야 한다.
2. Density estimation
    - 넣어준 이미지에 대하여 이미지가 우리가 원하는 것에 가깝다면 $p(x)$가 높아야 하며, 가깝지 않다면 $p(x)$는 낮아야 한다.
    - i.e. $x$가 개와 닮았다면 $p(x)$의 값이 높고, 아니면 낮다.   
    - 이를 통해 anomaly detection(이상 행동 감지)을 할 수 있다. 평소와 다른 행동에 대하여 $p(x)$의 값은 낮게 나타날 것이다.  
    - 이것은 마치 discriminative model(일반적인 분류 문제)과 같은 역할을 한다.
    - <strong>그래서 사실 generative model은 엄밀히 말하면 discriminative model을 포함하고 있다.</strong>
    - 이렇게 두 속성을 모두 가진 모델을 <strong>explicit model</strong>이라고 부르며, 단순히 generation만 수행하는 VAE, GAN 등의 모델은 implicit model이라고 나타낸다.
3. Unsupervised representation learning(feature learning)
    - 이 이미지가 보통 어떤 특성을 가지는지 스스로 학습해낸다.
    - i.e. 개에게 귀가 있고 꼬리가 있고, ...


<br/>


#### Discrete Distributions
우리는 확률분포를 explicit하게 나타낼 수 있는 explicit model(tractable density)에 대해 다뤄볼 것이다. 
학습을 어떻게 시키는지를 보기 이전에, 보고 지나가야할 것들이 많다.  
먼저 가장 대표적인 이산확률분포 몇 가지를 다시 짚고 넘어가자.  

- Bernoulli distribution
    + 나올 수 있는 경우가 2개인 경우의 확률 분포이다
    + $X \sim \text{Ber}(p)$
    + parameter $p$하나로 모든 확률을 표현할 수 있다.

- Categorical distribution
    + 나올 수 있는 경우가 $n$개인 확률 분포이다.
    + $Y \sim \text{Cat}(p\_1, \cdots, p\_n)$
    + 어떤 한 상황의 확률은 전체 확률 1에서 나머지 확률을 빼줌으로써 구할 수 있으므로 parameter는 $n-1$개 필요하다.

parameter 갯수 이야기를 하고 있는데, 그 이유는 <strong>주어진 학습 데이터의 확률 분포를 나타내기 위해 몇 개의 파라미터가 필요한지를 먼저 찾아야 하기 때문</strong>이다.   

<br/>

#### 필요한 parameter의 수  
다음 예시를 보자.
- 28 x 28 binary pixel 이미지의 확률 분포를 찾으면 이 때 parameter 수는?
    + 이 이미지는 binary pixel로 이루어져 있으므로 나타낼 수 있는 이미지의 경우의 수는 $2^{(28*28)} = 2^{768}$개이다.
    + 여기서 중요한건 이 이미지에 있는 픽셀들간 <strong>상호 독립적이라는 보장이 없다</strong>는 것이다.
    + 따라서 모든 픽셀이 서로 dependent하다고 가정하면 확률 분포 $p(x\_1, \cdots, x\_{2^{768}})$의 parameter 수는 $2^{768} - 1$개이다.

- 만약 위 문제에서 각 픽셀 $X\_1, \cdots, X\_n$이 서로 독립이면, 몇 개의 parameter가 필요할까?
    + 픽셀들이 서로 독립이므로 다음이 성립한다. 
    <center>
    $p(x_1, \cdots, x_{n}) = p(x_1)p(x_2)\cdots p(x_n)$
    </center>
    + 따라서 이 경우 각각에 대한 확률값만 parameter로 나타내면 되므로 $n$개의 parameter가 필요하다.
    + 하지만 실제로는 이렇게 모두가 독립인 경우는 존재할 수가 없다.
  
극단적인 두 가지 경우를 살펴보았다. 전자는 parameter가 말도 안되게 많고, 후자는 현실 세계에 적용할 수가 없다. 
따라서 우리는 이 사이 중간 어딘가를 찾아야한다.  
 
<br/>

#### Conditonal Independence
다음 세가지 rule을 짚고 넘어가자.
1. Chain rule
    <center>
    $$
    p(x_1, \cdots, x_n)=p(x_1)p(x_2 \vert x_1)p(x_3 \vert x_1,x_2) \cdots p(x_n \vert x_1,...,x_{n-1})
    $$
    </center>
    - 결합확률 분포를 위와 같이 나타낼 수 있다. 직관적으로 이해 가능하다.
    - Chain rule은 확률변수간 dependent 여부와 관계 없이 성립한다.
    - 아까 본 dependent binary pixel image에 chain rule을 적용하면 똑같이 parameter가 $2^n - 1$개 필요함을 확인할 수 있다. 

2. Bayes' rule
    <center>
    $$
    P(x \vert y) = \dfrac{p(x, y)}{p(y)} = \dfrac{P(y \vert x) p(x)}{P(y)}
    $$
    </center>
    - 베이즈 정리는 이전에 이미 배운 적이 있다.

3. Conditional Independence
    <center>
    $$
    \text{If} \;\; x \perp y \vert z, \;\;\text{then} \;\;\; p(x \vert y, z) = p(x \vert z)
    $$
    </center>
    - $\perp$는 독립을 의미한다. 즉, $z$가 주어졌을 때 $x$와 $y$가 서로 독립이 되면 이 때 $y$의 발생여부는 상관 없으니까 없애도 된다는 뜻이다. 

  
우리가 찾고자 하는 것은, 학습시킬 수 있는 적당한 parameter를 가진 확률분포이다.    
  
Markov assumption($i+1$번째 픽셀은 $i$번째 픽셀에만 dependent한 경우)에서는 다음이 성립한다.

<center>
$$
p(x_1, \cdots, x_{n}) = p(x_1)p(x_2 \vert x_1)p(x _3 \vert x _2) \cdots p(x_n \vert x_{n - 1})
$$
</center>

따라서 이 때 필요한 parameter는 $p(x_1)$, $p(x_2 \vert x_1 = 0)$, $p(x_2 \vert x_1 = 1)$, $p(x_3 \vert x_2 = 0)$, $p(x_3 \vert x_2 = 1)$, $\cdots$, $p(x_n \vert x_{n-1} = 0)$, $p(x_{n} \vert x_{n-1} = 1)$으로 총 $2n - 1$개이다.  

따라서 이러한 가정 안에서는 parameter 갯수가 대폭 줄어들게 되며, 이러한 조건부 독립성을 이용한 모델이 바로 <strong>Auto-regressive model(AR model)</strong>이다.   
  
참고로, 위에서 가정한 것은 <strong>AR-1 model</strong>에 속하며, 이것은 현재의 확률이 단순히 이전 1개에만 dependent한 경우이다. 
이전 $n$개에 dependent한(즉, $n$개를 고려하는) 모델을 AR-n 모델이라고 부른다. 
그리고 사실 어떤 식으로 conditional independence를 주느냐에 따라 각 픽셀이 dependent 변수의 갯수가 다를수도 있으므로, 이를 조절하는 것이 모델 자체에 큰 영향을 준다. 
따라서 이런 conditional independence를 잘 부여하는 것이 중요하다.  

<br/>

#### Auto regressive Model (AR model)
위에서 보았던 28 x 28 이미지의 joint distribution을 chain rule으로 나타내면 아래와 같다.

<center>
$$
p(x_{1:784}) = p(x_1)p(x_2 \vert x_1)p(x_3 \vert x_{1:2}) \cdots
$$
</center>
여기서 conditional independence를 어느정도 부여하게 되는데, 이러면 앞서 살펴본 3번 rule에 따라 해당 변수를 확률 식에서 지울 수 있으므로 그 부분의 필요 파라미터 갯수를 줄일 수 있다.

참고로 위에서 말한 '이전 픽셀'이라는게 픽셀들을 순서대로 나열한 이후에 정의될 수 있는데, 이 순서를 정하는 방법에도 여러가지가 있다.
그리고 당연히 순서를 정하는 방법론도 모델의 구조나 성능에 많은 영향을 준다. 

  
AR model에는 아래와 같은 모델이 존재한다.
- NADE(Neural Autoregressive Density Estimator)
    ![NADE](/img/posts/15-1.png){: width="100%" height="100%"}{: .center}  
    - $i$번째 픽셀이 $1$번째부터 $i-1$번째까지 픽셀에 모두 dependent한 모델이다.
    <center>
    $$
    p(x_i \vert x _ {1:i-1}) = \sigma (\alpha _i \mathrm{h}_i + b_i)
    $$
    $$
    \mathrm{h}_i = \alpha (W _{< i} x_{1:i-1} + c)
    $$
    </center>
    - 위 식을 말로 표현해보자.
        1. $i$번째 conditional probability를 구하고자 한다.
        2. $1 \sim i - 1$번째 값을 FC layer에 통과시킨 값을 시그모이드에 통과시켜 $\mathrm{h}_i$를 얻는다.
        3. 그 값을 다시 어떤 값 $\alpha$에 곱해서 시그모이드를 한번 더 통과시킨 값이 확률이 된다.
    - $1 \sim i - 1$번째 확률을 다 고려한다는 점, 그리고 input이 아래로 갈수록 늘어나니까 FC layer에서 곱해주는 가중치의 dimension도 올라간다는 점 정도를 생각해보면 될 것 같다.
        + 예를 들어 $x_{58}$에 대한 확률을 구하려고 하면 57개의 input을 받을 수 있는 가중치 $W$가 필요할 것이다.
    - NADE는 explicit 모델로, 784개 입력에 대한 확률 계산이 가능하다.
        <center>
        $$
        p(x_1, \cdots, x_784) = p(x_1)p(x_2 \vert x_1) \cdots p(x_{784} \vert x_{1:783})
        $$
        </center>
        + 각 $p(x\_i \vert x \_ {1:i-1})$는 독립적으로 앞에서 계산되었으니까 이것으로 특정상황에 대한 확률 계산이 가능하다.
    - 지금까지 계속 discrete variable에 대해서만 다루었는데, 만약 continuous variable을 다루고 싶다면 a mixture of Gaussian 분포를 사용하면 된다.
    - 장황하게 써놨는데.. $\alpha$가 무슨 값을 지칭하는지도 모르겠고, 이거 포함해서 아직 이걸 정확히 이해한건 아니다. 하지만 검색해도 정보가 많이 없고 수업에서도 비중을 적게 한걸로 봐선 굳이 더  자세히 찾아보진 않으려고 한다.
    - 여담으로, 이 모델처럼 density estimator라는 이름이 붙은 모델은 explicit model이라고 보면 된다. 단어부터 그런 뉘앙스를 준다.  

- Pixel RNN
    - RNN을 사용하여 정의된 AR model이다.
    - 예를 들어, n x n 이미지의 RGB를 나타내면 아래와 같다.
    <center>
    $$
    p(x) = \prod\limits _{i=1}^{n^2} p(x_{i,R} \vert x_{<i})p(x_{i,G} \vert x_{<i},x_{i,R})p(x_{i,B} \vert x_{<i},x_{i,R},x_{i,G})
    $$
    </center>
    - 지금까지 들어온 픽셀을 기반으로 다음 픽셀을 예측(RNN 관점) 혹은 생성(AR model 관점)한다.
    - RNN에서도 과거의 정보를 고려하니까 이러한 특성을 generative model을 쓸 때도 적용할 수 있는 것 같다.
        + 즉, 픽셀의 sequence를 그냥 픽셀이 쭉 이어져있는 시계열 데이터라고 보는 것이다.
    - 가장 고전적인 Pixel RNN은 말그대로 픽셀 sequence를 쭉 펴서 RNN처럼 돌린다.
    - 그런데 앞서 말했듯이, generative model에서는 순서를 정하는 방법도 중요한데, 이에 따라 아래와 같이 두 개의 모델이 나오게 된다.
        ![pixel_RNN](/img/posts/15-2.png){: width="100%" height="100%"}{: .center}  
        + 둘다 LSTM 구조를 기반으로 한다.
        + Row LSTM은 모든 픽셀을 보지 않고 직접적인 영향을 주는 픽셀(위쪽 triangular)들을 통해서만 학습한다. 근데 바로 옆 pixel도 영향을 줄 수 있으니까 이것까지 고려한다.
        + Diagnoal BiLSTM은 지금까지 들어온 이전 모든 정보를 활용하여 학습한다. 고전적인 RNN에서는 오른쪽에서 오는 픽셀 정보를 고려하기가 쉽지 않다고 하는데, 이 점을 개선한 것 같다.
        + Diagonal BiLSTM이 연산량이 더 많지만 빠르고, 반대로 Row LSTM은 연산량이 적다는 이점이 있다.


<br/>

## 작성중 쿠쿠쿠

<br />

## Reference  
[Pixel RNN](http://ai-hub.kr/post/98/)  