---
layout: post
title: "Day12. Optimizer, CNN"
subtitle: "Optimization, CNN의 Convolution"
date: 2021-02-02 23:13:12+0900
background: '/img/posts/bg-posts.png'
use_math: true
---

## 개요 <!-- omit in toc -->
> 모델을 현실 세계로 근사시키는 직접적인 역할을 바로 Optimizer가 수행한다. 오늘은 이러한 Optimizer의 발전과정과 종류를 살펴보았다. 그리고 CNN의 Convolution에 대해서도 간단하게 배워보았다. 
  

이 글은 아래와 같은 구성으로 이루어진다.
- [최적화(Optimization) 기법](#최적화optimization-기법)
    - [Optimization의 중요한 개념들](#optimization의-중요한-개념들)
    - [경사 하강법](#경사-하강법)
    - [경사 하강 최적화](#경사-하강-최적화)
    - [Regularization](#regularization)
- [Convolution](#convolution)
    - [왜 Convolution(CNN)을 쓰는가?](#왜-convolutioncnn을-쓰는가)
    - [Convolution 연산](#convolution-연산)
    - [다차원에서의 convolution](#다차원에서의-convolution)
    - [역전파](#역전파)
- [Reference](#reference)

<br/>

## 최적화(Optimization) 기법
짧은 시간에 여러 가지를 다루다보니, 모든 기법들을 매우 얕게 다루었다.  
<strong>따라서 추후 각각을 깊게 공부할 필요가 있어보인다. </strong> 
한편 optimization 파트에서는 많은 용어가 등장하는데, 각각의 용어들이 의미하는 바를 잘 이해하고 넘어가야 한다는 점을 기억하자.   

  
기본적으로 우리는 Gradient Descent 기법을 알고있다. 이것은 한 번(first-order)의 미분 및 이를 통한 최적화를 여러 번 반복하는 알고리즘이다.  
이를 통해 우리는 미분 가능한 함수의 local minimum을 도출해낼 수 있었다.  
오늘 다룰 알고리즘도 사실상 Gradient Descent에서 출발한 알고리즘들로, 모두 기본적으로 gradient 값에 기반한 경사하강을 시도한다. 
다만 원래의 것보다 더 다양한 parameter을(들을) 활용한다는 차이가 있다.  
  
<br />

#### Optimization의 중요한 개념들
먼저 이 optimization에서 공통적으로 많이 쓰이는 용어들을 짚고 넘어가자.  
- Generalization
    ![generalization](/img/posts/12-1.png){: width="90%" height="90%"}{: .center}  
    + 학습데이터에 대한 성능과 테스트데이터에 대한 성능 간의 gap을 줄이는 것을 뜻한다.
    + 여기서 특정 시점에서의 그 성능차를 generalization gap이라 부른다.
    + 보다시피, 학습이 너무 많이 되면 generalization gap이 증가할 수도 있다.
    + <strong>generalization이 잘 되었어도 학습데이터에 대한 성능이 안좋으면 테스트데이터에 대한 성능 역시 별로일 것이다.</strong> 이에 대한 혼동이 없도록 주의하자.
  
- Underfitting/Overfitting
    ![underfit_overfit](/img/posts/12-2.png){: width="90%" height="90%"}{: .center}  
    + Underfitting은 학습 데이터에 대한 학습이 덜된 상태를 말한다.
    + Overfitting은 너무 학습 데이터에만 fit되는 학습이 이루어진 상태를 말한다.
    + Balanced Model은 둘 사이의 적정선을 유지하는 모델이다.
    + <strong>다만 이는 결과론적인 이야기임에 주의하자. </strong>  
      예를 들어, overfitting 그림이 실제 우리가 원하는 모델일 수도 있다. (실제 세계가 학습데이터처럼 구불구불할 수 있다.) 
  

- Cross-validation
    ![cross_validation](/img/posts/12-3.png){: width="90%" height="90%"}{: .center}  
    + KFold 기법과 같이, train data를 실제 train data와 validation data로 나누는 기법이다.
    + 위 그림처럼 5개로 나누면 5개 중 하나는 validation set, 나머지 넷은 training set으로 활용하여 총 5번의 교차검증을 수행하게 될 것이다. 
    + <strong>당연히 test data는 validation에도 사용하면 안된다</strong>

- Bias and Variance
    ![bias_and_variance](/img/posts/12-4.png){: width="60%" height="60%"}{: .center}  
    + variance, 즉 분산이 낮다는 것은 출력이 일관적이라는 것을 뜻한다.
    + bias, 즉 편향이 높으면 예측값 자체가(즉, 예측값의 평균이) 정답과 멀리 떨어져 있음을 의미한다. 
    + cost는 간단하게 보면 bias$^2$ + Bias + Noise이다. 학습 데이터에 Noise가 껴있다고 가정하면 <strong>bias와 variance는 서로 tradeoff 관계</strong>로 하나를 줄이면 하나가 늘어나게 되므로 둘다 줄이기는 매우 힘든 작업이다. 

- Bootstrapping(Bagging vs. Boosting)
    + bootstrap은 신발끈을 의미하는데, 이 기법은 학습데이터를 여러 번 뽑아 이 데이터로 여러 개의 모델을 만든다. 흔히 앙상블 기법이라고도 불린다.
    + 이 여러 개의 모델을 어떻게 활용하느냐에 따라 Bagging과 Boosting으로 갈리게 된다.
    ![bootstrapping](/img/posts/12-5.png){: width="90%" height="90%"}{: .center}  
    + Bagging
        - Bootstrappng aggregating의 약자로, 여러 개의 모델들의 평균(분류면 voting, 회귀면 평균 등)을 output으로 낸다.
        - 실제로 처음부터 모델 한 개로 학습하는 것보다 여러 모델들의 출력의 평균을 이용하는 것이 더 좋은 성능을 낼 때가 많다.
    + Boosting 
        - 여러 개의 weak learner를 이어 붙여 하나의 strong learner를 만드는 기법이다.
        - 한 모델을 거쳐 얻어온 결과에서 잘못된 결과의 가중치를 조정하여 다음 모델에 넘긴다.
        - 성능은 상대적으로 좋은 방법이지만 속도가 느리고 overfitting이 일어날 가능성이 있다.

<br/>

#### 경사 하강법
- Gradient Descent에는 이전에 다루었듯이 batch size 1인 stochastic gradient descent, batch가 적당히 큰 mini-batch gradient descent, 
  batch 없이 전체를 한번에 학습시키는 batch gradient descent 등이 있다.
- Batch size를 어떻게 정하느냐도 학습에서 중요한 요인이다. 이와 관련하여
  <span class="link_button">
  좋은 논문 [On Large-batch Training for Deep Learning: Generalization Gap and Sharp Minima, 2017](https://openreview.net/pdf?id=H1oyRlYgg)이 있으니 추후 보도록 하자.
  </span> batch size가 너무 크면 sharp minimzer에 모이게 되어 안 좋으므로 웬만하면 flat minimizer에 도달할 수 있는 small-batch를 사용해야한다는 점, 그리고 배치사이즈가 클 때의
  최적화는 어떻게 하면 좋은지 등의 문제에 대해 다루고 있다고 한다.  
- 일단은 small-batch가 train과 test 간의 갭을 줄여주어 generalization에 더 유리하다는 점을 이해하고 넘어가도록 하자.  
- 그런데 실제로는 배치 사이즈를 키우는 방향으로 학습을 시키는 경우가 많다. 그 이유에 대해 <span class="link_button">[이 글](https://bit.ly/3azzXva)</span>에 자세히
  설명되어있다. 간단히 요약해보자면, 논문에서 소개하는 방향은 우리가 생각하는 배치사이즈보다 훨씬 큰(512 이상) 배치 사이즈에 대하여 다루고 있기 때문에 우리가 실제 학습시킬 때 사용하는 배치사이즈는 아무리 키워봤자 메모리 제한 때문에 일정 수준을 넘길 수 없으므로 generalization 성능에 큰 영향을 미치지 않는다는 것이다.

<br/>

#### 경사 하강 최적화
- Momentum
    + 이전에 이동한 벡터량에 관성(Momentum)을 적용하여 다음 가중치 업데이트에 이를 이용하는 기법이다.
    <center>
    $$
    v_t = \gamma v_{t-1} + \eta \nabla _{\theta} J(\theta)
    $$
    $$
    \theta := \theta - v_t 
    $$
    </center>
    + $\gamma$는 관성계수인데 보통 0.9 정도의 값을 사용한다고 한다.
    + 이전 이동값($v_{t - 1}$)을 기억해야하기 때문에, 당연히 메모리가 2배로 필요하다. 

- Nesterov Accelerated Gradient(NAG)
    + NAG는 Momentum과 형태가 매우 유사한데, gradient를 구하는 지점이 $\theta$에서 $\theta - \gamma v_{t-1}$로 변했다는 차이가 있다.
    <center>
    $$
    v_t = \gamma v_{t-1} + \eta \nabla _{\theta} J(\theta-\gamma v_{t-1})
    $$
    $$
    \theta := \theta - v_t 
    $$
    </center>
    + $\theta - \gamma v_{t-1}$는 무엇을 의미할까? 우리는 $\theta$로 이동하기 위해 $\gamma v_{t - 1}$를 이용할 것이다. 
      즉, $\theta - \gamma v_{t-1}$는 <strong>parameter $\theta$가 다음에 이동할 위치</strong>에 근사하는 값을 미리 던져준다.
    + 따라서 NAG는 현재 위치에서의 gradient가 아니라, 다음에 이동할 위치에 대한 gradient를 이용하여 parameter의 다음 값을 구한다.
    + Momentum은 다음 위치와 관계 없이 관성이 붙기 때문에 멈춰야할 곳에서도 계속 전진할 수도 있다.
      그런데 NAG는 관성은 붙는데, 멈춰할 시점에서는 그 관성을 줄일 수 있다(제동 효과)는 장점이 있다.
    + 여전히 0.9 정도의 관성 계수($\gamma$)가 사용된다.
  
- Adagrad
    + 자주 업데이트 되는 parameter은 적게 변화시키고, 적게 업데이트 된 parameter은 많이 변화시키는 알고리즘이다.
    + 자주 업데이트된 변수는 optimum에 가까이 있을 것이라는 추정에 근거한다. (실제로 이게 엄밀하게 증명이 되었는지는 모르겠다)
    + 따라서 각 parameter마다 step size를 다르게 적용시켜줄 수 있다.
    + 그리고 이를 위해 지금까지 gradient가 얼마나 변했는지 square(제곱)하여 저장하는 $G_t$라는 변수를 새로 도입한다.
    <center>
    $$
    G_{t} = G_{t-1} + (\nabla_{\theta}J(\theta_t))^2
    $$
    $$
    \theta := \theta - \frac{\eta}{\sqrt{G_t + \epsilon}} \nabla_{\theta}J(\theta)
    $$
    </center>
    + 식을 보면 위에서 했던 말이 직관적으로 이해가 된다. step size에서 분모의 $\epsilon$은 0으로 나누어지는 것을 방지하기 위해 사용된다.
    + epsilon 값도 어떻게 설정하는지가 중요한 요소가 될 수도 있는데 일단은 $10^{-8}$정도를 쓴다고 한다. 
    + $G_t$는 제곱을 계속 더해주기 때문에 쉽게 커질 수 있다. 당연히 이게 너무 커지면 학습이 잘 안될 것이다(단점).
    + 참고로, 위에서 $(\nabla_{\theta}J(\theta_t))^2$항은 element-wise operation이다.

- RMSprop
    + RMSprop는 Adagrad에서의 $G_t$ 저하에 따른 학습률 저하를 보완한 방법이다.
    + $G_{t}$를 sqaure sum 대신 지수평균<span class="link_button">([Exponential moving average](https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average))</span>으로 사용한다. 
    <center>
    $$
    G_{t} = \gamma G_{t-1} + (1-\gamma)(\nabla_{\theta}J(\theta_t))^2
    $$
    $$
    \theta := \theta - \frac{\eta}{\sqrt{G_t + \epsilon}} \nabla_{\theta}J(\theta)
    $$
    </center>
    - 이 방법은 square sum이 아닌 average를 사용하기 때문에 $G_t$가 무한정 커지지 않으면서, Adagrad에서의 상대적 업데이트로 얻는 장점도 그대로 가져갈 수 있다.
    - Geoff Hinton이 자신의 강의에서 제시한 방법으로, 이 분 말에 따르면 $\gamma$를 0.9로, $\eta$를 0.001정도로 설정하는게 좋다고 한다.

- Adadelta
    + 이것도 Adagrad에서 $G_t$ 저하에 따른 학습률 저하를 보완한 방법이다.
    + $G_t$에 과거의 모든 gradient square를 더하는 대신, Adadelta에서는 '과거의 크기'를 정한다.
    + 예를 들어, 과거의 크기(window size)를 10으로 정하면, 이전 10개만 더하는 것이다.
    <center>
    $$
    G_{t} = \gamma G_{t-1} + (1-\gamma)(\nabla_{\theta}J(\theta_t))^2
    $$
    $$
    \Delta \theta _ t = \frac{\sqrt{H_{t - 1}+\epsilon}}{\sqrt{G_t + \epsilon}} \nabla_{\theta}J(\theta _t)
    $$
    $$
    \theta _{t + 1} = \theta _ {t} - \Delta \theta _t
    $$
    $$
    H_{t} = \gamma H_{t-1} + (1-\gamma)(\Delta \theta) ^2
    $$
    </center>
    + 앞선 설명까지만 보고 위 식을 보면 의아함이 든다. 그래서 과거의 그래디언트를 어디서 더하는걸까?
        - 사실 window size가 조금만 커져도 그만큼의 벡터를 다 메모리에 저장하고 있는 것은 불가능하다.
        - 그래서 $G_{t}$를 $t$ 번째 gradient까지의 제곱의 평균($=E[g^2]_t$)이라고 하고, 이에 근사할 수 있는 값을 대신 찾는다.
        - 앞서 RMSprop에서 적용했던 지수 평균을 여기서도 사용한다.
    + 추가적으로, learning rate($\eta$) 대신 $H$로 이루어진 식이 분자에 들어가는데, <strong>이 알고리즘은 learning rate가 없다.</strong>
    + 대신 parameter의 변화값(미분)의 제곱에 대한 지수 평균을 구해서 사용하는데 유닛(?)이 일치하지 않아 그렇다고 한다.
    > The authors note that the units in this update (as well as in SGD, Momentum, or Adagrad) do not match, i.e. the update should have the same hypothetical units as the parameter. To realize this, they first define another exponentially decaying average.
    + ... 여기서 말하는 'unit'이 뭔진 잘 모르겠는데 :anguished: 그냥 gradient가 아닌 parameter 변화량의 제곱을 이용한다는 점을 기억하자.
    + 보다시피 learning rate를 직접 설정할 수 없어 모델 설계시 직접 바꿀 수 있는 요소가 거의 없기 때문에 잘 안쓴다고 한다.

- Adam(Adaptive Moment Estimation)
    + Adam은 <strong>현재 가장 널리 사용하는 옵티마이저로</strong>, RMSProp과 Momentum 방식을 합친 형태의 알고리즘이다.
    + momentum과 gradient 변화량 제곱에 대한 지수 평균을 모두 이용한다.
    + 그래서 이전처럼 momentum은 분자에 들어가고, gradient제곱 변화량은 분모에 square되어 들어간다.
    <center>
    $$
    m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t 
    $$
    $$
    v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
    $$
    $$
    \hat{m}_t = \dfrac{m_t}{1 - \beta^t_1},\;\; \hat{v}_t = \dfrac{v_t}{1 - \beta^t_2}
    $$
    $$
    \theta_{t+1} = \theta_{t} - \dfrac{\eta}{\sqrt{\hat{v}_t + \epsilon}} \hat{m}_t
    $$
    </center>
    + 그런데 $m_t$와 $v_t$만으로 이루어져 있을 것 같았던 식에 난데없는 베타들이 붙고, $m$과 $t$에도 hat(expectation)이 붙었다.
    + 이것은 $m_t$와 $v_t$가 0으로 초기화되어있어 초반에 이들을 unbiased하게 만들어주기 위해 보정을 해주는 작업이라고 한다.
    + $m_t$와 $v_t$ 식을 시그마 식으로 나타내고 전개한 후 expectation을 씌워 정리하면 저렇게 된다. (자세한 과정은 생략)
    + 여담으로, 위 식을 정리하여 한 줄로 쓰면 아래와 같다. (정리하면 expectation이 들어가지 않는다.)
    <center>
    $$
    \theta_{t+1} = \theta_{t} - \dfrac{\eta}{\sqrt{v_{t} + \epsilon}} \frac{\sqrt{1 - \beta_{2} ^{t}}}{1 - \beta_{1} ^{t}} m_{t}
    $$
    </center>
    - $\beta _1$은 0.9, $\beta _2$는 0.999, $\epsilon$은 $10 ^{-8}$ 정도의 값을 사용한다고 한다.

<br />

#### Regularization
학습이 잘 된 모델에 대하여 generalization을 잘 하는 것도 학습에서 중요한 요소이다.  
학습 과정에서 여러 방해(regularization)를 통해 generalization을 할 수도 있는데 여기서는 이를 위한 방법들을 소개한다.

- Early Stopping
    ![early_stopping](/img/posts/12-6.png){: width="90%" height="90%"}{: .center}  
    + 맨 처음 그림과 비슷한데, 여기서는 KFold의 validation error이 test error 대신 들어갔다.
    + KFold 등의 방법으로 validation을 수행하면소 loss가 어느 순간 커지기 시작하면 중단하는 방법을 쓸 수 있다.
    + 그래서 이 방법을 쓰려면 당연히 validation data를 따로 둬야한다. 

- Parameter Norm Penalty
    + parameter 중 하나가 너무 커지지 않게 해주는 것이다. 즉 함수를 부드럽게(adds smoothness to the function space) 만들어주는 것이다.
    + 부드러운 함수일수록 generalization performance가 높을 것이라는 가정에서 비롯된 방법이다.
    + 이 가정이 검증된건진 모르겠는데, 실제로 함수가 한번씩 툭툭 튀면 그렇게 튀는 부분에서 예측이 틀릴 가능성이 있을 것 같다는게 직관적으로 이해가 되기는 한다.
    + 아래에서 $W$(즉, $\theta$)의 L2-Norm을 구하는데 이를 통해 parameter의 총합을 구할 수 있다.
    <center>
    $$
    \text{total cost} = \text{loss}(\mathscr{D};W) + \dfrac{\alpha}{2} \Vert W \Vert _2 ^2
    $$
    </center>
    + 이를 통해 weight의 크기에 제한을 줄 수 있기 때문에 이상 parameter를 규제할 수 있다.

- Data Augmentation
    + 데이터셋을 늘리는 기법이다. 데이터셋이 적으면 당연히 딥러닝의 성능은 여타 머신러닝 기법에 비해 떨어지게 된다.
    + 데이터가 실제로 많으면 좋겠지만, 그렇지 못하면 데이터에 조금의 변화를 주어 데이터를 늘릴 수 있다.
    + 예를 들어 이미지를 회전시킨다거나 뒤집는다거나 하는 방법이 있는데 당연히 MNIST 같이 label preserving이 안되는 경우는 하면 안된다.

- Noise Robustness
    ![nosie_robustness](/img/posts/12-7.png){: width="90%" height="90%"}{: .center}  
    + 위와 같이 이미지 벡터에 노이즈를 주면 학습이 더 잘 될 수 있다.
    + 그 이유에 대해 명확하게 증명되지는 않았으나, 성능은 좋다고 한다.

- Label Smoothing
    ![label_smoothing](/img/posts/12-8.png){: width="90%" height="90%"}{: .center}  
    + 학습 데이터 2개를 섞어주는 방법으로, Mixup과 같이 아예 반반 섞는 방법도 있고, CutMix나 Cutout처럼 잘라서 무식하게 붙이거나 아예 잘라서 없애버리는 방법 등이 있다.

- Dropout
    ![dropout](/img/posts/12-9.png){: width="90%" height="90%"}{: .center}   
    + 학습시 뉴런의 일부를 0으로 만들고 (즉, 네트워크에 존재하지 않는 것으로 가정) 학습한다.

- Batch Normalization
    + <span class="link_button">[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://proceedings.mlr.press/v37/ioffe15.html)</span> 논문에서 발표된 방법으로, parameter에 대하여 정규화(normalization)를 하는 기법이다.
    + 논문에서는 internal covariate shift(레이어 통과시마다 분산, 즉 분포가 변하는 현상)을 줄이는 데에 좋다고 나와있으나 이에 대한 논란이 많다고 한다. (무슨 논란인지는 당연히 아직 모른다 :open_mouth:)
    + 단순히 각 parameter에서 입력값을 정규화해버리고 넣는다는게 아니고, 학습을 할 때 각 신경망 안에서 정규화하는 과정이 포함된 채로 학습한다. 
    + 즉, 각 레이어마다 정규화하는 레이어를 두어 <strong>분포가 변형되는 것을 막는 것이 주요 목적이다.</strong>
    <center>
    $$
    \mu _B = \dfrac{1}{m} \sum\limits ^m _{i=1} x_i
    $$
    $$
    \sigma ^2 _B = \dfrac{1}{m} \sum\limits ^m _{i=1} (x_i - \mu _B)^2
    $$
    $$
    \hat{x} _i = \dfrac{x_i - \mu _B}{\sqrt{\sigma ^2 _B + \epsilon}}
    $$
    </center>

지금까지 여러 optimization, generalization을 위한 기법/알고리즘들을 알아보았는데 중요한 것은 당연히 어떤 상황이 주어졌을 때 이들중 적절한걸 골라 잘 활용하는 능력이다.  

<br/>

## Convolution 
CNN을 다루기에 앞서 Convolution 연산의 의미와 쓰임에 대해 알아보자.

<br />

#### 왜 Convolution(CNN)을 쓰는가?
우선 우리가 왜 convolution(합성곱)을 써야하는지부터 알아보자.   
이전까지 배웠던 다층신경망(MLP)은 fully-connected 구조로, 가중치가 $m \times n$의 행렬로 구성되어있었다.  
즉, 입력벡터 $\mathrm{x}$의 $i$번째 값을 업데이트하기 위해 $W_i$라는 긴 행이 필요했던 것이다.  
요지는, 이게 당연하다고 생각했었는데 <strong>사실은 메모리 측면에서 상당히 비효율적이다.</strong>  

  
층이 한두개가 아닐텐데 이 fully-connected 구조가 수십개만 쌓여도 MLP구조는 메모리를 많이 차지하게 될 것이다.  
그래서 이제부터 우리는 커널(kernel)을 기존의 행렬 대신 사용할 것이다. 커널은 필터(filter)로도 불린다.  
kernel은 크기가 훨씬 작은데도 $\mathrm{x}$의 모든 요소에 대하여 적용할 수 있으므로 parameter size를 훨씬 줄일 수 있다.  
   
<center>

$h_i=\sigma\left(\sum\limits_{j=1}^k V_j x_{i+j-1}\right)$ 

</center>

![kernel](/img/posts/12-10.png){: width="30%" height="30%"}{: .center}

모든 $i$에 대하여 적용되는 커널은 $V$로 같고, 커널의 사이즈만큼 $\mathrm{x}$상에서 이동하면서 적용한다.  
이것도 선형식으로 나타내어지는 연산이기 때문에 convolution 연산도 당연히 선형변환에 속한다.  

#### Convolution 연산
Convolution 연산의 수학적인 의미는 <strong>신호를 커널을 이용해 국소적으로 증폭 또는 감소시켜서 정보를 추출 또는 필터링하는 것</strong>이다.
여기서 '국소적'이라는 말은 특정 영역으로 해석하면, 결국 kernel 사이즈만큼의 데이터를 증폭 또는 감소시킨다고 해석할 수 있을 것 같다.  
식으로는 아래와 같이 나타낼 수 있다.
  
- 연속형(continuous)
<center>

$[f*g](x)=\int_{\mathbb{R}^d}f(z)g(x-z)dz=\int_{\mathbb{R}^d}f(x - z)g(z)dz=[g*f](x)$

</center>

- 이산형(discrete)
<center>

$[f*g](i)=\sum_{a\in \mathbb{Z}^d}f(a)g(i-a)= \sum_{a\in \mathbb{Z}^d}f(i-a)g(a)=[g*f](i)$

</center>
  

$x-z$혹은 $i-a$가 들어간 항이 신호, $z$혹은 $a$가 들어간 항이 커널이다. $x$($i$)값에 뭘 넣어주느냐에 따라 신호는 달라질 수 있지만 커널은 똑같다는 점이 아까 본 그림과 같다.  
다만, CNN에서는 위에서 쓴 $-$ 부호가 모두 $+$ 부호로 바뀌어서 들어가게 되는데, CNN에서 사용하는 연산은 사실 convolution이 아니고 cross-correleation이라 그렇다.  
  

이것에 대해 자세한 내용을 알 필요는 없는 것 같아 찾아보진 않았다. 아무튼 딥러닝에서 쓰이는 연산은 엄밀하게 말하면 $+$부호가 들어가는게 맞다는 것을 기억하자.

- 연속형(continuous)
<center>

$[f*g](x)=\int_{\mathbb{R}^d}f(z)g(x+z)dz=\int_{\mathbb{R}^d}f(x + z)g(z)dz=[g*f](x)$

</center>

- 이산형(discrete)
<center>

$[f*g](i)=\sum_{a\in \mathbb{Z}^d}f(a)g(i+a)= \sum_{a\in \mathbb{Z}^d}f(i+a)g(a)=[g*f](i)$

</center>

<br />

#### 다차원에서의 convolution
당장 이미지만 해도, 2차원처럼 보이지만 RGB값이나 투명도 채널까지 고려하려면 3차원 이상의 데이터로 변하게된다.  
다양한 데이터를 처리하려면 다차원에서의 convolution 방법도 알아야 한다.  
방법은 1차원에서의 그것과 다를 바 없다.  

- 1D-conv
<center>

$[f*g](i)=\sum\limits_{p=1}^d f(p)g(i+p)$

</center>

- 2D-conv
<center>

$[f*g](i,j)=\sum\limits_{p,q} f(p,q)g(i+p, j+q)$

</center>

- 3D-conv
<center>

$[f*g](i,j,k)=\sum\limits_{p,q,r} f(p,q,r)g(i+p,j+q,k+r)$

</center>

여전히 $i$, $j$, $k$ 값이 바뀌어도 커널 $f$의 값은 바뀌지 않는다는 점에 주목하자.  

![2d_conv](/img/posts/12-11.png){: width="50%" height="50%"}{: .center}  
위와 같은 커널에서 만들어지는 출력은 $2 \times 2$ 크기이고 (1, 1) entry 값은 19, (1, 2) entry 값은 25임을 쉽게 알 수 있다.
  
그래서 2D conv에서 입력의 크기를 $(H, W)$, 커널 크기를 $(K_H, K_W)$, 출력의 크기를 $(O_H, O_W)$라고 하면 아래와 같은 식을 쉽게 유도해낼 수 있다.

<center>

$$
O_H = H - K_H + 1
$$
$$
O_W = W - K_W + 1
$$

</center>

예를 들어 $28 \times 28$ 입력을 $3 \times 3$ 커널로 2D-Conv 연산을 하면 $26 \times 26$이 된다. 

  
3차원 convolution의 경우 2차원 convolution을 3번 적용한다고 생각하면 된다.  
참고로 3차원부터는 입력이나 커널이 2차원 행렬로 나타내어지지 않기 때문에 이들을 텐서라고 부른다.  
여기서는 모든 채널에 대하여 2차원에서의 convolution을 똑같이 적용하고 그것을 다 더해주면 된다.  
<strong>물론 커널의 채널 수와 입력의 채널 수가 같아야 한다.</strong>  

![3d_conv](/img/posts/12-12.png){: width="80%" height="80%"}{: .center} 

위와 같이 출력이 1차원으로 나오게되는데, 만약 출력 채널을 $O_C$개로 늘리고 싶으면 커널을 $O_C$개 사용하면 된다.  

<br />

#### 역전파
Convolution 연산도 커널(필터)의 parameter 갱신이 필요하다.

![conv_backpropagation](/img/posts/12-13.png){: width="40%" height="40%"}{: .center} 
위와 같은 상황에서 역전파를 시뮬레이션해보자. 추후 여기(1D-conv)서 사용한 방법을 2D, 3D에서도 확장하여 적용하면 된다.

우선 입력이 커널을 거쳐 출력값을 내뱉는기 때문에 아래와 같이 세 식을 바로 얻을 수 있다.

<center>

$$
w_1 x_1 + w_2 x_2 + w_3 x_3 = o_1
$$
$$
w_1 x_2 + w_2 x_3 + w_3 x_4 = o_2
$$
$$
w_1 x_3 + w_2 x_4 + w_3 x_5 = o_3
$$
</center>


참고로 위 식을 일반화하면, $o\_i = \sum\limits \_j w\_j  x\_{i+j-1}$로도 나타낼 수 있는데 나한텐 별로 직관적이지 않아서 위와 같이 하나하나 나타냈다.
  
  
우리는 커널의 각 parameter인 $w_k$를 갱신해야한다. 따라서 손실함수 $\mathcal{L}$에 대한 각 parameter의 편미분 $\dfrac{\partial \mathcal{L}}{\partial w _1}$, $\dfrac{\partial \mathcal{L}}{\partial w _2}$, $\dfrac{\partial \mathcal{L}}{\partial w _3}$를 구하는 것이 목표이다.

이를 위해 아래와 같이 chain rule을 사용할 것이다.

<center>

$$
\dfrac{\partial \mathcal{L}}{\partial w _i} = \dfrac{\partial \mathcal{L}}{\partial o_j} \dfrac{\partial o_j}{\partial w_i}
$$

</center>

loss function으로 무엇을 사용할지는 미정이지만, 결국 loss function은 output에 대한 식으로 나타내질것이므로 $\dfrac{\partial \mathcal{L}}{\partial o_j}$은 쉽게 구할 수 있다.  
이렇게 구한 $\dfrac{\partial \mathcal{L}}{\partial o_1} = \delta _1$, $\dfrac{\partial \mathcal{L}}{\partial o_2} = \delta _2$ $\dfrac{\partial \mathcal{L}}{\partial o_3} = \delta _3$라 하자.   


예를 들어 우리가 $\dfrac{\partial \mathcal{L}}{\partial w _3}$을 구하고자 한다면, 아까 구한 $o_j$와 $w_i$ 사이의 관계식으로부터,

<center>

$$
\dfrac{\partial o_1}{\partial w _1} = x_3
$$
$$
\dfrac{\partial o_1}{\partial w _2} = x_2
$$
$$
\dfrac{\partial o_1}{\partial w _3} = x_1
$$


</center>

이므로 $\dfrac{\partial \mathcal{L}}{\partial w _3} = \delta _1 w_3 + \delta _2 w_2 + \delta _3 w_1$을 쉽게 유도해낼 수 있다.  
다른 파라미터들도 같은 방법으로 쉽게 유도할 수 있으며, 이를 아래와 같이 일반화 할 수 있다.

<center>

$$
\dfrac{\partial \mathcal{L}}{\partial w _i} = \sum\limits _j \delta _j x_{i + j - 1}
$$

</center>

<br />
  
## Reference  
[부트스트랩](https://bit.ly/3oCLMV3)   
[Batch Size in Deep Learning](https://blog.lunit.io/2018/08/03/batch-size-in-deep-learning/)  
[SGD에서 배치 사이즈가 학습과 성능에 미치는 영향](https://bit.ly/3azzXva)  
[Optimizer](https://ruder.io/optimizing-gradient-descent/index.html#rmsprop)  
[Optimizer2](http://shuuki4.github.io/deep%20learning/2016/05/20/Gradient-Descent-Algorithm-Overview.html)  
[Adam Optimizer](https://hiddenbeginner.github.io/deeplearning/2019/09/22/optimization_algorithms_in_deep_learning.html#Adam)  
[Batch Normalization](https://eehoeskrap.tistory.com/430)   
