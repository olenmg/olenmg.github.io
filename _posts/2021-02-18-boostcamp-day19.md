---
layout: post
title: "Day19. Transformer"
subtitle: "Transformer with Multi-Head Attention"
date: 2021-02-18 14:16:12+0900
background: '/img/posts/bg-posts.png'
use_math: true
---

## 개요 <!-- omit in toc -->
> 이전에 다루었던 트랜스포머 구조에 대해 복습하고 일부는 더 자세히 알아보는 시간을 가졌다.  
  
아래와 같은 순서로 작성하였다.  
- [Transformer: Scaled Dot-Product Attention](#transformer-scaled-dot-product-attention)
    - [Bi-Directional RNN](#bi-directional-rnn)
    - [Scaled Dot-Product Attention](#scaled-dot-product-attention)
- [Transformer: Multi-Head Attention](#transformer-multi-head-attention)
    - [Multi-Head Attention](#multi-head-attention)
    - [Complexity](#complexity)
- [Transformer: Other](#transformer-other)
    - [Layer Normalization](#layer-normalization)
    - [Positional Encoding](#positional-encoding)
    - [Warm-up Learning Rate Scheduler](#warm-up-learning-rate-scheduler)
    - [Decoder](#decoder)
    - [Performance](#performance)
- [Hyperparameter Tuning](#hyperparameter-tuning)
    - [Learning rate](#learning-rate)
    - [그 외 hyperparameter tuning TIP](#그-외-hyperparameter-tuning-tip)
- [Reference](#reference)


<br/>

<span class="link_button">[이전 포스트](/2021/02/04/boostcamp-day14.html)</span>에서 자세히 설명되지 않은 내용을 위주로 작성하였다.  

<br/>

## Transformer: Scaled Dot-Product Attention
Transformer 구조가 처음 제시된 논문의 제목은 'Attention is all you need(NeurlPs'17)'으로, 말그대로 이제는 **기존 RNN 구조 없이**
attention 모듈만으로도 같은 역할을 하는 모델을 만들겠다는 의미이다.  
  
#### Bi-Directional RNN
Transformer에 대해 알아보기 전에, RNN 구조를 응용한 **Bi-Directional RNN**에 대해 간단히 알아보자.   
  
복잡한건 없고, Bi-Directional RNN은 말그대로 양방향 RNN 구조를 의미한다.
기존 RNN 구조는 어떤 timestep에서 output을 낼 때 **뒤에 나오는 단어만을 고려할 수 있을 뿐, 앞에 오는 단어에 대한 정보는 고려하지 못한다는 문제가 있다.**
이를 해결하기 위해 양방향으로 RNN 구조를 두고 각각에서 나온 hidden state vector를 concat하여 이를 통해 최종 output을 만든다.  
  
![bi-directional-rnn](/img/posts/19-1.png){: width="80%" height="80%"}{: .center}   
위와 같이 forward, backward layer를 두고 각각에서 RNN 모델을 돌린다.
PyTorch에서는 해당 parameter만 주면 쉽게 구현 가능하다.

<br />

#### Scaled Dot-Product Attention
Bi-Directional RNN에서도 여전히 멀리 떨어진 단어에 대한 정보가 hidden state vector에서 vanishing되는 문제가 존재한다.  
 
Transformer 구조는 이전에도 보았듯이 sequence의 길이 및 방향에 상관 없이 attention 구조를 기반으로 돌아가기 때문에 위와 같은 문제가 전혀 발생하지 않는다.  
  
![transformer](/img/posts/19-2.png){: width="60%" height="60%"}{: .center}  
기준이 되는 input이 이전 seq2seq 모델의 decoder 내 hidden state vector의 역할(key vector)을 하며,
그 기준을 포함한 나머지 input들이 seq2seq 모델의 encoder 내 hidden state vector의 역할(query vector)을 한다.
  
즉, 따로 주어지는 벡터 없이 input만으로 스스로 attention을 수행해내기 때문에 self-attention이라고 불린다. 
seq2seq 구조에서처럼 $h\_k$는 $x\_k$가 기준이 되어 수행한 attention 작업의 결과(가중평균)가 된다.  
  
그런데 보통은 동일 벡터간의 내적이 가장 큰 값이므로 attention score를 구할 때 가중치가 기준 벡터에게 제일 많이 들어갈 우려가 있다. 
얼핏 생각하면 embedding vector를 변형 없이 query, key, value로 그대로 사용해도 되겠지만, 위에서 말한 문제를 해결하기 위해 query, key, value를 모두 새로 생성하여 이를 사용한다.
query, key, value 벡터는 모두 embedding vector에 대하여 서로 다른 linear transformation을 해주어 생성하게 된다.  
  
추가적으로 이와 같은 연산을 수행하려면 key와 value는 가중평균 계산시에 함께 사용되므로 서로 갯수가 같아야하며, query와 key는 내적을 수행해야하므로 서로 dimension이 일치해야한다. 
따라서 $d\_k = d\_q \neq d\_v$ 여도 관계는 없으나, 보통은 모두 같은 차원으로 맞춰준다고 한다. 달라도 되지만, 굳이 다를 필요는 없기 때문인 것 같다.  
  
위 과정은 모두 식 하나로 표현 가능하다.

<center>

$$
A(q, K, V) = \sum _i \frac{\exp \left(q \cdot k_i \right)}{\sum _j \exp \left(q \cdot k_i \right)} v_i
$$

</center>

$q$ 벡터 역시 column 방향으로 concat하여 연산이 가능하므로 이를 행렬 $Q$로 나타내면 식은 아래와 같다.

<center>

$$
A(Q, K, V) = \text{softmax}\left(QK^\intercal\right) V
$$

</center>

![transformer_calc](/img/posts/19-3.png){: width="80%" height="80%"}{: .center}  
행렬의 크기는 위와 같게 되는데, **softmax는 당연히 row-wise로 수행한다.**
query의 dimension은 key의 dimension과 반드시 같아야 하므로 편의를 위해 모두 $d\_k$로 표기되어있다.  
  
그런데 실제로는 아래와 같이 softmax를 취하기 전 가중평균이 된 벡터에 $\sqrt{d\_k}$를 나눠준다.  

<center>

$$
Z = \text{softmax}\left( \frac{QK^\intercal}{\sqrt{d_k}} \right) V
$$

</center>
  
이는 **normalize를 위한 작업으로, $d\_k$가 커짐에 따라 $q^\intercal k$의 분산이 커지기 때문에 사용한다.**
우리는 softmax를 취하기 전에 내적을 통해 scoring을 먼저 수행하는데, 벡터의 크기가 커질수록 결과 값의 분산이 커진다.  
  
예를 들어 2차원 벡터 $\[a, b\]$와 $\[x, y\]$ 간 내적을 수행하면 $ax + by$가 되는데 여기서 각 항을 $N(0, 1)$의 분포를 만족하는 확률변수라고 생각하면
두 확률변수 $X$, $Y$가 독립일 때 $V(X + Y) = V(X) + V(Y)$가 성립하므로 $ax + by$는 $N(0, 2)$를 만족하게 된다. 
이에 따라 $n$차원의 벡터간 내적 결과는 $N(0, n)$의 분포를 만족하게 된다.
실제로 직관적으로 생각해보면 sequence 길이가 3이고 $d\_k$가 2차원일 때 scoring(key, query 벡터간 내적)이 $\[1.1, -0.8, -1.7\]$ 정도의 값으로 기대될 때
동일 sequence길이(3)에서 $d\_k$가 100이라면 key, query 벡터간 내적은 $\[8, -11, 7\]$과 같이 그 편차가 매우 커질 것이라고 예상할 수 있다.
(구체적 수치 말고 그 편차에 집중해보자)  

그런데 이렇게 되면 softmax에 들어가는 벡터에서 한 entry가 유별나게 클 수 있고 이렇게 되면 softmax의 결과값이 그 큰 값에 쏠리는(very peaked) 문제가 발생한다. 
이에 따라 gradient가 vanishing되어 학습이 안될 가능성이 생긴다.  
  
따라서 우리는 내적의 결과에 표준편차로 기대되는 값 $\sqrt{d\_k}$를 나눠주어 normalize를 해준 후 softmax를 통과시킨다.

<br />

## Transformer: Multi-Head Attention
실제 Transformer 구조는 여러 head를 두고 각각에서 Q, K, V를 각각 따로 만들어 이를 전부 활용하는 방식으로 돌아간다. 
만약 head 1개로 돌아가게 된다면 문장을 여러 측면에서 고려하지 못하게 되는 문제가 발생한다. 
예를 들어 문장에는 누가, 언제, 어디서 등의 정보가 존재할텐데 multi head attention을 두면 각 attention이 누가, 언제, 어디서를 각각 집중적으로 담당할 수 있게 될 것이다.  
  
<br/>
  
#### Multi-Head Attention
![multi-head-attention](/img/posts/19-4.png){: width="50%" height="50%"}{: .center}  
구조 자체는 동일한데, 각 attention에서 생성된 output vector를 concat하여 마지막으로 linear layer를 통과시켜 원하는 차원으로 맞춰주는 방식으로 돌아가게 된다.

<center>

$$ 
\text{MultiHead} \left( Q, K, V \right) = \text{Concat} \left( \text{head}_1, \cdots, \text{head}_h \right) W ^O
$$
$$
\text{where} \;\; \text{head}_i = \text{Attention} \left( QW_i ^Q, KW_i ^K, VW_i ^V \right)
$$

</center>

위와 같이 여러 head 각각이 Q, K, V를 만들기 위한 행렬쌍을 독립적으로 가지고 있게 된다.  
  
![multi-head-attention2](/img/posts/19-5.png){: width="100%" height="100%"}{: .center}  
이후 concat된 벡터를 linear layer에 통과시켜 원하는 차원으로 맞춰준다.
여기서는 총 8개의 벡터니까 3 * 8 = 24, 즉 (3 x 24) 행렬을 linear layer에 통과시킬 것이고
만약 원논문에서처럼 Add&Norm 구조까지 활용하려면 output이 single head와 같아야하므로 linear layer의 크기가 24 x 3이 될 것이다. 
(concat되기 전과 size 일치) 

<br />

Multi-Head Attention은 앞서 말한 것과 같이 다양한 context를 동시에 고려할 수 있다는 장점이 있다.  
  
똑같은 word를 받더라도 서로 다른 head의 attention에서는 이 word에 대한 서로 다른 output을 내놓는다. 
또한 같은 word에 대하여 각 head가 주는 가중치의 크기가 제각기 다르다.

<br />

#### Complexity
![complexity](/img/posts/19-6.png){: width="100%" height="100%"}{: .center}  
$n$은 squence length, $d$는 hidden state vector나 key(query) vector 즉 주요 연산 대상인 벡터의 차원을 의미한다.  
  
우선 계산 복잡도를 측정하기 위해 주요 연산이 무엇인지를 살펴보도록 하자.  
  
self-attention의 주요 연산은 k, q의 내적으로 각각이 $n$개만큼 있으며 $d$개 entry에 대한 곱셈(내적)을 수행해야하므로 이에 소모되는 시간 복잡도는 레이어당 $O(n^2 \cdot d)$이다.
반면 RNN의 경우 hidden state vector가 매 timestep마다 $W\_hh$와 곱해지는 것이 주요 연산이고, $W\_hh$는 d x d 행렬이므로 계산량이 $O(n \cdot d^2)$이 된다.  
  
이렇게 되면 공간적 측면에서도 연산량에 비례하는 complexity를 보이게 되는데, backpropagation을 수행하려면 연산된 모든 값들을 저장하고 있어야 하기 때문이다. 
self-attention의 경우 sequence 길이가 늘어남에따라 저장해야하는 정보량이 $n^2$으로 늘어나게 되므로 문장의 길이가 길면 길수록 메모리가 매우 많이 필요하다.
반면 RNN은 $n$에 비례한 공간이 필요하므로 메모리 측면에서 RNN이 상대적으로 유리하다고 볼 수 있다.  
  
다만 RNN은 매 timestep마다 이전의 hidden state vector를 활용하므로 병렬연산이 불가하여 sequential operation(GPU가 충분히 많다고 가정)에 시간복잡도 $O(n)$이 필요하지만,
self-attention 모델은 input이 한번에 들어가며 이전 연산에 의존성이 없으므로 모두 한번에 병렬처리가 가능하여 $O(1)$에 연산이 가능하다.   
  
마지막으로 maximum path length는 long-term dependency와 관련이 되는데, $n$번째 정보는 첫번째 정보를 참조하기 위해 self-attention은 $O(1)$, 
RNN은 앞서 sequential operation에서와 같은 이유로 $O(n)$의 연산이 필요하다.  

<br />

## Transformer: Other
이전에 보았듯이 트랜스포머 구조는 아래와 같이, self-attention 구조가 여러개 쌓여있는 형태를 띤다. (Block-based Model)  
  
![transformer](/img/posts/14-16.jpg){: width="70%" height="70%"}{: .center}    
Multi-Head attention과 함께 two-layer로 이루어진 feed-forward NN(with ReLU)를 사용한다. 
이 구조에서 확인되는 Add & Norm 단계, Positional Encoding 단계에 대해 알아보도록 하자.  
  
<br />

#### Layer Normalization
Add & Norm 단계에서는 이전에 보았듯이 Residual connection 기법을 사용하여 gradient vanishing을 해결하고 학습을 안정화시킬 수 있다.
앞서 말했듯 이걸 쓰려면 Multi-Head Attention(이하 MHA)에서는 **최종 output이 input과 size가 같도록 마지막 linear layer를 구성해주어야 한다.**    
  
여기서 Layer Normalization도 추가적으로 행해지는데 이는 아래와 같은 두 단계로 이루어진다.  
![layer_normalization](/img/posts/19-7.png){: width="90%" height="90%"}{: .center}    
1. 각 word vector 하나하나가 $N(0, 1)$을 만족하도록 vector 내 entry들을 정규화해준다.  
2. **각 sequence에 대하여** 학습이 가능한 affine transformation(i.e. $y=ax+b$)을 행해준다. (normalization 대상과 affine transformation 대상이 다르다는 점에 주의한다)
   
여기서 학습해야할 parameter가 2개(affine transformation의 weight, bias) 늘어나게 된다.  
  
비슷한 유형인 batch normalization에 대해서도 한 번 짚고 가자.   
  
형태는 매우 비슷하다. 어떤 레이어를 통과하고 나온 output에 대하여 여러 batch들에서 같은 feature 값에 대해 각 feature가 $N(0, 1)$을 만족하도록 정규화를 해주고
이후 scale factor, shift factor를 이용하여 새로운 값을 (layer에서의 affine transformation과 비슷한 역할) 만든다.  
  
    
이렇게만 써두면 둘이 차이점을 알기가 어려운데, **layer norm은 같은 sample 내에서 서로 다른 feature들간 정규화를 해주는 것이고,**
**batch norm은 여러 sample에서 같은 feature들간의 정규화를 해주는 것이다.**    
  

![batch_layer_norm](/img/posts/19-8.png){: width="70%" height="70%"}{: .center}    
위 그림을 보면 이를 이해할 수 있다. reference에 쓰여있는 블로그에서 발췌하였으며 이에 대한 자세한 내용 역시 reference를 참조하자.  
  
보통 activation이 포함된 NN의 경우 activation을 통과하기 전에 normalization을 먼저 수행한다.   
  
아무튼 요약하면, transformer에서는 먼저 Residual connection을 해준 후 이 값에 대하여 LayerNorm을 수행한다. 
간단하게 식으로 표현하면 아래와 같다.

<center>

$$
LayerNorm(x + sublayer(x))
$$

</center>

<br />

#### Positional Encoding
Positional encoding은 이전 포스트에서도 썼지만, 이것에 대해서만 다룬 논문이 따로 나올 정도로 이걸 자세히 다루기는 벅차다.
따라서 간단하게만 알아보도록 하자.  
  
우선 attention 자체만으로는 **위치에 대한 정보를 담을 수 없기 때문에 이러한 인코딩 작업을 추가적으로 수행한다.**

<center>

$$
PE_{(pos, 2i)} = \sin(pos/10000^{2i/d_{model}})
​$$
$$
PE_{(pos, 2i + 1)} = \cos(pos/10000^{2i/d_{model}})
$$

</center>

식은 솔직히 완벽히 이해할 수 없는데, **sin, cos이라는 주기 함수를 사용한다는 내용을 담기 위해 위와 같이 적어보았다.**
주기 함수를 사용하여 순서를 특정지을 수 있는 unique한 vector를 생성하고 이를 각 position을 담당하는 input vector에 더해주는 것이다.

![positional_encoding](/img/posts/19-9.png){: width="100%" height="100%"}{: .center}    
좌측 그림은 dimension(4~7) 별로 담당하는 주기함수를 그린 것이다.
우측 그림은 128차원의 positional encoding vector인데 세로축(포지션) 기준 더해지는 vector가 달라지는 것을 확인할 수 있다.

<br />

#### Warm-up Learning Rate Scheduler  
![learning_rate_scheduler](/img/posts/19-10.png){: width="80%" height="80%"}{: .center}    
위와 같은 warm-up learning rate scheduler를 사용하기도 한다.  
1. 초반에는 최적 위치와 동떨어진, 즉 gradient가 매우 큰 위치에 있을 것으로 예상되므로 learning rate를 작게 가져간다.
2. 이후 어느정도 평탄한 위치에 도달할 때까지 점점 줄어드는 gradient를 보정해주기 위해 learning rate를 끌어올리며 학습한다.
3. 어느정도 평탄한 위치에 도달했으면 그 주변이 최적 위치일 것이므로 learning rate를 점점 감소시켜준다.

<center>

$$
learning\; rate\; = \;d ^{-0.5} _{model} \cdot \min(\# step ^{-0.5}, \#step \cdot warmup\_steps ^{-1.5})
$$

</center>

<br />

#### Decoder
Decoder는 encoder와 완전히 동일한 self-attention 구조로 이루어져있다. 
다만 Encoder에서 최종 출력으로 얻은 각 encoding vector에서 key/value vector를 뽑아내고 이를 decoder의 input에서 뽑아온 query vector와 함께 활용한다는 점만 다르다.  
  
**또한 여기서는 Masked Self-Attention이라는 구조를 추가적으로 활용한다.**  
  
**학습단계에서는 batch processing을 위해 전체 sequence를 decoder에 동시에 input으로 주지만**, 아래와 같이 뒷 단어에 대한 softmax 결과를 0 등을 활용하여 가려주어야(mask)한다.  
![masked_self_attention](/img/posts/19-11.png){: width="100%" height="100%"}{: .center}    
이후 각 row에 대하여 확률값을 다시 normalize한다. 예를 들어 첫번째 행은 \[1, 0, 0\]이 될 것이고 두번째 행은 약 \[0.47, 0.53, 0\] 정도의 값이 될 것이다.  
  
self-attention의 장점이 모든 시점을 vanishing 없이 고려할 수 있다는 점이었지만, 실제 추론(inference)에서는 미래 시점까지 고려할 수는 없으므로 이와 같이 decoder 단에서는 과거 시점의 정보만 고려한다고 볼 수 있다.  

<br />

#### Performance
Transformer 구조는 현재 거의 모든 NLP 모델의 기반이 될 정도로 성능이 좋으면서도 계산량이 그리 많지 않다.  
  
(2017년 기준)논문에도 나와있지만 실제로 BLEU scoring시 다른 모델들보다 높은 score를 보여주며, 연산량 역시 상위권(즉, 연산량이 적다는 뜻)에 속한다.

<br />

## Hyperparameter Tuning
오늘 과제 해설 시간에는 hyperparameter tuning에 대해 다루었다.  

<br />

#### Learning rate
learning rate는 정말 중요한 parameter 중 하나이다. 이를 튜닝하기 위해 아래와 같은 방법을 써볼 수 있다.  
1. learning rate scheduler를 끄고 early stage에서 어느정도 loss가 내려가는 learning rate를 찾는다.
2. learning rate scheduler를 켜고 early stage에서 어느정도 loss가 내려가는 learning rate를 찾는다. 배치 사이즈는 최대한 크게 잡는다.
3. 먼저 train set으로만 loss가 어떻게 변화하는지 판단해본다. 잘 움직이지 않는다면 initialization의 문제를 의심해보자. loss가 너무 안떨어지면 learning rate를 줄여보자.
4. 이후 어느정도 train set에 tuning이 되었으면, validation set을 함께 활용하여 식별되는 overfitting/underfitting 등을 regularization을 통해 해결한다.  

<br />

#### 그 외 hyperparameter tuning TIP
- 대개 중요한 hyper parameter는 learning rate, batch size 정도가 있다.
- hyperparameter tuning시 grid search보다는 random search를 하는 것이 보통 낫다.
- 전체적으로 random seed를 잘 두는 것도 중요한데, random seed는 고정해둘 수 없는 경우가 많다는 점을 주의하자. (하드웨어, OS 등의 gap으로 인해)
- 리소스가 얼마 없을 경우 cross-validation보다는 그냥 validation set 하나를 두고 그걸로 validation을 하는 것이 차라리 낫다.
- 배치 사이즈는 최대한 크게 놓는 것이 좋지만, 리소스가 부족하면 accumulation(작은 사이즈로 나누어 gradient를 구한 후 이에 대한 평균으로 업데이트) 기법을 사용하기도 한다.
  하지만 이 경우 같은 사이즈의 batch / accumulation을 사용하더라도 그냥 batch 자체로 두는 것이 성능이 더 낫다.
- 앙상블 기법은 최종적으로 예측 성능을 2~3% 정도 올릴 가능성이 보장되어있다.
  따라서 모델 성능을 어느정도 한계까지 올렸다고 생각되면 그때는 앙상블 기법을 적용해볼 수 있다.
  앙상블 적용시 random seed를 다르게 한다던지 등의 생각을 해볼 수 있다.
- 모델은 최대한 유명한 라이브러리/많이 사용된 라이브러리를 최대한 사용하고, 너무 어렵게 코딩되어있는 기법,(스스로 뜯어볼 수 없는) 너무 최신식인 기법은 기피하도록 하자.
- 어찌됐든 가장 좋은 것은 남들이 이미 해놓은 것을 참고하는 것이다.


<br />

## Reference  
[Batch Norm vs Layer Norm](https://yonghyuc.wordpress.com/2020/03/04/batch-norm-vs-layer-norm/)  
[The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)  