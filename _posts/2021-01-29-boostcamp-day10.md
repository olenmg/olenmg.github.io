---
layout: post
title: "Day10. 시각화 / 통계론"
subtitle: "mathplotlib, seaborn, 통계론 기초"
date: 2021-01-29 22:32:12+0900
background: '/img/posts/bg-posts.png'
use_math: true
---

## 개요 <!-- omit in toc -->
> 데이터의 분포가 어떻게 되는지 가장 직관적으로 파악하려면 시각화는 필수다. 오늘은 이러한 시각화를 위한 라이브러리들과 간단한 사용법을 살펴보았다. 
그리고 어제 배운 확률론에 이어 통계론의 기초적인 부분들을 배웠다.  
  
오늘 배운 내용은 아래와 같다.
- [데이터 시각화](#데이터-시각화)
    - [mathplotlib](#mathplotlib)
    - [seaborn](#seaborn)
- [통계론](#통계론)
    - [모수란?](#모수란)
    - [확률분포의 가정](#확률분포의-가정)
    - [중심극한정리](#중심극한정리)
    - [최대가능도 추정법](#최대가능도-추정법)
    - [딥러닝에서 최대가능도 추정법](#딥러닝에서-최대가능도-추정법)
    - [확률분포의 거리](#확률분포의-거리)
    - [쿨백-라이블러 발산](#쿨백-라이블러-발산)
- [그 외](#그-외)
- [Reference](#reference)
  

<br />

## 데이터 시각화
시각화는 다른 말로 그래프를 그리는 것과 같다. 물론 그래프의 종류도 여러가지가 있다. 오늘은 그래프를 그리기 위한 라이브러리인 mathplotlib와 seaborn에 대해 배웠다.  

#### mathplotlib
- seaborn도 그렇지만, 우리가 다루는 이러한 시각화 라이브러리는 pandas의 객체를 지원하며, 그렇기 때문에 쉽게 사용할 수 있다.
- mathplotlib을 사용하는 기본적인 방법은 아래와 같다.  
  ```python
  #mathplotlib.py
  import matplotlib.pyplot as plt

  X = range(100)
  Y = [value**2 for value in X]
  plt.plot(X, Y)
  plt.show()
  ```
- 이미지는 따로 첨부하지 않지만, $0 \leq x < 100$에서의 $x^{2}$ 그래프가 그려질 것이다. 
- 보다시피 mathplotlib의 <code>plot</code> 함수로 그래프를 그리게되는데, argument를 kwargs로 받는다는 단점이 있다.
- 따라서 argument로 무엇이 들어가야하는지 매번 찾아봐야한다는 단점이 존재한다.
- 그리고 $x$값이 모두 정렬된 상태로 들어가야한다. (당연히 대응하는 $y$값도 같은 순서로 들어가야한다.)
- 또한 맨 아래 <code>plt.show()</code>로 그래프를 출력하는데, 출력까지의 과정을 설명하면 아래와 같다.
  1. <code>pyplot</code> 객체에서 <code>plot()</code> 함수를 통해 생성한 그래프는 모두 'figure' 객체에 들어간다.
  2. figure 객체의 내용이 flush가 되기 전까지 <code>plot()</code> 함수가 여러번 호출되면 figure객체에 그래프가 그러진다.
  3. 사실 그래프 출력과 함께 <code>show()</code> 함수는 flush의 기능도 한다. <code>show()</code>를 호출하면 figure 객체가 비워지면서 그래프가 출력된다.

- <code>plot()</code>함수는 커다란 하나의 그래프를 그린다. 대신 <code>add_subplot()</code> 함수를 사용하면 여러 함수를 하나의 큰 도화지에 그릴 수 있다.  
  ```python
  #add_subplot.py
  import matplotlib.pyplot as plt

  fig = plt.figure()
  fig.set_size_inches(10, 5) # 크기 지정
  ax_1 = fig.add_subplot(1, 2, 1) # 1x2 그리드의 1번 위치
  ax_2 = fig.add_subplot(1, 2, 2) # 1x2 그리드의 2번 위치

  ax_1.plot(X_1, Y_1, c="b") #c는 color를 의미. blue
  ax_2.plot(X_2, Y_2, c="g") #green
  plt.show() #flush
  ```
- 위와 같이 <code>add_subplot(grid R, grid C, 순서)</code>의 형태로 사용한다. R과 C는 각각 열수, 행수를 의미한다.  
- 그 외 <code>ls</code> 혹은 <code>linestyle</code> 속성을 사용하여 선의 스타일을 지정할 수도 있다.
- <code>title</code> 속성을 수정하면 figure의 title을 subplot별로 입력할 수 있다. 여기에 LaTex 타입의 표현도 가능하다.
- <code>legend()</code> 함수로 범례를 표시할 수 있다. 여기에 <code>loc</code> 속성으로 범례의 위치를 조정할 수도 있다.
- 그래프에서 뒷배경에 grid 보조선을 긋는 것도 가능하다. <codE>grid()</code> 함수를 이용하면 된다.
- <code>plot()</code>은 가장 일반적인 선그래프를 표현한다. 그 외에도 다양한 종류의 그래프를 표현할 수 있다.
  + <code>scatter()</code>를 통해 선점도를 그릴 수 있다.
  + <code>bar()</code>를 통해 막대그래프를 그릴 수 있다.
  + <code>hist()</code>를 통해 히스토그램을 그릴 수 있다.
  + <code>boxplot()</code>을 통해 박스플롯을 그릴 수 있다. 박스플롯은 약간 주식 차트모양 연상하면 된다.

<br />

#### seaborn
- 기존 mathplotlib를 좀 더 개선한 라이브러리라고 보면 된다.
- 복잡한 그래프를 간단하게 만들 수 있는 wrapper 역할을 한다. 
- mathplotlib에서 <code>plot()</code>으로 그렸던 기본적인 plot은 아래와 같이 그릴 수 있다.  
  ```python
  #seaborn.py
  import seaborn as sns

  fmri = sns.load_dataset("fmri")
  sns.set_style("whitegrid") #style도 지정 가능
  sns.lineplot(x="timepoint", y="signal", data=fmri)
  ```
- 코드에서 보이듯이 seaborn은 라이브러리 자체적으로 몇 개의 연습용 데이터셋을 제공한다. <code>fmri</code>도 그 중 하나이다.
- style을 위와 같이 <code>set_style()</code> 함수로 쉽게 적용할 수 있다. 스타일의 종류는 구글링을 통해 알아보도록 하자.
- 기본적인 선그래프는 <code>lineplot()</code> 함수로 그릴 수 있다. 가지고있는 데이터에서 $x$와 $y$ 값을 무엇으로할지 직관적으로 지정할 수 있다는 장점이 있다.
- mathplotlib.pyplot의 <code>plot()</code>과 달리 argument가 복잡하지 않으므로 docstring을 보고 쉽게 파악할 수 있다.
- <code>lineplot()</code> 함수에서 <code>hue</code> 옵션을 주면, 지정한 column의 각 값들로 카테고리화된 그래프를 얻을 수 있다.
- 만약 <code>lineplot(x="A", y="B", hue="C", data=fmri)</code>라고 하면 <code>fmri["C"].unique()</code> 안에 있는 값들을 범례로 하는 x축 A, y축 B에 대한 그래프를 그릴 수 있다는 것이다.
- seaborn도 <code>scatterplot()</code>, <code>countplot()</code>, <code>barplot()</code>, <code>distplot()</code> 등으로 다양한 형태의 그래프를 그릴 수 있다.
- <code>violinplot()</code>은 boxplot과 distribution을 함께 표현하는 그래프이다.
- <code>FacetGrid()</code>로 특정 조건에 따른 다양한 plot을 grid 형태로 표현할 수 있다.

<br />

## 통계론
통계론에 대해 가볍게 다루어보았다. 오늘은 모수의 개념과 그런 모수를 추정하는 방법인 최대우도추정법에 대해 배웠다.

#### 모수란?
- 통계적 모델링은 <strong>적절한 가정 위에서 확률분포를 추정</strong>하는 것이 목표이다.
- 하지만 유한 개의 데이터를 관찰해서 모집단의 분포를 정확히 알아내는 것은 불가능하므로 <strong>근사적으로 확률분포를 추정</strong>할 수 밖에 없다.
- 따라서 우리의 목표는 모집단의 분포를 정확히 맞히는게 아니라 위험(오차)을 최소화하는 것이다.
- 데이터가 특정 확률분포를 따른다고 선험적으로(a priori) 가정한 후 그 분포를 결정하는 모수(parameter)를 추정하는 방법을 <strong>모수적(parametric) 방법론</strong>이라고 한다.
- 특정 확률분포를 가정하지 않고 데이터에 따라 모델의 구조 및 모수의 개수가 유연하게 바뀌면 비모수적(nonparametric) 방법론이라고 한다.
  + <strong>비모수적 방법론이라고 모수가 없는 것 아니라는 점에 주의하자.</strong>
  + 다만 모수가 무수히 많거나 모수의 개수가 데이터에 따라 바뀔 뿐이다.
- 위와 같이 모수적인지 비모수적인지는 확률분포를 미리 가정하는지 아닌지에 차이가 있다. (모수의 유무로 구분하는게 아니다)

<br />

#### 확률분포의 가정
- 우리는 데이터가 생성되는 원리를 고려하여 확률분포를 가정할 수 있다.
  + 데이터가 2개(0 또는 1)의 값만 가지면 베르누이 분포라고 가정해볼 수 있다.
  + 데이터가 $n$개의 이산적인 값을 가지면 카테고리 분포라고 가정해볼 수 있다.
  + 데이터가 $\mathbb{R}$ 전체에서 값을 가지면 정규분포라고 가졍해볼 수 있다.
- <strong>어디까지나 가정해볼 수 있다는 것이지 실제로 그런 분포인지는 검정을 해봐야만 알 수 있다. </strong>
- 각 분포마다 검정하는 방법이 있으므로 모수를 추정한 후에는 반드시 검정을 해야한다.  

<br />

#### 중심극한정리
모집단의 분포가 정규분포를 따르지 않아도 표집분포는 정규분포를 따르며 이를 중심극한정리(Central Limit Theorem)라고 한다.  
여기서 표집분포란 통계량의 확률분포를 말하며, 또한 여기서 통계량은 표본평균 등을 예시로 들 수 있다.  
<strong>표본분포와 표집분포는 서로 다른 것을 의미한다.<strong>

- 표본분포: 하나의 표본으로부터 계산 분포. 즉, 하나의 표본집단의 분포를 뜻한다. 정규분포를 따르지 않을 수도 있다.
- 표집분포: 통계량의 확률분포. 여러 표본추출로부터 얻어진 통계량의 통계치. 반드시 정규분포를 따른다.
- 옛날에 배운 <strong>표본평균($\bar{X}$), 표본분산($S^2$)</strong> 등은 표본분포의 통계량이라고 할 수 있으며, 
  <strong>표본평균의 평균($E[\bar{X}]$, $\mu$), 표본평균의 분산($V(\bar{X})$, $\dfrac{\sigma ^{2}}{N}$)</strong> 등은 표집분포의 통계량(정확히는 표본평균의 표집분포)이라고 할 수 있다.
- 그래서 실제로 표본평균의 표집분포는 $N$이 커질수록 정규분포 $N(\mu, \dfrac{\sigma ^2}{N})$을 따른다.
 
<br />

#### 최대가능도 추정법
최대가능도 추정법, 최대우도 추정법, 최대우도법, 최우추정법 등 여러 명칭으로 불리는데 영어로 하면 maximum likelihood estimation(MLE)이다.  
- 확률분포마다 사용하는 모수가 다르다. 우리가 아는 확률분포는 표본평균이나 표본분산을 모수로 하는 경우가 많은데 확률 분포에 따라 다른 통계량을 모수로 할 수도 있다.
- <strong>확률분포를 어떤 식으로 가정하는지와 관계 없이</strong> 이론적으로 가장 가능성이 높은 모수를 추정하는 방법으로써 MLE를 사용한다.

<center>

$$
\hat{\theta}_{\text{MLE}} = \underset{\theta}{\mathrm{argmax}} L(\theta;\text{x}) = \underset{\theta}{\mathrm{argmax}} P(\text{x} | \theta)
$$

</center>

- 한편, 여기서 쓰인 가능도 함수(likelihood function)는 $L(\theta;\text{x})(=P(\text{x} \vert \theta))$을 의미하며 이는 모수 $\theta$를 따르는 분포가 $\mathrm{x}$를 관찰할 가능성을 뜻할 뿐 확률을 뜻하는 것은 아니다.
- 원래 확률함수는 모수 $\theta$가 주어졌을 때 데이터 $\mathrm{x}$에 대한 함수로 해석한다.
- 그런데 가능도 함수는 주어진 $\mathrm{x}$에 대하여 모수 $\theta$를 변수로 둔 함수로 해석된다. (즉, $\theta$값에 따라 값이 바뀌는 함수이다.)  

  
- 데이터 집합 $X$가 독립적으로 추출되었을 경우 <strong>로그 가능도를 최적화</strong>한다.  

<center>

$$
L(\theta;\mathrm{X}) = \prod\limits _{i=1}^{n} P(\text{x}_{i} | \theta) \;\; \Rightarrow \;\; \log L(\theta;\mathrm{X}) = \sum\limits _{i=1}^{n} P(\text{x}_{i} | \theta)
$$

</center>

- 로그 가능도를 쓰는 이유는 크게 2가지가 있다.
  + 데이터의 숫자가 수 억개 단위로 많아지면 0과 1사이의 값을 수 억번 곱한다는 것인데, 이런 식에서 정확한 값을 컴퓨터가 계산하는 것은 불가능하다. <strong>하지만 이를 덧셈으로 바꾸면 계산할 수 있다.</strong>
  + 우리는 추후 미분을 통해 이 가능도함수를 최대로 하는 $\theta$를 찾을 것인데, <strong>곱의 미분으로 하면 시간복잡도가 $O(n^2)$이다. 더해진 것을 미분하는 시간은 $O(n)$밖에 안걸린다.</strong>
- 추가적으로 대개의 손실함수의 경우 경사하강법을 사용하므로 음의 로그가능도를 최적화하게 된다.
  + 경사하강법은 극솟점을 찾는 것이기 때문에 음의 부호를 붙여주면 가능도를 최대화하는 방향으로 경사하강법을 사용할 수 있다.

- 대개의 경우 MLE를 미분하여 그 값이 0이 되는 극점(의 $\theta$ 값)을 찾는다. 여기서 든 의문이 있는데, 이게 극대인지 극소인지는 그래프를 그려봐야 알 수 있지 않을까>
  + 대충 검색을 해본 결과 MLE에서는 미분을 통해 구한 극점이 극대이면서 최대라는 것을 대체로 전제하고 푼다고 한다.
  + 근데 어떠한 과정으로 이게 전제되는지는 아직 잘 모르겠다.
  + 검색으로 해결이 되지 않는다면 추후 피어세션에서 이 문제에 대해 다루려고 한다.

- 실제로 정규분포에서 최대가능도 추정법을 사용하면 다음과 같이 두 모수 $\mu$, $\sigma$의 값을 도출할 수 있다.

정규분포에서 
<center>

$$
\hat{\theta}_{\text{MLE}} = \underset{\theta}{\mathrm{argmax}} L(\theta;\text{x}) = \underset{\theta}{\mathrm{argmax}} P(\text{X} | \mu, \sigma ^{2})
$$

$$
\log L(\theta ; \mathrm{X}) = \sum\limits _{i=1} ^{n} \log P(x_i | \theta) = \sum\limits _{i=1} ^{n} \log \frac{1}{\sqrt{2\pi \sigma ^{2}}} e^{-\frac{\vert x_{i} - \mu \vert ^{2}}{2 \sigma ^{2}}}
$$

$\therefore \; \log L(\theta ; \mathrm{X}) = -\dfrac{n}{2} \log 2\pi \sigma ^{2} - \sum\limits _{i=1} ^{n} \frac{\vert x_{i} - \mu \vert ^{2}}{2\sigma^2}$

</center>

<center>
$$
\hat{\mu}_{\text{MLE}} = \frac{1}{n} \sum\limits _{i=1} ^{n} x_{i}
$$

$$
\hat{\sigma} ^{2} _{\text{MLE}} = \frac{1}{n} \sum\limits _{i=1} ^{n} (x_{i} - \mu) ^{2}
$$

</center>

- 미분하는 과정은 간단한 과정이라 생략하였다. 로그를 씌운 가능도 함수를 $\mu$와 $\sigma$로 한 번씩 편미분해주면 $\theta$값 즉 $\hat{\mu}_{\text{MLE}}$와 
  $\hat{\sigma} ^{2} _{\text{MLE}}$를 쉽게 유도해낼 수 있다.

- 강의에서는 카테고리 분포에 대한 최대가능도 추정도 다루었는데, 제약조건 때문에 라그랑주 승수법을 적용하는 부분이 있어 여기서는 생략한다.
- 라그랑주 승수법은 현재 시간이 없어 제대로 이해하지 못했는데, 이해하면 카테고리 분포에 대한 내용도 추가할 예정이다.

<br />

#### 딥러닝에서 최대가능도 추정법
- 딥러닝에서는 모수 $\theta$가 $\mathrm{W} ^{(1)}, \cdots, \mathrm{W} ^{(L)}$로 주어진다.
- 특히 분류 문제에서 softmax 벡터는 카테고리 분포의 모수 $(p_{1}, \cdots, p_{k})$를 모델링한다. <strong>즉, softmax 벡터는 카테고리 분포의 모수와 같은 형태이다. </strong>  
> 카테고리 분포는 one-hot 인코딩된 확률변수를 가지며, $\sum\limits \_{k=1} ^{n} p_{k} = 1$이다.  

- one-hot 벡터로 표현한 정답레이블 $\mathrm{y} = (y_1, \cdots, y_k)$을 관찰데이터로 이용해 확률분포인 소프트맥스 벡터의 로그가능도를 최적화할 수 있다.
- 쉬운 말로 하면 softmax가 카테고리 분포의 모수를 모델링하니까 카테고리 분포에서처럼 최대가능도 추정을 할 수 있다는 것이다.

<center>

$$
\hat{\theta}_{\text{MLE}} = \underset{\theta}{\mathrm{argmax}} \frac{1}{n} \sum\limits ^{n} _{i=1} \sum\limits ^{K} _{k=1} y_{i, k} \log (\mathrm{MLP} _{\theta} (\mathrm{x} _i) _k)
$$

</center>

- 맨 앞 시그마는 '모든 데이터에 대해'를 의미하고 그 뒤 시그마는 '모든 클래스의 개수에 대하여'를 의미한다.
- 그 뒤는 MLP(다층신경망) 예측 값에서 $k$번째 예측값의 로그값과 정답레이블 $\mathrm{y}$의 $k$번째 주소값 $y_{k}$를 곱한다는 의미이다.
- 식의 의미가 대충은 이해가 가는데, 아직은 확실히 이해가 안간다. 일단은 추후 카테고리 분포의 MLE를 이해한 후 다시 봐야할 것 같다.

<br />

#### 확률분포의 거리
- 기계학습에서 손실함수들은 모델이 학습하는 확률분포와 데이터에서 관찰되는 확률분포의 거리를 통해 유도한다
  + 거리라고 표현하긴했는데, 아무튼 실제 분포와 학습된 분포의 오차를 말하는 것 같다.
- <strong>데이터공간에 두 개의 확률 분포 $P(\mathrm{x})$, $Q(\mathrm{x})$가 있을 경우 두 확률분포 사이의 거리를 계산할 수 있다.</strong>
  + 이 과정에서 총변동 거리(Total Variation Distance, TV), 쿨백-라이블러 발산(Kullback-Leibler Divergence, KL), 바슈타인 거리(Wasserstein Distance) 등을 사용한다.
- 말하고자하는 바는 다음과 같다.
  + 두 확률 분포 사이의 오차라는 것은 사실 생각해보면 어떤 의미인지는 감이 오는데 어떻게 표현해야할지 의문이 든다.
  + 그런데 확률 분포 사이 오차(거리)를 위에서 언급한 세가지 방법 등으로 구할 수 있다는 것이다.
  + 우리는 <strong>실제 분포와 학습 분포 사이의 거리가 얼마나 되는지</strong> 궁금하다. 그래서 위 세가지 방법을 사용할 것이다.
  + 거리를 최소화하는 것이 우리의 목표일 것이다.

<br/>

#### 쿨백-라이블러 발산
- 강의에서는 3개 중 쿨백-라이블러 발산을 다루었다. 아무래도 가장 간단한 것 같기는 하다.  
  쿨백 라이블러 발산(KL Divergence)은 다음과 같이 정의된다.

<center>

$$
\text{dicrete} \; \mathbb{KL}(P \Vert Q) = \sum\limits_{x \in \mathcal{X}} P(\mathrm{x}) \log \left( \frac{P(\mathrm{x})}{Q(\mathrm{x})} \right)
$$

$$
\text{continuous} \; \mathbb{KL}(P \Vert Q) = \int _{\mathcal{X}} P(\mathrm{x}) \log \left( \frac{P(\mathrm{x})}{Q(\mathrm{x})} \right)
$$

</center>

- 쿨백-라이블러 발산은 다음과 같이 분해할 수 있다. 왼쪽은 크로스 엔트로피($\log Q(\mathrm{x})$의 기댓값), 오른쪽은 엔트로피($\log P(\mathrm{x})$의 기댓값)이다.

<center>

$$
\mathbb{KL}(P \Vert Q) = - \mathbb{E} _{\mathrm{x} \sim P(\mathrm{x})} [\log Q(\mathrm{x})] + \mathbb{E} _{\mathrm{x} \sim P(\mathrm{x})} [\log P(\mathrm{x})]
$$

</center>

- 쉽게 이해해보자. 엔트로피라는 것은 원래 아래와 같이 정의된다.

<center>
$$
H(x)=−\sum_{i=1}^{n}  p(x_i) log{p(x_i)}
$$
</center>

그리고 쿨백-라이블러 발산을 기댓값이 아닌 시그마가 포함된 식으로 풀어서 나타내보면 아래와 같다.

<center>

$$
\mathbb{KL}(P \Vert Q) = - \sum\limits _{i} P(i) \log Q(i) + \sum\limits _{i} P(i) \log P(i)
$$

</center>

- 정답 레이블을 P, 예측한 레이블을 Q라고 가정하면 왼쪽 부분은 정보량 부분에 예측확률($Q$)가 사용되었고 확률부분에는 실제확률($P$)가 사용(크로스 엔트로피)되었다.
  오른쪽 부분은 실제 엔트로피 식과 같다. 결국 <strong>두 분포간의 거리는 정답 레이블에 대한 두 정보의 엔트로피의 차</strong>로 나타낼 수 있었다.
- 엔트로피라는 것에 익숙하지 않아서 그렇지 사실 간단하게 생각하면 이 식으로 두 분포간의 차이를 알 수 있다는 것은 당연한 것이다.

<br/>

## 그 외
이렇게 통계론까지 가볍게 다뤄봤는데, 통계는 익숙하지 않은 부분이 많아 여러 번 다시 봐야할 것 같다. 그리고 솔직히 제대로 이해했는지도 아직은 잘 모르겠다.  
  

하지만 너무 깊게 파고들면 시간이 너무 오래 걸려 일단은 이만 줄이려고 한다. 그리고 실제로 해보면서 봐야 더 이해가 빠른 부분도 있을 것이다.
일단은 잘못 이해했더라도 공부를 지속하면서 잘못된 부분을 바로잡는 방향으로 학습을 진행하려고 한다.  

  
최근에 이쪽 분야를 공부하면서 느끼는건데 처음부터 너무 모든 것을 챙기려고 하는 태도는 좋지 않은 것 같다.
최대한 이해하려고 노력해보고 그래도 안되는 것은 다음에 다시 보는게 백번 나은 것 같다. :sweat_smile:

<br/>


## Reference
[카테고리 분포](https://bit.ly/3afr9sT)  
[크로스 엔트로피](https://bit.ly/3puBrvJ)  
[쿨백-라이블러 발산](https://brunch.co.kr/@chris-song/69)  