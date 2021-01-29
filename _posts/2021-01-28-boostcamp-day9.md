---
layout: post
title: "Day9. pandas와 확률론"
subtitle: "pandas(Part 2), 확률론 기초"
date: 2021-01-28 21:50:12+0900
background: '/img/posts/bg-posts.png'
use_math: true
---

## 개요 <!-- omit in toc -->
> 먼저 pandas 두 번째 파트를 다루었다. SQL에서의 여러 기능들과 엑셀에서의 여러 기능들을 모두 pandas에서 사용할 수 있었다. 
확률론도 배웠는데 모르는 개념이 많아 강의 외적으로도 많이 정리하는 시간이 필요했다.


  
오늘 배운 내용은 아래와 같다.
- [pandas (Part 2)](#pandas-part-2)
    - [groupby](#groupby)
    - [pivot table & crosstab](#pivot-table--crosstab)
      - [pivot table](#pivot-table)
      - [crosstab](#crosstab)
    - [merge & concat](#merge--concat)
      - [merge](#merge)
      - [concat](#concat)
    - [persistence](#persistence)
- [확률론 기초](#확률론-기초)
    - [확률론은 어디에 쓰는가?](#확률론은-어디에-쓰는가)
    - [확률분포와 데이터 공간](#확률분포와-데이터-공간)
    - [결합확률분포](#결합확률분포)
    - [주변확률분포](#주변확률분포)
    - [조건부확률분포](#조건부확률분포)
    - [기댓값과 조건부기댓값](#기댓값과-조건부기댓값)
    - [조건부확률과 기계학습](#조건부확률과-기계학습)
      - [회귀문제](#회귀문제)
      - [분류문제](#분류문제)
    - [몬테카를로 샘플링](#몬테카를로-샘플링)
- [그 외](#그-외)
- [Reference](#reference)

<br/>

## pandas (Part 2)
거의 막바지긴하지만.. 판다스에 대해 길게 기술하는게 별로 의미가 없는 것 같다. 이해가 필요한게 아니고 어차피 필요할 때 찾아서 쓰면 되는 부분이다.  
그래서 오늘부터 이런부분들은 간략하게 기술하도록 한다.  

#### groupby
SQL에서의 groupby명령어와 같다. split -> apply -> combine의 과정으로 이루어진다.  
좀 더 정리해서 말하면, 원래 테이블을 groupby의 기준이 되는 column 기준으로 split하고 sum이나 mean등의 apply과정을 거친 후 다시 하나로 combine한다.
- <code>df.groupby("기준 컬럼")["APPLY를 적용받는 컬럼"].연산()</code> 의 구조로 사용한다.  
  당연히 기준 컬럼이 여러 개일수 있으며, 이 경우 multi index로 사용된다.
- <code>unstack()</code> 함수를 통해 groupby으로 묶여진 데이터를 matrix 형태로 전환할 수 있다. 
- <code>stack()</code> 함수로 <code>unstack()</code>을 다시 원래대로 되돌릴 수 있다. 그런데 이건 사실 <code>reset_index()</code>와 같은 기능을 한다.
- <code>swaplevel()</code> 함수로 index level을 변경할 수 있다. 어떤 인덱스가 먼저 놓이느냐의 차이다.
- <code>sortlevel()</code> 함수로 parameter로 넣어준 index 값을 기준으로 테이블을 정렬한다.
- multi-indexed여도 combine된 부분에 value가 column 한 개이면 결국 Series 형식의 데이터이다. 따라서 이 경우 index level을 기준으로 기본 연산 수행이 가능하다.  
  예를 들어 <code>h_index.sum(level=0)</code>이라고 쓰면 level 0번 인덱스를 기준으로 value column의 sum을 구하는 것이다.
  
    
- 아래와 같이 groupby에 의해 split된 상태를 group별로 추출 가능하다.  
  name은 그룹의 이름, group은 각 group DataFrame을 의미한다. 역시 Tuple 형태로 추출되며 사실 key(name)-value(group) 형태이다.  

    ```python
    #group_split.py
    ...
    grouped = df.groupby("Team")

    for name, group in grouped:
        ...
    ```  

- 위 코드에서 특정 key값을 가진 그룹의 정보를 <code>grouped.get_group("그룹명")</code>으로 추출할 수 있다.
- 추출된 그룹에서 세 가지 유형의 apply가 가능하다.
    + Aggregation
        - 요약된 통계정보를 추출해준다. <code>grouped.agg(sum)</code>, <code>grouped.egg(np.mean)</code> 등으로 사용할 수 있다.
        - 특정 컬럼에 여러 개의 function을 apply할 수도 있다. i.e. <code>grouped['Points'].agg([np.sum, np.mean])</code>
    + Transformation
        - aggregation처럼 key값 별로 요약된 정보를 찾는게 아니고 개별 데이터의 변환을 지원한다.
        - 따라서 map에서처럼 lambda함수 등을 이용한다.
        - <code>grouped.transform(lambda x: (x))</code>의 형태로 사용할 수 있다. 이 경우 grouped의 기준열이 사라진 형태로 반환될 것이다. 
        - <code>grouped.transform(lambda x: (x.max()))</code> 의 형태로도 사용할 수 있다. 이렇게 <code>min</code>, <code>max</code>처럼 Series 데이터에 적용되는
          데이터들은 Key 값을 기준으로 grouped된 데이터 기준으로 값을 반환하게 된다.
    + Filter
        - 특정 조건으로 데이터를 검색할 때 이용한다.
        - <code>df.groupby("Team").filter(lambda x: len(x) >= 3)</code> 와 같은 형태로 사용한다.
        - filter 안에는 위처럼 boolean 조건이 존재해야하며 위 함수는 결국 grouped의 결과 행이 3개 이상인 행들만 반환할 것이다.  
    

  
- <code>df_phone.groupby("month", as_index=False).agg({"duration": "sum", "network_type": "count", "date": "first"})</code> 
  처럼 한 번에 여러 column에 대한 apply를 할 수도 있다. 코드가 두줄로 짤리는데 무슨 말인지 이해는 되니까 괜찮다. :smiley:

<br/> 

#### pivot table & crosstab
##### pivot table
엑셀에서 pivot 기능처럼 pandas에서도 pivot table을 만들 수 있다.  
그런데 앞서 봤던 groupby로도 사실 apply를 적절히 활용하면 같은 기능을 수행할 수 있다.  
편한걸로 사용하면 된다. 코드는 아래와 같이 쓴다.

```python
#pivot_table.py
...
df_phone.pivot_table(
    values=["duration"], # duration의 값들을 기준으로
    index=[df_phone.month, df_phone.item], # index를 month, item으로
    columns=df_phone.network, # column은 network의 값들을 깐다.
    aggfunc="sum", # index와 column에 대응하는 duration의 sum값을 가져온다
    fill_value=0, # NaN 값 대신 0을 사용한다. 
)
```

##### crosstab
두 column 간의 교차 빈도, 비율, 덧셈 등을 구할 때 사용한다.
pivot table의 특수한 형태인데, 사실 그냥 groupby나 pivot table이나 crosstab이나 모두 같은 기능을 수행한다.
crosstab은 함수의 형태가 pivot table의 그것과 매우 유사한데, 필요할시 직접 찾아보도록 하자.

<br/> 

#### merge & concat
##### merge
- SQL의 merge와 유사하다. 두 개의 데이터를 하나로 합치는 것이다.  
- 물론 합칠 때 서로 최소 하나의 column은 일치해야한다.
- <code>pd.merge(df_a, df_b, on='기준 column')</code>으로 사용한다.
- 같은 이름의 column을 <code>on</code>의 옵션으로 주면 되는 것이다.
- 만약 같은 column인데 두 테이블에서 이름이 다르면 <code>pd.merge(df_a, df_b, left_on='', right_on='')</code> 으로 사용하면 된다.
- SQL에서처럼 inner join, full join, left join, right join을 모두 제공한다.
- inner join이 merge의 default값이며, inner join 외 다른 방법에서는 없는 값에 대하여 NaN으로 처리한다.
- 만약 merge 과정에서 같은 이름의 column이 있으면 기준행이 아닌이상 _x, _y로 네이밍되니 주의하도록 하자.
- <code>pd.merge(df_a, df_b, right_index=True, left_index=True)</code>와 같이 쓰면 인덱스를 기준으로 같은 수의 인덱스끼리 merge하게 된다.

##### concat
- 같은 형태의 데이터를 붙이는 연산작업이다.
- 이것은 당연히 서로 같은 column을 가졌음을 전제로 한다.
- 만약 <code>axis</code> 옵션을 주어 <code> df_new = pd.concat([df_a, df_b], axis=1)</code>과 같이 쓰면 세로로 붙이게된다.
- 세로로 붙이게 될 경우 index값이 두 테이블 모두 0부터 시작하므로 key값으로 사용할 수 없게된다.
- 따라서 여기에 <code>df_new.reset_index(drop=True)</code>까지 해주면 금상첨화이다. <code>drop=True</code> 옵션은 기존 index column을 삭제해준다.

<br/> 

#### persistence
- SQL에 접속해서 <code>pd.read_sql_query("쿼리문", 커서)</code>의 형태로 데이터베이스에서 쿼리문을 적용한 데이터프레임을 받아올 수도 있다.  
  물론 SQL에 접속하는 것은 또 따로 <code>sqlite3</code> 등의 모듈을 불러와서 알아서 해야한다.
- <code>ExcelWriter</code>을 이용하여 엑셀 파일을 만들 수도 있다. 해본거니까 자세한 코드는 생략한다.
- 엑셀 외에 pickle 파일을 읽고 쓸 수도 있다. <code>df_routes.to_pickle("경로")</code>
- pickle 형식은 실제로 많이 사용하는 것이니 기억해두자.


<br />

## 확률론 기초
오늘은 통계론을 배우기 앞서 기초적인 확률론부터 배워보았다.  

#### 확률론은 어디에 쓰는가?
- 기계학습에서 사용되는 손실함수(loss function)들의 작동원리는 데이터 공간을 통계적으로 해석해서 유도하게 된다.
    + 만약 우리가 다루는 모델의 확률분포를 명시적으로 알 수 있다면 확률분포함수를 기반으로 기대값을 예측할 수 있다.
    + 확률분포를 정확히 모르더라도 우리는 통계적 지식을 기반으로 확률분포를 예측할 수 있다.
- 회귀 분석에서 손실함수로 사용되는 $L\_{2}$-norm은 <strong>예측오차의 분산을 가장 최소화하는 방향으로 학습</strong>하도록 유도한다.
    + 예측오차의 분산은 $\dfrac{1}{N} \sum\limits \_{i} (y\_{i} - f(x\_{i}))^2$이므로   
      $L\_{2}$-norm $=\Vert y - f(x) \Vert _{2}$와 형태가 매우 유사하다.
      따라서 $L\_{2}$-norm을 최소화하는것과 예측오차의 분산을 최소화하는 것은 같은 task이다.  

    + 왜 분산을 최소화하면 되는가?  
        - (첫번째 이해) 분산이 크다는 것은 예측 값과 실제 기대 값 사이의 오차가 크다는 것이다. 따라서 분산을 최소화해야한다.  
        - (두번째 이해) 신뢰구간을 생각해보면 신뢰구간 식에는 $\dfrac{\sigma}{\sqrt{n}}$가 곱해져있는 항이 있다. $\sigma$는 표준편차를 의미하는데, 표준편차의 제곱이 분산이기 때문에 결국 분산이 커지면 신뢰구간의 길이가 커지고 이러면 예측의 정확도가 떨어지게 된다.   
        - 사실 위 두 개는 개인적으로 생각해본 것인데 다 맞는 말인지는 모르겠다 :sweat: 일단 그렇게 이해하고 넘어가려고 한다.

- 분류 문제에서 사용되는 교차 엔트로피(cross-entropy)는 모델 예측의 불확실성을 최소화하는 방향으로 학습하도록 유도한다.
    + 교차 엔트로피는 아직 안다뤄봐서 따로 기재하지는 않겠다. 아무튼간에 불확실성을 최소화하기 위해 확률론을 사용한다는 것에 의미가 있다.

- 아무튼 분산 및 불확실성을 최소화하기 위해 확률이나 기댓값을 구하는 방법을 알아야한다.

<br/>

#### 확률분포와 데이터 공간
확률분포는 말그대로 데이터의 분포를 나타낸다. 보통 데이터공간을 $\mathscr{X} \times \mathscr{Y} \$라 표기하며 $\mathscr{D}$는 데이터공간에서 데이터를 추출하는 분포이다.
- 데이터는 확률변수로 $(x, y) \sim \mathscr{D}$라 표기한다. $\mathscr{D}$라는 분포에 있는 데이터 $(x, y)$쌍을 의미한다.
- 확률분포는 이산확률분포와 연속확률분포로 나뉜다.
    + 데이터 공간이 정수형이면 이산확률분포임은 확실하다. 하지만 실수공간이라고 꼭 연속확률분포는 아니다. 가령 입력으로 들어오는 확률변수가 0.5와 -0.5뿐이면 데이터 공간이 실수공간이지만
      이것은 이산확률분포이다.
    + 따라서 이산/연속은 데이터 공간이 아닌 확률분포의 종류로 구분한다.

<center>

$\text{dicrete} \; \mathbb{P}(X \in A) = \sum\limits _{\text{x} \in A} P(X = \text{x})$

<br/>
<br/>

$\text{continuous} \; \mathbb{P}(X \in A) = \int _{A} P(\text{x}) d\text{x}$


</center>

- 연속확률분포에서 $P(x)$는 확률밀도함수로, 당연히 적분을 해야 확률로 해석 가능하고 이 자체가 확률을 나타내는 것은 아니다.
- 또한 확률분포는 이산확률분포와 연속확률분포만 존재하는 것은 아니라는 점도 기억하자. 둘이 혼합된 형태가 있을 수도 있고 또 다른 확률분포가 존재할 수 있다.

<br/>

#### 결합확률분포
결합확률분포(disjoint probability distribution)는 $P(X, Y)$로 나타내며 서로 다른 두 확률변수 $X$와 $Y$에 의해 확률질량함수(확률밀도함수)가 결정되는 확률분포이다.  
어렵게 생각할 것 없이 그냥 확률변수가 2개 존재하는 확률분포라는 점만 기억하면 된다. 그것 말고는 차이점이 없다.  
물론 결합확률분포의 확률질량함수(이산분포)와 확률밀도함수(연속분포)의 형태는 약간씩 다르다.

<center>

$\text{dicrete} \; P(x, y) =  P(X = x, \; Y = y)$

<br/>
<br/>

$\text{continuous} \; P(X, Y \in A) = \int \int _{A} P(x, y) dx dy$

</center>

결합확률밀도함수의 경우 처음보면 긴가민가한데 사실 아래 식이랑 완전히 같은 의미이다.

<center>

$P(a \leq X \leq b, c \leq Y \leq d)=\int_{a}^{b}\int_{c}^{d}f(x, y)dxdy$

</center>

<br />

#### 주변확률분포
주변확률분포(marginal probability distribution)는 결합확률분포에서 변수 하나를 고정시킨 후 확률함수를 구하면 유도해낼 수 있다. 즉, 하나의 변수로만 이루어진 확률함수를 구하면 된다.  
확률분포식을 살펴보자.

<center>

$\text{dicrete} \; P(x) =  \sum\limits _{y} P(x, y)$

<br/>
<br/>

$\text{continuous} \; P(X) = \int _{ \mathcal{Y}} P(x, y) dy$

</center>

많은 블로그나 책에서 $P(x)$ 대신 $f_{\mathrm{X}} (x)$의 형태를 많이 사용하는데 실제로 이게 더 명확한 표현 같기는 하다.  
사실 $P(x)$는 확률값 그 자체이고 $f_{\mathrm{X}} (x)$와 같이 나타내는 것은 확률함수를 나타내는것인데 아무튼 둘다 결합확률분포에서 $y$를 신경쓰지 않은 $X=x$일 확률을 의미한다.      
   
만약 확률변수 $X$와 $Y$가 서로 독립이면 당연히 $P(x)$와 $P(y)$는 서로 영향을 받지 않을 것이므로

<center>

$f(x, y) = P(x)P(y) \;\;(=\, f_{\mathrm{X}} (x) f_{\mathrm{Y}} (y))$

</center>

가 성립한다.

<br/>

#### 조건부확률분포
조건부 확률과 매우 유사한 개념이다. 전제가 되는 변수의 주변확률함수를 구한 후 분자는 두 확률 변수 모두가 성립할 확률로 취하면 된다. 즉,

<center>

$P(y|x) = \dfrac{P(x, y)}{P(x)} \;\; \left(=f(y|x) = \dfrac{f(x, y)}{f_{\mathrm{X}}(x)} \right)$

</center>

이다. 자꾸 표현을 2개씩 쓰는데 많은 책과 블로그에서 실제로 확률함수를 $f$로 표현하고있기 때문이다 :cry: 우리 강의 자료에서는 그냥 확률이라는 의미로 $P$를 사용한다 :joy:

아무튼 둘다 $x$값이 고정되었을 때 $y$값이 나타날 <strong>조건부 확률질량(밀도)함수</strong>를 의미한다.  
확률분포가 연속일 경우 확률함수가 아니라 확률밀도함수라는 점에 주의하자.  
  
만약 연속확률분포에서의 조건부확률함수를 쓰면 아래와 같을 것이다.  
(이산확률분포이면 그냥 두 사건이 동시에 일어날 확률 구해서 확률질량함수에 넣으면 된다.)

<center>

$P(a < \text Y < b \vert \text X = x) = \int _a ^{b} f(y \vert x)dy = \dfrac{\int _a ^{b} f(x, \, y)dy}{f_{\mathrm{X}}(x)}$

</center>

<br/>

#### 기댓값과 조건부기댓값
기댓값(expectation)은 데이터를 대표하는 통계량이다. 우리가 흔히 하는 평균값과도 같은 의미이며 우리는 목적함수의 기댓값이 실제값과 같아지도록 만드는 것이 목표이다.  
일반적으로 확률변수가 $X$ 하나이고 확률분포함수가 $P(x)$일 때, 확률변수 $X$에 대한 기댓값 $E(X)$은 아래와 같다.

<center>

$\text{dicrete} \; E[X] =  \sum\limits _{x} xP(x)$

<br/>
<br/>

$\text{continuous} \; E[X] = \int _{-\infty} ^{\infty} xP(x) dx$

</center>

만약 확률변수 $X$를 어떤 함수 $f$에 넣은 값 즉 $f(X)$의 기댓값이 필요하면 아래 식을 따르면 된다.

<center>

$\text{dicrete} \; E[f(x)] =  \sum\limits _{x \in \chi} f(x)P(x)$

<br/>
<br/>

$\text{continuous} \; E[f(x)] = \int _{\chi} f(x)P(x) dx$

</center>

조건부기댓값도 구할 수 있는데, 이를 통해 입력값이 $x$일 때의 출력의 기댓값을 구할 수 있다.

<center>

$\text{dicrete} \; \mathbb{E} _{y \sim P(y | x)} [y | x]\;(=E[y | x]) = \sum\limits _{\mathcal{Y}} yP(y|x)$


<br/>
<br/>

$\text{continuous} \mathbb{E} _{y \sim P(y | x)} [y | x]\;(=E[y | x]) = \int _{\mathcal{Y}} yP(y|x) dy$

</center>

각각이 유도되는 과정은 일단 생략한다. 너무 길어진다 :unamused:  
기댓값을 이용하여 분산, 첨도, 공분산 등 여러 통계량을 계산할 수 있다.   

  
이제 기계학습에서 기댓값과 조건부확률이 어떻게 적용되는지 살펴보자.  

<br/>

#### 조건부확률과 기계학습 
기계학습에서는 대표적으로 회귀문제와 분류문제를 다루고 있으며 각각에 대하여 확률론과의 관계를 살펴보자.

##### 회귀문제
- 먼저 회귀문제에 대해 생각해보자. 회귀문제의 경우 입력값 $x$에 대한 출력값이 연속확률분포의 형태이며, 그렇기 때문에 <strong>입력값이 $x$일 때의 목표함수의 기댓값</strong>을 구해야한다.
  따라서 회귀문제는 그냥 조건부기댓값 $\mathbb{E}[y | x] \;(=E[y | x])$를 사용하면 되고, 이것이 우리가 찾고자했던 값이다.

  
- 조건부기댓값은 당연히 $\mathbb{E} \Vert y - f(x) \Vert _{2}$(=예측오차의 분산값)을 최소화하는 함수 $f(x)$와 일치한다. 
  기댓값이 실제 원하는 값과 같아야 하므로 위 $y$와 $f(x)$간의 square-error가 최소화되어야 하기 때문이다.  
  앞서 $L\_{2}$-norm을 최소화하는것과 예측오차의 분산을 최소화하는 것은 같은 task라고 설명했었는데 여기서 그 이유를 한번 더 짚고 넘어갈 수 있다.

<br/>

##### 분류문제
- 분류문제에서는 사실 softmax함수에서 조건부확률의 개념을 이미 사용하고 있었다. 데이터 $x$로부터 추출된 특징패턴 $\phi (x)$와 가중치행렬 $\text{W}$를 통해 만든 $\text{W} \phi(x) + \text{b}$를 softmax 함수에 통과시킨 값 $\text{softmax}\left(\text{W} \phi + \text{b}\right)$은 사실 조건부확률 $P(y \vert x)$를 의미한다. 이에 대한 자세한 설명은 <span class="link_button">[이 블로그](https://taeoh-kim.github.io/blog/softmax/)</span>에서 확인할 수 있다.
- 특징패턴을 어떻게 추출하는지는 아직 잘 모르겠다. 다층신경망을 이용하여 데이터로부터 특징패턴을 추출한다고 나와있기는 하다.  
- 우선은 활성화 함수를 통해 임계값을 넘은 부분만 추출하는 것도 특징패턴 추출 중 하나라고 생각되는데, 
  여기서 이해하고 넘어갈 점은 소프트맥스 함수에 조건부확률의 개념이 사용되었다는 점이므로 이부분은 나중에 더 알아보기로 한다.

<br/>

#### 몬테카를로 샘플링
사실 기계학습의 많은 문제들은 확률분포를 명시적으로 모를 때가 대부분이다.  
그래서 확률분포를 모를 때 데이터를 이용하여 기댓값을 계산하려면 몬테카를로(Monte Carlo) 샘플링 방법을 사용해야 한다.   
몬테카를로 방법은 이산형이든 연속형이든 상관없이 성립한다.  

<center>

$$
\mathbb{E} _{x \sim P(x)} [f(x)] \approx \dfrac{1}{N} \sum\limits _{i=1} ^{N} f(x^{(i)}), \; \; x^{(i)} \overset{\text{i.i.d.}}{\sim} P(x) 
$$

</center>

참고로 위 식에서 좌측에 i.i.d.라고 써있는 것은 $x^{(i)}$들은 서로 상호독립적이며 동일한 확률분포 $P(x)$를 가진다는 의미이다.  
이걸 독립항등분포라고 하는데 i.i.d.로 줄여서 쓴다.

  
식이 또 복잡해보이는데, 이건 진짜 간단하다! 
위 식과 같이 확률변수 $x^{(i)}$들간의 독립추출만 보장된다면 대수의 법칙(law of large number)에 의해 우측 식은 실제 확률분포에 수렴을 보장한다.  
즉 조건만 성립하면 그냥 확률변수의 범위 내에서 균등분포로 난수 $x$를 선정해 $f(x)$ 값들의 평균을 구하면 그게 $f(x)$의 기댓값이 되는 것이다.  

예를 들어, $f(x) = e^{-x^{2}}$의 [-1, 1]에서의 적분값을 구해보자.  
몬테카를로의 법칙에 의해 -1에서 1 사이의 난수를 충분히 많이 뽑아 함숫값을 다 더해 $N$으로 나눠주면 된다.   
  
주의할 점은, 이렇게 뽑아서 나온 기댓값은 당연히 함숫값 그 자체의 기댓값이므로 이걸 적분에 적용하려면 구간의 길이도 고려해주어야 한다.  
현재는 구간의 길이가 2이므로 구한 기댓값에 2를 곱하면 적분값을 구할 수 있다.

```python
#monte_carlo.py
import numpy as np

def mc_int(fun, low, high, sample_size=100, repeat=10):
    int_len = np.abs(high - low) #구간의 길이를 구한다.
    stat = [] #구한 f(x) 값을 여기에 저장한다.
    for _ in range(repeat):
        x = np.random.uniform(
                low=low, 
                high=high, 
                size=sample_size
            ) #난수를 sample_size개 만큼 생성
        fun_x = fun(x) #함숫값
        int_val = int_len * np.mean(fun_x)
        stat.append(int_val)
    return np.mean(stat), np.std(stat)

def f_x(x):
    return np.exp(-x**2)

print(mc_int(f_x, low=-1, high=1, sample_size=100000, repeat=100))    
#(1.4936840745776079, 0.0012555624582977366) 
# 오차 범위 1.49368 ± 0.0012
```

$\int ^{1} _{-1} e^{-x^{2}} dx \approx 1.49364$로 구한 값이 오차범위 이내에 있는 것을 확인할 수 있다.

그 외에 <span class="link_button">[여기](http://www.gisdeveloper.co.kr/?p=9039)</span>서 설명된 사분원 넓이로 $\pi$값 구하기도 몬테카를로 법칙을 적용한 좋은 예시가 될 수 있다.

<br/>


## 그 외
목적함수, 손실함수, 비용함수는 모두 비슷한 의미이다.  
약간의 차이가 있는데 <span class="link_button">[이 질문글](https://stats.stackexchange.com/questions/179026/objective-function-cost-function-loss-function-are-they-the-same-thing)</span>을 참고하면 좋을 것 같다.   
요약하면 Loss function이 Cost function에 포함되고, Cost function이 Objective function에 포함된다는 것이다.   
   

답변에 달린 댓글도 짚고 넘어가자.  
댓글에서 언급하고 있는 것은 Objective function은 말그대로 목표 함수라서 목표함수에 입력값을 넣은 결과를 최대화 해야할 수도 있고 최소화 해야할 수도 있는데 
다른 것들은 cost, loss라서 가급적 최소화해야되는 함수라는 의미로 사용된다고 한다.  

<br/>

## Reference
사실 이전까지 글들에서도 Reference를 달았어야했는데 포스팅이 처음이다보니 잊고 있었다 :cry: :cry: :cry:    

  
확률론에 대한 이야기를 할 때 아래 글들을 참고하였다.  
[확률밀도함수의 유도](https://bit.ly/3pqF5a1)  
[결합확률분포](https://destrudo.tistory.com/13)  
[주변확률분포](https://blog.naver.com/mykepzzang/220837645914)  
[조건부확률분포](https://blog.naver.com/mykepzzang/220837722214)  
[조건부기댓값](https://analysisbugs.tistory.com/8)  
[조건부기댓값(PDF)](https://imai.fas.harvard.edu/teaching/files/Expectation.pdf)  
[몬테카를로 법칙](http://www.gisdeveloper.co.kr/?p=9039)  