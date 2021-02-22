---
layout: post
title: "Day21. 그래프 이론, 패턴"
subtitle: "그래프, 작은 세상 효과, 연결성, 군집 계수, NetworkX"
date: 2021-02-22 15:13:12+0900
background: '/img/posts/bg-posts.png'
use_math: true
---

## 개요 <!-- omit in toc -->
> 오늘부터는 추천시스템에서 많이 활용되는 그래프 이론에 대해서 다루어보도록 한다.  
  
아래와 같은 순서로 작성하였다.  
- [그래프 기초](#그래프-기초)
    - [그래프 관련 인공지능 문제](#그래프-관련-인공지능-문제)
    - [그래프의 유형](#그래프의-유형)
    - [그래프의 표현](#그래프의-표현)
- [그래프 패턴](#그래프-패턴)
    - [실제 그래프, 랜덤 그래프](#실제-그래프-랜덤-그래프)
    - [작은 세상 효과](#작은-세상-효과)
    - [연결성의 두터운 꼬리분포](#연결성의-두터운-꼬리분포)
    - [군집 구조](#군집-구조)
    - [거대 연결 요소](#거대-연결-요소)
    - [그래프 별 군집 계수 및 지름 분석](#그래프-별-군집-계수-및-지름-분석)
- [그래프 구현을 위한 파이썬 라이브러리](#그래프-구현을-위한-파이썬-라이브러리)
- [Reference](#reference)


<br/>

## 그래프 기초
아래는 서로 대응되는 관계이다.  
- 그래프(Graph) - 네트워크(Network)
- 정점(Vertex) - 노드(Node)
- 엣지(Edge) - 링크(Link)  
  
세상의 많은 것들이 복잡계(Complex System)으로 이루어져있는데 이를 그래프로 표현할 수 있다.  
  
- 그래프는 보통 정점의 집합 $V$와 간선의 집합 $E$로 이루어져있으며 이를 $G = (V, E)$로 표기한다.
- 정점의 이웃은 $N(v)$ 혹은 $N\_v$로 적는다.
- 정점 $v$에서 나가는 이웃(Out-Neighbor)을 $N\_{out} (v)$로 적는다.
- 정점 $v$로 들어오는 이웃(In-Neighbor)을 $N\_{in} (v)$로 적는다

<br />

#### 그래프 관련 인공지능 문제
- 정점 분류(Node Classification)
    + 그래프의 연결관계를 통해 연결관계와 밀접하게 연관된 노드의 특성을 알 수 있고, 이 때 새로운 노드가 들어오면 해당 노드의 연결관계만으로 그 특성을 유추할 수 있다.
- 연결 예측(Link Prediction)
    + 거시적 관점의 연결 예측. i.e. 페이스북 소셜네트워크의 크기가 얼마나 더 크게 진화할지 유추할 수 있다.
    + 미시적 관점의 연결 예측. i.e. 사람과 상품 간 구매 그래프를 통해 어떤 사람이 어떤 물건을 구매할지 유추할 수 있다.
- 군집 분석(Community Detection)
    + 연결 관계로부터 사회적 무리(social circle)을 찾아낼 수 있다. 
- 랭킹(Ranking), 정보 검색(Information Retrieval)
    + 거대한 그래프인 웹(Web)에서 어떻게 중요한 웹페이지를 찾아낼 수 있을까?
- 정보 전파(Information Casting), 바이럴 마케팅(Viral Marketing)
    + 정보는 네트워크를 통해 어떻게 전달될 것이며, 정보의 전달을 어떻게 최대화할 수 있을까?

<br />

#### 그래프의 유형
- Undirected Graph - Directed Graph
- Unweighted Graph - Weighted Graph
- Unpartite Graph - Bipartite Graph

<br />

#### 그래프의 표현
- **간선 리스트(Edge list)**: 그래프를 간선들의 리스트로 저장
    + i.e. \[(1, 2), (1, 5), (2, 3)\]
    + 방향성이 있는 경우 (출발점, 도착점) 순서로 저장
- **인접 리스트(Adjacent list)**: 각 정점의 이웃들을 리스트로 저장
    + i.e. {1: \[2, 5\], 2: \[1, 3, 5\]}
    + 방향성이 있는 경우 out-neighbor(혹은 in-neighbor)만 저장하거나 둘 모두를 저장할 수 있다.
    + 간선 리스트로 표현할 때보다 탐색에 있어 시간복잡도가 좋다.
- **인접 행렬(Adjacent Matrix)**: 정점 수 x 정점 수 크기의 행렬로 저장
    + i.e. \[\[0, 1, 1\], \[0, 1, 0\], \[1, 0, 0\]\]
    + 탐색 시 시간복잡도는 좋지만 공간 복잡도면에서 비효율적이다.
    + 희소 행렬(sparse matrix)로 이를 나타내면 정점의 수에 비해 간선의 수가 매우 적을 때 공간복잡도 면에서 효율이 좋다.  
    > 희소 행렬은 행렬 전체를 저장하지 않고 행렬에서 1인 부분의 **인덱스**를 따로 저장하는 방식을 말한다.
    + 다만 표현하고자 하는 그래프가 밀집(dense) 그래프인 경우 일반 행렬로 나타내는 방식이 공간 복잡도 면에서도 유리하다.  

<br />

## 그래프 패턴
그래프의 형태와 이를 구분할 수 있는 요소들에 대해 알아보도록 한다.  
  
<br />

#### 실제 그래프, 랜덤 그래프
- 실제 그래프: 실제 복잡계로부터 얻어진 그래프
- 랜덤 그래프: 확률적 과정을 통해 생성한 그래프  
    + 에르되스-레니 랜덤 그래프(Erdős-Rényi Random Graph)
        - 에르되스-레니 랜덤 그래프 $G(n, p)$는 $n$개의 정점을 가지며, 임의의 두 정점 사이에 간선이 존재할 확률은 $p$이며 정점 간의 연결은 서로 독립적이다.

<br />

#### 작은 세상 효과
- 정점 $u$, $v$ 간 **경로(Path)**  
    + $u$에서 시작하여 $v$까지 가는 연결된 정점들의 순열을 말한다.
    + 경로의 길이는 해당 경로에 놓이는 간선의 수이다.
- 정점 $u$, $v$ 간 **거리(Distance)**  
    + $u$와 $v$ 사이의 최단 경로의 길이
- 그래프의 **지름(Diameter)**
    + 정점 간 거리의 최댓값
- **작은 세상 효과(Small-world Effect)**
    + 여섯 단계 분리(Six Degrees of Separation) 실험을 통해 얻은 결과
    + 실제 그래프에서 임의의 두 정점 간 거리는 생각보다 크지 않다.
    + 높은 확률로 랜덤 그래프에서도 작은 세상 효과가 나타난다.
    + **모든 그래프에서 작은 세상 효과가 나타나는 것은 아니다.** i.e. 체인, 사이클 그래프, 격자 그래프 등

<br />

#### 연결성의 두터운 꼬리분포
정점의 **연결성(Degree)**는 그 정점과 연결된 간선의 수를 의미한다. 
즉, $\vert N(v) \vert$이며, $d(v)$, $d\_v$로도 나타낸다.  

- 나가는 연결성(Out Degree) 
    + $\vert N\_{out}(v) \vert$를 의미하며 $d\_{out}(v)$로도 나타낸다.
- 들어오는 연결성(In Degree) 
    + $\vert N\_{in}(v) \vert$를 의미하며 $d\_{in}(v)$로도 나타낸다.  
  
  
실제 그래프의 연결성 분포는 두터운 꼬리(Heavy tail)을 갖는다.  
![heavy_tail](/img/posts/21-1.png){: width="80%" height="80%"}{: .center}   
위와 같이 **연결성이 매우 높은 허브(Hub) 정점이 존재함을 의미**한다.   
  
대부분의 정점들은 degree가 작지만 **극소수의 정점들의 degree**가 매우 커 위 그림과 같이 꼬리를 형성한다.  
  
**하지만 랜덤 그래프의 연결성 분포는 높은 확률로 정규 분포와 유사하다.** 
확률적으로 생성되기 때문에 연결성이 매우 높은 허브가 존재할 가능성은 0에 가깝다.  
![heavy_tail2](/img/posts/21-2.png){: width="80%" height="80%"}{: .center}    

<br />

#### 군집 구조
군집(Community)이란 어떤 집합에 속하는 정점 사이에는 많은 간선이 존재하고, 그렇지 않은 정점 사이에는 적은 수의 간선이 존재하는 정점들의 집합이다. 
우리가 일반적으로 생각하는 그 군집이 맞는데, 명확한 정의를 내리기에는 애매한 것 같다. 다음은 영어로 된 군집의 정의를 발췌해온 것인데, 역시나 비슷한 의미이다.  
> Qualitatively, a community is defined as a subset of nodes within the graph such that connections between the nodes are denser than connections with the rest of the network.  
  
**지역적 군집 계수(Local clustering coefficient)**는 한 정점에서 군집의 형성 정도를 측정한다. 
어떤 정점 $i$의 지역적 군집 계수는 정점 $i$의 이웃 쌍 중 간선으로 직접 연결된 것의 비율을 의미한다.
보통 이를 $C\_i$로 표기한다.  
  
한 정점을 기준으로 이웃들 간의 간선 개수가 늘어날수록 지역적 군집 계수도 증가하고, 반대로 이웃쌍 간 간선 개수가 줄어들면 지역적 군집 계수도 감소한다.
지역적 군집 계수가 매우 높다면 그 정점과 주변 정점들은 같은 군집을 형성하고 있을 가능성이 높다. 
**연결성이 0인 정점에서는 지역적 군집 계수가 정의되지 않는다.**   
  
**전역 군집 계수(Global clustering coefficient)**는 전체 그래프에서 군집의 형성 정도를 측정한다.
그래프 $G$의 전역 군집 계수는 각 정점에서의 지역적 군집 계수의 **평균**이다.
이 때 앞서 말한 지역적 군집 계수가 정의되지 않는 정점은 여기서 제외된다.  
  
실제 그래프의 군집 계수는 보통 높은 편이다. 즉, **많은 군집이 존재한다.**
여기에는 아래와 같은 이유가 있다.  
- **동질성(Homophily)**: 서로 유사한 정점끼리 간선으로 연결될 가능성이 높다.
- **전이성(Transitivity)**: 공통 이웃이 있는 경우, 공통 이웃이 매개 역할을 해줄 수 있다.  
  
하지만 랜덤 그래프에서는 간선이 **독립적으로** 형성되기 때문에 여기서는 동질성/전이성이 나타나지 않아 **지역적/전역 군집 계수가 높지 않다.**  

<br />

#### 거대 연결 요소
연결 요소(Connected component)는 연결이 가능한 최대 크기의 정점 집합이다.  
  
실제 그래프에서는 거대 연결 요소(Giant connected component)가 존재한다. 
거대 연결 요소는 대다수의 정점을 포함한다.
![giant_connected_component](/img/posts/21-3.png){: width="80%" height="80%"}{: .center}    
  
랜덤 그래프에도 높은 확률로 거대 연결 요소가 존재한다. 
그러나 이 때 **정점들의 평균 연결성이 1보다 충분히 커야한다.**  
![giant_connected_component](/img/posts/21-4.png){: width="80%" height="80%"}{: .center}   
   
왜 그런 것인지 수학적으로 상세하게 뜯어보도록 하자.  

**작성중**

<br />

#### 그래프 별 군집 계수 및 지름 분석  
![giant_connected_component](/img/posts/21-5.png){: width="80%" height="80%"}{: .center}   
> 작은 세상 그래프는 균일 그래프에서 일부 간선을 랜덤한 간선으로 대체함으로써 얻을 수 있다.  
  
균일 그래프나 작은 세상 그래프는 대부분의 정점들이 서로 연결되어있을 가능성이 높으므로 **군집 계수가 크다.** 
반면 랜덤 그래프의 경우 앞에서 말했듯이 간선이 독립적으로 생성되므로 **군집 계수가 낮다.**  

한편 지름의 측면에서, 균일 그래프는 멀리 있는 정점끼리 연결되는 간선이 없으므로 **지름이 작다.** 
작은 세상 그래프나 랜덤 그래프는 멀리 있는 정점끼리 연결되는 간선이 높은 확률로 존재하므로 **지름이 작다.**  
  
재차 말하지만 작은 세상 그래프가 실제 세상과 가장 유사한 형태를 띄므로, 실제 그래프 역시 일반적으로 군집 계수가 크고 지름이 작다고 볼 수 있다.  

<br />

## 그래프 구현을 위한 파이썬 라이브러리  
- **NetworkX**는 속도는 느리지만 사용이 간편하다.
- **Snap.py**는 속도는 빠르지만 사용이 비교적 불편하다.  
  
둘 모두를 잘 알아두는 것이 좋으며, 여기서는 **NetworkX**에 대해서만 다루어보도록 한다.  
  
```python
# graph.py
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

G = nx.Graph()           # 방향성이 없는 그래프
DiGraph = nx.DiGraph()   # 방향성이 있는 그래프

G.add_node(1)                    # 정점 추가
G.add_node(2)
print(str(G.number_of_nodes())   # 정점의 수 반환
# 2
print(str(G.nodes))              # 정점의 목록 반환
# [1, 2]

G.add_edge(1, 2)                 # 간선 추가

# 그래프를 시각화
# 정점의 위치 결정. spring_layout 메소드가 자동으로 결정
pos = nx.spring_layout(G) 

# 정점의 색과 크기를 지정하여 출력
nx.draw_networkx_nodes(G, pos, node_color="red", node_size=100)    

# 간선 출력
nx.draw_networkx_edges(G, pos)                                          

# 각 정점의 라벨 출력 
nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")       
plt.show()

# 그래프 저장(표현)
EdgeListGraph = nx.to_edgelist(G)                  # 간선 리스트
ListGraph = nx.to_dict_of_lists(G)                 # 인접 리스트
NumpyArrayGraph = nx.to_numpy_array(G)             # 인접 행렬(일반 행렬)
SparseMatrixGraph = nx.to_scipy_sparse_matrix(G)   # 인접 행렬(희소 행렬)

# 지름/군집계수 구하기
diameter = nx.diameter(G)
average_clustering = nx.average_clustering(G)
```


<br />

## Reference  
[Defining and identifying communities in networks](https://www.pnas.org/content/101/9/2658)  
[푸아송 분포, 직관적으로 이해하기](https://danbi-ncsoft.github.io/study/2019/07/15/poisson.html)  
[Network Science - Random Network (Erdös-Rényi Network)](http://sanghyukchun.github.io/50/)  