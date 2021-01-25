---
layout: post
title: "Day6. numpy 및 선형대수학"
subtitle: "numpy 패키지와 선형대수학 전반에 대한 활용"
date: 2021-01-25 19:34:22+0900
background: '/img/posts/bg-posts.png'
use_math: true
---

## 개요 <!-- omit in toc -->
> matrix, vector 연산을 위한 다양한 연산을 제공하는 numpy 패키지에 대해 배우고, 이후 선형대수학의 기초적인 내용을 전반적으로 훑었다.
특히 선형대수학의 경우 한 학기동안 배웠던 내용의 절반 이상을 1시간에 압축하여 배웠다.
따라서 유도과정보다는 결과를 위주로 강의가 나왔는데, 나중을 위해 그런 중간과정도 다시 한 번 되새길 필요는 있다.

  
오늘 배운 내용은 아래와 같다.
- [numpy](#numpy)
    - [Handling shape](#handling-shape)
    - [indexing & slicing](#indexing--slicing)
    - [creation function](#creation-function)
    - [operation functions](#operation-functions)
    - [array operations](#array-operations)
    - [comparison](#comparison)
    - [boolean & fancy index](#boolean--fancy-index)
- [선형대수학(벡터, 행렬)](#선형대수학벡터-행렬)
    - [벡터](#벡터)
    - [행렬](#행렬)
- [그 외(피어세션)](#그-외피어세션)

<br/>

## numpy
- matrix/vector 등의 array 연산의 사실상 표준
- <strong>반복문 없이</strong> 데이터 배열에 대한 처리를 지원 (일반 list에 비해 빠르다)
- <code>import numpy as np</code>로 호출하는게 거의 표준이다.

#### Handling shape
- 배열 생성시 <code>np.array</code>를 이용한다. Dynamic typing을 지원하지 않기 때문에 하나의 데이터 type만 배열에 넣을 수 있다.
- <code>shape</code>, <code>dtype</code> 함수를 이용하여 dimension 구성과 데이터 type을 알 수 있다.
    ```python
    #numpy_array.py
    a = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    test_array = np.array(a)
    print(test_array.shape) # (3, 3)
    print(test_array.ndim) # 2  ... 차원 수(축 수)
    print(test_array.size) # 9  ... element 수
    ```

- <code>reshape(row, column)</code>, <code>flatten()</code> 함수를 이용하여 shape을 변환할 수 있다. (크기는 동일)
    ```python
    #reshape_flatten.py
    test_matrix = [[1,2,3,4], [1,2,5,8]]
    np.array(test_matrix).shape # (2, 4)
    print(np.array(test_matrix).reshape(4, 2)) # 원본 배열은 변화 X
    # [[1 2]
    #  [3 4]
    #  [1 2]
    #  [5 8]]

    print(np.array(test_matrix).reshape(1, -1, 2).shape)
    # (1, 4, 2)
    # -1을 인자로 넣으면 자동으로 해당 부분을 계산해준다.

    test_matrix = [[[1,2,3,4], [1,2,5,8]], [[1,2,3,4], [1,2,5,8]]]
    print(np.array(test_matrix).flatten())
    # [1 2 3 4 1 2 5 8 1 2 3 4 1 2 5 8]
    print(np.array(test_matrix).flatten().shape))
    # (16, )
    ```

#### indexing & slicing
- numpy에서는 아래와 같은 표기법을 제공한다. 앞은 row, 뒤는 column을 의미한다.
    ```python
    #indexing.py
    test_example = np.array([[1,2,3], [4.5,5,6]], int)
    print(test_example[0][2]) # 3
    print(test_example[0, 2]) # 3
    ```
- 행과 열 부분을 나눠서 slicing이 가능하다. matrix의 부분집합 추출 시 유용하다. list의 slicing처럼 step도 지정할 수 있다.
    ```python
    #slicing.py
    a = np.array([[1,2,3,4,5], [6,7,8,9,10]], int)
    print(a[:, 2:]) # 모든 행, 2열부터~
    # [[ 3  4  5]
    # [ 8  9 10]]
    
    print(a[1, 1:3])
    # [7, 8]

    print(a[1:3]) 
    # [[ 6,  7,  8,  9, 10]])
    # 범위 추출을 했기 때문에 2d array가 반환됨
    ```

#### creation function
- <code>arange(::)</code>함수를 이용하여 범위 내의 값을 가지는 list를 생성할 수 있다.
    ```python
    #arange.py
    print(list(np.arange(0, 10, 0.5)))
    # [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, ... ]
    # step 값을 실수로 줄 수 있다.
    np.arange(30).reshape(5, 6) # 이렇게 많이 쓴다.
    # array([[ 0,  1,  2,  3,  4,  5],
    #        [ 6,  7,  8,  9, 10, 11],
    #        [12, 13, 14, 15, 16, 17],
    #        [18, 19, 20, 21, 22, 23],
    #        [24, 25, 26, 27, 28, 29]])
    ```

- <code>zeros(shape, dtype, order)</code> 함수를 이용하여 0으로 가득찬 ndarray를 생성할 수 있다.
    ```python
    #zeros.py
    np.zeros(shape=(10, ), dtype=np.int8)
    # array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int8)
    ```
- <code>ones()</code>, <code>empty()</code> 함수도 <code>zeros()</code>처럼 동작하는데 각각 1로 채운 행렬, 아무것도 채우지 않은 행렬을 얻을 수 있다.
- <code>ones_like(array)</code> 함수를 이용하여 인자로 넣은 matrix와 동일한 크기의 1로 채운 행렬을 얻을 수 있다. zeros와 empty도 동일하게 사용 가능하다.
    ```python
    #ones_like.py
    test_matrix = np.arange(30).reshape(5, 6)
    np.ones_like(test_matrix)
    # array([[1, 1, 1, 1, 1, 1],
    #        [1, 1, 1, 1, 1, 1],
    #        [1, 1, 1, 1, 1, 1],
    #        [1, 1, 1, 1, 1, 1],
    #        [1, 1, 1, 1, 1, 1]])
    ```
- <code>identity(n, dtype)</code> 함수를 이용하여 단위행렬을 만들 수 있다.
    ```python
    #identity.py
    np.identity(n=3, dtype=np.int8)
    # array([[1, 0, 0],
    #        [0, 1, 0],
    #        [0, 0, 1]], dtype=int8)
    
    np.identity(5)
    # array([[1., 0., 0.],
    #        [0., 1., 0.],
    #        [0., 0., 1.]])
    ```
- <code>eye(rsize, csize, k)</code> 함수를 이용하여 rsize x csize 크기의 행렬에 k번 열부터 시작하는 대각성분의 값이 1인 행렬을 얻을 수 있다.
    ```python
    #eye.py
    np.eye(3)
    # array([[1., 0., 0.],
    #        [0., 1., 0.],
    #        [0., 0., 1.]])

    np.eye(N=3, M=5, dtype=np.int8)
    # array([[1, 0, 0, 0, 0],
    #        [0, 1, 0, 0, 0],
    #        [0, 0, 1, 0, 0]], dtype=int8)

    np.eye(3, 5, k=2)
    # array([[0., 0., 1., 0., 0.],
    #        [0., 0., 0., 1., 0.],
    #        [0., 0., 0., 0., 1.]])
    ```
- <code>diag(array)</code> 함수를 이용하여 대각 행렬의 값을 추출할 수 있다.
    ```python
    #diag.py
    matrix = np.arange(9).reshape(3, 3)
    # [[0, 1, 2], 
    #  [3, 4, 5], 
    #  [6, 7, 8]]
    np.diag(matrix)
    # array([0, 4, 8])

    np.diag(matrix, k=1)
    # array([1, 5])
    ```
- <code>random</code> 모듈 내의 메소드들로 array를 생성할 수 있다.
    ```python
    #random.py
    np.random.uniform(0, 1, 10).reshape(2, 5) # 균등분포
    # array([[0.95293434, 0.89947041, 0.9439255 ... ], [...]])

    np.random.normal(0, 1, 10).reshape(2, 5) # 정규분포
    # array([[-0.02499249, -2.32161813,  0.61860 ... ], [...]]),
    # 정규분포라서 0과 1 사이 값 외에 다른 것도 나온다

    np.random.exponential(scale=2, size=10) # 지수분포
    # array([0.01469851, 1.92952912, 0.26667749, 0.09419233 ... ])
    ```

#### operation functions
- <code>sum()</code> 함수로 element들의 합을 구할 수 있다. 
- <code>mean()</code> 함수, <code>std()</code> 함수로 평균, 표준편차 등을 구할 수 있다.
- axis 옵션을 주어 기준이 되는 dimension 축을 지정할 수 있다.
- 앞의 것부터 0번 축인데, 예를 들어 2darray에서 세로 부분이 0번 axis, 가로 부분이 1번 axis이다. 
- 축을 지정하여 해당 축 방향 성분들의 수학 연산 결과를 각각 구할 수 있다.
    ```python
    #operation_functions.py
    test_array = np.arange(1, 13).reshape(3, 4)
    # array([[ 1,  2,  3,  4],
    #        [ 5,  6,  7,  8],
    #        [ 9, 10, 11, 12]])

    test_array.sum(axis=1), test_array.sum(axis=0)
    # 해당 axis '방향'으로 합하는 것이기 때문에 갯수는 반대 axis 만큼 나온다.
    # (array([10, 26, 42]), array([15, 18, 21, 24]))

    third_order_tensor = np.array([test_array, test_array, test_array])
    # array([[[ 1,  2,  3,  4],
    #         [ 5,  6,  7,  8],
    #         [ 9, 10, 11, 12]],

    #        [[ 1,  2,  3,  4],
    #         [ 5,  6,  7,  8],
    #         [ 9, 10, 11, 12]],

    #        [[ 1,  2,  3,  4],
    #         [ 5,  6,  7,  8],
    #         [ 9, 10, 11, 12]]])

    third_order_tensor.sum(axis=1)
    # array([[15, 18, 21, 24],
    #        [15, 18, 21, 24],
    #        [15, 18, 21, 24]])

    third_order_tensor.sum(axis=0)
    # array([[ 3,  6,  9, 12],
    #        [15, 18, 21, 24],
    #        [27, 30, 33, 36]])
    ```
    
- concatenate 함수(<code>vstack</code>, <code>hstack</code>를 통해 numpy array를 붙일 수 있다.
    ```python
    #concatenate.py
    a = np.array([1, 2, 3])
    b = np.array([2, 3, 4])
    np.vstack((a, b)) # vertical
    # array([[1, 2, 3],
    #        [2, 3, 4]])

    a = np.array([ [1], [2], [3]])
    b = np.array([ [2], [3], [4]])
    np.hstack((a, b)) # horizontal
    # array([[1, 2],
    #        [2, 3],
    #        [3, 4]])

    a=np.array([[1,2,3]]) # 행벡터
    b=np.array([[2,3,4]]) # 행벡터
    np.concatenate((a,b), axis=0)
    # axis 축을 기준으로 붙인다. 
    # 붙였을때 생성되는 결과값의 axis가 0이 된다.
    # array([[1, 2, 3],
    #        [2, 3, 4]])

    a=np.array([[1,2], [3,4]])
    b=np.array([[5,6]])

    np.concatenate((a,b.T), axis=1) # T ... traspose
    # array([[1, 2, 5],
    #        [3, 4, 6]])
    ```
- 위 코드처럼 concatenate 함수들은 두 행렬을 붙일 때 축이 하나 더 있어야한다. (없으면 오류 발생)  
  그래서 계속 1darray로 가능한 것을 2darray로 썼었다. 이를 쉽게 하려면 아래와 같은 방법이 있다.
    ```python
    #concatenate_preprocessing.py
    a = np.array([[1,2], [3,4]])
    b = np.array([5, 6])
    
    #방법 1
    b.reshape(-1, 2) # array([[5, 6]])
    
    #방법 2
    b[np.newaxis, :] # array([[5, 6]])
    #newaxis 객체를 이용하여 축 추가

    np.concatenate((a, b.T), axis=1)
    # array([[1, 2, 5],
    #        [3, 4, 6]])
    ```

#### array operations
- element-wise operations들을 제공한다. (합, 차, 곱, 나누기, 모듈로 등)
- Dot production도 제공하며 <code>dot()</code> 함수를 이용한다.
    ```python
    #dot_product.py
    a = np.arange(1, 5).reshape(2, 2)
    b = np.arange(5, 9).reshape(2, 2)
    print(a.dot(b))
    # [[19 22]
    #  [43 50]]
    ```
- <code>T</code> attribute를 이용하면 transpose된 matrix를 얻을 수 있다.
    ```python
    #transpose.py
    a = np.arange(1, 7).reshape(2, 3)
    print(a)
    print(a.T) #a.transpose()
    # [[1 2 3]
    #  [4 5 6]]
    # [[1 4]
    #  [2 5]
    #  [3 6]]
    ```
- 서로 다른 크기의 행렬간 operation을 하면 broadcasting이 일어나 연산값이 퍼져나가게 된다. 
    ```python
    #broadcasting_1.py
    test_matrix = np.arange(1, 7).reshape(2, 3)
    # array([[-2, -1,  0],
    #        [ 1,  2,  3]])
    scalar = 3
    
    print(test_matrix - scalar) 
    # [[-2 -1  0]
    #  [ 1  2  3]]

    print(test_matrix * 5)
    # [[ 5 10 15]
    #  [20 25 30]]

    print(test_matrix / 5)
    # [[0.2 0.4 0.6]
    #  [0.8 1.  1.2]]

    print(test_matrix ** 5)
    # [[   1   32  243]
    #  [1024 3125 7776]]

    print(test_matrix // 5)
    # [[0 0 0]
    #  [0 1 1]]
    ```
    
    ```python
    #broadcasting_2.py
    test_matrix = np.arange(1, 13).reshape(4, 3)
    test_row = np.arange(10, 40, 10)
    test_matrix + test_row # 자동으로 확장된다.
    # array([[11, 22, 33],
    #        [14, 25, 36],
    #        [17, 28, 39],
    #        [20, 31, 42]])
    ```

#### comparison
- <code>any()</code> 함수, <code>all()</code> 함수 등을 이용하여 조건 만족 여부를 반환받을 수 있다.
- 배열 크기가 동일하면 각 element간 비교의 결과를 Boolean type으로 반환한다.
    ```python
    #any_all.py
    a = np.arange(10)
    # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    a < 4 # Boolean array가 반환된다
    # array([ True,  True,  True,  True,  False, 
    #         False, False, False, False, False])

    print(np.any(a < 4)) # True
    print(np.any(a < 0)) # False

    print(np.all(a < 4)) # False
    print(np.all(a < 10)) # True

    a = np.array([1, 3, 0], float)
    b = np.array([5, 2, 1], float)
    a > b # array([False, True, False])

    a = np.array([1, 3, 0], float)
    np.logical_and(a > 0, a < 3) # TTF & TFT = TFF
    # array([True, False, False])

    b = np.array([True, False, True], bool)
    np.logical_not(b) # TFT -> FTF
    # array([False, True, False])
    ```
- <code>where(condition, TRUE, FALSE)</code> 함수를 이용하면 각 원소별로 삼항연산자를 적용하는 효과를 얻을 수 있다. 
    ```python
    # where.py
    a = np.array([1, 3, 0], float)
    np.where(a > 0, 3, 2) # where(condition, TRUE, FALSE)
    # array([3, 3, 2])
    
    a = np.arange(10)
    np.where(a > 5) # true인 값의 index값 반환)
    # (array([6, 7, 8, 9], dtype=int64),)

    a = np.array([1, np.NaN, np.Inf], float)
    np.isnan(a) # NaN(Not Number)을 찾는다
    # array([False,  True, False])

    np.isfinite(a) # finite number을 찾는다.
    # array([ True, False, False])
    ```
- <code>argmin()</code>, <code>argmax</code> 함수를 이용하여 최댓값/최솟값의 index를 반환받을 수 있다.
    ```python
    #argmin_argmax.py
    a = np.array([1, 2, 4, 5, 8, 78, 23, 3])
    np.argmax(a), np.argmin(a) # index 값 반환
    # 5 / 0

    print(a.argsort()) # 오름차순으로 index 반환
    # [0 1 7 2 3 4 6 5]

    print(a.argsort()[::-1]) # 반대로
    # [5 6 4 3 2 7 1 0]
    ```
- axis 번호를 지정하여 axis 기반의 반환도 받을 수 있다.
    ```python
    #arg_with_axis.py
    a = np.array([[1,2,4,7], [9,88,6,45], [9,76,3,4]])
    # array([[ 1,  2,  4,  7],
    #        [ 9, 88,  6, 45],
    #        [ 9, 76,  3,  4]])

    print(np.argmax(a, axis=1))
    # [3 1 1]
    print(np.argmin(a, axis=0))
    # [0 0 2 2]
    ```

#### boolean & fancy index
- boolean index는 특정 조건에 따른 값을 배열 형태로 추출한다. 조건이 True인 index의 element만 추출한다.  
  <strong>boolean list를 사용하며 원래 대상 array와 shape이 같아야 한다.</strong>
- fancy index는 array내의 값을 index value로 사용해서 값을 추출한다.  
  <strong>integer list를 사용하며 shape은 상관 없으나, list 안의 값이 대상 array에 대하여 out of bound가 나면 안된다.</strong>
    ```python
    #boolean_fancy_index.py
    a = np.arange(0, 11)

    ## boolean index
    condition = a < 3
    a[condition] # True인 값들만 뽑아준다.
    # array([ 4,  5,  6,  7,  8,  9, 10])

    ## fancy index
    a = np.array([2, 4, 6, 8], float)
    b = np.array([0, 0, 1, 3, 2, 1], int) # index이므로 반드시 integer
    a[b] #bracket index, b 배열의 값을 index로 하여 a의 값들 추출
    # array([2., 2., 4., 8., 6., 4.])
    a.take(b) # take함수는 bracket index와 동일
    # array([2., 2., 4., 8., 6., 4.])

    # matrix 형태의 데이터도 가능하다.
    a = np.array([[1,4], [9,16]], float)
    b = np.array([0, 0, 1, 1, 0], int)
    c = np.array([0, 1, 1, 1, 1], int)
    a[b, c] #(0, 0), (0, 1), (1, 1), (1, 1), (0, 1)성분을 차례대로 추출 
    # array([ 1.,  4., 16., 16.,  4.])
    a[b] # row값만 넣어주면 row 가져옴
    # array([[ 1.,  4.],
    #        [ 1.,  4.],
    #        [ 9., 16.],
    #        [ 9., 16.],
    #        [ 1.,  4.]])
    ```
  

<br/>


## 선형대수학(벡터, 행렬)
선형대수학 전반에 대해 훑었다. 앞서 개요에서 언급했듯이 중간과정이 모두 생략되었으므로 그런 부분들은 알아서 메꿀 필요가 있다.  

#### 벡터
- 벡터는 숫자를 원소로 가지는 리스트(배열)이다.  
- 원래는 열벡터(column)를 많이 사용하였으나 numpy에서 벡터라고 하면 보통 행백터(row)이다. 즉, <strong>행벡터를 기준으로 많이 쓴다.</strong>  
- 벡터에 들어있는 entry 갯수는 벡터의 차원을 나타낸다.  
- Hadamard product(element-wise operation)는 같은 모양을 가진 벡터 간의 성분곱이다.  
  
  
- 벡터의 노름(norm)은 원점에서부터의 거리이며, $\left \Vert \cdot  \right \Vert$ 로 표기한다.  
- $L_{1}$-norm은 각 성분의 변화량의 절댓값을 모두 더한다.
- $L_{2}$-norm은 피타고라스의 정리에 의한 유클리드 거리를 계산한다.
  + $L_{1}$-norm :
  $\left \Vert x \right \Vert_{1} = \sum |x_{i}|$
  + $L_{2}$-norm :
  $\left \Vert x \right \Vert_{2} = \sqrt{\sum |x_{i}|^{2}}$
- $L_{2}$-norm에 의한 두 벡터간 각도 계산은 제2코사인법칙을 통해 한다. 
<center>  $\cos\theta = \frac{\left \Vert x \right \Vert^{2} + \left \Vert y \right \Vert^{2} - \left \Vert x-y \right \Vert^{2}}{2{\left \Vert x \right \Vert}{\left \Vert  y \right \Vert}}$ 이고  
  ${\left \Vert x \right \Vert}^{2} + {\left \Vert y \right \Vert}^{2} - {\left \Vert x-y \right \Vert}^{2} =\sum x_{i}^{2} + \sum y_{i}^{2} - \sum (x_{i} - y_{i})^{2}=2\sum x_{i} y_{i}$  
  <br/>따라서 $\cos\theta = \frac{ \langle x, y \rangle}{ {\left \Vert x \right \Vert}{\left \Vert y \right \Vert}}$ 이다. </center>   
    

- 아래와 같이 코사인 값을 구한 후, 아크코사인(코사인 역함수)을 거치면 라디안 각도값을 구할 수 있다.
    ```python
    #find_angle.py
    def angle(x, y):
        v = np.inner(x, y) / (l2_norm(x) * l2_norm(y))
        theta = np.arccos(v)
        return theta
    ```

- $\langle x, y \rangle = \lVert x \rVert \lVert y \rVert \cos \theta$이다.
- 따라서 정사영된 벡터의 길이는 $\lVert x \rVert \cos \theta$이다.

#### 행렬
- 행렬은 벡터를 원소로 가지는 <strong>2차원 배열</strong>이다.  
- <strong>numpy에서는 행(row)이 기본 단위이다.</strong>  
- $\text{X} = x_{ij}$ 에서 $i$ 는 행번호, $j$ 는 열번호를 나타낸다.  
- 행렬의 행벡터 $x_{i}$는 $i$ 번째 데이터를 의미한다.  
  같은 의미에서 $x_{ij}$ 는 $i$ 번째 데이터의 $j$ 번째 변수의 값을 말한다.  
- 행렬곱은 $i$ 번째 행벡터와 $j$ 번째 열벡터 사이의 내적을 성분으로 가지는 행렬을 찾는 것이다.  
<center>   $\text{XY} = \left( \displaystyle\sum\limits_{k} x_{ik} y_{kj} \right)$ </center>  
- numpy에서는 행렬곱을 위해 <code>@</code> 연산을 사용한다.
  
    
- numpy의 <code>np.inner</code>는 $i$번째 행벡터와 $j$번째 행벡터 사이의 내적을 성분으로 하는 행렬을 의미한다.  
<center>   $\text{XY}^{\text{T}} = \left( \displaystyle\sum\limits_{k} x_{ik} y_{jk} \right)$ </center> 
- 수학에서의 내적과는 좀 다르므로 이점에 유의한다.
  
    
- 행렬곱을 통해 벡터를 다른 차원의 공간으로 보낼 수 있다. (Linear transformation, 선형 변환) m x n 행렬 A를 이용하여 n차원 벡터 x를 m차원 벡터 z로 만들 수 있다.  
<center>  $\text{A} \text{x} = \text{z}$ </center>
- 선형 변환을 이용하여 특정 행렬에서 패턴을 추출할 수도 있고 데이터를 압축할 수도 있다.


- 역행렬은 곱해서 단위행렬(항등행렬)이 되는 행렬을 말하며, 정사각행렬이며 행렬식(det)이 0이 아닐때만 존재한다.
- 역행렬은 <code>numpy.linalg.inv</code> 함수로 구할 수 있다.

  
- 만약 역행렬을 계산할 수 없다면 유사역행렬(pseudo-inverse) 또는 무어-펜로즈(Moore-Penrose) 역행렬 $\text{A}^{+}$을 이용한다.
  + $n \geq m$인 경우(행이 더 많은 경우): $\text{A}^{+} = ( \text{A}^\text{T} \text{A} )^{-1} \text{A} ^\text{T}$
  + $n \leq m$인 경우(열이 더 많은 경우): $\text{A}^{+} =  \text{A} ^\text{T} ( \text{A}^\text{T} \text{A} )^{-1}$
- 행과 열의 갯수에 따라 곱해지는 순서가 위와 같이 달라지므로 이 점에 유의해야 한다.
- pseudo inverse matrix는 <code>numpy.linalg.pinv()</code> 함수로 구할 수 있다.



- 연립방정식의 해 및 선형회귀식도 응용하면 $\text{x} = \text{A} ^{+} \text{b}$ 로 찾을 수 있다.

<br/>



## 그 외(피어세션)
오늘은 피어세션에서 민혁님이 좋은 이야기를 많이 해주셨다. :smile:   
물론 아직 다들 배우는 입장이기 때문에 민혁님께서 해주신 말씀들이 모두 정답은 아니라고 생각한다.  
그러나 나처럼 아직 이 분야에 대해 아무런 사전지식이 없는 사람에게는 많은 참고가 되었고 감사했다.  
또한 해주신 말씀들을 들으면서 기술적 측면 외에 스스로의 게으름에 대한 경각심도 생기게 되었다.  


나도 6일차에 들어서면서 이미 느낀 부분이지만, 이 캠프는 나아갈 방향성을 제공해줄뿐 학습자에게 A부터 Z까지 모든걸 제공해주지는 않는다.  
물론 이건 비단 이 캠프만의 문제라기보다는, 세상 어느 교육 프로그램이든 마찬가지일 것이다. 학습은 결국 우리가 스스로 한다.  
5개월, P Stage를 제외하면 단 2~3개월 내에 한 분야를 총망라한다는 것은 당연히 불가능하다. :smirk:  
그래서 U Stage가 끝난 이후로도 이 분야에 대한 학습을 지속해야하며, 특히 후반부가 되기 전에 나중에 다룰 내용들에 대한 예습이 필요할 것 같다.  
  
  
민혁님이 말씀해주신 내용을 한 마디로 요약하면, 나중에는 코스 내용에 퀀텀점프가 일어나 매우 힘들거라는 말이다.  
사실 3개월에 이 내용을 다 다룬다고 했을 때부터 이건 당연한 이야기긴 했다.  
그런데 큰 기관에 시험까지 쳐서 붙었는데 교육 컨텐츠가 다 메워주겠지라는 안일한 생각 때문에 이런 점을 잊고 있었다.  
따로 학습이 확실히 필요한 것 같고, 좀 더 부지런히 움직여야 할 것 같다. :sweat_smile:


그 외에 강의에서 다루지 않은 부분에 대한 추가적인 설명을 들었고, 학습을 위한 좋은 컨텐츠들도 추천받았다.
데이터 처리를 할 때 사용할 수 있는 데이터 타입은 우리가 지금까지 배운 것보다 훨씬 많다.  
GPU가 동반되어 사용할 수 있는 데이터 타입이 물론 성능이 더 좋으나, 그렇게 사용하지 못하는 경우가 많으므로 여러가지를 알고 있는 것이 좋다.  
이 <span class="link_button">[캐글 사이트](https://www.kaggle.com/rohanrao/tutorial-on-reading-large-datasets)</span>에서 대량 데이터를 위한 현존하는 여러 데이터 타입들에 대해 소개해준다.  
pandas로는 raw data를 램 위에 올리는 것부터 막힐 수 있으므로 최적화도 많이 해보고, 그래도 안되면 여러 데이터 타입을 활용해보자. 
(근데 아마 이걸 활용하게 될 시점이 올 때까지는 아직 먼 것 같다 :sweat_smile: )  


학습하기에 좋은 컨텐츠는 아래와 같다. 다 한 번쯤은 들어본 유명한 강의/책이긴 하다. 난이도순 나열이다.   


<strong>Basic level</strong>   
<span class="link_button">
[Machine Learning by Andrew Ng](https://ko.coursera.org/learn/machine-learning) - 사실상 머신러닝계의 바이블처럼 취급되는 앤드류 응의 머신러닝 강의.   
[파이썬 머신러닝 완벽 가이드](https://bit.ly/3c7Syj3) - 캐글 커뮤니티에서 유명한 책이라고 한다.   
[인공지능 및 기계학습 개론](https://bit.ly/2LYPzyE) - KAIST 문일철 교수님의 강의로 논문 리딩을 위해 많은 도움이 된다고 한다. 어느 범위까지 수학을 공부해야하는지에 대한 궁금증을 해소할 수 있다. </span>

<strong>Advanced level</strong>   
<span class="link_button"> 
[cs231n](http://cs231n.stanford.edu/) - 컴퓨터 비전   
[cs224](http://web.stanford.edu/class/cs224n/) - 자연어처리(cs224d, cs224u)  
[cs229](http://cs229.stanford.edu/) - 머신러닝에 대한 수학적인 접근  
[cs230](https://cs230.stanford.edu/) - 딥러닝 전반에 대해 빠르게 훑고 싶을 때 </span>  


마지막으로 덧붙이자면 학습정리를 꼼꼼히 하기보다 시간을 아껴 한번이라도 더 실습을 돌리는게 좋을 것 같다.  
처음에 지나치게 꼼꼼하게 하지 말자고 다짐했는데 시간이 지날수록 점점 아니게 된다 :disappointed:  
어느정도까지만 써놓고 실습한 코드 위주로 업로드하도록 하자. :sleepy:
