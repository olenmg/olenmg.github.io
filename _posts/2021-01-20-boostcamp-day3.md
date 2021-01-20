---
layout: post
title: "Day3. 자료구조와 파이써닉"
subtitle: "파이썬에서의 자료구조와 파이썬다운 코드"
date: 2021-01-20 12:18:13+0900
background: '/img/posts/bg-posts.png'
---

## 개요
> 오늘 배운 내용도 마찬가지로 concept은 모두 아는 내용이었다. 하지만 역시 여러모로 미세하게 다른 부분들이 많았다.
파이썬에만 존재하는 함수들이 여럿 있었으며 이런 함수들은 다소 생소했다.
자료구조의 경우 같은 컨셉 구현을 위해 여러 다른 자료구조를 사용할 수도 있었고 다소 생소한 자료구조도 한 두개 있었던 것 같다. 
파이썬다운 코드를 짜고, 파이썬다운 코드를 이해하기 위해서는 위에 언급한 파이썬만의 고유한 것들을 알아야한다. 
기존 C/C++ 스타일로 코드를 짜면 파이썬을 사용하는 의미가 무색해질 것이다.  
  
  
오늘은 아래 2가지 주제를 다루었는데, 각자에서 배운 내용이 상당히 많았다.
+ [파이썬 자료구조](#파이썬-자료구조)
+ [Pythonic code](#Pythonic-code)


<br/>


## 파이썬 자료구조
스택, 큐, 튜플, 셋, 딕셔너리, 컬렉션(모듈)에 대하여 다루었다.

#### 스택/큐(Stack/Queue)
- Stack(스택)  
    + Stack은 LIFO(Last In First Out)로 작동한다.  
    + Python에서는 Stack의 경우 <code>append()</code>, <code>pop()</code>함수를 사용하여 단순 list로도 구현 가능하다.  

- Queue(큐)  
    + Queue는 FIFO(First In First Out)로 작동한다.  
    + 나머지는 Stack과 동일한데, <code>pop()</code>의 경우 맨 처음 들어온 인자를 제거해야하므로 <code>pop(0)</code>과 같이 <code>pop</code>함수의 파라미터를 0으로 줘야한다.
  
  
#### 튜플/셋(Tuple/Set)
- 튜플(Tuple)  
    + Tuple은 값 변경이 불가한 리스트이다. <code>const</code>로 선언된 어레이와 비슷하다고 보면 될 것 같다. 따라서 튜플은 변경되지 않는 값들을 주고 받을 때 사용자의 실수를 사전에 방지할 수 있다.
    + 선언시 소괄호로 선언하며(i.e. <code>t = (1, 2, 3)</code>) 메소드도 리스트에 있는 것과 거의 같다.  
    + 추가적으로 <code>t = (1)</code>과 같이 쓰는 것은 일반적인 연산시 괄호를 붙이는 것으로 인식되므로 원소가 한 개인 튜플 선언이 필요할 시 <code>t = (1, )</code>와 같이 사용해야한다.

- 셋(set)
    + 여기서 set은 집합으로, 중복을 허용하지 않는 저장 공간이다.
    + 선언시 중괄호로 선언한다. i.e. <code>s = {1, 2, 3}</code>
    + 교집합, 합집합, 차집합 등의 연산을 수행할 수 있다.

        ```python
        #set_example.py
        s1 = {1, 2, 3} # s = set([1, 2, 3])으로도 선언 가능하다.
        s2 = {2, 3, 5}
        s1.intersection(s2) #{2, 3}
        s1.union(s2) #{1, 2, 3, 5}
        s1.difference(s2) #{1}
        ```
        <code>intersection</code>, <code>union</code>, <code>difference</code> 함수 대신 <code>&</code>, <code>|</code>, <code>-</code> 연산자를 사용해도 된다.  
        i.e. <code>s1 & s2</code>

    + set에는 <code>remove</code>, <code>update</code>, <code>discard</code>, <code>clear</code> 등의 메소드가 존재한다. 
    + <code>remove</code>는 존재하지 않는 원소를 지우려하면 에러가 발생하지만 <code>discard</code>는 같은 상황에서 에러가 발생하지 않는다는 차이가 있다. 
  
#### 딕셔너리(Dictionary)
- 해시테이블과 유사한 역할을 한다. 모든 원소가 key와 value로 이루어져있다.
- dictionary에서 <code>for</code>문을 돌리면 tuple 형태로 key-value 쌍이 나오게 된다.
- 주로 <code>for</code>문에서 key, value를 뽑아낼 때 value, key 순서로 뽑아내는 것 같다. (파이썬의 관습인듯 ..? :sweat:)
- 아래와 같이 언팩킹도 할 수 있으며, key값을 index로 하여 value 수정도 된다.

    ```python
    #dictionary_example.py
    dic = {1: "car", 2: "train", 3: "bus", 4: "airplane"}
    dic[2] = "walk"
    for k, v in dic.items(): #key는 keys(), value는 values(), 둘다는 items()
        print(k, v)
    ```


#### 컬렉션(Collections)
자바에서의 컬렉션과 비슷한 것 같다. list, tuple, dict에 대한 python built-in 확장 자료구조(모듈)이다. collections를 import해서 사용한다.  

```python
#import_deque.py
from collections import deque
```
  
collections에서는 deque, defaultdict, counter, namedtuple 등을 일단 알고가자.

- deque
    + 일단 원래 알고있던 deque이랑 같긴한데, linked list 구현에도 사용하는 것 같다.
        ```python
        #deque.py
        from collections import deque
        deque_list = deque()
        for i in range(5):
            deque_list.append(i)
        deque_list.appendleft(10) #deque([10, 0, 1, 2, 3, 4])
        ```
    + <code>append</code>, <code>appendleft</code>, <code>extend</code>, <code>extendleft</code>, <code>pop</code>, <code>popleft</code>, <code>rotate</code> 등의 메소드가 존재한다.
    + <code>rotate</code>의 경우 iterate 연산시의 시작 원소의 위치를 바꾸게 된다. 양의 방향이 오른쪽 방향이다.

- defaultdict
    + 딕셔너리와 같은데, 딕셔너리에 없는 키값에 접근해도 에러가 발생하지 않는다. 지정하지 않은 키에 접근하려하면 그 값이 default value로 지정된다.
    + 다만 defaultdict는 선언시 초기값 지정이 필요하다. 자료형을 인자로 넣으면 해당 자료형의 default value가 들어가며, 그 외 직접 지정하고싶으면 <code>lambda</code>를 이용하면 된다.

        ```python
        #defaultdict.py
        from collections import defaultdict
        d_dic1 = defaultdict(int) # d = defaultdict(object)가 기본 선언 형태
        print(d_dic1["a"]) #0

        d_dic2 = defaultdict(lambda: 5)
        print(d_dic2["b"]) #5
        ```

- Counter
    + 이름 그대로 각 value가 list에 총 몇 개인지 카운팅할 수 있는 클래스이다. 별도의 반복문 없이 바로 각 단어의 반복횟수를 구할 수 있다.
    
        ```python
        #counter.py
        from collections import Counter
        counter = Counter('hello world')
        print(counter)
        #Counter({'l': 3, 'o': 2, 'h': 1, 'e': 1, ' ': 1, 'w': 1, 'r': 1, 'd': 1})
        print(counter.most_common(n=2))
        #[('l', 3), ('o', 2)]
        ```
    
    + Counter간의 union, intersection 연산 등도 가능하다.


- namedtuple
    + C/C++에서의 구조체와 비슷하다.
    + 다만 어차피 주로 클래스를 사용할 것이기 때문에 namedtuple의 형태가 필요한 경우 클래스를 사용하면 되고, 존재만 알고 있으면 된다고 한다.

        ```python
        #namedtuple.py
        from collections import namedtuple
        Point = namedtuple('Point', ['x', 'y'])
        p = Point(5, y = 3)
        print(p) #Point(x=5, y=3)
        print(p[0] + [1]) #8
        x, y = p
        print(p.x + p.y) #8
        print(x) #5
        ```
<br/>


## Pythonic code  
말그대로 파이썬다운 코드를 말한다. Pythonic한 코드를 짤수록 대체로 속도가 빠르다. 또한 코드 자체가 간결하여 가독성도 좋아진다. 파이썬에 익숙하지 않기 때문에 앞으로 특히 Pythonic한 코드를 짜기 위해 신경써야 할 것 같다.
  
작성중