---
layout: post
title: "Day3. 자료구조와 Pythonic"
subtitle: "파이썬에서의 자료구조와 파이썬다운 코드"
date: 2021-01-20 19:27:13+0900
background: '/img/posts/bg-posts.png'
---

## 개요 <!-- omit in toc -->
> 오늘 배운 내용도 마찬가지로 concept은 모두 아는 내용이었다. 하지만 역시 여러모로 미세하게 다른 부분들이 많았다.
파이썬에만 존재하는 함수들이 여럿 있었으며 이런 함수들은 다소 생소했다.
자료구조의 경우 같은 컨셉 구현을 위해 여러 다른 자료구조를 사용할 수도 있었고 다소 생소한 자료구조도 한 두개 있었던 것 같다. 
파이썬다운 코드를 짜고, 파이썬다운 코드를 이해하기 위해서는 위에 언급한 파이썬만의 고유한 것들을 알아야한다. 
기존 C/C++ 스타일로 코드를 짜면 파이썬을 사용하는 의미가 무색해질 것이다.  
  
  
오늘은 아래 2가지 주제를 다루었는데, 각자에서 배운 내용이 상당히 많았다.
- [파이썬 자료구조](#파이썬-자료구조)
    - [스택/큐(Stack/Queue)](#스택큐stackqueue)
    - [튜플/셋(Tuple/Set)](#튜플셋tupleset)
    - [딕셔너리(Dictionary)](#딕셔너리dictionary)
    - [컬렉션(Collections)](#컬렉션collections)
- [Pythonic code](#pythonic-code)
    - [split & join](#split--join)
    - [list comprehension](#list-comprehension)
    - [enumerate](#enumerate)
    - [zip](#zip)
    - [lambda/map/reduce](#lambdamapreduce)
    - [iterable object](#iterable-object)
    - [generator](#generator)
    - [function passing argument](#function-passing-argument)


<br/>


## 파이썬 자료구조
스택, 큐, 튜플, 셋, 딕셔너리, 컬렉션(모듈)에 대하여 다루었다.


#### 스택/큐(Stack/Queue)
- Stack(스택)  
    + Stack은 LIFO(Last In First Out)로 작동한다.  
    + Python에서는 Stack의 경우 <code>append()</code>, <code>pop()</code>함수를 사용하여 단순 list로도 구현 가능하다.  

- Queue(큐)  
    + Queue는 FIFO(First In First Out)로 작동한다.  
    + 나머지는 Stack과 동일한데, <code>pop()</code>의 경우 맨 처음 들어온 인자를 제거해야하므로 <code>pop(0)</code>과 같이 <strong><code>pop</code>함수의 파라미터를 0으로</strong> 줘야한다.
  
   
  
#### 튜플/셋(Tuple/Set)
- 튜플(Tuple)  
    + Tuple은 값 변경이 불가한 리스트이다. C/C++에서 <code>const</code>로 선언된 배열과 비슷하다고 보면 될 것 같다. 따라서 튜플은 변경되지 않는 값들을 주고 받을 때 사용자의 실수를 사전에 방지할 수 있다.
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
        > <code>intersection</code>, <code>union</code>, <code>difference</code> 함수 대신 <code>&</code>, <code>|</code>, <code>-</code> 연산자를 사용해도 된다. i.e. <code>s1 & s2</code>

    + set에는 <code>remove</code>, <code>update</code>, <code>discard</code>, <code>clear</code> 등의 메소드가 존재한다. 
    + <strong><code>remove</code>는 존재하지 않는 원소를 지우려하면 에러가 발생하지만 <code>discard</code>는 같은 상황에서 에러가 발생하지 않는다는 차이가 있다.</strong> 
    + <strong>set을 활용하면 아래와 같이 list의 중복 원소를 제거하는 것도 가능하다.</strong>
        ```python
        #remove_duplicated.py
        mylist = [1, 1, 3, 4, 5, 5]
        mylist = list(set(mylist))
        print(mylist) #[1, 3, 4, 5]
        ```  
    


#### 딕셔너리(Dictionary)
- 해시 테이블과 유사한 역할을 한다. 모든 원소가 key와 value로 이루어져있다.
- dictionary에서 <code>for</code>문을 돌리면 tuple 형태로 key-value 쌍이 나오게 된다.
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
  
collections에는 많은 클래스가 있는데, 일단은 deque, defaultdict, counter, namedtuple 에 대해서 먼저 알고 가기로 한다.

- deque
    + 원래 알고있던 deque이랑 같긴한데, linked list 구현에도 사용하는 것 같다.
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
    + 딕셔너리와 같은데, <strong>딕셔너리에 없는 키값에 접근해도 에러가 발생하지 않는다.</strong> 지정하지 않은 키에 접근하려하면 그 값이 default value로 지정된다.
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
    
    + Counter간의 <code>union</code>, <code>intersection</code> 연산 등도 가능하다.


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
파이썬다운 코드를 말한다. Pythonic한 코드를 짤수록 대체로 속도가 빠르다. 또한 코드 자체가 간결하여 가독성도 좋아진다. 파이썬에 익숙하지 않기 때문에 앞으로 특히 Pythonic한 코드를 짜기 위해 신경써야 할 것 같다.

#### split & join
어제 과제를 진행하면서 <code>split</code>, <code>join</code> 함수를 적극적으로 사용해야겠다고 생각했는데 역시 이 파트에서도 언급되었다.  
  
<code>split</code> 함수는 파라미터로 들어온 문자를 기준으로 주어진 문자열을 잘라 리스트로 만드는 함수이다. <code>join</code> 함수는 반대로 파라미터로 들어온 문자를 기준으로 주어진 리스트를 문자열로 합치는 함수이다.  
    
```python
#split_and_join.py
a = "I am groot"
b = a.split() #parameter를 안주면 공백을 기준으로 자른다.
print(b) #['I', 'am', 'groot']
c = ' '.join(b)
print(c) #I am groot
```

피어세션때 알게된건데, split에 인자를 주지 않은 상태, 즉 <strong>공백으로 자르게되면 연속되는 공백은 모두 무시하게 된다. 그런데 parameter로 특정 문자를 주게 되면 그 특정 문자가 연속할 때 모두 공백으로 처리되어 리스트에 들어가게 된다.</strong>

```python
#split_more.py
a = "___EXAMPLE___WORD___"
b = "    EXTRA   SPACE   "
print(a.split('_'))
#['', '', '', 'EXAMPLE', '', '', 'WORD', '', '', '']

print(b.split())
#['EXTRA', 'SPACE']
```

#### list comprehension
list comprehension은 딱히 번역이 없어 영어로 많이 쓴다고 한다.  
리스트를 즉각적으로 생성해야 할 때 사용되며 속도도 더 빠르다고.  

List comprehension시 중첩 <code>for</code>문도 사용할 수 있으며 <code>if</code>문을 통해 filtering도 즉각적으로 가능하므로 아주 강력한 도구인 것 같다.  
안쪽 <code>for</code>문을 괄호로 감싸줌으로써 2차원 리스트도 만들 수 있다.

```python
#list_comprehension.py
result = [i for i in range(10)]
print(result) #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

word_1 = "abc"
word_2 = "abc"
result = [i + j for i in word_1 for j in word_2] #Nested For
print(result)
#['aa', 'ab', 'ac', 'ba', 'bb', 'bc', 'ca', 'cb', 'cc']

result = [i + j if not(i == j) else "X" for i in word_1 for j in word_2] #Filter
print(result)
#['X', 'ab', 'ac', 'ba', 'X', 'bc', 'ca', 'cb', 'X']

case_1 = ["A", "B", "C"]
case_2 = ["D", "E", "A"]
result = [i + j for i in case_1 for j in case_2]
print(result)
#['AD', 'AE', 'AA', 'BD', 'BE', 'BA', 'CD', 'CE', 'CA']

result = [[i + j for i in case_1] for j in case_2]
print(result)
#[['AD', 'BD', 'CD'], ['AE', 'BE', 'CE'], ['AA', 'BA', 'CA']]
```  


#### enumerate
python에서는 대부분의 <code>for</code>문에 index 값을 사용하지 않기 때문에 index를 다루면서 무언가를 하기가 쉽지 않았다. <strong>enumerate를 사용하면 index 값을 지정할 수 있다.</strong>

```python
#enumerate.py
alphabet = ['a', 'b', 'c', 'd']
#여기서 나오는 (i, v)쌍은 기본적으로 tuple 구조이다.
for i, v in enumerate(alphabet):
    print(i, v)
#0 a
#1 b
#2 c
#3 d

#아래와 같이 바로 딕셔너리를 만들 수도 있다.
dic = {v : i for i, v in enumerate(a)} #v, i의 순서를 바꿀 수도 있다.
print(dic) #{'a': 0, 'b': 1, 'c': 2, 'd': 3}
```

#### zip
zip은 두 list를 병렬적으로 추출하는 데에 활용된다. 이것도 아래 코드를 보면 바로 이해가 된다. 

```python
#zip.py
alist = ["a1", "a2", "a3"]
blist = ["b1", "b2", "b3"]
[ [a, b] for a, b in zip(alist, blist) ]
#[['a1', 'b1'], ['a2', 'b2'], ['a3', 'b3']]
[ c for c in zip(alist, blist) ] #zip은 기본적으로 tuple 구조를 뱉는다.
#[('a1', 'b1'), ('a2', 'b2'), ('a3', 'b3')]
list(enumerate(zip(alist, blist))) #enumerate와 함께 활용
#아래 결과값을 보면 enumerate도 tuple 구조를 뱉는다는 것을 다시 확인할 수 있다.
#[(0, ('a1', 'b1')), (1, ('a2', 'b2')), (2, ('a3', 'b3'))]
```


#### lambda/map/reduce
- lambda
    + lambda는 이름 없는 익명함수를 선언하여 사용하고 싶을 때 쓴다.
    + 사용법은 아래 코드와 같이 매우 간단하다. 그래서 많이 되는 것 같다. 
    + 현재 버전의 파이썬에서는 lambda 사용을 <strong>권장하지 않는다고</strong> 한다. 다만 편하기 때문에 아직 많이 사용한다고 한다. :fearful:
        ```python
        #lambda.py
        f = (lambda x, y : x + y)
        print(f(10, 50)) #60
        ```
- map
    + map은 리스트를 함수라는 필터를 거쳐 새로운 리스트에 매핑해주는 함수이다.
    + javascript의 map과 비슷한 것 같다. 두 개 이상의 리스트에도 적용할 수 있다.
    + 추가적으로 if 필터도 사용 가능하다.
        ``` python
        #map.py
        target = [1, 2, 3, 4, 5]
        #lambda 함수를 활용한 매핑
        result = map(lambda x : x ** 2, target)
        #list로의 타입캐스팅이 필요
        list(result) # [1, 4, 9, 16, 25]

        def f(x):
            return x + 5
        list(map(f, target)) # [6, 7, 8, 9, 10]

        def f2(x, y):
            return x + y
        list(map(f2, target, target)) # [2, 4, 6, 8, 10]
        ```

    + map 역시도 사용이 권장되지는 않으나 아직 쓰는 사람이 많으므로 알고 있어야 한다. 그런데 map 보면 알겠지만 사실 list comprehension으로도 모두 처리할 수 있는 내용이다. :anguished:

- reduce
    + 그러고보니 map/reduce는 hadoop 공부할때도 본 것 같다. 원리는 똑같다.
    + reduce는 인자로 들어온 함수를 이용하여 리스트의 원소들을 통합한다. (즉, 더 작은 단위로 줄인(reduce)다.)
        ```python
        #reduce.py
        from functools import reduce
        print(reduce(lambda x, y: x + y, [1,2,3,4,5])) # 15
        ```

    + reduce는 잘 안쓰지만 <strong>대용량 데이터 처리에 유용하니</strong> map/reduce 개념을 알고 있어야 한다.
    

#### iterable object
sequence 자료형에서 데이터를 차례대로 가리킬 수 있는 객체이다. (반복 가능한 객체)  
loop이 돌 때마다 매번 iterator는 다음 원소를 가리킨다.

```python
#iterator.py
cities = ["seoul", "busan", "jeju"]
memory_address_cities = iter(cities)
next(memory_address_cities) #'seoul'
next(memory_address_cities) #'busan'
next(memory_address_cities) #'jeju'
next(memory_address_cities) #ERROR!
```


#### generator
- iterator를 생성해주는 함수이며, 함수 안에 <code>yield</code> 키워드를 사용하여 loop이 돌 때마다 데이터를 올려준다. 
- 순서의 다음 값은 필요에 따라 계산된다. 미리 계산되지 않는다.
- 즉, 필요할 때마다 불러와서 쓰기 때문에 iterating하는 데이터가 많아져도(무한한 순서가 있어도) 메모리 사용량이 커지지 않는다.

    ```python
    #generator_1.py
    def test_generator():
        yield 1
        yield 2
        yield 3

    gen = test_generator()
    type(gen) #<class 'generator'>
    next(gen) #1
    next(gen) #2
    next(gen) #3
    next(gen) #StopIteration (ERROR)
    ```

- 아래 코드를 보면 generator에 대한 느낌을 확 받을 수 있다. 내부의 변수는 유지되며 필요할 때마다 generator 내부의 iterator가 돌면서 하나씩 값을 받아올 수 있다.

    ```python
    #generator_2.py
    def infinite_generator():
        count = 0
        while True:
            count += 1
            yield count

    gen = infinite_generator() 
    next(gen) #1
    next(gen) #2
    ...
    next(gen) #13
    ...
    ```

- python 3.3부터는 <code>yield from</code>을 사용할 수 있게 되면서 아래와 같은 코드 작성이 가능하다.

    ```python
    #generator_3.py
    def three_generator():
        a = [1, 2, 3]
        for i in a:
            yield i

    #위 함수를 python 3.3 이상부터는 아래와 같이 작성 가능하다.
    def three_generator():
        a = [1, 2, 3]
        yield from a
    ```

- list comprehension의 형태로 generator list를 생성할 수 있다. 이 때, 대괄호 대신 소괄호를 사용한다.

    ```python
    #generator_4.py
    a = (n * n for n in range(500))
    next(a) #0
    next(a) #1
    next(a) #4
    ...
    ```

- :exclamation: generator는 <strong>대용량 데이터에서 실사용되는 메모리를 크게 줄일 수 있으므로</strong> 적극적으로 사용하는 것이 좋으며, 잘 이해하고 있어야 한다.


#### function passing argument
- keyword argument
    + 아래와 같이 함수를 호출할 때 파라미터에 변수명을 써줄 수 있다. 따라서 명시만 해준다면, 들어오는 순서가 바뀌어도 관계가 없다.
        ```python
        #keyword_argument.py
        def get_rectangle_area(width, height):
            return width * height

        get_rectangle_area(width=5, height=8) #OK
        w = 4, h = 7
        get_rectangle_area(height=h, width=w) #순서가 바뀌어도 가능
        ```

- default argument
    + 미리 변수의 기본값을 지정할 수 있다. 만약 파라미터가 들어오지 않을 경우 기본 값을 사용한다.   
        ```python
        #default_argument.py
        def get_rectangle_area(width, height=5):
            return width * height

        get_rectangle_area(8) #40
        get_rectangle_area(4, 7) #28
        ```

    + 단, <code>def ..(width=5, height)</code>와 같이 default argument가 non-default argument보다 우선되어서는 안된다. 반드시 default argument는 맨 마지막에 나와야한다.

- variable-length argument
    + 들어오는 parameter의 갯수가 정해져있지 않을 때(즉, 가변인자일 때) 사용한다.
    + <strong>가변인자는 asterisk(<code>*</code>)를 사용하여 나타낸다.</strong>
    + 가변인자로 들어오는 값들은 tuple형태로 사용 가능하다.
        ```python
        #variable_length_argument.py
        def asterisk_test(a, b, *args):
            return a * b + sum(args)

        asterisk_test(1, 2, 3, 4, 5) #14
        ```
    
    + parameter 이름도 아래와 같이 지정할 수 있으며, 이 경우 dict type으로 사용된다.
        ```python
        #kwargs.py
        def kwargs_test_1(**kwargs):
            print(kwargs)
            print(type(kwargs))

        kwargs_test_1(first=3, second=4, third=5)
        # {'first': 3, 'second': 4, 'third': 5}
        # <class 'dict'>

        def kwargs_test_3(one, two, *args, **kwargs):
            print(one * two + sum(args)) 
            print(kwargs) 

        kwargs_test_3(3,4,5,6,7,8,9, first = 3, second = 4, third = 5)
        # 47
        # {'first': 3, 'second': 4, 'third': 5}
        ```

- asterisk(<code>*</code>)
    + 지금까지 본 바와 같이 asterisk는 다양한 용도로 사용된다. 곱 연산자, 제곱 연산자, 가변인자 등.. 
    + 마지막으로 asterisk는 unpacking container로도 사용된다.
    + 아래와 같이 tuple, dict 등에 있는 값을 풀어서 인자로 넣어줄 수 있다.

    ```python
    #asterisk_for_unpacking.py
    def asterisk_test(a, *args):
        print(a, *args)
        print(a, args)
        print(type(args))

    test = (2,3,4,5,6)
    asterisk_test(1, *test) #Unpacking 이후(총 5개) 들어감
    asterisk_test(1, test) #그냥 tuple 자체(총 1개)가 들어감

    # 1 2 3 4 5 6
    # 1 (2, 3, 4, 5, 6)
    # <class 'tuple'>
    # 1 (2, 3, 4, 5, 6)
    # 1 ((2, 3, 4, 5, 6),)
    # <class 'tuple'>
    ```

    + asterisk 1개는 sequence형을 unpacking 할 때, 2개는 dictionary에서 key/value를 unpacking 할 때 사용된다.

<br/>