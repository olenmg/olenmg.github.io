---
layout: post
title: "Day2. 파이썬 기초 문법"
subtitle: "변수, 함수, I/O, 조건문, 문자열 처리"
date: 2021-01-19 21:18:13+0900
background: '/img/posts/bg-posts.png'
---

## 개요 <!-- omit in toc -->
> 파이썬의 기본 문법에 대한 내용을 배웠다. 어려운건 딱히 없었으나 아직 Python에 익숙하지 않아 문법상으로 헷갈리는 점이 많다.
아직도 과제 제출 도중 <code>if</code>에 괄호를 붙이고 콜론을 안붙이는 등의 실수가 잦아 익숙해지는 과정이 필요한 것 같다. :cry:
오늘 배운 내용에서 concept들은 이미 모두 알고 있는 것이기 때문에 C++과의 차이점을 위주로 기록하려고 한다.
  

오늘은 아래의 내용을 다루었다.
- [변수](#변수)
- [함수 및 입출력](#함수-및-입출력)
    - [함수](#함수)
    - [입출력](#입출력)
- [조건문 및 반복문](#조건문-및-반복문)
- [문자열과 함수 작성](#문자열과-함수-작성)
    - [문자열](#문자열)
    - [함수 작성](#함수-작성)
- [과제 풀이](#과제-풀이)
    
<br/>

## 변수
operator에서 기억할 점은 아래와 같다.  
1. Python에서는 <code>//</code>와 <code>/</code>이 모두 사용된다. <code>//</code>로는 int형 나눗셈 결과(즉, 몫)를 얻을 수 있으며 <code>/</code>로는 float형 나눗셈 결과를 얻을 수 있다.  
2. 또한 거듭제곱 연산 시에 <code>**</code> 연산자를 사용할 수 있다.  
3. <code>++</code>, <code>--</code> 연산자는 사용할 수 없다. <code>a += 1</code>, <code>a -= 1</code>을 대신 사용한다.  
4. 자료형 변환시에 <code>float(a)</code>, <code>str(a)</code>와 같은 방식으로 변환한다.
5. <code>long</code> 자료형의 크기에 제약이 없다 (크기가 8바이트로 정해져있지 않은 듯하다)

또한 list라는 자료구조를 배웠는데, list의 경우 array와 비슷하지만 python에서 훨씬 다루기 위한 도구가 많은 것 같다는 느낌을 받았다.

1. slicing

    ```python
    # list.py
    num = [1, 2, 3, 4, 5, 6, 7, 8]
    print(num[0:6], " AND ", num[-3:]) #[1, 2, 3, 4, 5, 6]  AND  [6, 7, 8]
    print(num[:]) #[1, 2, 3, 4, 5, 6, 7, 8]

    # 범위를 넘어갈 경우 자동으로 최대 범위 지정
    print(num[-50:50]) #[1, 2, 3, 4, 5, 6, 7, 8]

    # name[start:end:step]을 의미한다. 
    print(num[::2]) #[1, 3, 5, 7] / start, end가 생략되고 step 2인 경우 
    print(num[::-1]) #[8, 7, 6, 5, 4, 3, 2, 1]
    ```

2. concatenate

    ```python
    # concatenate.py
    num = [1, 2, 3, 4]
    print(num * 2) #[1, 2, 3, 4, 1, 2, 3, 4]
    num2 = [5, 6]
    print(num + num2) #[1, 2, 3, 4, 5, 6]
    ```

3. packing/unpacking

    ```python
    # packing_unpacking.py
    t = [1, 2, 3]
    a, b, c = t
    print(a, b, c) #1 2 3
    ```

4. 그 외
    - <code>in</code>, <code>append</code>, <code>extend</code>, <code>remove</code> 등의 함수가 있으며 이 함수들은 
    직관적으로 이해할 수 있으며 이미 알고 있는 함수이기 때문에 자세한 설명은 생략한다.
    - Python에서도 <strong>배열을 <code>=</code> 연산자로 바로 복사하는 것은 불가능</strong>하다. 주소값을 복사할 뿐이다. 따라서 이를 위해 <code>a = b[:]</code>와 같은 방식으로 복사를 해주어야한다.
    - 2차원 이상의 배열인 경우 주소값 구조 때문에 저런 방식도 완전한 복사를 해내지는 못하므로 이때는 <code>deepcopy</code> 함수를 이용한다.


<br/>


## 함수 및 입출력  
함수 및 입출력 모두 기본적인 컨셉은 같았다. 특히 강의에서 함수의 Call by value, Call by reference에 관한 내용을 길게 다루었는데, 역시 기존에 알고 있던 내용과 동일했다. 몇 가지 추가적으로 기억할 점을 아래에 메모하였다.

#### 함수
- 코드 블럭은 indentation으로 구분하며 <strong>모든 indentation의 형태는 같아야</strong> 한다.  
실제로 옛날에 django를 쓸 때 스페이스 4번과 탭 한 번을 혼용해서 한 시간을 고생했었다(...)  
옛 경험이 있어 요즘은 절대 잊지 않으려 하지만 그럼에도 실수하기 쉬운 부분이기 때문에 다시 한 번 메모해둔다. :innocent:

- 함수와 함수 사이에는 <strong>두 줄</strong>을 띄워야 한다.

    ``` python
    # function_code_convention.py
    def f1():
        ...
        ...
    # 두 줄을
    # 띄운다
    def f2():
        ...
        ...
    ```

#### 입출력 
입력의 경우 <code>input()</code> 함수를 사용한다. 그런데 <strong>input함수는 기본적으로 입력된 값을 string 타입으로 읽어오기 때문에</strong> 다음과 같이 type casting 처리해준다.

``` python
# get_input.py
int(input("Input number: ")) #input을 정수형으로 받는다.
float(input("Input float: ")) #input을 실수형으로 받는다.
```

출력의 경우 원하는 형태로 출력하는 방법에 대한 내용인데, 기본적으로 C/C++과 비슷하다.  
  
<code>%5d</code>는 정수형을 다섯 자리로 출력, <code>%8.2f</code>는 실수형을 8자리로 출력하되, 소수점 뒤는 2자리 출력하고 싶을 때 쓰는 서식문자이다. 물론 존재하지 않는 자리는 공백처리된다.  
  
다음과 같은 세가지의 출력 방법이 있는데 <string>주로 마지막(fstring)을 많이 사용</string>한다고 한다.  

``` python
# print_output.py
#1. % formatting
print("%d .... %d .... %s"%(10, 5, "abc")) # 10 .... 5 .... abc
print("%5d .... %3.3f ... %s"%(32, 3.458, "abc")) #   32 .... 3.46 ... abc

#2. format function
print("{0} and {1} and {2}".format(10, 5, "abc")) #10 and 5 and abc
print("{0} and {2} and {1}".format(10, 5, "abc")) #10 and abc and 5
print("{0: 5d} and {1: 10.2f}".format(32, 2.4589)) #   32 and       2.46

#3. fstring
a = 5
b = 2.4589
c = "abc"
print(f"{a} and {b} and {c}") #5 and 2.4589 and abc
print(f"{a:5d} and {b:10.2f}") #    5 and        2.46
```

<br/>


## 조건문 및 반복문
조건문/반복문도 이미 알고 있는 내용과 동일했다. 강의는 이 부분이 더 긴데 사실 새로 배운 점은 앞 강의들에서 더 많았던 것 같다. :unamused: 기억해야할 점은 다음과 같다.

- else if는 elif로 사용한다.
- 삼항 연산자는 <code>VALUE IF TRUE if CONDITION else VALUE IF FALSE</code>로 사용된다.
    ex) <code>c = (a == b ? 5 : 3)</code>   :arrow_right:   <code>c = 5 if a == b else 3</code>
- <code>for a in list</code> a에 list에 있는 값들을 하나씩 차례대로 할당하며 loop을 돈다.
- <code>for a in range(0, 5)</code> 0이상 5미만의 값들을 차례대로 a에 할당하며 loop을 돈다.
    <code>range(0, 5)</code>는 <code>range(5)</code>와 같은 표현이다.
- <code>for l in "abcdef"</code>, <code>for i in ["abc", "bcd", "cde"]</code>등의 표현도 물론 허용된다.
- <code>for i in range(10, 0, -1)</code>과 같이 감소도 물론 가능하다. 10에서 시작하여 0을 초과하는 동안 -1씩 감소하며 loop을 돈다.


<br/>


## 문자열과 함수 작성
여기서는 문자열과 함수의 가독성 향상을 위한 여러가지 방법에 대해 배웠다.

#### 문자열
기본적으로 문자열도 시퀀스 자료형이기 때문에 배열과 사용하는 함수나 형태가 매우 비슷하다. 다만 과제를 하면서 알게된건데, 문자열 자료형은 중간에 있는 값을 <strong>인덱스를 이용하여 수정하는게 불가능</strong>하다..  
<code>replace</code>함수를 사용하여 특정 문자를 수정하는 것은 가능한데 인덱스 사용이 불가하니 여간 불편한 점이 아니다.  
물론 다른 방법이 있을 것 같긴 한데 아직 확실한 방법은 찾지 못했다. :angry:  
일단 과제에서는 슬라이싱을 적극적으로 활용하여 인덱스 접근을 못하는 한계점들을 해결하였다.  

그 외에 기억할만한 내용은 아래와 같다.

- 여러 줄을 변수에 저장할때는 아래와 같이 한다.  
    ``` python
    # multi_line.py
    a = """ abc
        def
        ghi """
    print(a) 
    #abc
    #def
    #ghi
    ```
   
- 이스케이프 문자 없이 슬래시 등의 문자를 표현하고 싶을 때는 아래와 같이 한다.  
    ``` python
    # raw_string.py
    print(r"\/medfds") #\/medfds
    ```

<br/>

#### 함수 작성
함수를 작성할 때 가독성을 높이는 방법에 대해 다루었다. 함수 작성시 다음과 같은 점을 유의하자.
1. 함수는 짧게 여러 번 작성하는 것이 좋다. 함수 하나하나의 코드를 최대한 짧게 작성하자. 기능별로 분리하는 것이 바람직하다.
2. 함수 이름에 이 함수를 작성한 역할과 목적 등을 드러낸다. 보통 <strong>V + O(동사 + 목적어) 구조가 바람직하다.</strong> 띄어쓰기가 필요하면 언더바를 사용한다.
3. 인자로 받아온 값 자체를 바꾸지 말고 웬만하면 복사해서 사용한다. 
4. 여러번 반복되는 작업이 있거나 복잡한 수식을 쓰는 부분이 있으면 함수화해서 사용한다.

또한 코드 내에서도 가독성을 높이는 방법이 있는데, 다음과 같이 함수를 작성할 수 있다.

``` python
#good_function.py
def get_rectangle_area(width: float, height: float)->float:
    # '''
    # Input:
    #   -width: 밑변 길이 (float)
    #   -height: 높이 길이 (float)
    # Output:
    #   -사각형의 넓이 (float)
    # Examples:
    #   >>> get_rectangle_area(4.0, 2.0)
    #   8.0
    # '''
    result = width * height
    return result
```

위와 같이 <strong>parameter로 들어오는 변수들의 자료형과 반환되는 자료형을 미리 명시할 수 있다.</strong>  
또한 아래 함수의 기능을 설명하는 <strong>docstring</strong>을 덧붙이면 더 좋다. :smile:  


그 외 python coding convention을 위해 flake8, black 등의 module을 conda를 통해 설치하여 이용할 수 있을듯하다.  


<br/>


## 과제 풀이 
<strong>:exclamation: 과제의 세부 내용은 정책상 여기에 세부적으로 공유하지는 않는다. </strong>  
저녁에는 과제 리뷰 시간을 가졌는데, 미흡했다고 생각했던 점은 다음과 같다.  
  
- <code>join</code>을 사용하여 리스트를 문자열로 변환
- <code>split</code>을 사용하여 원하는 문자 삭제(띄어쓰기도 가능)
- set/dictionary 자료구조를 이용함으로써 query에서 time complexity 향상
- list comprehension 사용하여 바로 리스트 생성 (list comprehension을 사용하는 것은 권장하지만 2번 이상의 중복 사용 시에는 가독성이 떨어지기 때문에 2번 이상의 사용은 지양해야 한다.)
- <code>strip</code> 사용 (strip은 인자로 전달된 문자를 string의 왼쪽, 오른쪽에서 제거한다)
  
전체적으로 문자열과 리스트 처리에 대한 숙련도가 많이 부족함을 느꼈다. 특히 <code>join</code>과 <code>split</code>을 적극적으로 활용해야할 것 같다는 생각을 했다.  
아직 파이썬을 사용하기 시작한지 얼마 되지 않았지만 이러한 편한 도구들에 빠르게 적응해야할 것 같다.   

  
추가적으로, 부스트캠프 입과 전부터 code convention의 중요성을 여러모로 많이 느꼈었다. python code convention에 대하여 이번주 내에 쭉 훑어보고 앞으로 해당 convention에 입각하여 코드를 짜야할 것 같다. 