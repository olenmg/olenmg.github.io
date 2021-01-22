---
layout: post
title: "Day5. 예외처리, 데이터 추출"
subtitle: "try-except, 정규표현식, 로그 핸들링, 데이터 핸들링"
date: 2021-01-22 20:01:22+0900
background: '/img/posts/bg-posts.png'
---

## 개요 <!-- omit in toc -->
> try-exception(try-catch와 유사)과 파일 입출력, 로그 파일 처리, 그리고 다양한 형식의 데이터 파일을 어떻게 핸들링하는지에 대해 다루었다. 오늘 '이 주제들에 대해 10%는 배웠다'라고 말하면, 그것도 잘 쳐준 것이라고 생각한다. 그만큼 이 파트는 내용이 굉장히 방대하고, 목적을 구현하기 위한 수단이 다양하기 때문에 앞으로 메꿔야할 부분이 아주 많다.

  
오늘 배운 내용은 아래와 같다.
- [예외처리, 로그 핸들링](#예외처리-로그-핸들링)
    - [예외처리](#예외처리)
    - [파일 핸들링](#파일-핸들링)
    - [로그 핸들링](#로그-핸들링)
- [데이터 핸들링](#데이터-핸들링)
    - [CSV](#csv)
    - [html(정규표현식)](#html정규표현식)
    - [XML](#xml)
    - [JSON](#json)

<br/>

## 예외처리, 로그 핸들링
파이썬에서의 예외처리는 원래 알고 있던 것과 거의 다른 것이 없다. 다만 문제는 그 뒤인데, 파일 입출력과 로그 핸들링 관련 내용은 생소한 것이 많았다. 이를 위해 여러 모듈들이 사용되는데 내용이 좀 많았다. 피어세션을 통해 알게된건데 파일 입출력의 경우 <strong>추후 거의 pandas 모듈을 사용하여 이루어진다고</strong> 한다. 따라서 이 부분은 문법을 외우기보다는 그 과정을 이해하는 것이 중요할 것 같다.

#### 예외처리
- <code>try</code>~<code>except</code>를 이용한다.
    ```python
    #try_except.py
    try: #예외처리가 필요한 코드
        ...
    except <Exception_Type> as <alias>: #예외 발생 시
        ...(alias)...
        ...
    else: #예외 발생 안할 시
        ...
    finally: #마지막에 무조건 실행
        ...
    ```
    위 코드에서 <code>else</code>부와 <code>finally</code>부는 안써도되고, 실제로 <strong>안쓰는게 더 깔끔할 것 같다.</strong>
- 아래는 기본적으로 제공하는 예외들로 어떤 것들이 있는지만 짚고 넘어가자.  


|  Exception 이름   |              내용               |
| :---------------: | :-----------------------------: |
|    IndexError     |    OutOfBound와 동일한 에러     |
|     NameError     | 존재하지 않는 변수를 호출할 때  |
| ZeroDivisionError |          0으로 나눌 때          |
|    ValueError     | 변환할 수 없는 문자를 변환할 때 |
| FileNotFoundError |      없는 파일을 호출할 때      |
  
  
- <code>raise</code> 함수로 강제 exception을 발생시킬 수 있다.
    ```python
    #raise.py
    while ...:
        ...
        if ...:
            raise ...Error(Contents)
    ```

- <code>assert</code> 함수로 특정 조건을 만족하지 않을 시 예외를 발생시킬 수 있다.
    ```python
    #assert.py
    def get_binary_number(decimal_number)
        assert isinstance(decimal_number, int)
        # True/False를 반환하며 True가 아니면 코드를 멈춘다.
        return bin(decimal_number)
        # cf. bin()은 decimal->binary 변환 함수
    
    print(get_binary_number(10)) # 0b1010
    ```

#### 파일 핸들링
1. 파일의 종류
   - 파일은 크게 text파일과 binary파일로 나눌 수 있다.
   - text파일 처리시 컴퓨터는 text파일을 binary파일로 변환한다.
   - 모든 text파일도 실제로는 binary파일이지만, ASCII/Unicode 문자열 집합으로 저장되어 사람이 읽을 수 있는 것이다.
   - 사람이 이해할 수 있는지, 메모장을 통해 정상적으로 불러올 수 있는지 등의 기준으로 두 파일을 구분할 수 있다.

2. 파일 입출력
   - 파일 열기 모드에는 읽기모드 r, 쓰기모드 w, 추가모드 a 등이 있다.
        > 해당 모드 뒤에 b를 붙여 binary 파일을 읽고 쓸 수도 있다. 
   - 아래 두 방법을 통해 파일을 열고 닫을 수 있다.
        ```python
        #file_io_1.py
        f = open("file.txt", "r") #상대경로 입력
        contents = f.read()
        ...
        f.close()
        ```
        ```python
        #file_io_2.py
        with open("file.txt", "r") as my_file: #상대경로 입력
            contents = my_file.read()
            ...
        #close가 따로 필요하지 않다.
        ```
   - 아래와 같이 <code>readlines</code> 함수로 파일을 한 줄씩 잘라 리스트에 넣을 수 있다.
        ```python
        #readlines.py
        with open("file.txt", "r") as my_file:
            content_list = my_file.readlines()
            print(type(content_list)) # list
            print(content_list[0]) # ...\n 
        ```
   - 아래와 같이 <code>readline</code>을 통해 실행시마다 한 줄씩 읽어올 수 있다. 이렇게 하면 큰 데이터를 다룰 때 <code>readline</code> 함수를 사용할 때보다 메모리를 효율적으로 관리할 수 있을 것이다.
        ```python
        #readline.py
        with open("file.txt", "r") as my_file:
            while True:
                line = my_file.readline()
                if not line:
                    break
                line = line.replace("\n", "")
                ...
        ```
   - 파일을 불러올 때, 특히 한글이면, 아래와 같이 <strong>encoding 형태를 cp949 혹은 utf8로 입력해야 한다는 점을 기억</strong>하자.
        ```python
        f = open("text.txt", "w", encoding="utf8")
        # f = open("text.txt", "w", encoding="cp949")
        ...
        f.close()
        ```

3. 디렉토리를 다루는 모듈
   - <code>os</code> 모듈이나 <code>pathlib</code> 모듈을 통해 디렉토리를 다룰 수 있다. 특히 <code>pathlib</code> 모듈을 사용하면 path를 객체로 다룰 수 있다는 장점이 있다. 
   - <span class="link_button">[OS 모듈](https://bit.ly/3p9NzBZ)과 [pathlib 모듈](https://bit.ly/3c3dQyq)</span>은 내부 메소드가 많으니 코드를 보는 것보다도 때마다 필요한 것을 찾아 쓰는 것이 중요할 것 같다.  
   - log 파일도 위 모듈을 활용하여 폴더나 파일을 만들고, 필요할 때마다 로그파일에 지정된 텍스트를 입출력하는 방식으로 구현할 수 있다.

4. Pickle
    - 메모리에 올라온 객체가 인터프리터의 동작이 끝나도 남아있게 하기 위해(<strong>파이썬 개체의 영속화</strong>) pickle을 활용한다. 
    - built-in 객체이며 객체를 파일로 저장한다(*.pickle)
        ```python
        #pickle.py
        import pickle

        class Mutltiply(object):
            def __init__(self, multiplier):
                self.multiplier = multiplier
            def multiply(self, number):
                return number * self.multiplier

        muliply = Mutltiply(5)
        muliply.multiply(10)

        f = open("multiply_object.pickle", "wb")
        pickle.dump(muliply, f)
        f.close()

        del muliply

        f = open("multiply_object.pickle", "rb")
        multiply_pickle = pickle.load(f)
        f.close()
        multiply_pickle.multiply(5)
        ```
> 'dump'는 CS분야에서 메모리의 내용을 출력하라는 뜻도 가진다. 

#### 로그 핸들링
- 일반적으로 <strong>실행시점과 개발시점 모두에서 로그를 사용</strong>한다. (개발의 편의성/정확성 향상을 위한 로그가 필요)
- Python built-in 모듈로 <code>logging</code> 모듈이 존재한다. <code>logging</code> 모듈에는 로그를 남기는 5가지 단계가 아래와 같이 존재한다. 자세한 것은 역시 <span class="link_button">[공식 문서](https://bit.ly/3qIOiui)</span>를 보도록하자.
  
|  Level   | 개요                                                                                       |
| :------: | ------------------------------------------------------------------------------------------ |
|  debug   | 개발 시 처리 기록을 남겨야하는 로그정보를 남김                                             |
|   info   | 처리가 진행되는 동안의 정보를 알림                                                         |
| warning  | 사용자가 잘못 입력한 정보나 처리는 가능하나 원래 개발시 의도치 않은 정보가 들어왔을때 알림 |
|  error   | 잘못된 처리로 인해 에러가 났으나, 프로그램은 동작할 수 있음을 알림                         |
| critical | 잘못된 처리로 데이터 손실이나 더이상 프로그램이 동작할 수 없음을 알림                      |

- 정보를 설정하는 모듈
    + 로그를 찍을 때 필요한 정보에는 무엇이 있을까? 로그를 어느 데이터로부터 가져오는지, 로그 파일은 어디에 저장하는지 등의 정보가 있어야 할 것이다.
    + 설정할 정보가 많기 때문에 정보 설정을 간편하게 해줄 모듈이 또 필요하다.
    + 이러한 설정을 돕는 모듈에는 <code>configparser</code>, <code>argparser</code> 등이 있다.
      - <code>configparser</code>의 경우 config 파일을 저장하여 dict type으로 불러와 활용한다. 간단하지만, 또 메소드가 많기 때문에 <span class="link_button">[공식 문서](https://bit.ly/3iMPPgt)</span>를 참조하자.
      - <code>argparser</code>은 console에서 setting 정보를 입력한다. 설정을 바꾸어가며 무언가 실험을 여러 번 하고 싶을 때도 좋을 것 같다. <span class="link_button">[공식 문서](https://bit.ly/3sGComy)</span>를 보면, 이 모듈도 사용 자체는 간단하다.
  
뭔가 내용이 많아 대부분이 링크로 대체되었지만(...) 어려워서라기보다는, 사용 자체는 어렵지 않으나 내부 옵션이나 메소드가 방대하여 굳이 여기에 쓰는 것이 의미가 없다고 생각했다. :sweat_smile: 따로 이해가 필요한 부분은 거의 없기 때문에 필요할 때마다 각 모듈에 대한 정보를 찾아서 쉽게 활용할 수 있을 것이다.
<br/>

## 데이터 핸들링
실제로 다루게 될 데이터의 형식은 훨씬 다양하지만, 우선 이미 어느정도 정형화되어있어 다루기가 비교적 간단한 <strong>csv, html, xml, json 형식의 데이터를 파싱하는 방법</strong>에 대해 알아보도록 하자.

#### CSV
Comma separate Values의 약자로, 필드를 쉼표(,)로 구분한 텍스트 파일로, 테이블(엑셀) 형식의 데이터가 들어있다.  
- 쉼표 대신 탭(TSV), 빈칸(SSV) 등으로 구분해서 만들기도 하는데 모두 통칭하여 CSV라 부른다.  
- 마찬가지로 <code>with open("NAME.txt") as alias</code>으로 데이터를 읽어올 수 있을 것이다. 당연히 쉼표(,)로 데이터를 나누는 전처리 과정도 필요하다. 가장 간단하게는 <code>split(',')</code> 함수도 활용할 수 있을 것이다.
- <code>csv</code> 객체를 활용하여 더욱 섬세한 처리가 가능하다. 데이터에 쉼표가 들어가는 경우 등을 방지하기 위해 다른 delimiter(구분문자)를 활용할 수도 있고, quoting 처리를 해줄 수도 있다.

    ```python
    #csv_parsing.py
    import csv
    reader = csv.reader(f, 
            delimiter=',', quotechar='"', 
            quoting=csv.QUOTE_ALL)
    ```

|   Attribute    |    Default    |                        Meaning                        |
| :------------: | :-----------: | :---------------------------------------------------: |
|   delimiter    |       ,       |                  글자를 나누는 기준                   |
| lineterminator |     \r\n      |                      줄바꿈 기준                      |
|   quotechar    |       "       |              문자열을 둘러싸는 신호 문자              |
|    quoting     | QUOTE_MINIMAL | 데이터를 나누는 기준이 quotechar에 의해 둘러싸인 레벨 |
    
> 코드에서의 quoting 옵션 '<code>QUOTEALL</code>'은 모든 필드에 대하여 quoting을 하라는 뜻이다. 기본 옵션은 필요한 부분만(즉, 쉼표가 있는 필드에만) quoting 처리한다.  

- 추후 pandas 모듈을 활용하여 csv파일을 다룰 것이기 때문에 이 부분은 이해만 하고 넘어가면 된다.

#### html(정규표현식)
html 파일은 트리구조의 태그와 태그 내부의 컨텐츠로 이루어져있다. html파일에는 매우 많은 텍스트가 존재하며 여기서 원하는 정보를 뽑아내야 한다. (크롤링) 이를 위해 다음 세가지 방법이 존재한다.

1. 문자열 함수로 처리
2. 정규표현식(regex) 활용
3. BeautifulSoup(XML에서 많이 활용)

사실 html 전체를 문자열로 보고 그대로 함수로 처리하는 것은 직관성이 떨어지고 복잡한 처리가 필요하다. 따라서 html 처리시에는 정규표현식을 많이 활용한다.

##### 정규표현식? <!-- omit in toc -->
복잡한 문자열 패턴을 정의하는 문자 표현 공식이다. 특정한 규칙을 가진 문자열 집합을 추출하는데 유용하다. html 역시 태그로 이루어진 트리구조라는 일정한 규칙이 있기 때문에 정규표현식 적용에 적합하다.  
- 정규 표현식의 문법을 하나하나 적기에는 너무나 많다.
- <span class="link_button">[이 곳](https://bit.ly/39V8cLV)</span>에서 간단한 문법을 배우고, <span class="link_button">[이 곳](https://regexr.com/)</span>에서 연습을 해보자.
- <code>re</code> 모듈을 <code>import</code>하여 사용한다. 
- 한 개만 찾을 때는<code>search()</code> 함수를, 모두 찾을 때는 <code>findall()</code> 함수를 사용한다. 추출된 패턴은 tuple 구조로 반환된다.
- 아래는 예시로 삼성전자 주식에 대한 정보를 크롤링하는 코드이다.
    ```python
    #html_parsing.py
    import urllib.request
    import re

    url = "http://finance.naver.com/item/main.nhn?code=005930"
    html = urllib.request.urlopen(url)
    # 인코딩/디코딩 형식은 ms949/cp949/utf-8임을 기억하자.
    html_contents = str(html.read().decode("ms949"))

    stock_results = re.findall(
        "(\<dl class=\"blind\"\>)([\s\S]+?)(\<\/dl\>)", html_contents)
    samsung_stock = stock_results[0] # 두 개 tuple 값중 첫번째 패턴
    samsung_index = samsung_stock[1] # 세 개의 tuple 값중 두 번째 값

    # 하나의 괄호가 tuple index가 됨
    index_list= re.findall(
        "(\<dd\>)([\s\S]+?)(\<\/dd\>)", samsung_index)
    for index in index_list:
        print (index[1]) # 세 개의 tuple 값중 두 번째 값

    #    
    # 삼성전자 주식의 금일 거래정보들이 출력된다.
    #
    ```

#### XML
eXtensible Markup Language의 약자로, 마크업 언어이기 때문에 마찬가지로 정규표현식을 이용하여 정보를 파싱할 수 있다.  
그러나 <strong>beautifulsoup 등의 parser로 파싱하는 것이 간편해서 좀 더 일반적</strong>이다.

##### beautifulsoup <!-- omit in toc -->
- 마크업 언어 스크래핑을 위한 대표적인 도구이다.
- lxml, html5lib 등의 parser를 함께 사용한다.
- <strong>속도는 비교적 느리나, 간편하다는게 장점</strong>이다.
- 설치
    ```
    activate [ENV_NAME]
    conda install lxml
    conda install beautifulsoup
    ```
- 사용
    ```python
    # xml_parsing.py
    from bs4 import BeautifulSoup

    with open("books.xml", "r", encoding="utf8") as books_file:
        books_xml = books_file.read()
    
    #객체 생성
    soup = BeautifulSoup(books_xml, "lxml")
    #'author'태그를 찾는 함수. 이 태그가 쓰인 곳 모두를 리스트로 반환
    for book_info in soup.find_all("author"):
        ...
    ```

#### JSON
- JavaScript Object Notation의 약자로, 자바스크립트에서 사용하는 데이터 객체 표현 방식이다. 실제로 여러 사이트에서 제공하는 API로 데이터를 request하면 JSON 형식으로 response가 오는 경우가 많다.  
- 간결하며 데이터 용량이 적고 코드로의 전환이 쉽다는 장점이 있다.  
- <strong>dict type의 구조를 띤다.</strong> 간결하기 때문에 파싱이 쉽다. 아래 예시를 보면.. 바로 이해가 된다.
    ```json
    {
        "employees":[
            {
                "name":"Shyam",
                "email":"shyamjaiswal@gmail.com"
            },
            {
                "name":"Bob",
                "email":"bob32@gmail.com"
            },
            {
                "name":"Jai",
                "email":"jai87@gmail.com"
            }
        ]
    } 
    ```
- json 파일을 읽어올 수도 있지만, json 파일을 쓸 수도 있다.

    ```python
    #read_json.py
    import json
    with open("jsonfile.json", "r", encoding="utf8") as f:
        contents = f.read()
        json_data = json.loads(contents) #dict type으로 불러오기
        ... # 이후 dict type으로 활용할 수 있다.
    ```
    ```python
    #write_json.py
    import json

    dict_data = {'Name': 'Zara', 'Age': 7, 'Class': 'First'}

    with open("data.json", "w") as f:
        json.dump(dict_data, f)
    ```

<br>

## 마치며 <!-- omit in toc -->
오늘 배운 내용은 꽤 많은 것 같다. 이걸 쓰지 않더라도 추후 같은 기능을 수행하는 다른 도구를 사용할 수도 있으므로 각 모듈의 메소드를 외우기보다도 각 모듈이 동작하는 구조에 대해 잘 알아두면 좋을 것 같다. 하지만 정규표현식은 외우면 좋을 것 같다(...) :tired_face:  
