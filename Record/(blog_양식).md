---
title: '[밑바닥 딥러닝] chapter 01 - 헬로 파이썬'  
description: '[책] 밑바닥부터 시작하는 딥러닝'  
publishDate: '2025-08-29'  
author: 'Boram Son'  
tags: ['AI', 'DeepLearning', 'Python', 'Numpy', 'Matplotlib']  
# image:  
#     src: '/assets/stock-1.jpg'  
#     alt: 'Iridescent ripples of a bright blue and pink liquid'  
---


# 2025년 08월 29일 금요일 (89일차)  

# 🟩 1장. 헬로 파이썬  

## 1.3 파이썬 인터프리터  


```python
1 + 2  

# 터미널에서 파이썬 인터프리터를 사용하여 대화식으로 프로그래밍할 수 있습니다.  
# 하지만 저는 jupyter notebook을 통해서 실습을 해보겠습니다.  
```

3  



### 1.3.1 산술 연산  
```python
1 - 2  # 빼기: 1 - 2 = -1  
```
-1  


```python
4 * 5  # 곱셈: 4 x 5 = 20  
```
20  


```python
7 / 5  
# 기본적인 나누기  
# return type = float (부동소수점)  
```

1.4  




```python
7 // 5  
# 소숫점을 버린 int type으로 반환됩니다.  
```




1  




```python
type(7 // 5)  # type 확인 시 int type 확인가능  
```




int  




```python
7 % 5  # 몫이 나머지 값을 return  
```




2  




```python
3**2  # 거듭제곱 (3의 2제곱)  
```




9  



### 1.3.2 자료형  


```python
type(10)  # 10이라는 것에 타입이 뭐야? = 10이라는 숫자 즉, int(integer)입니다.  
```




int  




```python
type(2.718)  # 2.718이라는 것의 타입이 뭐야? = 소수는 float type입니다.  
```




float  




```python
type(  
"hello"  
)  # 따옴표, 쌍따옴표로 감싸진 것은 글자, 글씨 입니다. str(string) type입니다.  
```




str  



### 1.3.3 변수  


```python
x = 10  # x라는 변수를 만들었습니다. 숫자 10을 x라는 변수에 넣습니다.  
print(x)  # x라는 변수를 출력하면 10이라는 숫자가 나옵니다.  
```

10  



```python
x = 100  # 기존 x라는 변수에 100이라는 숫자를 새롭게 덮어쓰기하듯이 넣습니다.  
print(x)  # x라는 변수를 출력하면 100이라는 숫자가 들어가있습니다.  
```

100  



```python
y = 3.14  # float type의 3.14값을 y라는 변수에 넣습니다.  
x * y  
```




314.0  




```python
type(x * y)  # 정수와 실수를 곱한 결과는 float로 자동 형변환 됩니다.  
```




float  




```python
# 그렇다면 정수와 실수를 빼거나 더했을 때는 어떨까?  
print(f"빼기 ->  x - y 결과: {x - y} | type: {type(x - y)}")  
print(f"더하기 ->  x + y 결과: {x + y} | type: {type(x + y)}")  
# = 결과적으로 float 형변환이 된다.  
```

빼기 ->  x - y 결과: 96.86 | type: <class 'float'>  
더하기 ->  x + y 결과: 103.14 | type: <class 'float'>  


### 1.3.4 리스트  


```python
a = [1, 2, 3, 4, 5]  # 리스트 생성  
print(a)  # 리스트의 내용 출력  
```

[1, 2, 3, 4, 5]  



```python
len(a)  # length로 리스트의 길이 출력  
```




5  




```python
a[0]  
# 대괄호를 통해 리스트의 인덱스에 접근한다.  
# 첫 원소에 접근  
```




1  




```python
a[4]  # 다섯 번째 원소에 접근  
```




5  




```python
a[4] = 99  
# a라는 변수에 있는 리스트 내에 4번째 인덱스 값을 99로 만든다.  
# 값 대입  
print(a)  
```

[1, 2, 3, 4, 99]  



```python
# 슬라이싱 ( ~부터 : 이 값 전까지  )  
a[0:2]  # 인덱스 0부터 2까지 얻기(2번째는 포함하지 않는다!)  
```




[1, 2]  




```python
a[1:]  
# '~까지' 라는 조건이 없다면 '끝까지' 가져옴.  
# 인덱스 1부터 끝까지 얻기.  
```




[2, 3, 4, 99]  




```python
a[:3]  
# '~부터'라는 조건이 없다면 '처음부터' 가져옴.  
# 처음부터 인덱스 3까지 얻기(3번째는 포함하지 않는다!)  
```




[1, 2, 3]  




```python
a[:-1]  
# '~부터'라는 조건이 없다면 '처음부터' 가져옴.  
# 처음부터 마지막 원소의 1개 앞까지 얻기  
```




[1, 2, 3, 4]  




```python
a[:-2]  
# 처음부터 마지막 원소의 2개 앞까지 얻기  
# 뒤에서 2번째까지 (2번째는 포함하지 않는다)  
```




[1, 2, 3]  



### 1.3.5 딕셔너리  
- {}중괄호를 사용합니다.  
- 키:값 형태를 갖습니다.  


```python
me = {"height": 180}  # me라는 변수에 딕셔너리 넣습니다.(생성)  
me["height"]  # 원소에 접근 (키를 통해서 값을 알 수 있습니다.)  
```




180  




```python
me["weight"] = (  
70  # me라는 변수(dict type)에 새로운 dictionary를 추가합니다. (새 원소 추가)  
)  
print(me)  # 키와 몸무게 존재합니다.  
```

{'height': 180, 'weight': 70}  


### 1.3.6 bool  


```python
hungry = True  # 배가 고프다 라는 의미가 진실이다.  
sleepy = False  # 졸리지 않다 라는 의미가 거짓이다.  
```


```python
type(hungry)  # boolean type이다.  
```




bool  




```python
# not - 해당 boolean의 반대  
not hungry  
```




False  




```python
# and - 모두 다 True여야 True입니다.  
hungry and sleepy  # '배가 고프다' 그리고 '졸리지 않다'  
```




False  




```python
# or - 하나만 True라면 True입니다.  
hungry or sleepy  # '배가 고프다' 또는 '졸리지 않다'  
```




True  



### 1.3.7 if 문  


```python
hungry = True  

if hungry:  # 만약 hugry라는 변수가 True라면  
    print("I'm hungry")  # 이 글씨(str)를 출력해라.  
```

I'm hungry  



```python
hungry = False  

if hungry:  
    print("I'm hungry")  
else:  # 만약 hugry가 True가 아니라면  
    print("I'm not hungry")  # 이 동작을 해라.  
    print("I'm sleepy")  
```

    I'm not hungry  
    I'm sleepy  


### 1.3.8 for 문  


```python
for i in [1, 2, 3]:  # 1,2,3 각각 원소를 가진 list가 있다.  
    # for문은 iterable 특성을 가진 것들의 내부 값을 하나씩 가져와서 동작한다. or 테이터 집합의 각 원소에 차례로 접근.  
    print(i)  # 단순히 출력하고 다음 값으로 넘어갑니다.  
```

    1  
    2  
    3  


### 1.3.9 함수  


```python
def hello():  # 특정 동작을 하는 코드를 블록같은 것에 넣어 하나의 기능처럼 만든다.  
    print("Hello World!")  
    # hello() 함수를 실행하면 해당 텍스트(str)을 출력하는 기능이 전부이다.  


hello()  # Hello World! (str)을 출력함  
```

Hello World!  



```python
def hello(object):  # 함수에 매개변수를 주었다.  
    print("Hello " + object + "!")  
    # 함수 실행 시 주어진 매개변수가 들어와서 계산 및 동작함.  


hello("cat")  # 매개변수 지정 필수 (cat이라는 텍스트를 넣음)  
# print("Hello " + "cat" + "!") = Hello cat! 이 나옵니다.  
```

Hello cat!  


## 1.4 파이썬 스크립트 파일  

### 1.4.1 파일로 저장하기  

- <직접 실습하기>  
    1. 현재 디렉토리에 hungry.py 파일을 만든다.  
    2. `print("I'm hungry!")` 해당 문장 하나만 작성한다.  
    3. 저장한다.  
    4. 터미널을 열고, 터미널에서 현재 디렉토리까지 이동한다.  
    5. 터미널에 `python hungry.py`라고 입력한다.  
    6. 코드에 맞게 출력된 텍스트를 확인한다.  


```python
print("I'm hungry!")  
```

I'm hungry!  


### 1.4.2 클래스  

***ch01/man.py***  


```python
class Man:  
    def __init__(self, name):  
        # constructor(생성자)라고 합니다. name이라는 매개변수가 있습니다. (초기화 메서드)  

        self.name = name  
        # name 매개변수를 통해 들어온 name 인수를 self.name에 초기화함.  

        print("Initilized!")  # 생성자 동작 마지막 출력됨.  

    def hello(self):  
        print("Hello " + self.name + "!")  # class 내 self.name 변수를 가져와서 사용  

    def goodbye(self):  
        print("Good-bye " + self.name + "!")  # class 내 self.name 변수를 가져와서 사용  


m = Man("David")  
m.hello()  
m.goodbye()  
```

Initilized!  
Hello David!  
Good-bye David!  


<br><br>

---

## 1.5 넘파이  

### 배열의 속성들  
| 속성           | 의미               | 예시 값              |  
| ------------ | ---------------- | -------------------- |
| **dtype**    | 원소의 자료형이 무엇인지 알려줌    |   int32, float64, bool                     |  
| **shape**    | 배열의 구조 (행, 열, …)을 알려줌 |   (2, 3) → 2행 3열                          |  
| **ndim**     | 차원 수             | 1차원, 2차원, 3차원 (얼마나 감싸졌는지로 쉽게 확인 가능)       |  
| **size**     | 전체 원소 개수         | 6  (차원 구분 없이 모든 원소 갯수 세기)                   |  
| **itemsize** | 원소 하나의 크기(byte)  | 8  (배열의 각 원소가 메모리에서 차지하는 바이트 크기)        |  


### 1.5.1 넘파이 가져오기  


```python
# !uv add numpy  
```


```python
import numpy as np  
```

### 1.5.2 넘파이 배열 생성하기  


```python
x = np.array([1.0, 2.0, 3.0])  
# 파이썬 리스트를 인수로 받아 넘파이 라이브러리가 제공하는 특수한 형태의 배열(numpy.ndarray)을 반환  
# 1차원 배열(벡터) 생성  

print(x)  
```

[1. 2. 3.]  



```python
type(x)  
```




numpy.ndarray  



### 1.5.3 넘파이의 산술 연산  


```python
x = np.array([1.0, 2.0, 3.0])  # x라는 벡터  
y = np.array([2.0, 4.0, 6.0])  # y라는 벡터  

x + y  # 원소별 덧셈  
```




array([3., 6., 9.])  




```python
x - y  # 원소별 뺄셈  
```




array([-1., -2., -3.])  




```python
x * y  # 원소별 곱셈  
```




array([ 2.,  8., 18.])  




```python
x / y  # 원소별 나눗셈  
```




array([0.5, 0.5, 0.5])  




```python
x = np.array([1.0, 2.0, 3.0])  
x / 2.0  
```




array([0.5, 1. , 1.5])  



### 1.5.4 넘파이의 N차원 배열  


```python
A = np.array([[1, 2], [3, 4]])  # 2차원 배열인 행렬을 만듬 (2행 2열)  
print(A)  
```

[[1 2]  
 [3 4]]  



```python
A.shape  # 해당 배열(행렬)의 행과 열을 수를 tuple 형태로 반환합니다.  
# = 2행 2열  
```




(2, 2)  




```python
A.dtype  # 내부 원소들의 자료현이 무엇인지 파악합니다.  
```




dtype('int64')  




```python
B = np.array([[3, 0], [0, 6]])  # B라는 2차원 배열인 행렬을 만듬  
A + B  # 기본 사칙 연산 + 더하기  
```




array([[ 4,  2],  
   [ 3, 10]])  




```python
A * B  # 기본 사칙 연산 * 곱하기  
```




array([[ 3,  0],  
   [ 0, 24]])  




```python
print(A)  # 단순 A라는 배열을 출력합니다.  
```

[[1 2]  
 [3 4]]  



```python
# 브로트 캐스트  

A * 10  # A라는 행렬 * 값이 하나뿐인 스칼라  
# 형상이 다른 행렬끼리 계산이 가능합니다.  
# 브로캐스트를 통해 자동으로 numpy가 스칼라의 shape를 [[10,10], [10,10]]으로 만들어서 계산합니다.  
```




array([[10, 20],  
   [30, 40]])  



### 1.5.5 브로드캐스트  


```python
A = np.array([[1, 2], [3, 4]])  # 2차원 배열 = 행렬  
B = np.array([10, 20])  # 이 벡터의 shape가 브로드캐스트되면 [[10,20], [10,20]]입니다.  
A * B  # 행렬 shape가 (2,2) * (2,2) 와 같이 계산이 되어 정상적인 연산 결과를 확인할 수 있습니다.  
```




array([[10, 40],  
   [30, 80]])  



### 1.5.6 원소 접근  


```python
X = np.array([[51, 55], [14, 19], [0, 4]])  
print(X)  
```

[[51 55]  
 [14 19]  
 [ 0  4]]  



```python
# 마치 python list에서 인덱스로 각각의 원소에 접근했듯이 인덱스를 통해 행렬에 접근합니다.  
X[0]  # 현재 행렬의 0번쨰 값을 확인할 수 있습니다.  
```




array([51, 55])  




```python
X[0][1]  # (0, 1) 위치의 원소  
# X라는 행렬의 0번째 원소의 1번째 원소를 의미합니다.  
```




np.int64(55)  




```python
for row in X:  
    print(row)  
```

[51 55]  
[14 19]  
[0 4]  



```python
# 🆘 flatten  
X = X.flatten()  # X를 1차원 배열로 변환(평탄화)  

print(X)  

# 궁금증 test  
XXX = np.array(  
    [  
        [[51, 55], [14, 19], [0, 4]],  
        [[51, 55], [14, 19], [0, 4]],  
        [[51, 55], [14, 19], [0, 4]],  
    ]  
)  
XXX = XXX.flatten()  
print(XXX)  
```

[51 55 14 19  0  4]  
[51 55 14 19  0  4 51 55 14 19  0  4 51 55 14 19  0  4]  



```python
# X는 현재 1차원 배열임  
# 🆘 X 배열에서 인덱스 0,2,4에 해당되는 값만 별도 배열로 만듭니다.  
X[np.array([0, 2, 4])]  # 인덱스가 0, 2, 4인 원소 얻기  
# 원본을 그대로 두고, 새로운 배열을 반환.  

# ------------------------------------  

XXXXX = X[np.array([0, 2, 4])]  # 원본을 그대로 두고, 새로운 배열을 반환.  
print(XXXXX)  # 원본을 그대로 두고, 새로운 배열을 반환.  

print(X)  # 원본이 변경되지 않음을 확인.  
```

[51 14  0]  
[51 55 14 19  0  4]  



```python
# X는 현재 1차원 배열임  
X > 15  
# 🆘 X 배열에서 15보다 큰 것은 True, 작으면 False를 반환  
```




array([ True,  True, False,  True, False, False])  




```python
X[X > 15]  
# 🆘 X 배열에서 15보다 큰 값의 인덱스를 파악 -> 해당 인덱스의 값만 가지고 새로운 배열을 반환  
```




array([51, 55, 19])  



<br><br>

---

## 1.6 맷플롯립  

### 1.6.1 단순한 그래프 그리기  

***ch01/sin_graph.py***  


```python
# !uv add matplotlib  
```


```python
import numpy as np  

# np.arange(시작점(생략 시 0), 끝점(끝점 값은 미포함), step size(생략 시 1))  
x1 = np.arange(4)  
print(f"파악하기 : {x1}")  

x2 = np.arange(1, 4)  
print(f"파악하기 : {x2}")  

x3 = np.arange(1, 20, 3)  
print(f"파악하기 : {x3}")  

x4 = np.arange(30, -20, -3)  
print(f"파악하기 : {x4}")  

# -------------------------------------  

x = np.arange(0, 6, 0.1)  # 0에서 6까지 0.1 간격으로 값 생성 (배열 반환)  
print(f"\n\n{x}")  
```

파악하기 : [0 1 2 3]  
파악하기 : [1 2 3]  
파악하기 : [ 1  4  7 10 13 16 19]  
파악하기 : [ 30  27  24  21  18  15  12   9   6   3   0  -3  -6  -9 -12 -15 -18]  


[0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.  1.1 1.2 1.3 1.4 1.5 1.6 1.7  
    1.8 1.9 2.  2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.  3.1 3.2 3.3 3.4 3.5  
    3.6 3.7 3.8 3.9 4.  4.1 4.2 4.3 4.4 4.5 4.6 4.7 4.8 4.9 5.  5.1 5.2 5.3  
    5.4 5.5 5.6 5.7 5.8 5.9]  



```python
y = np.sin(x)  # x라는 배열의 값마다 sin radian방식으로 계산  
print(y)  
```

[ 0.          0.09983342  0.19866933  0.29552021  0.38941834  0.47942554  
0.56464247  0.64421769  0.71735609  0.78332691  0.84147098  0.89120736  
0.93203909  0.96355819  0.98544973  0.99749499  0.9995736   0.99166481  
0.97384763  0.94630009  0.90929743  0.86320937  0.8084964   0.74570521  
0.67546318  0.59847214  0.51550137  0.42737988  0.33498815  0.23924933  
0.14112001  0.04158066 -0.05837414 -0.15774569 -0.2555411  -0.35078323  
-0.44252044 -0.52983614 -0.61185789 -0.68776616 -0.7568025  -0.81827711  
-0.87157577 -0.91616594 -0.95160207 -0.97753012 -0.993691   -0.99992326  
-0.99616461 -0.98245261 -0.95892427 -0.92581468 -0.88345466 -0.83226744  
-0.77276449 -0.70554033 -0.63126664 -0.55068554 -0.46460218 -0.37387666]  



```python
import numpy as np  
import matplotlib.pyplot as plt  # pyplot 스타일을 사용하겠습니다.  

# 데이터 준비  
x = np.arange(0, 6, 0.1)  # 0에서 6까지 0.1 간격으로 값 생성 (배열 반환)  
y = np.sin(x)  # x라는 배열의 값마다 sin radian방식으로 계산  

# 그래프 그리기  
plt.plot(x, y)  # 그래프의 x축과 y축을 정의  
plt.show()  # 그래프를 보여줘  
```

    
![png](01_%ED%97%AC%EB%A1%9C_%ED%8C%8C%EC%9D%B4%EC%8D%AC_files/01_%ED%97%AC%EB%A1%9C_%ED%8C%8C%EC%9D%B4%EC%8D%AC_93_0.png)  


<br>

### 1.6.2 pyplot의 기능  

***ch01/sin_cos_graph.py***  


```python
import numpy as np  
import matplotlib.pyplot as plt  # pyplot 스타일을 사용하겠습니다.  

# 데이터 준비  
x = np.arange(0, 6, 0.1)  # 0에서 6까지 0.1 간격으로 값 생성 (배열 반환)  
y1 = np.sin(x)  # x라는 배열의 값마다 sin radian방식으로 계산  
y2 = np.cos(x)  # x라는 배열의 값마다 cos radian방식으로 계산  

# 그래프 그리기  
plt.plot(x, y1, label="sin")  
# x 배열과 y1 배열의 각각 인덱스별로 그려라  

plt.plot(x, y2, linestyle="--", label="cos")  
# x 배열과 y2 배열의 각각 인덱스별로 그려라  
# cos 함수는 점선으로 그리기  

plt.xlabel("x")  # x축 이름  
plt.ylabel("y")  # y축 이름  
plt.title("sin & cos")  # 제목  
plt.legend()  # 그래프 좌측 하단에 범례 추가하기  
plt.show()  # 그래프를 보여줘  
```

![png](01_%ED%97%AC%EB%A1%9C_%ED%8C%8C%EC%9D%B4%EC%8D%AC_files/01_%ED%97%AC%EB%A1%9C_%ED%8C%8C%EC%9D%B4%EC%8D%AC_96_0.png)  

<br>

### 1.6.3 이미지 표시하기  

***ch01/img_show.py***  


```python
# [노트] 맷플롯립의 imread() 메서드는 URL을 직접 처리하지 못하고,  
# 로컬 파일 경로에서 이미지를 가져오는 기능만 지원합니다.  
# 따라서 이 코드는 책 본문과 달리, urllib 라이브러리를 사용하여  
# 깃허브상의 이미지를 다운로드한 후 불러오도록 했습니다.  

import matplotlib.pyplot as plt  
from matplotlib.image import imread  # 이미지를 읽어올 수 있음.  

# import matplotlib.image as mpimg  
import urllib.request  # 특정 url에 요청을 넣을 수 있음.  

# 이미지가 존재하는 URL  
# url = "https://github.com/WegraLee/deep-learning-from-scratch/blob/master/dataset/cactus.png?raw=true"  

# 위 URL에서 이미지 다운로드 / 다운로드 할 이미지 이름은 "downloaded_image.png"로 한다.  
# urllib.request.urlretrieve(url, "downloaded_image.png")  

# 특정 경로에 존재 또는 다운로드한 이미지 읽어오기  
img = imread("practice_image/downloaded_image.jpg")  

plt.imshow(img)  # 이미지 보여줘  
# plt.show()  
```



<matplotlib.image.AxesImage at 0x114f91bd0>

![png](01_%ED%97%AC%EB%A1%9C_%ED%8C%8C%EC%9D%B4%EC%8D%AC_files/01_%ED%97%AC%EB%A1%9C_%ED%8C%8C%EC%9D%B4%EC%8D%AC_99_1.png)  



<br><br><br>

---

## Chapter 1 끝  