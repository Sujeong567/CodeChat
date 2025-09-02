# Dummy 암호화 데이터 생성
pythontexts = [
    "print('Hello, world!')",
    "for i in range(5):\n    print(i)",
    "x = 10\nif x > 5:\n    print('big')",
    "def add(a, b):\n    return a + b",
    "while True:\n    break",
    "numbers = [1, 2, 3]\nfor n in numbers:\n    print(n)",
    "try:\n    val = int('abc')\nexcept ValueError:\n    print('error')",
    "with open('test.txt', 'w') as f:\n    f.write('data')",
    "import math\nprint(math.sqrt(16))",
    "class Dog:\n    def bark(self):\n        print('woof')",
    "if x == 5: return True",   # 문법 오류
    "def foo()\n    print('missing colon')",    # 문법 오류
    "print(unknown_var)",   # NameError
    "for i in range(3): print(i)",  # 한 줄 for문
    "x = [i*i for i in range(5)]",
    "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)",
    "lambda x: x**2",
    "import os\nos.listdir('.')",
    "def greet(name):\n    print(f'Hello {name}')",
    "raise Exception('manual error')"   # 의도적 예외
]

# 1 = 정상 코드, 0 = 오류 코드
labels = [
    1,  # print 정상
    1,  # for loop 정상
    1,  # if 정상
    1,  # 함수 정의 정상
    1,  # while 정상
    1,  # for-in 정상
    1,  # try-except 정상
    1,  # with open 정상
    1,  # import + math 정상
    1,  # class 정의 정상
    0,  # return 들여쓰기 오류
    0,  # 함수 정의 오류
    0,  # NameError
    1,  # 단일 for문 정상
    1,  # list comprehension 정상
    1,  # 재귀 factorial 정상
    1,  # lambda 정상
    1,  # os.listdir 정상
    1,  # 함수 greet 정상
    0   # Exception 강제 발생
]

def load_data():
    return pythontexts, labels
