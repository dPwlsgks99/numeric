# %%
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
plt.rc('axes', unicode_minus=False)
print(plt.rcParams['font.family'])

# %% [markdown]
# ### 확률적 경사 하강법(SGD) 동작 원리

# %% [markdown]
# 1. class로 함수 생성
# 2. keras에서 가져오기

# %%
# class로 SGD 생성
"""랜덤하게 추출한 일부 데이터에 대해 가중치를 조절한다."""
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr # 초기화 때 받는 인수
    def update(self, params, grads): # SGD과정 반복해서 불림
        for key in params.keys():
            params[key] -= self.lr * grads[key]

# %%
# class를 이용한 인공신경망 구현
network = TwoLayerNet(. . .)
optimizer = SGD() # 매개변수 갱신은 optimizer가 책임지고 수행
for i in range(10000):
. . .
x_batch, t_batch = get_mini_batch(. . .) #미니배치
grads = network.gradient(x_batch, t_batch)
params = network.params
optimizer.upate(params, grads)
. . .

# %%
# keras 패키지에서 SGD 이용
import keras
keras.optimizers.SGD(lr=0.1)

# %% [markdown]
# ### 확률적 경사 하강법(SGD) 구현

# %% [markdown]
# 간단한 학습 스케줄을 사용함.

# %%
import numpy as np

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]

# %%
theta_path_sgd = []
m = len(X_b)
np.random.seed(42)

# %%
n_epochs = 50
t0, t1 = 5, 50  # 학습 스케줄 하이퍼파라미터

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2,1)  # 무작위 초기화

for epoch in range(n_epochs):
    for i in range(m):
        if epoch == 0 and i < 20:                    
            y_predict = X_new_b.dot(theta)           
            style = "b-" if i > 0 else "r--"         
            plt.plot(X_new, y_predict, style)        
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
        theta_path_sgd.append(theta)                 

plt.plot(X, y, "b.")                                 
plt.xlabel("$x_1$", fontsize=18)                     
plt.ylabel("$y$", rotation=0, fontsize=18)           
plt.axis([0, 2, 0, 15])                                                           
plt.show()                            

# %%
theta # 정규방정식

# %%
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1, random_state=42)
sgd_reg.fit(X,y.ravel())

sgd_reg.intercept_,sgd_reg.coef_ # 정규방정식과 비슷한 값

# %% [markdown]
# ### 확률적 경사 하강법(SGD) 단위 테스트

# %%
import unittest
import os


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr 
    def update(self, params, grads): 
        for key in params.keys():
            params[key] -= self.lr * grads[key]
            
            
# TestCase를 작성
class CustomTests(unittest.TestCase):

    def setUp(self):
        """테스트 시작되기 전 파일 작성"""
        self.file_name = 'test_file.txt'
        with open(self.file_name, 'wt') as f:
            f.write("""테스트 파일""".strip())

    def tearDown(self):
        """테스트 종료 후 파일 삭제 """
        try:
            os.remove(self.file_name)
        except:
            pass

    def test_runs(self):
        """단순 실행여부 판별하는 테스트 메소드"""

        SGD(self.file_name)

    #def test_line_count(self):
    #    self.assertEqual(SGD(self.file_name), SGD)


# unittest를 실행
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

# %% [markdown]
# ### 확률적 경사 하강법(SGD) 구체화

# %% [markdown]
# keras를 사용해 2층 피드백 신경망 모델을 만들고 학습함.

# %%
import numpy as np
# 데이터 생성
np.random.seed(seed=1) # 난수를 고정
N = 200 # 데이터의 수
K = 3 # 분포의 수
T = np.zeros((N, 3), dtype=np.uint8)
X = np.zeros((N, 2))
X_range0 = [-3, 3] # X0의 범위, 표시용
X_range1 = [-3, 3] # X1의 범위, 표시용
Mu = np.array([[-.5, -.5], [.5, 1.0], [1, -.5]]) # 분포의 중심
Sig = np.array([[.7, .7], [.8, .3], [.3, .8]]) # 분포의 분산
Pi = np.array([0.4, 0.8, 1]) # 각 분포에 대한 비율
for n in range(N):
    wk = np.random.rand()
    for k in range(K):
        if wk < Pi[k]:
            T[n, k] = 1
            break
    for k in range(2):
        X[n, k] = np.random.randn() * Sig[T[n, :] == 1, k] + \
        Mu[T[n, :] == 1, k]

# %%
# 2 분류 데이터를 테스트 훈련 데이터로 분할
TestRatio = 0.5
X_n_training = int(N * TestRatio)
X_train = X[:X_n_training, :]
X_test = X[X_n_training:, :]
T_train = T[:X_n_training, :]
T_test = T[X_n_training:, :]

# %%
import time
import keras.optimizers
from keras.models import Sequential 
from keras.layers.core import Dense, Activation

# 난수 초기화
np.random.seed(1)


# Sequential 모델 작성
model = Sequential()
model.add(Dense(2, input_dim=2, activation='sigmoid',
                kernel_initializer='uniform'))
model.add(Dense(3,activation='softmax',
                kernel_initializer='uniform'))
sgd = keras.optimizers.SGD(lr=1, momentum=0.0,
                           decay=0.0, nesterov=False) 
model.compile(optimizer=sgd, loss='categorical_crossentropy',
              metrics=['accuracy'])


# 학습
startTime = time.time()
history = model.fit(X_train, T_train, epochs=1000, batch_size=100,
                    verbose=0, validation_data=(X_test, T_test)) 


# 모델 평가
score = model.evaluate(X_test, T_test, verbose=0) 
print('cross entropy {0:3.2f}, accuracy {1:3.2f}'\
      .format(score[0], score[1]))
calculation_time = time.time() - startTime
print("Calculation time:{0:.3f} sec".format(calculation_time))

# %% [markdown]
# ### 확률적 경사 하강법(SGD) 수치적 최적화

# %% [markdown]
# 1차함수와 2차함수의 최적화(진동 현상)

# %%
# 1차원 목적함수
def f1(x):
    return (x - 2) ** 2 + 2

# 목적함수 직접 미분
def f1d(x):
    """f1(x)의 도함수"""
    return 2 * (x - 2.0)

# 2차원 로젠브록 함수에 SGD 적용
def f2(x, y):
    return (1 - x)**2 + 100.0 * (y - x**2)**2

# 목적함수 미분
def f2g(x, y):
    """f2(x, y)의 도함수"""
    return np.array((2.0 * (x - 1) - 400.0 * x * (y - x**2), 200.0 * (y - x**2)))

# %%
xx = np.linspace(-1, 4, 100)

plt.plot(xx, f1(xx), 'k-')

# step size
mu = 0.4

# k = 0
x = 0
plt.plot(x, f1(x), 'go', markersize=10)
plt.text(x + 0.1, f1(x) + 0.1, "1차 시도")
plt.plot(xx, f1d(x) * (xx - x) + f1(x), 'b--')
print("1차 시도: x_1 = {:.2f}, g_1 = {:.2f}".format(x, f1d(x)))

# k = 1
x = x - mu * f1d(x)
plt.plot(x, f1(x), 'go', markersize=10)
plt.text(x - 0.2, f1(x) + 0.4, "2차 시도")
plt.plot(xx, f1d(x) * (xx - x) + f1(x), 'b--')
print("2차 시도: x_2 = {:.2f}, g_2 = {:.2f}".format(x, f1d(x)))

# k = 2
x = x - mu * f1d(x)
plt.plot(x, f1(x), 'go', markersize=10)
plt.text(x - 0.2, f1(x) - 0.7, "3차 시도")
plt.plot(xx, f1d(x) * (xx - x) + f1(x), 'b--')
print("3차 시도: x_3 = {:.2f}, g_3 = {:.2f}".format(x, f1d(x)))

plt.xlabel("x")
plt.ylabel("$f_1(x)$")
plt.title("확률적 경사 하강법을 사용한 1차함수의 최적화")
plt.ylim(0, 10)
plt.show()

# %% [markdown]
# ### 확률적 경사 하강법(SGD) 검증

# %% [markdown]
# 로젠브록 함수에 대해 확률적 경사 하강법 적용

# %%
xx = np.linspace(0, 4, 800)
yy = np.linspace(0, 3, 600)
X, Y = np.meshgrid(xx, yy)
Z = f2(X, Y)

levels = np.logspace(-1, 4, 20)

plt.contourf(X, Y, Z, alpha=0.2, levels=levels)
plt.contour(X, Y, Z, colors="green", levels=levels, zorder=0)
plt.plot(1, 1, 'ro', markersize=10)

mu = 1.8e-3  # 스텝 사이즈
s = 0.95  # 화살표 크기

x, y = 1.5, 1.5
for i in range(15):
    g = f2g(x, y)
    plt.arrow(x, y, -s * mu * g[0], -s * mu * g[1],
              head_width=0.04, head_length=0.04, fc='k', ec='k', lw=2)
    x = x - mu * g[0]
    y = y - mu * g[1]

# 그레디언트 벡터가 최저점을 가리키고 있지 않은 경우 진동 현상 발생
# SGD의 단점
plt.xlim(0, 3)
plt.ylim(0, 2)
plt.xticks(np.linspace(0, 3, 4))
plt.yticks(np.linspace(0, 2, 3))
plt.xlabel("x")
plt.ylabel("y")
plt.title("확률적 경사 하강법을 사용한 2차함수의 최적화 (진동 현상)")
plt.show()


