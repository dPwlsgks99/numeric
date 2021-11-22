# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

# %%
tf.disable_v2_behavior()

# %%
x = np.array([1,2,3])
y = np.array([1,2,3])
m = len(x)

w = tf.placeholder(tf.float32) # w=0 전수검사

# %%
#hypothesis = tf.mul(W, X)
hypothesis = w*x # y=ax 형태

cost = tf.reduce_sum(tf.pow(hypothesis-y, 2)) / m

init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)

# %%
# 그래프로 표시하기 위해 데이터를 누적할 리스트
w_val, cost_val = [], []

# 0.1 단위로 증가할 수 없어서 -30부터 시작. 그래프에는 -3에서 5까지 표시됨.
for i in range(-30, 50):
    
    xPos = i*0.1                                    # x 좌표. -3에서 5까지 0.1씩 증가

    yPos = sess.run(cost, feed_dict={w: xPos})      # x 좌표에 따른 y 값

    print('{:3.1f}, {:3.1f}'.format(xPos, yPos))

    # 그래프에 표시할 데이터 누적. 단순히 리스트에 갯수를 늘려나감
    w_val.append(xPos)

    cost_val.append(yPos)

sess.close()

# %%
plt.plot(w_val, cost_val)

plt.xlabel('W')
plt.ylabel('cost')
plt.grid()
plt.show()

# %%



