# %%
# 2차원 로젠브록 함수
def f2(x,y):
    return(1-x)**2 + 100*(y-x**2)**2

# %%
import numpy as np
xx = np.linspace(-4,4,800)

# %%
yy=np.linspace(-3,3,600)

# %%
X,Y = np.meshgrid(xx,yy)

# %%
Z = f2(X,Y)

# %%
np.round(Z,3)

# %%
np.size(Z) # 480000개의 데이터

# %%
levels = np.logspace(-1,3,10)

# %%
import matplotlib.pyplot as plt
plt.contourf(X,Y,Z,alpha=0.2, levels=levels)

# %%
plt.contour(X,Y,Z, colors='gray', levels=[0.4,3,15,50,150,500,1500,5000])

# %%
plt.plot(1,1,'ro', markersize=10)

# %%



