import numpy as np
import autograd as ag


base = np.random.rand(12).reshape(3,4)
z = ag.Var(base, require_grad=True)
mx = ag.max(z, axis=1)
mx.require_grad = False
tmp = (z.transpose(1,0) - mx).transpose(1,0)
a = (ag.exp(tmp).transpose(1,0) / ag.sum(ag.exp(tmp), axis=1)).transpose(1,0)
a.value = np.maximum(a.value, 1e-8)

t = (base.T - np.max(base, axis=1)).T
t = (np.exp(t).T/np.sum(np.exp(t), axis=1)).T
t = np.maximum(t, 1e-8)
print(a.value==t)

y = np.array([1,0,0,0,0,1,0,0,1,0,0,0]).reshape(3,4)
Y = ag.Var(y)
L = -ag.sum(Y*ag.log(a))

l = -np.sum(y*np.log(t))/y.shape[0]
print(l==L.value)

print(a.value - y)
L.backward()
print(z.gradient)

d_l = -y/t
g = np.zeros(shape=(3,4))
d_a = np.zeros(shape=(3,4,4))
for i in range(3):
    d_a[i] = np.diagflat(t[i]) - np.outer(t[i], t[i])
    g[i] = np.dot(d_a[i], d_l[i])
print(g)

print(a.gradient)
print(d_l)