#fp = struct('A', [Nil]*10, 'Z', [Nil]*10)
import namespaces as ns
fp = [ns.Namespace(A=None, Z=None) for _ in range(10)]
for l in range(10):
  fp[l].A = nn.eval(l)

#fp = struct('A', [Nil]*10, 'Z', [Nil]*10)
import namespaces as ns
# option 1
fp = [ns.FrozenNamespace(A=nn.eval(l), Z=nn.eval(l)) for l in range(10)]

# option 2
def activation(l):
    Z = nn.eval(l)
    A = nn.act(Z)
    return ns.FrozenNamespace(A=A, Z=Z)
fp = map(activation, range(10))

for l in range(10):
  fp[l].A = nn.eval(l)
  fp[l].Z = nn.eval(l)

# option 3
fp = []
for l in range(10):
    Z = nn.eval(l)
    A = nn.act(Z)
    fp.append(ns.FrozenNamespace(A=A, Z=Z)
