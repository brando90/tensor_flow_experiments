>>> def f(*args):
...     for arg in args:
...             print arg
...
>>> f(1)
1
>>> f(1,2)
1
2
>>> f(1,2, 3)
1
2
3
>>> def g(**kwargs):
...     for k,v in kwargs.iteritems():
...             print k,v
...
>>> g(a=1, b=2)
a 1
b 2
>>> g(**{'a':1, 'b':2})
a 1
b 2
>>> def h(*args, **kwargs):
...     pass
...
>>> h(1,2,3,4,a=7)
>>> def wrapper(myarg1, myarg2, **kwargs):
...     return pandas(myarg1, **kwargs) + myarg2
...
>>> def k(apple, **kwargs):
...     print apple
...     print kwargs
...
>>> k(1, **{'a': 2})
1
{'a': 2}
>>> k(1, **{'apple': 2})
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: k() got multiple values for keyword argument 'apple'
>>>
