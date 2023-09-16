from graphviz import Digraph
import math


class Value:
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self._prev = set(_children)
    self._op = _op
    self.grad = 0
    self._backward = lambda: None
    self.label = label

  def __repr__(self):
    if (len(self.label) > 10):
      return f"Value({self.data})"
    else:
      return f"Value({self.label})"

  def __add__(self, other):
    other = other if isinstance(
        other, Value) else Value(other, label=str(other))
    lab = self.label + "+" + other.label
    out = Value(self.data + other.data, (self, other), '+', lab)

    def _backward():
      self.grad += out.grad
      other.grad += out.grad

    out._backward = _backward
    return out

  def __mul__(self, other):
    other = other if isinstance(
        other, Value) else Value(other, label=str(other))
    lab = self.label + "*" + other.label
    out = Value(self.data * other.data, (self, other), '*', lab)

    def _backward():

      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward

    return out

  def exp(self):
    x = math.exp(self.data)
    out = Value(x, (self,), 'exp', f'exp({self})')

    def _backward():
      self.grad = x * out.grad

    out._backward = _backward

  def __pow__(self, other):
    assert isinstance(other, (int, float)
                      ), "only ints and floats are allowed for power"
    out = Value(self.data**other, (self,),
                f'**{other}', f'{self.label}**{other}')

    def _backward():
      self.grad += other * (self.data**(other - 1)) * out.grad
    out._backward = _backward
    return out

  def tanh(self):
    lab = "tanh(" + self.label + ")"
    x = self.data
    t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
    out = Value(t, (self,), 'tanh', lab)

    def _backward():
      self.grad += (1 - t**2) * out.grad

    out._backward = _backward
    return out

  def relu(self):
    out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

    def _backward():
      self.grad += (out.data > 0) * out.grad
    out._backward = _backward

    return out

  def __radd__(self, other):
    return self + other

  def __rmul__(self, other):
    return self * other

  def __truediv__(self, other):
    out = self * other**-1
    lab = other.label if isinstance(other, Value) else str(other)
    out.label = self.label + "/" + lab
    return out

  def __neg__(self):
    return self * -1

  def __sub__(self, other):
    lab = other.label if isinstance(other, Value) else str(other)
    out = self + (-other)
    out.label = self.label + '-' + lab
    return out

  def __rsub__(self, other):
    return other + (-self)

  def backward(self):
    topo = []
    vis = set()

    def build_topo(v):
      if v not in vis:
        vis.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)
    self.grad = 1
    for node in reversed(topo):
      node._backward()


def trace(root):
  nodes, edges = set(), set()

  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges


def draw_dot(root, format='svg', rankdir='LR'):
  """
  format: png | svg | ...
  rankdir: TB (top to bottom graph) | LR (left to right)
  """
  assert rankdir in ['LR', 'TB']
  nodes, edges = trace(root)
  # , node_attr={'rankdir': 'TB'})
  dot = Digraph(format=format, graph_attr={'rankdir': rankdir})

  for n in nodes:
    dot.node(name=str(id(n)), label="{%s| data %.4f | grad %.4f }" % (
        n.label, n.data, n.grad), shape='record')
    if n._op:
      dot.node(name=str(id(n)) + n._op, label=n._op)
      dot.edge(str(id(n)) + n._op, str(id(n)))

  for n1, n2 in edges:
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  return dot
