from micrograd.engine import Value
import random


class Module:
  def zero_grad(self):
    for p in self.parameters():
      p.grad = 0

    def parameters(self):
      return []


class Neuron(Module):

  def __init__(self, nin, nonlin=True):
    self.weights = [
        Value(random.uniform(-1, 1), label=f'Random{x}') for x in range(nin)]
    self.bias = Value(random.uniform(-1, 1), label=f'Bias')
    self.nonlin = nonlin

  def __call__(self, x):
    activation = sum((xi * wi for xi, wi in zip(self.weights, x)), self.bias)
    # print(self.nonlin)
    return activation.tanh() if self.nonlin else activation

  def parameters(self):
    return self.weights + [self.bias]


class Layer(Module):
  def __init__(self, nin, nout, **kwargs):
    self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

  def __call__(self, x):
    out = [n(x) for n in self.neurons]
    return out[0] if len(out) == 1 else out

  def parameters(self):
    return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):

  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    hidden_sz = len(nouts)
    self.layers = [Layer(sz[i], sz[i + 1], nonlin=True)
                   for i in range(hidden_sz)]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]
