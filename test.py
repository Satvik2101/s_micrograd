from micrograd.nn import MLP

n = MLP(3, [4, 4, 1])
xs = [
    [2, 3, -1],
    [3, -1, 0.5],
    [0.5, 1, 1],
    [1, 1, -1],

]
ys = [1, -1, -1, 1]


i = 0
l = 1
iterations = 100
while (i < iterations):
  ypred = [n(x) for x in xs]
  loss = sum((des - pred)**2 for des, pred in zip(ys, ypred))

  n.zero_grad()
  loss.backward()

  learning_rate = 1 * (1 - 0.9 * i / iterations)
  # learning_rate=0.07
  for p in n.parameters():
    p.data -= learning_rate * p.grad
  print(learning_rate, loss.data, ypred)
  i += 1
  l = loss

print(l)
print(ypred)
