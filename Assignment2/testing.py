from sklearn.neural_network import MLPClassifier

dimenX = 64

X = []
Y = []

with open("test.txt", "rt") as f:
    for line in f:
        x = [float(line.split(',')[i]) for i in range(dimenX)]
        y = int(line.split(',')[dimenX])
        X.append(x)
        Y.append(y)

clf = MLPClassifier(solver = 'sgd', alpha = 1e-4, activation = 'logistic',
                    hidden_layer_sizes = (4), random_state = 1,
                    batch_size = 100, max_iter = 3000)
clf.fit(X,Y)

error = 0
for i in range(len(X)):
    if clf.predict([X[i]])[0] != Y[i]:
        error += 1

print(error)
print(clf.n_iter_)
