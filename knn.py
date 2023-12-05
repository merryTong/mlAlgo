import numpy as np
from collections import Counter
import random
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle


def compute_distances(X, X_train):
    num_test = X.shape[0]
    num_train = X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 

    M = np.dot(X, X_train.T)
    te = np.square(X).sum(axis=1)
    tr = np.square(X_train).sum(axis=1)
    dists = np.sqrt(-2 * M + tr + np.matrix(te).T)    
    return dists

def predict_labels(y_train, dists, k=1):
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)    
    for i in range(num_test):
        # 注意 argsort 函数的用法
        labels = y_train[np.argsort(dists[i, :])].flatten()
        closest_y = labels[:k]

        c = Counter(closest_y)
        y_pred[i] = c.most_common(1)[0][0]    
    return y_pred

iris = datasets.load_iris()
X, y = shuffle(iris.data, iris.target, random_state=13)
X = X.astype(np.float32)
# 训练集与测试集的简单划分
offset = int(X.shape[0] * 0.7)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]
y_train = y_train.reshape((-1,1))
y_test = y_test.reshape((-1,1))

print('X_train=', X_train.shape)
print('X_test=', X_test.shape)
print('y_train=', y_train.shape)
print('y_test=', y_test.shape)

dists = compute_distances(X_test, X_train)
print(dists.shape)

plt.imshow(dists, interpolation='none')
plt.show()

y_test_pred = predict_labels(y_train, dists, k=1)
y_test_pred = y_test_pred.reshape((-1, 1))
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / X_test.shape[0]
print('Got %d / %d correct => accuracy: %f' % (num_correct, X_test.shape[0], accuracy))


num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []

X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)
k_to_accuracies = {}
for k in k_choices:    
    for fold in range(num_folds): 
        # 对传入的训练集单独划出一个验证集作为测试集
        validation_X_test = X_train_folds[fold]
        validation_y_test = y_train_folds[fold]
        temp_X_train = np.concatenate(X_train_folds[:fold] + X_train_folds[fold + 1:])
        temp_y_train = np.concatenate(y_train_folds[:fold] + y_train_folds[fold + 1:])       
        # 计算距离
        temp_dists = compute_distances(validation_X_test, temp_X_train)
        temp_y_test_pred = predict_labels(temp_y_train, temp_dists, k=k)
        temp_y_test_pred = temp_y_test_pred.reshape((-1, 1))       
        # 查看分类准确率
        num_correct = np.sum(temp_y_test_pred == validation_y_test)
        num_test = validation_X_test.shape[0]
        accuracy = float(num_correct) / num_test
        k_to_accuracies[k] = k_to_accuracies.get(k,[]) + [accuracy]

# 打印不同 k 值不同折数下的分类准确率
for k in sorted(k_to_accuracies):    
    print('k = %d, k fold accuracy = ' % (k), k_to_accuracies[k])

for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()

best_k = k_choices[np.argmax(accuracies_mean)]
print('best k =', best_k)