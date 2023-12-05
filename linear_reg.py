import numpy as np

def forward(X, w, b):
    # 模型公式
    y_hat = np.dot(X, w) + b 
    return y_hat

def cal_loss(y, y_hat):
    # 损失函数
    loss = np.mean((y_hat-y)**2)
    return loss

def cal_grad(X, y, y_hat):
    num_train = X.shape[0]
    # 参数的偏导
    dw = np.dot(X.T, (y_hat-y)) / num_train
    db = np.mean((y_hat-y))    
    return dw, db

def initialize_params(dims):
    w = np.zeros((dims, 1))
    b = 0
    return w, b


def linear_train(X, y, learning_rate, epochs):
    w, b = initialize_params(X.shape[1])  
    loss_list = []  
    for i in range(1, epochs):        
        # 计算当前预测值、损失和参数偏导
        y_hat = forward(X, w, b)
        loss = cal_loss(y, y_hat) 
        loss_list.append(loss)      
        dw, db = cal_grad(X, y, y_hat)

        # 基于梯度下降的参数更新过程
        w += -learning_rate * dw
        b += -learning_rate * db   

        # 打印迭代次数和损失
        if i % 10000 == 0:
            print('epoch %d loss %f' % (i, loss)) 
               
        # 保存参数
        params = {            
            'w': w,            
            'b': b
        }        
        
        # 保存梯度
        grads = {            
            'dw': dw,            
            'db': db
        }    
            
    return loss_list, loss, params, grads

def predict(X, params):
    w = params['w']
    b = params['b']

    y_pred = np.dot(X, w) + b    
    return y_pred


def cross_validation(data, k, randomize=True):   
    from sklearn.utils import shuffle
    if randomize:
        data = list(data)
        shuffle(data)

    slices = [data[i::k] for i in range(k)]      
    for i in range(k):
        validation = slices[i]
        train = list()
        for s in slices:
            if s is not validation:
                train.extend(s)

        train = np.array(train)
        validation = np.array(validation)            
        yield train, validation

from sklearn.datasets import load_boston
import pandas as pd
import numpy as np

dataset = load_boston()

dataframe = pd.DataFrame(dataset["data"])
dataframe.columns = dataset["feature_names"]
dataframe["price"] = dataset["target"]
X, y = dataframe["RM"].values, dataframe["price"]
X, y = np.array(X), np.array(y)

# 训练集与测试集的简单划分
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]
X_train = X_train.reshape((-1,1))
X_test = X_test.reshape((-1,1))
y_train = y_train.reshape((-1,1))
y_test = y_test.reshape((-1,1))

print('X_train=', X_train.shape)
print('X_test=', X_test.shape)
print('y_train=', y_train.shape)
print('y_test=', y_test.shape)

loss_list, loss, params, grads = linear_train(X_train, y_train, 0.001, 100000)
print(params)

y_pred = predict(X_test, params)
print(y_pred[:5])


import matplotlib.pyplot as plt
f = X_test.dot(params['w']) + params['b']

plt.scatter(X_test, y_test)
plt.plot(X_test, f, color = 'darkorange')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

plt.plot(loss_list, color = 'blue')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

# K折交叉验证
data = np.concatenate((X.reshape(-1,1), y.reshape(-1,1)), axis=1) 
print(data.shape)

r_square_list = []
for train, validation in cross_validation(data, 5):
    X_train = train[:, :-1].reshape(-1,1)
    y_train = train[:, -1].reshape((-1, 1))
    X_valid = validation[:, :-1].reshape(-1,1)
    y_valid = validation[:, -1].reshape((-1, 1))

    loss_list, loss, params, grads = linear_train(X_train, y_train, 0.001, 100000)
    y_pred = predict(X_valid, params)
    r_square = np.mean((y_valid-y)**2)
    r_square_list.append(r_square)
print('five kold cross validation score is', r_square_list)
print('valid score is', np.mean(np.array(r_square_list)))