'''
2020.12.19
这段代码使用tensorflow解决y=wx+b线性回归问题
y=wx+b需要使用回归损失函数
代码使用subclass的编程方法实现
history手动编写
custom loss

'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

a = np.array([1,3,5,-2,-1])

b = np.clip(a, -1, float('inf'))

c = np.mean(a)

x = np.random.uniform(0,1,(100,1))
rand = np.random.uniform(0,1,(100,1))
y = x*0.34 + 0.78 + rand*0.0001
# print(x[0][0]*0.34+0.78+x[0][1])

print(f'x:{x} \n x shape:{x.shape}')
print(f'y:{y} \n y shape:{y.shape} y: {y}')

dataset = tf.data.Dataset.from_tensor_slices((x,y))

class Reg(tf.keras.Model):
    def __init__(self):
        super(Reg, self).__init__()
        self.layer1 = tf.keras.layers.Dense(1)
    def call(self, inputs):
        x = self.layer1(inputs)
        return x

# dataset = dataset.batch(4)

reg = Reg()

loss_fn = tf.keras.losses.MeanSquaredError()
# loss_self = tf.keras.losses.MeanSquaredLogarithmicError()
def loss_self(y_pred, y_true):
    # y_true = tf.cast(y_pred, tf.float32)
    # y_pred = y_pred - y_true
    diff = tf.subtract(y_pred,y_true)
    t = tf.square(diff)
    tt = tf.math.reduce_mean(t)
    return tt

optimizer = tf.keras.optimizers.SGD()
epochs = 20000
X,Y=[],[]
# print(reg(x))
# for x,y in dataset:
#     X.append(x)
#     Y.append(y)
# print(np.array(X).shape)

@tf.function
def train(X,Y):
    with tf.GradientTape() as tape:
        loss_ = loss_fn(reg(X),Y)
        '''
        非常奇怪的是，如果在函数内修改tensor的数据类型，函数会报错
        '''
        loss_self_ = loss_self(tf.cast(reg(X),tf.float32),tf.cast(Y,tf.float32))
    gradients = tape.gradient(loss_, reg.trainable_weights)
    optimizer.apply_gradients(zip(gradients,reg.trainable_weights))
    return loss_, tf.cast(loss_self_,tf.float64)-loss_
loss_his = []
for epoch in range(epochs):
    loss, loss_self_t = train(x,y)
    loss_his.append(loss)
    if epoch % 100 == 0:
        print(f'epoch:{epoch:4d}\t\tloss:{loss}\t\tcustom loss:{loss_self_t}')

print(reg.get_weights())
# reg.summary()

history = {'loss':loss_his}

plt.plot(history['loss'])
plt.title('Model accuracy')
plt.ylabel('Loss')
# plt.legend(['Train', 'Test'], loc='upper left')
plt.show()



