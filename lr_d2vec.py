from keras import Sequential
from keras import layers
from sklearn.model_selection import train_test_split
from keras import optimizers
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import numpy
numpy.set_printoptions(precision=2)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import L1L2

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

classes = {"Software_Developer": 0, "Front_End_Developer": 1, "Network_Administrator": 2,
           "Web_Developer": 3, "Project_manager": 4, "Database_Administrator": 5, "Security_Analyst": 6,
           "Systems_Administrator": 7, "Python_Developer": 8, "Java_Developer": 9}
curr_class = "Java_Developer"

dataset='resume_d2v_dataset.csv'
data = numpy.loadtxt(open(dataset), delimiter=",")
print(data.shape)
numpy.random.shuffle(data)
ones_class_indices = numpy.where(data[:, (classes[curr_class]-10)] == 1)[0]
print(len(ones_class_indices))
zeros_class_indices = numpy.where(data[:, (classes[curr_class]-10)] == 0)[0]
d = len(ones_class_indices)
#x = numpy.where(data[:, 0] == 17088)[0]
#print(x)
#print(data[x,-10:])
#exit(0)
indices = numpy.concatenate((ones_class_indices,zeros_class_indices[:d]))
x_others = data[-indices,1:-10]
#x_others = x_others.reshape((x_others.shape[0], 500, 100))
y_others = data[-indices, (classes[curr_class]-10)]
data = data[indices]
print(data[0,0])
numpy.random.shuffle(data)
print(data.shape)
#print(data)
#dataset1 = data[:, :-10]
y = data[:, (classes[curr_class]-10)]

#train_set = (data[:800, 1:-7], data[:800, -7])
#test_set = (data[800:1200, 1:-7], data[500:1200, -7])
#valid_set = (data[1200:, 1:-7], data[1200:, -7])
batch_size = 10
X_train, X_test, y_train, y_test = train_test_split(
data, y, test_size=0.25, random_state=1000)

"""
Set up the logistic regression model
"""
model = Sequential()
model.add(layers.Dense(100, activation='relu',input_dim=300))
model.add(Dense(1,  # output dim is 2, one score per each class
                #activation='softmax',
                activation='sigmoid',
                #kernel_regularizer=L1L2(l1=0.0, l2=0.1),
                ))  # input dimension = number of features your data has
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.1, nesterov=True)
model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train[:,1:-10], y_train, epochs=100, validation_data=(X_test[:,1:-10], y_test))
loss, accuracy = model.evaluate(X_test[:,1:-10], y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

#classifier = LogisticRegression()
#classifier.fit(X_train[:,1:-10], y_train)
#score = classifier.score(X_test[:,1:-10], y_test)
#print('Accuracy : {:.4f}'.format(score))

model_json = model.to_json()
with open("d2v_models/"+curr_class+".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("d2v_models/"+curr_class+".h5")

numpy.save("d2v_models/"+curr_class,X_test)