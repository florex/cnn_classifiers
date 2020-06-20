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
import itertools
import csv

def loadMatrix(file_name) :
    cv_size = 100*300
    n_class = 10
    matrix = numpy.empty((0, cv_size + n_class + 1), dtype=numpy.float16)
    i = 0
    with open(file_name, newline='') as f :
        reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for row in reader :
            #matrix = numpy.append(matrix,numpy.array([row],dtype=numpy.float16), axis=0)
            if i%1000 == 0 :
                print("Line "+ str(i))
            i+=1
    return matrix


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

classes = {0:"Front_End_Developer", 1:"Network_Administrator", 2:"Web_Developer", 3:"Project_manager", 4:"Database_Administrator",
 5:"Security_Analyst", 6:"Software_Developer", 7:"Systems_Administrator", 8:"Python_Developer", 9:"Java_Developer"}

def split_data_set(dataset) :
    writers = {}
    for k,v in classes.items() :
        writers[k] = csv.writer(open(classes[k],'w', newline=''),delimiter=',')
    i = 1
    with open(dataset, newline='') as f:
        reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            # matrix = numpy.append(matrix,numpy.array([row],dtype=numpy.float16), axis=0)
            for i in range(10) :
                if row[i] == 1 :
                    writers[i].writerow(row)
            if i % 1000 == 0:
                print("Line " + str(i))
            i += 1

    for k in writers :
        writers[k].close()

print("Start loading the matrix....")
dataset='resume_dataset.csv'
#data = numpy.loadtxt(dataset, delimiter=",", dtype=numpy.float16)
split_data_set(dataset)
data = loadMatrix(dataset)
print(data.shape)
numpy.random.shuffle(data)
ones_class_indices = numpy.where(data[:, -1] == 1)[0]
zeros_class_indices = numpy.where(data[:, -1] == 0)[0]
d = len(ones_class_indices)
indices = numpy.concatenate((ones_class_indices,zeros_class_indices[:d]))
x_others = data[-indices,1:-10]
x_others = x_others.reshape((x_others.shape[0], 300, 100))
y_others = data[-indices, -1]
data = data[indices]

numpy.random.shuffle(data)
print(data)
dataset1 = data[:, 1:-10]
y = data[:, -1]

#train_set = (data[:800, 1:-7], data[:800, -7])
#test_set = (data[800:1200, 1:-7], data[500:1200, -7])
#valid_set = (data[1200:, 1:-7], data[1200:, -7])
batch_size = 64
X_train, X_test, y_train, y_test = train_test_split(
dataset1, y, test_size=0.25, random_state=1000)
test_ones_indices = numpy.where(y_test == 1)[0]
Y_test_ones = y_test[test_ones_indices];
X_test_ones = X_test[test_ones_indices];
Y_test_zeros = y_test[numpy.where(y_test == 0)[0]];
X_test_zeros = X_test[numpy.where(y_test == 0)[0]];
print(X_test.shape)
X_train = X_train.reshape((X_train.shape[0], 300, 100))
X_test = X_test.reshape((X_test.shape[0], 300, 100))
X_test_ones = X_test_ones.reshape((X_test_ones.shape[0], 300, 100))
X_test_zeros = X_test_zeros.reshape((X_test_zeros.shape[0], 300, 100))
#classifier = LogisticRegression()
#classifier.fit(X_train, y_train)
#score = classifier.score(X_test, y_test)
print(X_test_ones.shape)
print(X_test_zeros.shape)
print(Y_test_ones)
print(Y_test_zeros)
input_dim = X_train.shape[1]
model = Sequential()
#model.add(layers.Embedding(300,100,input_length=30000))
#model.add(layers.Conv1D(10,100,activation='relu'))
#model.add(layers.MaxPool1D(pool_size=2, strides=1))
#model.add(layers.Flatten())
model.add(layers.Conv1D(40,100,activation='relu'))
model.add(layers.GlobalMaxPool1D())
#model.add(layers.Flatten())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.1, nesterov=True)
model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])

history = model.fit(X_train,y_train,epochs=50,verbose=True,validation_data=(X_test,y_test),batch_size=batch_size)
model.summary()
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test_ones, Y_test_ones, verbose=True)
print("Testing Accuracy of class-1:  {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test_zeros, Y_test_zeros, verbose=True)
print("Testing Accuracy of class-0:  {:.4f}".format(accuracy))
#plot_history(history)


pred = model.predict(X_test[numpy.where(y_test== 0)[0],:])

pred = [0 if i<0.5 else 1 for i in pred]
for i in pred :
    print (str(i)+" ")

for i in y_test[numpy.where(y_test== 0)[0]] :
    print (str(i)+" ")

#print(y_test[numpy.where(X_test[:, -7] == 1)[0]])
#matrix = confusion_matrix(y_test, pred)
#print (matrix)
#print ("Accuracy :",score)

