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

dataset_dir = 'E:/Thèse/datasets/tf_idf/'
models_dir = 'E:/Thèse/models/tf_idf/'
curr_class = "Java_Developer"
cv_height = 500

for curr_class in classes :
#curr_class = "Python_Developer"
#for n_filters in [100] :
    print("Start loading the matrix....")
    dataset = dataset_dir+str(cv_height)+'/'+curr_class+'.csv'
    #dataset='datasets/clean/'+str(cv_height)+'/'+curr_class+'.csv'
    data = numpy.loadtxt(dataset, delimiter=",", dtype=numpy.float32)
    print(data.shape)
    numpy.random.shuffle(data)
    ones_class_indices = numpy.where(data[:, (classes[curr_class]-10)] == 1)[0]
    zeros_class_indices = numpy.where(data[:, (classes[curr_class]-10)] == 0)[0]
    d = len(ones_class_indices)
    #x = numpy.where(data[:, 0] == 17088)[0]
    #print(x)
    #print(data[x,-10:])
    #exit(0)
    indices = numpy.concatenate((ones_class_indices,zeros_class_indices[:d]))
    x_others = data[-indices,1:-10]
    x_others = x_others.reshape((x_others.shape[0], cv_height, 100))
    y_others = data[-indices, (classes[curr_class]-10)]
    data = data[indices]
    print(data[0,0])
    numpy.random.shuffle(data)
    #print(data)
    #dataset1 = data[:, :-10]
    y = data[:, (classes[curr_class]-10)]
    #train_set = (data[:800, 1:-7], data[:800, -7])
    #test_set = (data[800:1200, 1:-7], data[500:1200, -7])
    #valid_set = (data[1200:, 1:-7], data[1200:, -7])
    batch_size = 32
    X_train_1, X_test_1, y_train, y_test = train_test_split(
    data, y, test_size=0.25, random_state=1000)

    print(X_test_1)
    test_ones_indices = numpy.where(y_test == 1)[0]
    Y_test_ones = y_test[test_ones_indices];
    X_test_ones = X_test_1[test_ones_indices, 1:-10];
    Y_test_zeros = y_test[numpy.where(y_test == 0)[0]];
    X_test_zeros = X_test_1[numpy.where(y_test == 0)[0],1:-10];
    print(X_test_1.shape)
    X_train = X_train_1[:, 1:-10]
    X_test = X_test_1[:, 1:-10]
    X_train = X_train.reshape((X_train.shape[0], cv_height, 100))
    X_test = X_test.reshape((X_test.shape[0], cv_height, 100))
    X_test_ones = X_test_ones.reshape((X_test_ones.shape[0], cv_height, 100))
    X_test_zeros = X_test_zeros.reshape((X_test_zeros.shape[0], cv_height, 100))
    #classifier = LogisticRegression()
    #classifier.fit(X_train, y_train)
    #score = classifier.score(X_test, y_test)
    #print(X_test_ones.shape)
    #print(X_test_zeros.shape)
    #print(Y_test_ones)
    #print(Y_test_zeros)
    input_dim = X_train.shape[1]
    model = Sequential()
    #model.add(layers.Embedding(300,100,input_length=30000))
    #model.add(layers.Conv1D(10,100,activation='relu'))
    #model.add(layers.MaxPool1D(pool_size=2, strides=1))
    #model.add(layers.Flatten())
    model.add(layers.Conv1D(100,1,activation='relu'))
    model.add(layers.GlobalMaxPool1D())
    #model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.1, nesterov=True)
    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=100, verbose=False, validation_data=(X_test,y_test), batch_size=batch_size)
    model.summary()
    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("************************Results for class :"+str(curr_class)+"*********************")
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test_ones, Y_test_ones, verbose=False)
    print("Testing Accuracy of class-1:  {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test_zeros, Y_test_zeros, verbose=False)
    print("Testing Accuracy of class-0:  {:.4f}".format(accuracy))
    #plot_history(history)
    # serialize model to JSON
    model_json = model.to_json()
    with open(models_dir+str(cv_height)+"/"+curr_class+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(models_dir+str(cv_height)+"/"+curr_class+".h5")
    print("Saved model to disk")
    numpy.save(models_dir+str(cv_height)+"/"+curr_class,X_test_1)

prod_t = model.predict(X_test)
pred_t = [0 if i<0.5 else 1 for i in prod_t]

pos = 0
for y,t,p in zip(pred_t,y_test,prod_t) :
    if y != t :
        print("cv_id = {:.1f} target = {:.1f} predicted = {:.1f}  probability = {:.3f}".format(X_test_1[pos,0],t,y,prod_t[pos,0]))
    pos += 1
pred = model.predict(X_test[numpy.where(y_test== 0)[0],:])


#pred = [0 if i<0.5 else 1 for i in pred]
#for i in pred :
#    print (str(i)+" ")

#errors_indices = numpy.where(pred == 0)[0]

#print(len(errors_indices))
#ref_errors = data[errors_indices,0]
#for ref in ref_errors :
#    print (str(ref))

#for i in y_test[numpy.where(y_test== 0)[0]] :
#    print (str(i)+" ")

#print(y_test[numpy.where(X_test[:, -7] == 1)[0]])
#matrix = confusion_matrix(y_test, pred)
#print (matrix)
#print ("Accuracy :",score)


