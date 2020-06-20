from keras import Sequential
from keras import layers
from sklearn.model_selection import train_test_split
from keras import optimizers
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from keras import backend as K
import json
import numpy
numpy.set_printoptions(precision=2)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import itertools
import csv
import hashlib
from keras.models import Model
from keras.models import model_from_json
from gensim.models import KeyedVectors
from collections import Counter

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

def loadDicModel(file):
    with open(file) as json_file:
        return json.load(json_file)

#from tp4 import classes
#from tp4 import curr_class
classes = {"Front_End_Developer": 1, "Network_Administrator": 2, "Project_manager": 4, "Database_Administrator": 5, "Security_Analyst": 6,
           "Systems_Administrator": 7, "Python_Developer": 8, "Java_Developer": 9}

labels = ["Front_End_Developer", "Network_Administrator", "Project_manager", "Database_Administrator", "Security_Analyst",
           "Systems_Administrator", "Python_Developer", "Java_Developer"]
curr_class = "Systems_Administrator"

print("Start loading the matrix....")
#dataset='datasets/clean/500/Security_Analyst.csv'
#data = numpy.loadtxt(dataset, delimiter=",", dtype=numpy.float32)
def build_multi_label_model() :
    multi_label_model = dict()
    for label in labels :
        json_file = open('d2v_models/'+label+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.1, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        # load weights into new model
        model.load_weights('d2v_models/' + label + '.h5')
        multi_label_model[label] = model
    return multi_label_model


def predict(multi_model, data) :
    prod_t = dict()
    test_data = data[:, 1:-10]
    #test_data = test_data.reshape((test_data.shape[0], 500, 100))
    for label in labels :
        prod_t[label] = multi_model[label].predict(test_data)
    predicted = numpy.empty((0, len(classes)), dtype=numpy.float32)
    for i in range(test_data.shape[0]) :
        if i == 0 :
            print(prod_t[label][i,0])
        v = [1 if prod_t[l][i,0] >= 0.5 else 0 for l in labels]
        predicted = numpy.append(predicted, numpy.array([v], dtype=numpy.float32), axis=0)
    return predicted


def compute_precision(multi_model, test_data, pred_t) :
    targets = test_data[:, (-9,-8,-6,-5,-4,-3,-2,-1)]
    avg = 0.0;
    for i in range(targets.shape[0]) :
        a = len([p for p,t in zip(pred_t[i], targets[i]) if p==t and p==1])
        b = len([v for v in pred_t[i] if v==1])
        if b!=0 :
            avg += a*1.0/b
        else :
            avg += 1.0
    avg = avg/len(targets)
    return avg

def compute_recall(multi_model, test_data, pred_t) :
    targets = test_data[:, (-9,-8,-6,-5,-4,-3,-2,-1)]
    avg = 0.0;
    for i in range(targets.shape[0]):
        a = len([p for p, t in zip(pred_t[i], targets[i]) if p == t and p==1])
        b = len([v for v in targets[i] if v == 1])
        if b != 0 :
            avg += a * 1.0 / b
        else :
            avg += 1.0
    avg = avg / targets.shape[0]
    return avg

def compute_accuracy(multi_model, test_data, pred_t) :
    print(test_data.shape)
    targets = test_data[:, (-9,-8,-6,-5,-4,-3,-2,-1)]
    print(targets.shape)
    print(pred_t.shape)
    avg = 0.0;
    for i in range(targets.shape[0]):
        a = len([p for p, t in zip(pred_t[i], targets[i]) if p == t and p==1])
        b = len([v for v,z in zip(targets[i],pred_t[i]) if v == 1 or z==1])
        if b!=0 :
            avg += a * 1.0 / b
        else :
            avg += 1.0
        if i%100==0 :
            print(pred_t[i])
            print(targets[i])
            print(test_data[i,0])
            print(a)
            print(b)
    avg = avg / targets.shape[0]
    return avg

def evaluate_multilabel_model(multi_model, test_data) :
    pred_t = predict(multi_model, test_data)
    accuracy = compute_accuracy(multi_model,test_data,pred_t)
    precision = compute_precision(multi_model, test_data, pred_t)
    recall = compute_recall(multi_model, test_data, pred_t)
    print("recall={:0.4f} precision={:0.4f} accuracy={:0.4f}".format(recall, precision, accuracy))

#dataset='datasets/clean/500/'+curr_class+'.csv'
dataset='resume_d2v_dataset.csv'
data = numpy.loadtxt(dataset, delimiter=",", dtype=numpy.float32)
ones_class_indices = numpy.where(data[:, (classes[curr_class]-10)] == 1)[0]
zeros_class_indices = numpy.where(data[:, (classes[curr_class]-10)] == 0)[0]
d = len(ones_class_indices)
numpy.random.shuffle(ones_class_indices)
indices = numpy.concatenate((ones_class_indices[:400],zeros_class_indices[:d]))
data = data[indices,:]
print(data.shape)
model = build_multi_label_model()
evaluate_multilabel_model(model, data)