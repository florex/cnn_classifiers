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
classes = {"Software_Developer": 0, "Front_End_Developer": 1, "Network_Administrator": 2,
           "Web_Developer": 3, "Project_manager": 4, "Database_Administrator": 5, "Security_Analyst": 6,
           "Systems_Administrator": 7, "Python_Developer": 8, "Java_Developer": 9}

curr_class = "Project_manager"

print("Start loading the matrix....")
#dataset='datasets/clean/500/Security_Analyst.csv'
#data = numpy.loadtxt(dataset, delimiter=",", dtype=numpy.float32)

#numpy.random.shuffle(data)
#ones_class_indices = numpy.where(data[:, -1] == 1)[0]
#zeros_class_indices = numpy.where(data[:, -1] == 0)[0]

def compute_contributions(model, data):
    c2 = compute_contrib_maxpool(model,"global_max_pooling1d_1",data)
    return c2

def compute_contrib_maxpool(model,layer_name, data) :
    weights = model.layers[2].get_weights()[0]
    c1 = compute_contrib_dense1(model, "dense_1", data)
    max_pool = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
    max_out = max_pool.predict(data)
    print("***************************")
    print(max_out.shape)
    print(weights.shape)
    for w in weights :
        print(w)
    i=0
    contribs = numpy.empty((0, max_out.shape[1]), dtype=numpy.float32)
    for (out,c) in zip(max_out,c1) :
        out_1 = out.reshape((out.shape[0],1))
        contrib_mat = out_1*weights
        contrib_mat = contrib_mat/abs(contrib_mat).sum(axis=0)
        contrib = contrib_mat.dot(c)
        #contrib = contrib.reshape((contrib.shape[0],))
        contribs = numpy.append(contribs, [contrib], axis=0)
        if i%100 == 0 :
            print("*********************out**************")
            print(out)
            print("********************contrib_mat***************")
            contrib_mat
            print("******************contrib***********")
            print(contrib)
        i+=1
    return contribs

def compute_contrib_dense1(model,layer_name, data) :
    ow = model.layers[3].get_weights()[0]
    dense1 = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    dense1_out = dense1.predict(data)
    print(dense1_out.shape)
    i=0
    contribs = numpy.empty((0, dense1_out.shape[1]), dtype=numpy.float32)
    for out in dense1_out :
        ow = ow.reshape((ow.shape[0],))
        contrib = out*ow
        contrib = contrib/sum(abs(contrib))
        contribs = numpy.append(contribs, [contrib], axis=0)
        if i%100 == 0 :
            print(out.shape)
            print(type(out))
            print(out)
            print(ow.shape)
            print(type(ow))
            print(ow)
            print(contrib)
        i+=1
    return contribs

    #max_pool = Model(inputs=model.input,
    #               outputs=model.get_layer(layer_name).output)

def compute_words_contributions(model, data) :
    wv_file = "embeddings/vectors.kv"
    wv = KeyedVectors.load(wv_file, mmap='r')
    test = data[:, 1:-10]
    # ones_class_indices = numpy.where(test_x[:, (classes[curr_class]-10)] == 1)[0]
    # test = test[ones_class_indices,:]
    # zeros_class_indices = numpy.where(data[:, -1] == 0)[0]

    test = test.reshape(test.shape[0], 500, 100)
    layer_name = 'conv1d_1'
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(test)
    pos = 0
    words_selected = [0] * intermediate_output[0].shape[1]
    for i in range(len(words_selected)):
        words_selected[i] = dict()
    # wv.init_sims()
    # print(len(wv.vocab))
    inverted_wv = loadDicModel("models/inverted_wv.json")
    k = 0
    contribs = compute_contributions(model,test)
    print(contribs.shape)
    for (input,c) in zip(intermediate_output,contribs):
        if k==7 :
            print("iteration" + str(k))
            max_indices = numpy.argmax(input, axis=0)
            filters_words = test[k][max_indices, :]
            for i in range(len(filters_words)):
                current_w = filters_words[i, :]
                formated_wv = ["{:0.3f}".format(x) for x in current_w]
                hashed = hashlib.sha1(str.encode(str(formated_wv)))
                word = inverted_wv.get(hashed.hexdigest(), "")
                if word != '':
                    if word in words_selected[i] :
                        entry = words_selected[i].get(word)
                        print (entry)
                        entry.update({'count':entry.get('count')+1,'contrib':entry.get('contrib')+contribs[k,i]})
                    else :
                        words_selected[i].update({word:{'count':1,'contrib':contribs[k,i]}})
                else:
                    print(formated_wv)
            break

        k += 1
    return words_selected

wv_file="embeddings/vectors.kv"
wv = KeyedVectors.load(wv_file, mmap='r')
# load json and create model
json_file = open('models/'+curr_class+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.1, nesterov=True)
model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
# load weights into new model
model.load_weights('models/'+curr_class+'.h5')
print("Loaded model from disk")

#inp = model.input                                           # input placeholder
#outputs = [layer.output for layer in model.layers]          # all layer outputs
#functors = [K.function([inp], [out]) for out in outputs]    # evaluation functions

# Testing
model.summary()
#test = numpy.random.random(input_shape)[numpy.newaxis,...]
test_x = numpy.load('models/'+curr_class+'.npy')
test = test_x[:,1:-10]
print(test_x.shape)

#ones_class_indices = numpy.where(test_x[:, (classes[curr_class]-10)] == 1)[0]
#test = test[ones_class_indices,:]
#zeros_class_indices = numpy.where(data[:, -1] == 0)[0]
words = compute_words_contributions(model,test_x)
print(words)
exit(0)
test = test.reshape(test.shape[0],500,100)
#compute_contributions(model,test)
print(test.shape)
layer_name = 'conv1d_1'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(test)

print(type(intermediate_output[1]))

pos = 0
words_selected = [0]*intermediate_output[0].shape[1]
for i in range(len(words_selected)) :
    words_selected[i] = []
#wv.init_sims()
#print(len(wv.vocab))
inverted_wv = loadDicModel("models/inverted_wv.json")
k = 0
for input in intermediate_output :
    if test_x[k,0] == 1889 :
        print("iteration"+str(k))
        max_indices = numpy.argmax(input,axis=0)
        filters_words = test[k][max_indices,:]
        for i in range(len(filters_words)) :
            current_w = filters_words[i,:]
            formated_wv = ["{:0.3f}".format(x) for x in current_w]
            hashed = hashlib.sha1(str.encode(str(formated_wv)))
            word = inverted_wv.get(hashed.hexdigest(),"")
            if word != '' :
                words_selected[i].append(word)
            else :
                print(formated_wv)
        break
    k += 1
#print(Counter(words_selected[5]))
#print (words_selected)
count_vector = [0]*len(words_selected)
for i in range(len(words_selected)) :
    count_vector[i] = Counter(words_selected[i])
    print(count_vector[i])

s = set()
for i in range(len(count_vector)) :
    for key in dict(count_vector[i]) :
        s.add(key)

print(" ".join(s))

#print(test_x[ones_class_indices[0],0])

#layer_outs = [func([test]) for func in functors]
#print (len(layer_outs[0][0][0]))
#print (len(layer_outs[0][0][0][1]))
#print (layer_outs[0][0][0][1])
