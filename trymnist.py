from uwnet import *

def softmax_model():
    l = [make_connected_layer(784, 10, SOFTMAX)]
    return make_net(l)

def neural_net():
    l = [   make_connected_layer(784, 32, LRELU),
            make_connected_layer(32, 10, SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("mnist/mnist.train".encode('utf-8'), "mnist/mnist.labels".encode('utf-8'))
test  = load_image_classification_data("mnist/mnist.test".encode('utf-8'), "mnist/mnist.labels".encode('utf-8'))
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = .01
momentum = .9
decay = .0

m = softmax_model()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

