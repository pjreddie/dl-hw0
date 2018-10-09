from uwnet import *
import pandas as pd
import numpy as np

def softmax_model():
    l = [make_connected_layer(784, 10, SOFTMAX)]
    return make_net(l)

def neural_net():
    l = [   make_connected_layer(784, 32, LRELU),
            make_connected_layer(32, 10, SOFTMAX)]
    return make_net(l)


iters_attempts = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]

df = pd.DataFrame()
for iters in iters_attempts:
    for batch in [int(math.pow(2, x)) for x in range(0, 12)]:
        for momentum in [x * .05 + .8 for x in range(0, 5)]:
            for decay in [x * 0.01 for x in range(0, 11)]:
                for rate in [x * 0.005 for x in range(1, 11)]:
                    print("loading data...")
                    train = load_image_classification_data("mnist/mnist.train".encode('utf-8'), "mnist/mnist.labels".encode('utf-8'))
                    test  = load_image_classification_data("mnist/mnist.test".encode('utf-8'), "mnist/mnist.labels".encode('utf-8'))
                    print("done")

                    # batch = 128
                    # iters = 1000
                        # momentum = .9
                    # decay = .1
                    print("making model...")
                    print(f"iters: {iters}")
                    print(f"batch: {batch}")
                    print(f"momentum: {momentum}")
                    print(f"decay: {decay}")
                    print(f"rate: {rate}")

                    m = softmax_model()
                    print("training...")
                    train_image_classifier(m, train, batch, iters, rate, momentum, decay)
                    print("done")

                    print("evaluating model...")
                    train = accuracy_net(m, train)
                    test = accuracy_net(m, test)
                    print("training accuracy: %f", train)
                    print("test accuracy:     %f", test)
                    entry = pd.Series(np.array([iters, batch, momentum, decay, rate, train, test]),
                    index=["iters", "batch", "momentum", "decay", "rate", "train", "test"])
                    df = df.append(entry, ignore_index=True)

df.to_csv("results.csv")
print("printed results to results.csv")

