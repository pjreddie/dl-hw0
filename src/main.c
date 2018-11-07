#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "uwnet.h"
#include "image.h"
#include "test.h"
#include "args.h"
#include "matrix.h"

void try_mnist()
{
    data train = load_image_classification_data("mnist/mnist.train", "mnist/mnist.labels");
    data test  = load_image_classification_data("mnist/mnist.test", "mnist/mnist.labels");

    net n = {0};
    n.layers = calloc(2, sizeof(layer));
    n.n = 2;
    n.layers[0] = make_connected_layer(784, 32, LRELU);
    n.layers[1] = make_connected_layer(32, 10, SOFTMAX);

    int batch = 128;
    int iters = 5000;
    float rate = .01;

    float momentum = .9;
    float decay = .0;

    train_image_classifier(n, train, batch, iters, rate, momentum, decay);
    printf("Training accuracy: %f\n", accuracy_net(n, train));
    printf("Testing  accuracy: %f\n", accuracy_net(n, test));
}

int main(int argc, char **argv)
{
  matrix w = random_matrix(3, 4, .9);
  matrix x = random_matrix(4, 1, .9);
  matrix b = random_matrix(4, 1, .9);
  printf("-------------\n");
  print_matrix(b);
  matrix m = matrix_dot(w, x);
  print_matrix(m);
  matrix_add(m, b);
  printf("-------------\n");
  print_matrix(b);
  return 0;
}
