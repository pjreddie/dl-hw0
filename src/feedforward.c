#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"
#include "list.h"
#include "mnist.h"

typedef struct Network{
  int num_layers;
  int *sizes;
  list *weights;
  list *biases;
} Network;

Network NetworkInit()
{
  
  Network network;

  network.num_layers = 3;
  network.sizes = calloc(3, sizeof(int));

  network.sizes[0] = 784;
  network.sizes[1] = 30;
  network.sizes[2] = 10;
  
  network.weights = make_list();
  network.biases = make_list();
  
  for (int i=0; i<(network.num_layers-1); i++)
    {
      matrix w = random_matrix(network.sizes[i], network.sizes[i+1], .9);
      matrix b = random_matrix(network.sizes[i+1], 1, .9);

      list_insert(network.weights, &w);
      list_insert(network.biases, &b);
    }

  return network;
}


int main()
{

  Network network = NetworkInit();
  //print_matrix(*((matrix*)network.biases->back->val));
  //data train = load_image_classification_data("../../Data/MNIST/t10k-images-idx3-ubyte", "../../Data/MNIST/t10k-labels-idx1-ubyte");

  Input input = load_mnist("./t10k-images-idx3-ubyte", "./t10k-labels-idx1-ubyte");
  void **images = list_to_array(input.images);
  random_batch(images, input.labels, 10000);
  
  // DO NO DELTE
  //   matrix *m = (matrix*)(network.weights->front->val); /* ; */
  //   print_matrix(*((matrix*)network.weights->front->val));
  // DO NOT DELTE
  
}
