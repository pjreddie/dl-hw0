#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"
#include "mnist.h"

#define batch_size 4

Network NetworkInit()
{
  
  Network network;

  network.num_layers = 3;
  
  network.sizes = malloc(network.num_layers * sizeof(int));
  network.weights = (matrix**)malloc(network.num_layers * sizeof(matrix*));
  network.biases = (matrix**)malloc(network.num_layers * sizeof(matrix*));
  
  network.sizes[0] = 784;
  network.sizes[1] = 30;
  network.sizes[2] = 10;
   
  for (int i=0; i<(network.num_layers-1); i++)
    {
      network.weights[i] = random_matrix(network.sizes[i], network.sizes[i+1], .9);
      network.biases[i] = random_matrix(network.sizes[i+1], 1, .9);
      
    }

  network.training_data = load_mnist("./train-images-idx3-ubyte", "./train-labels-idx1-ubyte");
  network.test_data = load_mnist("./t10k-images-idx3-ubyte", "./t10k-labels-idx1-ubyte");

  return network;
}

Delta *backprop(Network *network, matrix *x, matrix *y)
{
  Delta *d;
  
  matrix activation;
  matrix **activations, **nabla_w, **nabla_b, **zs;
  matrix *deriv, *sp, *delta, *z;
  
  zs  = (matrix**)calloc(network->num_layers-1, sizeof(matrix*));
  activations = (matrix**)calloc(network->num_layers, sizeof(matrix*));

  nabla_w = zero_copy_matrix_array(network->weights, network->num_layers-1);
  nabla_b = zero_copy_matrix_array(network->biases, network->num_layers-1);
  
  d = malloc(sizeof(Delta));

  /* forward pass */
  activation = *x;
  activations[0] = x;

  for (int i=0; i<(network->num_layers-1); i++)
    {
      z = matrix_add(matrix_dot(network->weights[i], &activation), network->biases[i]);
      zs[i] = copy_matrix(z);
      activations[i+1] = sigmoid(z);
    }

  /* backward pass */
  deriv = cost_derivative(activations[network->num_layers-1], y);
  sp = sigmoid_prime(zs[network->num_layers-1]);

  delta = matrix_mul(deriv, sp);

  nabla_b[network->num_layers-1] = copy_matrix(delta);
  nabla_w[network->num_layers-1] = matrix_dot(delta, transpose_matrix(activations[network->num_layers-2]));

  for (int j = 2; j < network->num_layers; j++)
    {
      z  = zs[(network->num_layers-1)-j];
      sp = sigmoid_prime(z);
      delta = matrix_mul(transpose_matrix(network->weights[network->num_layers-(j+1)]), delta);

      nabla_b[network->num_layers-j] = copy_matrix(delta);
      nabla_w[network->num_layers-j] = matrix_dot(delta, transpose_matrix(activations[network->num_layers-(j+1)]));

    }

  d->weights = nabla_w;
  d->biases = nabla_b;
  return d;

}

void sgd(Network *network, int num_epochs, float learning_rate)
{
  int n, start, end;
  matrix **nabla_w, **nabla_b;
  Delta *d;

  n = network->training_data->nsamples;

  matrix **images, **labels;
  
  nabla_w = zero_copy_matrix_array(network->weights, network->num_layers-1);
  nabla_b = zero_copy_matrix_array(network->biases, network->num_layers-1);

  print_matrix(*nabla_b[1]);

  for (int i = 0; i < num_epochs; i++)
    {
      
      shuffle_data(network->training_data->images, network->training_data->labels, n);
      start = 0;
      
      while (end < network->training_data->nsamples)
	{
	  end = start + batch_size;
	  
	  images = copy_matrix_pointers(network->training_data->images, start, end);
	  labels = copy_matrix_pointers(network->training_data->labels, start, end);

	  for (int k = 0; k < batch_size; k++)
	    {
	      matrix *x = images[k];
	      matrix *y = labels[k];

	      d = backprop(network, x, y);
	      for (int l = 0; l < network->num_layers; l++)
		{
		  
		}
	    }
	    start = end;

	    //break;
	}

      //      break;
    }
}

int main()
{

  Network network = NetworkInit();
  //sgd(&network, 2, 1.0);
  //print_matrix(*((matrix*)network.biases->back->val));
  
  //input input = 
  //sgd(&input, &input, 2, 0.5, 10);

  //list l;
  //matrix *m = random_matrix(3, 4,.9);
  //matrix *m1 = random_matrix(5, 3,.9);

  //list_insert(&l, &m);
  //list_insert(&l, &m1);
  
  //matrix *copy = zero_copy_list_of_matrices(&l);

  //print_matrix(*network.biases[0]);
  //printf("----------------------\n");
  //print_matrix(*network.biases[1]);
  
  //shuffle_data(images, input.labels, 10000);
  
  // do no delte
  //   matrix *m = (matrix*)(network.weights->front->val); /* ; */
  //   print_matrix(*((matrix*)network.weights->front->val));
  // do not delte
  
}
