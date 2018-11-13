#include <sys/types.h>
#include <string.h>

#include "matrix.h"

typedef struct Data{
  matrix **images;
  matrix **labels;
  int nsamples;
} Data;

typedef struct Network{

  int num_layers;
  int *sizes;

  matrix **weights;
  matrix **biases;

  Data *training_data;
  Data *validation_data;
  Data *test_data;

} Network;

typedef struct Delta{
  matrix **weights;
  matrix **biases;
  
} Delta;

void read_next_image(Data *data, int f, u_char *image, int nr, int nc, int index);

Data *load_mnist(char *ifile, char *lfile);

matrix* get_next_label(int f);

matrix *one_hot_encode(u_char label);

void shuffle_data(matrix **images, matrix **labels, int count);

void write_image(matrix image, u_char label, int x);

void SGD(Network network, int num_epochs, float learning_rate);

Delta *backprop(Network *network, matrix *x, matrix *y);

matrix** zero_copy_list_of_matrices(matrix **ms, int num);
