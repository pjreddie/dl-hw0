#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "matrix.h"

matrix *sigmoid(matrix *z)
{
  matrix *m = make_matrix(z->rows, z->cols);
  for (int i = 0; i < (z->rows*z->cols); i++)
    {
      m->data[i] = 1.0/(1.0+exp(-z->data[i]));
    }
  return m;
}

matrix *sigmoid_prime(matrix *z)
{
  float one = 1.0;

  matrix *s = sigmoid(z);
  matrix *ones = make_matrix(z->rows, z->cols);

  for (int i = 0; i < (z->rows*z->cols); i++)
    {
      ones->data[i] = one;
    }

  
  return matrix_mul(s, matrix_sub(ones, s));
}


matrix *cost_derivative(matrix *output_activations, matrix *y)
{
  return matrix_sub(output_activations, y);
}

// Make empty matrix filled with zeros
// int rows: number of rows in matrix
// int cols: number of columns in matrix
// returns: matrix of specified size, filled with zeros
matrix *make_matrix(int rows, int cols)
{
  matrix *m = calloc(1, sizeof(matrix));
  m->rows = rows;
  m->cols = cols;
  m->shallow = 0;
  m->data = calloc(m->rows*m->cols, sizeof(float));
  return m;
}

// Make a matrix with uniformly random elements
// int rows, cols: size of matrix
// float s: range of randomness, [-s, s]
// returns: matrix of rows x cols with elements in range [-s,s]
matrix *random_matrix(int rows, int cols, float s)
{
  matrix *m = make_matrix(rows, cols);

  for (int i=0; i < rows*cols; i++)
    {
      m->data[i] = 2*s*(rand()%1000/1000.0) - s;
    }

  return m;
}

// Free memory associated with matrix
// matrix m: matrix to be freed
void free_matrix(matrix m)
{
    if (!m.shallow && m.data) {
        free(m.data);
    }
}

// Copy a matrix
// matrix m: matrix to be copied
// returns: matrix that is a deep copy of m
matrix *copy_matrix(matrix *m)
{
  matrix *c = make_matrix(m->rows, m->cols);

  for (int i=0; i < m->rows*m->cols; i++)
    {
      c->data[i] = m->data[i];
    }

  return c;
}

matrix **copy_matrix_pointers(matrix** data, int start, int end)
{
  matrix **copy = (matrix**)malloc((end-start) * sizeof(matrix*));

  for (int i = start; i < end; i++)
    {
      copy[i-start] = data[i];
    }

  return copy;
}

// copy the weights and biases into new arrays
// and init their matrices to 0's  
matrix** zero_copy_matrix_array(matrix **ms, int num)
{
  matrix **copy;

  copy = (matrix**)calloc(num, sizeof(matrix*));

  for (int i = 0; i < num; i++)
    {
      copy[i] = make_matrix(ms[i]->rows, ms[i]->cols);
    }
 
  return copy;
}

float matrix_get(matrix *m, int row_index, int col_index)
{
  return m->data[((m->cols*row_index)+col_index)];
}

// Transpose a matrix
// matrix m: matrix to be transposed
// returns: matrix, result of transposition
matrix *transpose_matrix(matrix *m)
{
  matrix *t = make_matrix(m->cols, m->rows);

  for (int j=0; j < m->cols; j++)
    {
      for (int i=0; i < m->rows; i++)
	{
	  t->data[((j*m->rows)+i)] = matrix_get(m, i, j);
	}
    }

  return t;
}

void matrix_set(matrix *m, int row_index, int col_index, float val)
{
  m->data[((m->cols*row_index)+col_index)] = val;
}

matrix *matrix_dot(matrix *w, matrix *x)
{
  assert(w->cols == x->rows);
  
  float tmp;
  matrix *m = make_matrix(w->rows, x->cols);
  
  for (int i=0; i < w->cols; i++)
    {
      tmp = 0.0;
      for (int j=0; j < x->rows; j++)
	{
	  tmp += matrix_get(w, i, j) * matrix_get(x, j, 0);
	}
      matrix_set(m, i, 0, tmp);
    }

  return m;
}


matrix *matrix_add(matrix *w, matrix *b)
{
  assert(w->rows == b->rows);
  assert(w->cols == b->cols);
  
  matrix *m = make_matrix(w->rows, w->cols);
  
  for (int i = 0; i < (w->rows*w->cols); i++)
    {
      m->data[i] = w->data[i] + b->data[i];
    }
  
  return m;
}

matrix *matrix_mul(matrix *a, matrix *b)
{
  assert(a->cols == b->cols);
  assert(a->rows == b->rows);
  
  matrix *m = make_matrix(a->rows, a->cols);

  for (int i = 0; i < (a->rows*a->cols); i++)
    {
      m->data[i] = a->data[i]*b->data[i];
    }

  return m;
}

matrix *matrix_sub(matrix *a, matrix *b)
{
  assert(a->cols == b->cols);
  assert(a->rows == b->rows);
  
  matrix *m = make_matrix(a->rows, a->cols);

  for (int i = 0; i < (a->rows*a->cols); i++)
    {
      m->data[i] = a->data[i] - b->data[i];
    }

  return m;
}

// Print a matrix
void print_matrix(matrix m)
{
    int i, j;
    printf(" __");
    for(j = 0; j < 16*m.cols-1; ++j) printf(" ");
    printf("__ \n");

    printf("|  ");
    for(j = 0; j < 16*m.cols-1; ++j) printf(" ");
    printf("  |\n");

    for(i = 0; i < m.rows; ++i){
        printf("|  ");
        for(j = 0; j < m.cols; ++j){
            printf("%15.7f ", m.data[i*m.cols + j]);
        }
        printf(" |\n");
    }
    printf("|__");
    for(j = 0; j < 16*m.cols-1; ++j) printf(" ");
    printf("__|\n");
}

