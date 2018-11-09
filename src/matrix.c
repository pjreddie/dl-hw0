#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

matrix sigmoid(matrix z)
{
  matrix m = make_matrix(z.rows, z.cols);
  for (int i = 0; i < (z.rows*z.cols); i++)
    {
      m.data[i] = 1.0/(1.0+exp(-z.data[i]));
    }
  return m;
}

matrix sigmoid_prime(matrix z)
{
  float one = 1.0;

  matrix s = sigmoid(z);
  matrix ones = make_matrix(z.rows, z.cols);

  for (int i = 0; i < (z.rows*z.cols); i++)
    {
      ones.data[i] = one;
    }

  return matrix_mul(s, matrix_sub(ones, s));
}

matrix matrix_mul(matrix a, matrix b)
{
  assert(a.cols == b.cols);
  assert(a.rows == b.rows);
  matrix m = make_matrix(a.rows, a.cols);
  for (int i = 0; i < (a.rows*a.cols); i++)
    {
      m.data[i] = a.data[i]*b.data[i];
    }
  return m;
}



matrix matrix_sub(matrix a, matrix b)
{
  assert(a.cols == b.cols);
  assert(a.rows == b.rows);
  matrix m = make_matrix(a.rows, a.cols);
  for (int i = 0; i < (a.rows*a.cols); i++)
    {
      m.data[i] = a.data[i] - b.data[i];
    }
  return m;
}


matrix cost_derivative(matrix output_activations, matrix y)
{
  return matrix_sub(output_activations, y);
}

// Make empty matrix filled with zeros
// int rows: number of rows in matrix
// int cols: number of columns in matrix
// returns: matrix of specified size, filled with zeros
matrix make_matrix(int rows, int cols)
{
    matrix m;
    m.rows = rows;
    m.cols = cols;
    m.shallow = 0;
    m.data = calloc(m.rows*m.cols, sizeof(float));
    return m;
}

// Make a matrix with uniformly random elements
// int rows, cols: size of matrix
// float s: range of randomness, [-s, s]
// returns: matrix of rows x cols with elements in range [-s,s]
matrix random_matrix(int rows, int cols, float s)
{
    matrix m = make_matrix(rows, cols);
    for (int i=0; i < rows*cols; i++)
      {
	m.data[i] = 2*s*(rand()%1000/1000.0) - s;
	//m.data[i] = (float)i+1;
	//printf(">>> %f \n", m.data[i]);
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
matrix copy_matrix(matrix m)
{
    matrix c = make_matrix(m.rows, m.cols);
    // TODO: 1.1 - Fill in the new matrix
    for (int i=0; i < m.rows*m.cols; i++)
      {
	c.data[i] = m.data[i];
      }
    return c;
}

float matrix_get(matrix m, int row_index, int col_index)
{
  return m.data[((m.cols*row_index)+col_index)];
}

// Transpose a matrix
// matrix m: matrix to be transposed
// returns: matrix, result of transposition
matrix transpose_matrix(matrix m)
{
    // TODO: 1.2 - Make a matrix the correct size, fill it in
    matrix t = make_matrix(m.cols, m.rows);
    for (int j=0; j < m.cols; j++)
      {
	for (int i=0; i < m.rows; i++)
	  {
	    t.data[((j*m.rows)+i)] = matrix_get(m, i, j); 
	  }
      }
    return t;
}

void matrix_set(matrix m, int row_index, int col_index, float val)
{
  m.data[((m.cols*row_index)+col_index)] = val;
}

matrix matrix_dot(matrix w, matrix x)
{
  float tmp;
  matrix m = make_matrix(w.rows, x.cols);
  
  assert(w.cols == x.rows);
  for (int i=0; i < w.cols; i++)
    {
      tmp = 0.0;
      for (int j=0; j < x.rows; j++)
	{
	  tmp += matrix_get(w, i, j) * matrix_get(x, j, 0);
	}
      matrix_set(m, i, 0, tmp);
    }
  return m;
}


void matrix_add(matrix w, matrix b)
{
  for (int i = 0; i < (w.rows*w.cols); i++)
    {
      b.data[i] += w.data[i];
    }
}

// Perform y = ax + y
// float a: scalar for matrix x
// matrix x: left operand to the scaled addition
// matrix y: unscaled right operand, also stores result
void axpy_matrix(float a, matrix x, matrix y)
{
    assert(x.cols == y.cols);
    assert(x.rows == y.rows);
    // TODO: 1.3 - Perform the weighted sum, store result back in y
}

// Perform matrix multiplication a*b, return result
// matrix a,b: operands
// returns: new matrix that is the result
matrix matmul(matrix a, matrix b)
{
    matrix c = make_matrix(a.rows, b.cols);
    // TODO: 1.4 - Implement matrix multiplication. Make sure it's fast!



    return c;
}

// In-place, element-wise scaling of matrix
// float s: scaling factor
// matrix m: matrix to be scaled
void scal_matrix(float s, matrix m)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        for(j =0 ; j < m.cols; ++j){
            m.data[i*m.cols + j] *= s;
        }
    }
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

// Used for matrix inversion
matrix augment_matrix(matrix m)
{
    int i,j;
    matrix c = make_matrix(m.rows, m.cols*2);
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            c.data[i*c.cols + j] = m.data[i*m.cols + j];
        }
    }
    for(j = 0; j < m.rows; ++j){
        c.data[j*c.cols + j+m.cols] = 1;
    }
    return c;
}

// Invert matrix m
matrix matrix_invert(matrix m)
{
    int i, j, k;
    //print_matrix(m);
    matrix none = {0};
    if(m.rows != m.cols){
        fprintf(stderr, "Matrix not square\n");
        return none;
    }
    matrix c = augment_matrix(m);
    //print_matrix(c);
    float **cdata = calloc(c.rows, sizeof(float *));
    for(i = 0; i < c.rows; ++i){
        cdata[i] = c.data + i*c.cols;
    }


    for(k = 0; k < c.rows; ++k){
        float p = 0.;
        int index = -1;
        for(i = k; i < c.rows; ++i){
            float val = fabs(cdata[i][k]);
            if(val > p){
                p = val;
                index = i;
            }
        }
        if(index == -1){
            fprintf(stderr, "Can't do it, sorry!\n");
            free_matrix(c);
            return none;
        }

        float *swap = cdata[index];
        cdata[index] = cdata[k];
        cdata[k] = swap;

        float val = cdata[k][k];
        cdata[k][k] = 1;
        for(j = k+1; j < c.cols; ++j){
            cdata[k][j] /= val;
        }
        for(i = k+1; i < c.rows; ++i){
            float s = -cdata[i][k];
            cdata[i][k] = 0;
            for(j = k+1; j < c.cols; ++j){
                cdata[i][j] +=  s*cdata[k][j];
            }
        }
    }
    for(k = c.rows-1; k > 0; --k){
        for(i = 0; i < k; ++i){
            float s = -cdata[i][k];
            cdata[i][k] = 0;
            for(j = k+1; j < c.cols; ++j){
                cdata[i][j] += s*cdata[k][j];
            }
        }
    }
    //print_matrix(c);
    matrix inv = make_matrix(m.rows, m.cols);
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            inv.data[i*m.cols + j] = cdata[i][j+m.cols];
        }
    }
    free_matrix(c);
    free(cdata);
    //print_matrix(inv);
    return inv;
}

matrix solve_system(matrix M, matrix b)
{
    matrix none = {0};
    matrix Mt = transpose_matrix(M);
    matrix MtM = matmul(Mt, M);
    matrix MtMinv = matrix_invert(MtM);
    if(!MtMinv.data) return none;
    matrix Mdag = matmul(MtMinv, Mt);
    matrix a = matmul(Mdag, b);
    free_matrix(Mt); free_matrix(MtM); free_matrix(MtMinv); free_matrix(Mdag);
    return a;
}

void test_matrix()
{
    int i;
    for(i = 0; i < 100; ++i){
        int s = rand()%4 + 3;
        matrix m = random_matrix(s, s, 10);
        matrix inv = matrix_invert(m);
        if(inv.data){
            matrix res = matmul(m, inv);
            print_matrix(res);
        }
    }
}
