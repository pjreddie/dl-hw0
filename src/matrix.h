// Include guards and C++ compatibility
#ifndef MATRIX_H
#define MATRIX_H
#ifdef __cplusplus
extern "C" {
#endif

// A matrix has size rows x cols
// and some data stored as an array of floats
// storage is row-major order:
// https://en.wikipedia.org/wiki/Row-_and_column-major_order
typedef struct matrix{
    int rows, cols;
    float *data;
    int shallow;
} matrix;

matrix *transpose_matrix(matrix *m);

matrix **copy_matrix_pointers(matrix** data, int start, int end);
  
float matrix_get(matrix *m, int row_index, int col_index);

void  matrix_set(matrix *m, int row_index, int col_index, float val);

matrix *matrix_add(matrix *w, matrix *b);

matrix *matrix_dot(matrix *w, matrix *x);

matrix *sigmoid(matrix *z);

matrix *sigmoid_prime(matrix *z);

matrix *cost_derivative(matrix *output_activations, matrix *y);

matrix *matrix_sub(matrix *a, matrix *b);

matrix *matrix_mul(matrix *a, matrix *b);

matrix **zero_copy_matrix_array(matrix** ms, int num);
  
// Make empty matrix filled with zeros
// int rows: number of rows in matrix
// int cols: number of columns in matrix
// returns: matrix of specified size, filled with zeros
matrix *make_matrix(int rows, int cols);

// Make a matrix with uniformly random elements
// int rows, cols: size of matrix
// float s: range of randomness, [-s, s]
// returns: matrix of rows x cols with elements in range [-s,s]
matrix *random_matrix(int rows, int cols, float s);

// Free memory associated with matrix
// matrix m: matrix to be freed
void free_matrix(matrix m);

// Copy a matrix
// matrix m: matrix to be copied
// returns: matrix that is a deep copy of m
matrix *copy_matrix(matrix *m);

// Print a matrix
void print_matrix(matrix m);

#ifdef __cplusplus
}
#endif
#endif
