#include "matrix.h"
#include "matrix.c"
#include <stdio.h>
int main(int argc, char * argv[])
{
	matrix m = random_matrix(3, 2, 0.5);
	print_matrix(m); 
	matrix copy_m = copy_matrix(m);
	print_matrix(copy_m);
	matrix transpose_m = transpose_matrix(m);
	print_matrix(transpose_m);
	// axpy_matrix(3.0, m, copy_m);
	// print_matrix(copy_m);
	matrix multiply_m = matmul(transpose_m, m);
	print_matrix(multiply_m);
}