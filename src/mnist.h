#include <sys/types.h>

#include "matrix.h"

typedef struct Input{
  list *images;
  u_char *labels;
}Input;

void read_next_image(list *images, int f, u_char *image, int nr, int nc);

Input load_mnist();

u_char get_next_label(int f);

void random_batch(void **images, u_char *labels, int count);
