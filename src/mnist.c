#include <stdio.h>
#include <stdlib.h>

#include <unistd.h>
#include <sysexits.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <math.h>
#include <time.h>

#include <arpa/inet.h>
#define betoh32 ntohl

#include "matrix.h"
#include "list.h"
#include "mnist.h"

#define BATCH_SIZE 100

void write_image(matrix image, u_char label, int x)
{
  FILE *fp;
  u_char w;
  char pname[70];

  snprintf(pname, 72, "%d-%d.pgm", label, x);
      fp = fopen(pname, "w");
      fprintf(fp, "P5\n");
      fprintf(fp, "# %s\n", pname);
      fprintf(fp, "%u %u\n", image.rows, image.cols);
      fprintf(fp, "255\n");
      fflush(fp);
      
      for (int i = 0; i < image.rows; i++) {
	for (int j = 0; j < image.cols; j++) {
	  w = image.data[j + (i * image.cols)];
	  write(fileno(fp), &w, 1);
	}
	fprintf(fp, "\n");
      }
      fprintf(fp, "\n");
      fclose(fp);
}

void shuffle_data(matrix **images, matrix **labels, int count)
{
  int x1,x2;
  srand ( time(NULL) );
  for (int z = 0; z < count; z++)
    {
      x1 = rand()%count;
      x2 = rand()%count;

      matrix *img = images[x2];
      matrix *lbl = labels[x2];

      images[x2] = images[x1];
      labels[x2] = labels[x1];

      images[x1] = img;
      labels[x1] = lbl;
    }
}

matrix *one_hot_encode(u_char label)
{
  matrix *m = make_matrix(10, 1);
  m->data[label] = 1;
  return m;
}


matrix *get_next_label(int f)
{
  int c;
  u_char byte;

  c = read(f, &byte, sizeof(byte));
  if (c < 0) {
    fprintf(stderr, "read image %s\n", strerror(errno));
    exit(EX_IOERR);
  }
  
  return one_hot_encode(byte);
}

void
read_next_image(Data *data, int f, u_char *image, int nr, int nc, int index)
{
  int c, ilen, i, j;
  ilen = nr * nc;
  c = read(f, image, ilen);

  if (c < 0) {
    fprintf(stderr, "read image %s\n", strerror(errno));
    exit(EX_IOERR);
  }

  matrix *m = make_matrix(nr, nc);
  
  for (i = 0; i < nr; i++) {
    for (j = 0; j < nc; j++) {
      m->data[(i * nc) +j ] =  (u_char) (*(image + j + (i * nc)));
      //printf("%d",*(image + j + (i * nc)));
      //matrix_set(*m, i, j, *(image + j + (i * nc)));
    }
  }

  data->images[index] = m;
}

Data *load_mnist(char *ifile, char *lfile)
{
  
  int c, fl, fi;

  u_int32_t lmagic, imagic, lnum, inum, nr, nc, ilen, n;
  u_char *image;

  Data *data = calloc(1, sizeof(Data));
  
  fl = open(lfile, O_RDONLY, 0);
  fi = open(ifile, O_RDONLY, 0);
  
  /* get the magic numbers */
  c = read(fl, &lmagic, sizeof(lmagic));
  lmagic = betoh32(lmagic); /* idx files are big endian */
  c = read(fi, &imagic, sizeof(imagic));
  imagic = betoh32(imagic); /* idx files are big endian */

  /* read the number of items */
  c = read(fl, &lnum, sizeof(lnum));
  lnum = betoh32(lnum);
  c = read(fi, &inum, sizeof(inum));
  inum = betoh32(inum);
  
  /* At the very least lnum and inum are equal */
  if (lnum != inum) {
    fprintf(stderr, "Please use label and image files that at least have an equal number of items!\n");
    exit(1);
  }

  /* read the number of rows */
  c = read(fi, &nr, sizeof(nr));
  nr = betoh32(nr);
  
  /* read the number of columns */
  c = read(fi, &nc, sizeof(nc));
  nc = betoh32(nc);

  /* allocate the image buffer */
  ilen = nr * nc;
  image = (u_char *)malloc(ilen);
  if (image == NULL) {
    fprintf(stderr, "malloc(image): %s\n", strerror(errno));
    exit(EX_OSERR);
  }
  
#if 0
  printf("magic numbers: %u %u\n", lmagic, imagic);
  printf("#items: %u %u\n", lnum, inum);
  printf("#rows: %u #columns: %u\n", nr, nc);
#endif

  n = 0;

  data->images = calloc(inum, sizeof(matrix*));
  data->labels = calloc(lnum, sizeof(matrix*));

  //while (n != lnum) {
  while (n != 10) {
    data->labels[n] = get_next_label(fl);
    read_next_image(data, fi, image, nr, nc, n);
    n++;
  }

  data->nsamples = lnum;
  
  close(fi);
  close(fl);

  return data;
}

