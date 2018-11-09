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

void random_batch(void **images, u_char *labels, int count)
{
  int x;
  matrix image;
  FILE *fp;
  srand ( time(NULL) );
  u_char w;
  char pname[70];
  for (int z = 0; z < 10; z++)
    {
      x = rand()%count;
      image = *((matrix*) images[x]);
      printf("label = %d and nrows = %d and ncols = %d\n", labels[x], image.rows, image.cols);
      snprintf(pname, 72, "%d.pgm", labels[x]);
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
	//break;
    }
}

u_char
get_next_label(int f)
{
  int c;
  u_char byte;

  c = read(f, &byte, sizeof(byte));
  if (c < 0) {
    fprintf(stderr, "read image %s\n", strerror(errno));
    exit(EX_IOERR);
  }
  
  return(byte);
}

void
read_next_image(list *images, int f, u_char *image, int nr, int nc)
{
  int c, ilen, i, j;
  ilen = nr * nc;
  c = read(f, image, ilen);

  if (c < 0) {
    fprintf(stderr, "read image %s\n", strerror(errno));
    exit(EX_IOERR);
  }

  matrix m = make_matrix(nr, nc);
  
  for (i = 0; i < nr; i++) {
    for (j = 0; j < nc; j++) {
      m.data[(i * nc) +j ] =  (u_char) (*(image + j + (i * nc)));
      //printf("%d",*(image + j + (i * nc)));
      //matrix_set(*m, i, j, *(image + j + (i * nc)));
    }
  }

  list_insert(images, &m);
}

Input load_mnist(char *ifile, char *lfile)
{
  /* image and label idx files */
  /* char *ifile = "./t10k-images-idx3-ubyte"; */
  /* char *lfile = "./t10k-labels-idx1-ubyte"; */

  int c, fl, fi;
  u_int32_t lmagic, imagic, lnum, inum, nr, nc, ilen, n;
  u_char *image, *labels;
  
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
  list *images = make_list();
  labels = calloc(lnum, sizeof(u_char));

  while (n != lnum) {
    labels[n] = get_next_label(fl);
    read_next_image(images, fi, image, nr, nc);
    n++;
  }

  close(fi);
  close(fl);
  Input input;
  input.images = images;
  input.labels = labels;
  return input;
}

