#include <stdlib.h>
#include <stdio.h>
#include "uwnet.h"

matrix forward_net(net m, matrix X)
{
    int i;
    for (i = 0; i < m.n; ++i) {
        layer l = m.layers[i];
        X = l.forward(l, X);
    }
    return X;
}

void backward_net(net m)
{
    int i;
    for (i = m.n-1; i >= 0; --i) {
        layer l = m.layers[i];
        matrix delta = {0};
        if(i > 0) delta = m.layers[i-1].delta[0];
        l.backward(l, delta);
    }
}

void update_net(net m, float rate, float momentum, float decay)
{
    int i;
    for(i = 0; i < m.n; ++i){
        layer l = m.layers[i];
        l.update(l, rate, momentum, decay);
    }
}

void write_matrix(matrix m, FILE *fp)
{
    fwrite(m.data, sizeof(float), m.rows*m.cols, fp);
}

void read_matrix(matrix m, FILE *fp)
{
     fread(m.data, sizeof(float), m.rows*m.cols, fp);
}

void file_error(char *filename)
{
    fprintf(stderr, "Couldn't open file %s\n", filename);
    exit(-1);
}

void save_weights(net m, char *filename)
{
    FILE *fp = fopen(filename, "wb");
    if(!fp) file_error(filename);
    int i;
    for(i = 0; i < m.n; ++i){
        layer l = m.layers[i];
        if(l.b.data) write_matrix(l.b, fp);
        if(l.w.data) write_matrix(l.w, fp);
    }
    fclose(fp);
}

void load_weights(net m, char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);
    int i;
    for(i = 0; i < m.n; ++i){
        layer l = m.layers[i];
        if(l.b.data) read_matrix(l.b, fp);
        if(l.w.data) read_matrix(l.w, fp);
    }
    fclose(fp);
}
