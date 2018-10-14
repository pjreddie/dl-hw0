#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "uwnet.h"

// Add bias terms to a matrix
// matrix m: partially computed output of layer
// matrix b: bias to add in (should only be one row!)
void forward_bias(matrix m, matrix b)
{
    assert(b.rows == 1);
    assert(m.cols == b.cols);
    int i,j;
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            m.data[i*m.cols + j] += b.data[j];
        }
    }
}

// Calculate bias updates from a delta matrix
// matrix delta: error made by the layer
// matrix db: delta for the biases
void backward_bias(matrix delta, matrix db)
{
    int i, j;
    for(i = 0; i < delta.rows; ++i){
        for(j = 0; j < delta.cols; ++j){
            db.data[j] += delta.data[i*delta.cols + j];
        }
    }
}

// Run a connected layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer: f(wx + b)
matrix forward_connected_layer(layer l, matrix in)
{
    // TODO: 3.1 - run the network forward
    matrix out = make_matrix(in.rows, l.w.cols); // Going to want to change this!

    // Saving our input and output and making a new delta matrix to hold errors
    // Probably don't change this
    l.in[0] = in;
    free_matrix(l.out[0]);
    l.out[0] = out;
    free_matrix(l.delta[0]);
    l.delta[0] = make_matrix(out.rows, out.cols);
    return out;
}

// Run a connected layer backward
// layer l: layer to run
// matrix delta: 
void backward_connected_layer(layer l, matrix prev_delta)
{
    matrix in    = l.in[0];
    matrix out   = l.out[0];
    matrix delta = l.delta[0];

    // TODO: 3.2
    // delta is the error made by this layer, dL/dout
    // First modify in place to be dL/d(in*w+b) using the gradient of activation
    
    // Calculate the updates for the bias terms using backward_bias
    // The current bias deltas are stored in l.db

    // Then calculate dL/dw. Use axpy to add this dL/dw into any previously stored
    // updates for our weights, which are stored in l.dw

    if(prev_delta.data){
        // Finally, if there is a previous layer to calculate for,
        // calculate dL/d(in). Again, using axpy, add this into the current
        // value we have for the previous layers delta, prev_delta.
    }
}

// Update 
void update_connected_layer(layer l, float rate, float momentum, float decay)
{
    // TODO
}

layer make_connected_layer(int inputs, int outputs, ACTIVATION activation)
{
    layer l = {0};
    l.w  = random_matrix(inputs, outputs, sqrtf(2.f/inputs));
    l.dw = make_matrix(inputs, outputs);
    l.b  = make_matrix(1, outputs);
    l.db = make_matrix(1, outputs);
    l.in = calloc(1, sizeof(matrix));
    l.out = calloc(1, sizeof(matrix));
    l.delta = calloc(1, sizeof(matrix));
    l.activation = activation;
    l.forward  = forward_connected_layer;
    l.backward = backward_connected_layer;
    l.update   = update_connected_layer;
    return l;
}

