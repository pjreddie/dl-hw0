#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "uwnet.h"
#include "image.h"
#include "test.h"
#include "args.h"

void tryml()
{
    data train = load_image_classification_data("mnist.train", "mnist.labels");
    data test  = load_image_classification_data("mnist.test", "mnist.labels");

    net n = {0};
    n.layers = calloc(2, sizeof(layer));
    n.n = 2;
    n.layers[0] = make_connected_layer(784, 32, LRELU);
    n.layers[1] = make_connected_layer(32, 10, SOFTMAX);

    int batch = 128;
    int iters = 5000;
    float rate = .01;
    float momentum = .9;
    float decay = .0;

    train_image_classifier(n, train, batch, iters, rate, momentum, decay);
    printf("Training accuracy: %f\n", accuracy_net(n, train));
    printf("Testing  accuracy: %f\n", accuracy_net(n, test));
}

int main(int argc, char **argv)
{
    char *in = find_char_arg(argc, argv, "-i", "data/dog.jpg");
    char *out = find_char_arg(argc, argv, "-o", "out");
    //float scale = find_float_arg(argc, argv, "-s", 1);
    if(argc < 2){
        printf("usage: %s [test | grayscale]\n", argv[0]);  
    } else if (0 == strcmp(argv[1], "test")){
        run_tests();
    } else if (0 == strcmp(argv[1], "grayscale")){
        image im = load_image(in);
        image g = rgb_to_grayscale(im);
        save_image(g, out);
        free_image(im);
        free_image(g);
    }
    return 0;
}
