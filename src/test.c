#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include "uwnet.h"
#include "matrix.h"
#include "image.h"
#include "test.h"
#include "args.h"

int tests_total = 0;
int tests_fail = 0;


int within_eps(float a, float b){
    return a-EPS<b && b<a+EPS;
}

int same_matrix(matrix a, matrix b)
{
    int i;
    if(a.rows != b.rows || a.cols != b.cols) {
        printf ("first matrix: %dx%d, second matrix:%dx%d\n", a.rows, a.cols, b.rows, b.cols);
        return 0;
    }
    for(i = 0; i < a.rows*a.cols; ++i){
        if(!within_eps(a.data[i], b.data[i])) {
            printf("differs at %d, %f vs %f\n", i, a.data[i], b.data[i]);
            return 0;
        }
    }
    return 1;
}

double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void test_copy_matrix()
{
    matrix a = random_matrix(32, 64, 10);
    matrix c = copy_matrix(a);
    TEST(same_matrix(a,c));
    free_matrix(a);
    free_matrix(c);
}

void test_transpose_matrix()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix at = load_matrix("data/test/at.matrix");
    matrix atest = transpose_matrix(a);
    matrix aorig = transpose_matrix(atest);
    TEST(same_matrix(at, atest) && same_matrix(a, aorig));
    free_matrix(a);
    free_matrix(at);
    free_matrix(atest);
    free_matrix(aorig);
}

void test_axpy_matrix()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix y = load_matrix("data/test/y.matrix");
    matrix y1 = load_matrix("data/test/y1.matrix");
    axpy_matrix(2, a, y);
    TEST(same_matrix(y, y1));
    free_matrix(a);
    free_matrix(y);
    free_matrix(y1);
}

void test_matmul()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix b = load_matrix("data/test/b.matrix");
    matrix c = load_matrix("data/test/c.matrix");
    matrix mul = matmul(a, b);
    TEST(same_matrix(c, mul));
    free_matrix(a);
    free_matrix(b);
    free_matrix(c);
    free_matrix(mul);
}

void test_activate_matrix()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix truth_alog = load_matrix("data/test/alog.matrix");
    matrix truth_arelu = load_matrix("data/test/arelu.matrix");
    matrix truth_alrelu = load_matrix("data/test/alrelu.matrix");
    matrix truth_asoft = load_matrix("data/test/asoft.matrix");
    matrix alog = copy_matrix(a);
    activate_matrix(alog, LOGISTIC);
    matrix arelu = copy_matrix(a);
    activate_matrix(arelu, RELU);
    matrix alrelu = copy_matrix(a);
    activate_matrix(alrelu, LRELU);
    matrix asoft = copy_matrix(a);
    activate_matrix(asoft, SOFTMAX);
    TEST(same_matrix(truth_alog, alog));
    TEST(same_matrix(truth_arelu, arelu));
    TEST(same_matrix(truth_alrelu, alrelu));
    TEST(same_matrix(truth_asoft, asoft));
    free_matrix(a);
    free_matrix(alog);
    free_matrix(arelu);
    free_matrix(alrelu);
    free_matrix(asoft);
    free_matrix(truth_alog);
    free_matrix(truth_arelu);
    free_matrix(truth_alrelu);
    free_matrix(truth_asoft);
}

void test_gradient_matrix()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix y = load_matrix("data/test/y.matrix");
    matrix truth_glog = load_matrix("data/test/glog.matrix");
    matrix truth_grelu = load_matrix("data/test/grelu.matrix");
    matrix truth_glrelu = load_matrix("data/test/glrelu.matrix");
    matrix truth_gsoft = load_matrix("data/test/gsoft.matrix");
    matrix glog = copy_matrix(a);
    matrix grelu = copy_matrix(a);
    matrix glrelu = copy_matrix(a);
    matrix gsoft = copy_matrix(a);
    gradient_matrix(y, LOGISTIC, glog);
    gradient_matrix(y, RELU, grelu);
    gradient_matrix(y, LRELU, glrelu);
    gradient_matrix(y, SOFTMAX, gsoft);
    TEST(same_matrix(truth_glog, glog));
    TEST(same_matrix(truth_grelu, grelu));
    TEST(same_matrix(truth_glrelu, glrelu));
    TEST(same_matrix(truth_gsoft, gsoft));
    free_matrix(a);
    free_matrix(y);
    free_matrix(glog);
    free_matrix(grelu);
    free_matrix(glrelu);
    free_matrix(gsoft);
    free_matrix(truth_glog);
    free_matrix(truth_grelu);
    free_matrix(truth_glrelu);
    free_matrix(truth_gsoft);
}

void test_connected_layer()
{
    matrix x = load_matrix("data/test/a.matrix");
    matrix w = load_matrix("data/test/b.matrix");
    matrix dw = load_matrix("data/test/dw.matrix");
    matrix db = load_matrix("data/test/db.matrix");
    matrix delta = load_matrix("data/test/delta.matrix");
    matrix prev_delta = load_matrix("data/test/prev_delta.matrix");
    matrix truth_prev_delta = load_matrix("data/test/truth_prev_delta.matrix");
    matrix truth_dw = load_matrix("data/test/truth_dw.matrix");
    matrix truth_db = load_matrix("data/test/truth_db.matrix");
    matrix updated_dw = load_matrix("data/test/updated_dw.matrix");
    matrix updated_db = load_matrix("data/test/updated_db.matrix");
    matrix updated_w = load_matrix("data/test/updated_w.matrix");
    matrix updated_b = load_matrix("data/test/updated_b.matrix");

    matrix b = load_matrix("data/test/bias.matrix");
    matrix truth_out = load_matrix("data/test/out.matrix");
    layer l = make_connected_layer(64, 16, LRELU);
    free_matrix(l.w);
    free_matrix(l.b);
    free_matrix(l.dw);
    free_matrix(l.db);
    l.w = w;
    l.b = b;
    l.dw = dw;
    l.db = db;
    matrix out = l.forward(l, x);
    TEST(same_matrix(truth_out, out));

    free_matrix(*l.delta);
    l.delta[0] = delta;
    l.backward(l, prev_delta);
    TEST(same_matrix(truth_prev_delta, prev_delta));
    TEST(same_matrix(truth_dw, l.dw));
    TEST(same_matrix(truth_db, l.db));

    l.update(l, .01, .9, .01);
    TEST(same_matrix(updated_dw, l.dw));
    TEST(same_matrix(updated_db, l.db));
    TEST(same_matrix(updated_w, l.w));
    TEST(same_matrix(updated_b, l.b));

    free_matrix(x);
    free_matrix(out);
    free_matrix(truth_out);
    free_layer(l);
    free_matrix(prev_delta);
    free_matrix(truth_prev_delta);
    free_matrix(truth_db);
    free_matrix(truth_dw);
    free_matrix(updated_db);
    free_matrix(updated_dw);
    free_matrix(updated_b);
    free_matrix(updated_w);
}


void test_matrix_speed()
{
    int i;
    int n = 128;
    matrix a = random_matrix(512, 512, 1);
    matrix b = random_matrix(512, 512, 1);
    double start = what_time_is_it_now();
    for(i = 0; i < n; ++i){
        matrix d = matmul(a,b);
        free_matrix(d);
    }
    printf("Matmul elapsed %lf sec\n", what_time_is_it_now() - start);
    start = what_time_is_it_now();
    for(i = 0; i < n; ++i){
        matrix at = transpose_matrix(a);
        free_matrix(at);
    }
    printf("Transpose elapsed %lf sec\n", what_time_is_it_now() - start);
}

void run_tests()
{
    //make_matrix_test();
    test_copy_matrix();
    test_axpy_matrix();
    test_transpose_matrix();
    test_matmul();
    test_activate_matrix();
    test_gradient_matrix();
    test_connected_layer();
    //test_matrix_speed();
    //printf("%d tests, %d passed, %d failed\n", tests_total, tests_total-tests_fail, tests_fail);
    printf("%d tests, %d passed, %d failed\n", tests_total, tests_total-tests_fail, tests_fail);
}

