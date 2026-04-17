#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <nnstreamer_customfilter.h> 

typedef struct {
    int dummy;
} blaze_priv;

/*
 * INIT
 */
static void *blaze_init(const char *option)
{
    printf("[blaze_decode] init\n");
    blaze_priv *p = (blaze_priv *)malloc(sizeof(blaze_priv));
    return p;
}

/*
 * INVOKE
 */
static int blaze_invoke(void *data,
                         const float *input,
                         float *output)
{
    float score = input[0];

    if (score < 0.5f) {
        memset(output, 0, sizeof(float) * 4);
        return 0;
    }

    float cx = 320.0f;
    float cy = 240.0f;

    output[0] = cx - 90;
    output[1] = cy - 90;
    output[2] = 180;
    output[3] = 180;

    printf("[blaze_decode] bbox OK\n");
    return 0;
}

/*
 * DEINIT
 */
static void blaze_deinit(void *data)
{
    free(data);
}

/*
 * REGISTER
 */
NNStreamerCustomFilter blaze_custom = {
    .name = "blaze_decode",
    .init = blaze_init,
    .invoke = blaze_invoke,
    .deinit = blaze_deinit
};