#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <nnstreamer_plugin_api_filter.h>

typedef struct {
    int dummy;
} blaze_ctx;

static void* blaze_init(const char* model_path)
{
    printf("blaze_decode init: %s\n", model_path);
    blaze_ctx *ctx = (blaze_ctx*) malloc(sizeof(blaze_ctx));
    return ctx;
}

static int blaze_invoke(void *data,
                        const GstTensorMemory *input,
                        GstTensorMemory *output)
{
    float *in = (float*) input[0].data;
    float *out = (float*) output[0].data;

    float score = in[0];

    if (score < 0.5) {
        memset(out, 0, sizeof(float) * 4);
        return 0;
    }

    out[0] = 200;
    out[1] = 150;
    out[2] = 200;
    out[3] = 200;

    printf("bbox: %f %f %f %f\n",
           output[0], output[1],
           output[2], output[3]);

    return 0;
}

static void blaze_exit(void *data)
{
    free(data);
}

NNSTREAMER_CUSTOM_FILTER_DESC(blaze_decode) = {
    .name = "blaze_decode",
    .init = blaze_init,
    .exit = blaze_exit,
    .invoke = blaze_invoke,
};