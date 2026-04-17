#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * =========================
 * INIT
 * =========================
 */
void *init(const char *model_path)
{
    printf("[blaze_decode] init model: %s\n", model_path ? model_path : "NULL");
    return NULL;
}

/*
 * =========================
 * INVOKE
 * =========================
 *
 * input:
 *   float array from blazeface tflite output
 *
 * output:
 *   bbox = [x, y, w, h]
 */
int invoke(void *private_data,
           const float *input,
           float *output)
{
    if (!input || !output)
        return -1;

    float score = input[0];

    // ===== NO FACE =====
    if (score < 0.5f) {
        output[0] = 0.0f;
        output[1] = 0.0f;
        output[2] = 0.0f;
        output[3] = 0.0f;
        return 0;
    }

    // ===== SIMPLE DEMO BBOX =====
    // (center of image fallback)

    float img_w = 640.0f;
    float img_h = 480.0f;

    float w = 180.0f;
    float h = 180.0f;

    float cx = img_w / 2.0f;
    float cy = img_h / 2.0f;

    output[0] = cx - w / 2.0f;  // x
    output[1] = cy - h / 2.0f;  // y
    output[2] = w;              // width
    output[3] = h;              // height

    printf("[blaze_decode] bbox = %.2f %.2f %.2f %.2f\n",
           output[0], output[1],
           output[2], output[3]);

    return 0;
}

/*
 * =========================
 * DEINIT
 * =========================
 */
void deinit(void *private_data)
{
    printf("[blaze_decode] deinit\n");
}