#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <nnstreamer_plugin_api_filter.h>

typedef struct {
    int dummy;
} blaze_priv;

/* * OPEN (Thay cho init)
 * Lưu ý: Nhận vào GstTensorFilterProperties và private_data
 */
static int blaze_open(const GstTensorFilterProperties *prop, void **private_data)
{
    printf("[blaze_decode] open\n");
    blaze_priv *p = (blaze_priv *)malloc(sizeof(blaze_priv));
    if (!p) return -1;
    
    *private_data = p; // Lưu pointer vào hệ thống của NNStreamer
    return 0;
}

/* * INVOKE
 * Sử dụng GstTensorMemory cho input/output thay vì float*
 */
static int blaze_invoke(const GstTensorFilterProperties *prop, void **private_data,
                        const GstTensorMemory *input, GstTensorMemory *output)
{
    // Lấy dữ liệu float từ GstTensorMemory
    float *in_ptr = (float *)input[0].data;
    float *out_ptr = (float *)output[0].data;

    float score = in_ptr[0];

    if (score < 0.5f) {
        // Giả sử output tensor có kích thước 4 floats
        memset(out_ptr, 0, sizeof(float) * 4);
        return 0;
    }

    float cx = 320.0f;
    float cy = 240.0f;

    out_ptr[0] = cx - 90;
    out_ptr[1] = cy - 90;
    out_ptr[2] = 180;
    out_ptr[3] = 180;

    printf("[blaze_decode] bbox OK\n");
    return 0;
}

/* * CLOSE (Thay cho deinit)
 */
static void blaze_close(const GstTensorFilterProperties *prop, void **private_data)
{
    printf("[blaze_decode] close\n");
    if (*private_data) {
        free(*private_data);
        *private_data = NULL;
    }
}

/*
 * REGISTER
 * Định nghĩa theo đúng struct GstTensorFilterFramework bạn vừa tìm thấy
 */
static GstTensorFilterFramework blaze_custom = {
    .version = 1, 
    .open = blaze_open,
    .close = blaze_close,
    {
        .name = "blaze_decode",
        .allow_in_place = 0,
        .allocate_in_invoke = 0,
        .run_without_model = 1,
        .verify_model_path = 0,
        .invoke_NN = blaze_invoke,
        /* Các hàm khác có thể để NULL hoặc không khai báo */
    }
};

/* Entry point để NNStreamer load plugin */
GstTensorFilterFramework * get_filter_info (void)
{
    return &blaze_custom;
}