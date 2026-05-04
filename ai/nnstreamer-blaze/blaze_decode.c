# include <math.h>
# include <stdio.h>
# include <string.h>
# include <glib.h>
# include <nnstreamer_plugin_api_filter.h>

#define NUM_ANCHORS 896
#define BOX_SIZE 16

// x, y, w, h per detection
#define OUTPUT_DIM 4

#define BBOX_SCALE 1.45f

#define CLAMP(x) (fmaxf(0.0f, fminf(1.0f, x)))

typedef struct {
    float x;
    float y;
} Anchor;

void init_filter_blaze (void) __attribute__ ((constructor));
void fini_filter_blaze (void) __attribute__ ((destructor));

typedef struct {
    gchar *model_path;
    Anchor anchors[NUM_ANCHORS];
    float width_img;
    float height_img;
} blaze_pdata;

/**
 * Create anchors
 */
static void generate_anchors(Anchor *anchors)
{
    int idx = 0;

    // Grid 16x16 (2 anchors each)
    for (int y = 0; y < 16; y++) {
        for (int x = 0; x < 16; x++) {
            for (int i = 0; i < 2; i++) {
                anchors[idx].x = (x + 0.5f) / 16.0f;
                anchors[idx].y = (y + 0.5f) / 16.0f;
                idx++;
            }
        }
    }

    // Grid 8x8 (6 anchors each)
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            for (int i = 0; i < 6; i++) {
                anchors[idx].x = (x + 0.5f) / 8.0f;
                anchors[idx].y = (y + 0.5f) / 8.0f;
                idx++;
            }
        }
    }
}

/**
 * Sigmoid
 */
static float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-fmaxf(fminf(x, 88.0f), -88.0f)));
}


static void blaze_close (const GstTensorFilterProperties * prop,
    void **private_data);

/**
 * Check condition to reopen model.
 */
static int blaze_reopen (const GstTensorFilterProperties * prop,
    void **private_data)
{
    blaze_pdata *pdata = *private_data;

    if (prop->num_models > 0 &&
        pdata->model_path &&
        strcmp(prop->model_files[0], pdata->model_path) != 0) {
        return 1;
    }

    return 0;
}

/**
 * Init sub-plugin
 */
static int blaze_open (const GstTensorFilterProperties * prop,
    void **private_data)
{
    blaze_pdata *pdata;

    if (*private_data != NULL) {
        if (blaze_reopen(prop, private_data) != 0) {
            blaze_close(prop, private_data);
        } else {
            return 1;
        }
    }

    pdata = g_new0(blaze_pdata, 1);
    if (pdata == NULL)
        return -ENOMEM;

    *private_data = (void *) pdata;

    if (prop->num_models > 0)
        pdata->model_path = g_strdup(prop->model_files[0]);

    pdata->width_img = 640.0f;
    pdata->height_img = 480.0f;

    generate_anchors(pdata->anchors);

    g_print("[blaze_decode] Loaded model: %s\n", pdata->model_path);

    return 0;
}

/**
 * @brief The standard tensor_filter callback for static input/output dimension.
 * @note If you want to support flexible/dynamic input/output dimension,
 *       read nnstreamer_plugin_api_filter.h and supply the
 *       setInputDimension callback.
 */
static int blaze_getInputDim (const GstTensorFilterProperties * prop,
    void **private_data, GstTensorsInfo * info)
{
    blaze_pdata *pdata = (blaze_pdata *) (*private_data);

    if (prop->custom_properties) {
        if (sscanf(prop->custom_properties, "%f,%f",
            &pdata->width_img,
            &pdata->height_img) != 2) {
            pdata->width_img = 640.0f;
            pdata->height_img = 480.0f;
        }
    }

    info->num_tensors = 2;

    /* boxes */
    info->info[0].type = _NNS_FLOAT32;
    info->info[0].dimension[0] = 16;
    info->info[0].dimension[1] = NUM_ANCHORS;
    info->info[0].dimension[2] = 1;
    info->info[0].dimension[3] = 1;

    /* scores */
    info->info[1].type = _NNS_FLOAT32;
    info->info[1].dimension[0] = 1;
    info->info[1].dimension[1] = NUM_ANCHORS;
    info->info[1].dimension[2] = 1;
    info->info[1].dimension[3] = 1;

    return 0;
}

/**
 * @brief The standard tensor_filter callback for static input/output dimension.
 * @note If you want to support flexible/dynamic input/output dimension,
 *       read nnstreamer_plugin_api_filter.h and supply the
 *       setInputDimension callback.
 */
static int blaze_getOutputDim (const GstTensorFilterProperties * prop,
    void **private_data, GstTensorsInfo * info)
{
    info->num_tensors = 1;

    // info->info[0].type = _NNS_UINT32;
    info->info[0].type = _NNS_FLOAT32;

    // num detections + (x,y,w,h)*N
    info->info[0].dimension[0] = 4;
    info->info[0].dimension[1] = 1;
    info->info[0].dimension[2] = 1;
    info->info[0].dimension[3] = 1;

    return 0;
}

/**
 * Invoke sub-plugin
 */
static int blaze_invoke (const GstTensorFilterProperties * prop,
    void **private_data,
    const GstTensorMemory * input,
    GstTensorMemory * output)
{
    blaze_pdata *pdata = (blaze_pdata *) (*private_data);

    float *boxes = (float *) input[0].data;
    float *scores = (float *) input[1].data;
    float *out_ptr = (float *) output[0].data;

    int best_idx = -1;
    float max_score = -1e10f;

    for (int i = 0; i < NUM_ANCHORS; i++) {
        if (scores[i] > max_score) {
            max_score = scores[i];
            best_idx = i;
        }
    }

    float conf = sigmoid(max_score);
    if (best_idx < 0 || conf < 0.75f) {
        out_ptr[0] = 0;
        out_ptr[1] = 0;
        out_ptr[2] = 0;
        out_ptr[3] = 0;
        return 0;
    }

    float *raw_box = &boxes[best_idx * BOX_SIZE];
    Anchor a = pdata->anchors[best_idx];

    float cx = raw_box[1] / 128.0f + a.x;
    float cy = raw_box[0] / 128.0f + a.y;
    float w  = raw_box[3] / 128.0f;
    float h  = raw_box[2] / 128.0f;

    float xmin = CLAMP(cx - w / 2.0f);
    float ymin = CLAMP(cy - h / 2.0f);
    float xmax = CLAMP(cx + w / 2.0f);
    float ymax = CLAMP(cy + h / 2.0f);

    // ===== Convert to pixel =====
    float x1 = xmin * pdata->width_img;
    float y1 = ymin * pdata->height_img;
    float x2 = xmax * pdata->width_img;
    float y2 = ymax * pdata->height_img;

    // center
    float cx_box = (x1 + x2) * 0.5f;
    float cy_box = (y1 + y2) * 0.5f;

    // width/height pixel
    float box_w = x2 - x1;
    float box_h = y2 - y1;

    // Make bbox square and wider
    float side = fmaxf(box_w, box_h) * BBOX_SCALE;

    float half = side * 0.5f;

    float new_x1 = cx_box - half;
    float new_y1 = cy_box - half;
    float new_x2 = cx_box + half;
    float new_y2 = cy_box + half;

    if (new_x1 < 0) new_x1 = 0;
    if (new_y1 < 0) new_y1 = 0;
    if (new_x2 > pdata->width_img)  new_x2 = pdata->width_img;
    if (new_y2 > pdata->height_img) new_y2 = pdata->height_img;

    // ===== output =====
    out_ptr[0] = new_x1;
    out_ptr[1] = new_y1;
    out_ptr[2] = new_x2 - new_x1;
    out_ptr[3] = new_y2 - new_y1;

    return 0;
}

/**
 * Close sub-plugin
 */
static void blaze_close (const GstTensorFilterProperties * prop,
    void **private_data)
{
    blaze_pdata *pdata = (blaze_pdata *) (*private_data);

    if (pdata) {
        g_print("[blaze_decode] Closing: %s\n", pdata->model_path);

        g_free(pdata->model_path);
        pdata->model_path = NULL;

        g_free(pdata);
        *private_data = NULL;
    }
}

/**
 * Register sub-plugin
 */
static gchar filter_subplugin_blaze[] = "blaze_decode";

static GstTensorFilterFramework blaze_custom = {
#ifdef GST_TENSOR_FILTER_API_VERSION_DEFINED
    .version = GST_TENSOR_FILTER_FRAMEWORK_V0,
#else
    .name = filter_subplugin_blaze,
    .allow_in_place = FALSE,
    .allocate_in_invoke = FALSE,
    .run_without_model = TRUE,
    .invoke_NN = blaze_invoke,
    .getInputDimension = blaze_getInputDim,
    .getOutputDimension = blaze_getOutputDim,
#endif
    .open = blaze_open,
    .close = blaze_close,
};

void init_filter_blaze (void)
{
#ifdef GST_TENSOR_FILTER_API_VERSION_DEFINED
    blaze_custom.name = filter_subplugin_blaze;
    blaze_custom.allow_in_place = FALSE;
    blaze_custom.allocate_in_invoke = FALSE;
    blaze_custom.run_without_model = TRUE;
    blaze_custom.invoke_NN = blaze_invoke;
    blaze_custom.getInputDimension = blaze_getInputDim;
    blaze_custom.getOutputDimension = blaze_getOutputDim;
#endif

    nnstreamer_filter_probe(&blaze_custom);
}

void fini_filter_blaze (void)
{
    nnstreamer_filter_exit(blaze_custom.name);
}