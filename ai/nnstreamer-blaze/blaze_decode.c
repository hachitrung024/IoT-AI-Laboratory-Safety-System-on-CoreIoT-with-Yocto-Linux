# include <math.h>
# include <stdio.h>
#include <string.h>
#include <glib.h>
#include <nnstreamer_plugin_api_filter.h>

#define NUM_ANCHORS 896
#define BOX_SIZE 16
#define SCORE_IDX 14336
// #define OUTPUT_DIM 16   // 4 (bbox) + 12 (6 landmarks * 2)
#define OUTPUT_DIM 4   // 4 (bbox) for pipeline

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

/* Create anchors */
static void generate_anchors(Anchor *anchors) {
    int idx = 0;
    // Grid 16x16
    for (int y = 0; y < 16; y++) {
        for (int x = 0; x < 16; x++) {
            for (int i = 0; i < 2; i++) {
                anchors[idx].x = (x + 0.5f) / 16.0f;
                anchors[idx].y = (y + 0.5f) / 16.0f;
                idx++;
            }
        }
    }
    // Grid 8x8
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

/* Sigmoid function */
static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-fmaxf(fminf(x, 88.0f), -88.0f)));
}

static void blaze_close (const GstTensorFilterProperties * prop,
    void **private_data);

/**
 * Check condition to reopen model.
 */
static int
blaze_reopen (const GstTensorFilterProperties * prop, void **private_data)
{
  blaze_pdata *pdata = *private_data;

  if (prop->num_models > 0 && pdata->model_path && strcmp (prop->model_files[0], pdata->model_path) != 0) {
    return 1;
  }

  return 0;
}

/**
 * Init sub-plugin
 */
static int
blaze_open (const GstTensorFilterProperties * prop, void **private_data)
{
  blaze_pdata *pdata;

  if (*private_data != NULL) {
    if (blaze_reopen (prop, private_data) != 0) {
      blaze_close (prop, private_data);
    } else {
      return 1; 
    }
  }

  pdata = g_new0 (blaze_pdata, 1);
  if (pdata == NULL)
    return -ENOMEM;

  *private_data = (void *) pdata;

  if (prop->num_models > 0)
    pdata->model_path = g_strdup (prop->model_files[0]);

  generate_anchors(pdata->anchors);

  g_print ("[blaze_decode] Loaded model: %s\n", pdata->model_path);

  return 0;
}

/**
 * @brief The standard tensor_filter callback for static input/output dimension.
 * @note If you want to support flexible/dynamic input/output dimension,
 *       read nnstreamer_plugin_api_filter.h and supply the
 *       setInputDimension callback.
 */
static int
blaze_getInputDim (const GstTensorFilterProperties * prop,
    void **private_data, GstTensorsInfo * info)
{
  blaze_pdata *pdata = (blaze_pdata *) (*private_data);

  if (prop->custom_properties) {
      if (sscanf(prop->custom_properties, "%f,%f", &pdata->width_img, &pdata->height_img) != 2) {
          pdata->width_img = 640.0f;
          pdata->height_img = 480.0f;
      }
  } else {
      pdata->width_img = 640.0f;
      pdata->height_img = 480.0f;
  }

  // Split the array: the first 14,336 numbers are Box, and the remaining 896 numbers are Score.
  info->num_tensors = 2;
  
  // Tensor 0: Boxes (896 anchors * 16 values)
  info->info[0].type = _NNS_FLOAT32;
  info->info[0].dimension[0] = 16;
  info->info[0].dimension[1] = 896;
  info->info[0].dimension[2] = 1;
  info->info[0].dimension[3] = 1;

  // Tensor 1: Scores (896 anchors * 1 value)
  info->info[1].type = _NNS_FLOAT32;
  info->info[1].dimension[0] = 1;
  info->info[1].dimension[1] = 896;
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
static int
blaze_getOutputDim (const GstTensorFilterProperties * prop,
    void **private_data, GstTensorsInfo * info)
{
  info->num_tensors = 1;
  info->info[0].type = _NNS_FLOAT32;
  info->info[0].dimension[0] = OUTPUT_DIM; 
  info->info[0].dimension[1] = 1;
  info->info[0].dimension[2] = 1;
  info->info[0].dimension[3] = 1;
  return 0;
}

/**
 * Invoke sub-plugin
 */
static int
blaze_invoke (const GstTensorFilterProperties * prop, void **private_data,
              const GstTensorMemory * input, GstTensorMemory * output)
{
  blaze_pdata *pdata = (blaze_pdata *) (*private_data);
  
  float *boxes = (float *)input[0].data;
  float *scores = (float *)input[1].data;
  float *out_ptr = (float *)output[0].data;
  
  int best_idx = -1;
  float max_score = -1e10f;

  /* Find best score */
  for (int i = 0; i < NUM_ANCHORS; i++) {
      if (scores[i] > max_score) {
          max_score = scores[i];
          best_idx = i;
      }
  }

  float confidence = sigmoid(max_score);
  float threshold = 0.75f;

  if (best_idx == -1 || confidence < threshold) {
      memset(out_ptr, 0, sizeof(float) * OUTPUT_DIM);
      return 0;
  }

  /* Decode box */
  float *raw_box = &boxes[best_idx * BOX_SIZE];
  Anchor anchor = pdata->anchors[best_idx];

  float width_img = pdata->width_img;
  float height_img = pdata->height_img;

  float cx = (raw_box[1] / 128.0f + anchor.x) * width_img;
  float cy = (raw_box[0] / 128.0f + anchor.y) * height_img;
  float w = (raw_box[3] / 128.0f) * width_img;
  float h = (raw_box[2] / 128.0f) * height_img;

  out_ptr[0] = cx - w / 2.0f; // xmin
  out_ptr[1] = cy - h / 2.0f; // ymin
  out_ptr[2] = w;
  out_ptr[3] = h;

  if (OUTPUT_DIM == 16) {
    // 2. Decode 6 Landmarks
    for (int i = 0; i < 6; i++) {
      float kx_raw = raw_box[4 + i * 2];
      float ky_raw = raw_box[4 + i * 2 + 1];
      
      // out_ptr[4...15]
      out_ptr[4 + i * 2]     = (kx_raw / 128.0f + anchor.x) * width_img; // landmark_x
      out_ptr[4 + i * 2 + 1] = (ky_raw / 128.0f + anchor.y) * height_img; // landmark_y
    }
  }

  return 0;
}

/**
 * Close sub-plugin
 */
static void
blaze_close (const GstTensorFilterProperties * prop, void **private_data)
{
  blaze_pdata *pdata = (blaze_pdata *) (*private_data);

  if (pdata) {
    g_print ("[blaze_decode] Closing sub-plugin: %s\n", pdata->model_path);

    g_free (pdata->model_path);
    pdata->model_path = NULL;

    g_free (pdata);
    *private_data = NULL;
  }
}

/**@brief Name of this subplugin */
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

/**@brief Initialize this object for tensor_filter subplugin runtime register */
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
  nnstreamer_filter_probe (&blaze_custom);
}

/**@brief Destruct the subplugin */
void fini_filter_blaze (void)
{
  nnstreamer_filter_exit (blaze_custom.name);
}