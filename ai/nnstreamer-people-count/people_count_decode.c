# include <math.h>
# include <stdio.h>
#include <string.h>
#include <glib.h>
#include <nnstreamer_plugin_api_filter.h>

#define MAX_DETECTION 10

typedef struct {
    float x;
    float y;
} Anchor;

void init_filter_people_count (void) __attribute__ ((constructor));
void fini_filter_people_count (void) __attribute__ ((destructor));

typedef struct {
  gchar *model_path;
  float width_img;
  float height_img;
} people_count_pdata;

static void people_count_close (const GstTensorFilterProperties * prop,
    void **private_data);

/**
 * Check condition to reopen model.
 */
static int
people_count_reopen (const GstTensorFilterProperties * prop, void **private_data)
{
  people_count_pdata *pdata = *private_data;

  if (prop->num_models > 0 && pdata->model_path && strcmp (prop->model_files[0], pdata->model_path) != 0) {
    return 1;
  }

  return 0;
}

/**
 * Init sub-plugin
 */
static int
people_count_open (const GstTensorFilterProperties * prop, void **private_data)
{
  people_count_pdata *pdata;

  if (*private_data != NULL) {
    if (people_count_reopen (prop, private_data) != 0) {
      people_count_close (prop, private_data);
    } else {
      return 1; 
    }
  }

  pdata = g_new0 (people_count_pdata, 1);
  if (pdata == NULL)
    return -ENOMEM;

  *private_data = (void *) pdata;

  if (prop->num_models > 0)
    pdata->model_path = g_strdup (prop->model_files[0]);

  g_print ("[people_count_decode] Loaded model: %s\n", pdata->model_path);

  return 0;
}

/**
 * @brief The standard tensor_filter callback for static input/output dimension.
 * @note If you want to support flexible/dynamic input/output dimension,
 *       read nnstreamer_plugin_api_filter.h and supply the
 *       setInputDimension callback.
 */
static int
people_count_getInputDim (const GstTensorFilterProperties * prop,
    void **private_data, GstTensorsInfo * info)
{
  people_count_pdata *pdata = (people_count_pdata *) (*private_data);

  if (prop->custom_properties) {
      if (sscanf(prop->custom_properties, "%f,%f", &pdata->width_img, &pdata->height_img) != 2) {
          pdata->width_img = 640.0f;
          pdata->height_img = 480.0f;
      }
  } else {
      pdata->width_img = 640.0f;
      pdata->height_img = 480.0f;
  }

  info->num_tensors = 4;

  /* detection_boxes */
  info->info[0].type = _NNS_FLOAT32;
  info->info[0].dimension[0] = 4;
  info->info[0].dimension[1] = MAX_DETECTION;
  info->info[0].dimension[2] = 1;
  info->info[0].dimension[3] = 1;

  /* detection_classes */
  info->info[1].type = _NNS_FLOAT32;
  info->info[1].dimension[0] = MAX_DETECTION;
  info->info[1].dimension[1] = 1;
  info->info[1].dimension[2] = 1;
  info->info[1].dimension[3] = 1;

  /* detection_scores */
  info->info[2].type = _NNS_FLOAT32;
  info->info[2].dimension[0] = MAX_DETECTION;
  info->info[2].dimension[1] = 1;
  info->info[2].dimension[2] = 1;
  info->info[2].dimension[3] = 1;

  /* num_detections */
  info->info[3].type = _NNS_FLOAT32;
  info->info[3].dimension[0] = 1;
  info->info[3].dimension[1] = 1;
  info->info[3].dimension[2] = 1;
  info->info[3].dimension[3] = 1;

  return 0;
}

/**
 * @brief The standard tensor_filter callback for static input/output dimension.
 * @note If you want to support flexible/dynamic input/output dimension,
 *       read nnstreamer_plugin_api_filter.h and supply the
 *       setInputDimension callback.
 */
static int
people_count_getOutputDim (const GstTensorFilterProperties * prop,
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
people_count_invoke (const GstTensorFilterProperties * prop, void **private_data,
              const GstTensorMemory * input, GstTensorMemory * output)
{
  people_count_pdata *pdata = (people_count_pdata *) (*private_data);

  float *in_ptr = (float *)input[0].data; // Tensor 0: FaceMesh output is [468 * 3]
  float *score_ptr = (float *)input[1].data; // Tensor 1: Score (1 value)
  float *out_ptr = (float *)output[0].data;

  float score = score_ptr[0];
  float threshold = 0.5f;

  if (score < threshold) {
      memset(out_ptr, 0, sizeof(float) * OUTPUT_DIM); 
      return 0; 
  }

  // float width_img = pdata->width_img;
  // float height_img = pdata->height_img;

  for (int i = 0; i < NUM_LANDMARKS; i++) {
      float x = in_ptr[i * 3 + 0];
      float y = in_ptr[i * 3 + 1];
      // float z = in_ptr[i * 3 + 2];

      // --- Return pixel coordinate ---
      // out_ptr[i * 2 + 0] = x * width_img;
      // out_ptr[i * 2 + 1] = y * height_img;

      // --- Return normalized coordinate ---
      out_ptr[i * 2 + 0] = x;
      out_ptr[i * 2 + 1] = y;
  }

  return 0;
}

/**
 * Close sub-plugin
 */
static void
people_count_close (const GstTensorFilterProperties * prop, void **private_data)
{
  people_count_pdata *pdata = (people_count_pdata *) (*private_data);

  if (pdata) {
    g_print ("[people_count_decode] Closing sub-plugin: %s\n", pdata->model_path);

    g_free (pdata->model_path);
    pdata->model_path = NULL;

    g_free (pdata);
    *private_data = NULL;
  }
}

/**@brief Name of this subplugin */
static gchar filter_subplugin_people_count[] = "people_count_decode";

static GstTensorFilterFramework people_count_custom = {
#ifdef GST_TENSOR_FILTER_API_VERSION_DEFINED
  .version = GST_TENSOR_FILTER_FRAMEWORK_V0,
#else
  .name = filter_subplugin_people_count,
  .allow_in_place = FALSE,
  .allocate_in_invoke = FALSE,
  .run_without_model = TRUE,
  .invoke_NN = people_count_invoke,
  .getInputDimension = people_count_getInputDim,
  .getOutputDimension = people_count_getOutputDim,
#endif
  .open = people_count_open,
  .close = people_count_close,
};

/**@brief Initialize this object for tensor_filter subplugin runtime register */
void init_filter_people_count (void)
{
#ifdef GST_TENSOR_FILTER_API_VERSION_DEFINED
  people_count_custom.name = filter_subplugin_people_count;
  people_count_custom.allow_in_place = FALSE;
  people_count_custom.allocate_in_invoke = FALSE;
  people_count_custom.run_without_model = TRUE;
  people_count_custom.invoke_NN = people_count_invoke;
  people_count_custom.getInputDimension = people_count_getInputDim;
  people_count_custom.getOutputDimension = people_count_getOutputDim;
#endif
  nnstreamer_filter_probe (&people_count_custom);
}

/**@brief Destruct the subplugin */
void fini_filter_people_count (void)
{
  nnstreamer_filter_exit (people_count_custom.name);
}