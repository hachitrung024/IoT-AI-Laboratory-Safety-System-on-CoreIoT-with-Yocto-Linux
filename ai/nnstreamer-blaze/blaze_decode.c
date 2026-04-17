#include <string.h>
#include <glib.h>
#include <nnstreamer_plugin_api_filter.h>

void init_filter_blaze (void) __attribute__ ((constructor));
void fini_filter_blaze (void) __attribute__ ((destructor));

typedef struct {
  gchar *model_path;
} blaze_pdata;

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
  info->num_tensors = 1;
  info->info[0].type = _NNS_UINT8;
  info->info[0].dimension[0] = 3;   // Channels
  info->info[0].dimension[1] = 640; // Width
  info->info[0].dimension[2] = 480; // Height
  info->info[0].dimension[3] = 1;   // Batch
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
  info->info[0].dimension[0] = 4;
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
  
  uint8_t *in_ptr = (uint8_t *) input[0].data;
  float *out_ptr = (float *) output[0].data;

  float score = (float)in_ptr[0] / 255.0f;

  if (score < 0.5f) {
    memset (out_ptr, 0, sizeof (float) * 4);
    return 0;
  }

  float cx = 320.0f;
  float cy = 240.0f;

  out_ptr[0] = cx - 90.0f; // xmin
  out_ptr[1] = cy - 90.0f; // ymin
  out_ptr[2] = 180.0f;     // width
  out_ptr[3] = 180.0f;     // height

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