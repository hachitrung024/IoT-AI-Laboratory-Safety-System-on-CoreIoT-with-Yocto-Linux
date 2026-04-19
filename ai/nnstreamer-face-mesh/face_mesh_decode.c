# include <math.h>
# include <stdio.h>
#include <string.h>
#include <glib.h>
#include <nnstreamer_plugin_api_filter.h>

#define NUM_ANCHORS 896
#define NUM_LANDMARKS 468
#define LANDMARK_DIM 3
#define OUTPUT_DIM (NUM_LANDMARKS * 2)

typedef struct {
    float x;
    float y;
} Anchor;

void init_filter_facemesh (void) __attribute__ ((constructor));
void fini_filter_facemesh (void) __attribute__ ((destructor));

typedef struct {
  gchar *model_path;
  Anchor anchors[NUM_ANCHORS];
  float width_img;
  float height_img;
} facemesh_pdata;

/**
 * Check condition to reopen model.
 */
static int
facemesh_reopen (const GstTensorFilterProperties * prop, void **private_data)
{
  facemesh_pdata *pdata = *private_data;

  if (prop->num_models > 0 && pdata->model_path && strcmp (prop->model_files[0], pdata->model_path) != 0) {
    return 1;
  }

  return 0;
}

/**
 * Init sub-plugin
 */
static int
facemesh_open (const GstTensorFilterProperties * prop, void **private_data)
{
  facemesh_pdata *pdata;

  if (*private_data != NULL) {
    if (facemesh_reopen (prop, private_data) != 0) {
      facemesh_close (prop, private_data);
    } else {
      return 1; 
    }
  }

  pdata = g_new0 (facemesh_pdata, 1);
  if (pdata == NULL)
    return -ENOMEM;

  *private_data = (void *) pdata;

  if (prop->num_models > 0)
    pdata->model_path = g_strdup (prop->model_files[0]);

  g_print ("[facemesh_decode] Loaded model: %s\n", pdata->model_path);

  return 0;
}

/**
 * @brief The standard tensor_filter callback for static input/output dimension.
 * @note If you want to support flexible/dynamic input/output dimension,
 *       read nnstreamer_plugin_api_filter.h and supply the
 *       setInputDimension callback.
 */
static int
facemesh_getInputDim (const GstTensorFilterProperties * prop,
    void **private_data, GstTensorsInfo * info)
{
  facemesh_pdata *pdata = (facemesh_pdata *) (*private_data);

  if (prop->custom_properties) {
      if (sscanf(prop->custom_properties, "%f,%f", &pdata->width_img, &pdata->height_img) != 2) {
          pdata->width_img = 640.0f;
          pdata->height_img = 480.0f;
      }
  } else {
      pdata->width_img = 640.0f;
      pdata->height_img = 480.0f;
  }

  info->num_tensors = 1;

  // FaceMesh output is [468 * 3]
  info->info[0].type = _NNS_FLOAT32;
  info->info[0].dimension[0] = NUM_LANDMARKS * LANDMARK_DIM;
  info->info[0].dimension[1] = 1;
  info->info[0].dimension[2] = 1;
  info->info[0].dimension[3] = 1;

  return 0;
}

/**
 * @brief The standard tensor_filter callback for static input/output dimension.
 * @note If you want to support flexible/dynamic input/output dimension,
 *       read nnstreamer_plugin_api_filter.h and supply the
 *       setInputDimension callback.
 */
static int
facemesh_getOutputDim (const GstTensorFilterProperties * prop,
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
facemesh_invoke (const GstTensorFilterProperties * prop, void **private_data,
              const GstTensorMemory * input, GstTensorMemory * output)
{
  facemesh_pdata *pdata = (facemesh_pdata *) (*private_data);

  float *in_ptr = (float *)input[0].data;
  float *out_ptr = (float *)output[0].data;

  float width_img = pdata->width_img;
  float height_img = pdata->height_img;

  for (int i = 0; i < NUM_LANDMARKS; i++) {
      float x = in_ptr[i * 3 + 0];
      float y = in_ptr[i * 3 + 1];
      // float z = in_ptr[i * 3 + 2];

      out_ptr[i * 2 + 0] = x * width_img;
      out_ptr[i * 2 + 1] = y * height_img;
  }

  return 0;
}

/**
 * Close sub-plugin
 */
static void
facemesh_close (const GstTensorFilterProperties * prop, void **private_data)
{
  facemesh_pdata *pdata = (facemesh_pdata *) (*private_data);

  if (pdata) {
    g_print ("[facemesh_decode] Closing sub-plugin: %s\n", pdata->model_path);

    g_free (pdata->model_path);
    pdata->model_path = NULL;

    g_free (pdata);
    *private_data = NULL;
  }
}

/**@brief Name of this subplugin */
static gchar filter_subplugin_facemesh[] = "face_mesh_decode";

static GstTensorFilterFramework facemesh_custom = {
#ifdef GST_TENSOR_FILTER_API_VERSION_DEFINED
  .version = GST_TENSOR_FILTER_FRAMEWORK_V0,
#else
  .name = filter_subplugin_facemesh,
  .allow_in_place = FALSE,
  .allocate_in_invoke = FALSE,
  .run_without_model = TRUE,
  .invoke_NN = facemesh_invoke,
  .getInputDimension = facemesh_getInputDim,
  .getOutputDimension = facemesh_getOutputDim,
#endif
  .open = facemesh_open,
  .close = facemesh_close,
};

/**@brief Initialize this object for tensor_filter subplugin runtime register */
void init_filter_facemesh (void)
{
#ifdef GST_TENSOR_FILTER_API_VERSION_DEFINED
  facemesh_custom.name = filter_subplugin_facemesh;
  facemesh_custom.allow_in_place = FALSE;
  facemesh_custom.allocate_in_invoke = FALSE;
  facemesh_custom.run_without_model = TRUE;
  facemesh_custom.invoke_NN = facemesh_invoke;
  facemesh_custom.getInputDimension = facemesh_getInputDim;
  facemesh_custom.getOutputDimension = facemesh_getOutputDim;
#endif
  nnstreamer_filter_probe (&facemesh_custom);
}

/**@brief Destruct the subplugin */
void fini_filter_facemesh (void)
{
  nnstreamer_filter_exit (facemesh_custom.name);
}