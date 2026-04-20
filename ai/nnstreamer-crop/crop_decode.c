#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <glib.h>
#include <math.h>

#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_util.h>

#define DEFAULT_OUT_W 192
#define DEFAULT_OUT_H 192
#define DEFAULT_OUT_C 3

void init_filter_crop (void) __attribute__ ((constructor));
void fini_filter_crop (void) __attribute__ ((destructor));

typedef struct
{
  gchar *model_path;
  guint out_w;
  guint out_h;
  guint out_c;
} crop_pdata;


/* ---------- Helpers ---------- */
static void
parse_custom_dims (const gchar * s, guint * w, guint * h, guint * c)
{
  guint tw = 0, th = 0, tc = 0;
  int n;

  if (!s || !*s)
    return;

  n = sscanf (s, "%u,%u,%u", &tw, &th, &tc);
  if (n >= 2) {
    *w = tw;
    *h = th;
    if (n >= 3 && tc > 0)
      *c = tc;
  }
}

static inline guint
clamp_u (gint v, guint lo, guint hi)
{
  if (v < (gint) lo)
    return lo;
  if (v > (gint) hi)
    return hi;
  return (guint) v;
}

static void
resize_rgb_u8_to_f32_nn (const guint8 * src,
    guint sw, guint sh, guint sc,
    gfloat * dst, guint dw, guint dh, guint dc)
{
  guint x, y, c;

  for (y = 0; y < dh; y++) {
    guint sy = (guint) (((gfloat) y * sh) / dh);
    if (sy >= sh)
      sy = sh - 1;

    for (x = 0; x < dw; x++) {
      guint sx = (guint) (((gfloat) x * sw) / dw);
      if (sx >= sw)
        sx = sw - 1;

      const guint src_idx = (sy * sw + sx) * sc;
      const guint dst_idx = (y * dw + x) * dc;

      if (sc == 1 && dc == 3) {
        gfloat v = ((gfloat) src[src_idx]) / 255.0f;
        dst[dst_idx + 0] = v;
        dst[dst_idx + 1] = v;
        dst[dst_idx + 2] = v;
      } else {
        for (c = 0; c < dc; c++) {
          guint8 sv = src[src_idx + (c < sc ? c : (sc - 1))];
          dst[dst_idx + c] = ((gfloat) sv) / 255.0f;
        }
      }
    }
  }
}

static void crop_close (const GstTensorFilterProperties * prop,
    void **private_data);

/**
 * Check condition to reopen model.
 */
static int crop_reopen (const GstTensorFilterProperties * prop,
    void **private_data)
{
    crop_pdata *pdata = *private_data;

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
static int crop_open (const GstTensorFilterProperties * prop,
    void **private_data)
{
    crop_pdata *pdata;

    if (*private_data != NULL) {
        if (crop_reopen(prop, private_data) != 0) {
            crop_close(prop, private_data);
        } else {
            return 1;
        }
    }

    pdata = g_new0(crop_pdata, 1);
    if (pdata == NULL)
        return -ENOMEM;

    pdata->out_w = DEFAULT_OUT_W;
    pdata->out_h = DEFAULT_OUT_H;
    pdata->out_c = DEFAULT_OUT_C;

    if (prop && prop->custom_properties) {
        /* custom="192,192" or "192,192,3" */
        parse_custom_dims (prop->custom_properties,
            &pdata->out_w, &pdata->out_h, &pdata->out_c);
    }

    *private_data = (void *) pdata;

    if (prop->num_models > 0)
        pdata->model_path = g_strdup(prop->model_files[0]);

    g_print("[crop_decode] Loaded model: %s\n", pdata->model_path);

    return 0;
}

/**
 * @brief The standard tensor_filter callback for static input/output dimension.
 * @note If you want to support flexible/dynamic input/output dimension,
 *       read nnstreamer_plugin_api_filter.h and supply the
 *       setInputDimension callback.
 */
static int crop_setInputDim (const GstTensorFilterProperties * prop,
    void **private_data, const GstTensorsInfo * in_info, GstTensorsInfo * out_info)
{
    crop_pdata *pdata = (crop_pdata *) (*private_data);

    if (!pdata || !in_info || !out_info)
        return -EINVAL;

    if (in_info->num_tensors < 1)
        return -EINVAL;

    memset (out_info, 0, sizeof (GstTensorsInfo));
    out_info->num_tensors = 1;

    /* output for face landmark: float32 RGB, 192x192 */
    out_info->info[0].type = _NNS_FLOAT32;
    out_info->info[0].dimension[0] = pdata->out_c;  /* 3 */
    out_info->info[0].dimension[1] = pdata->out_w;  /* 192 */
    out_info->info[0].dimension[2] = pdata->out_h;  /* 192 */
    out_info->info[0].dimension[3] = 1;

    return 0;
}

/**
 * @brief The standard tensor_filter callback for static input/output dimension.
 * @note If you want to support flexible/dynamic input/output dimension,
 *       read nnstreamer_plugin_api_filter.h and supply the
 *       setInputDimension callback.
 */
static int crop_getOutputDim (const GstTensorFilterProperties * prop,
    void **private_data, GstTensorsInfo * info)
{
    crop_pdata *pdata = (crop_pdata *) (*private_data);

    if (!pdata || !info)
        return -EINVAL;

    memset (info, 0, sizeof (GstTensorsInfo));
    info->num_tensors = 1;
    info->info[0].type = _NNS_FLOAT32;
    info->info[0].dimension[0] = pdata->out_c;
    info->info[0].dimension[1] = pdata->out_w;
    info->info[0].dimension[2] = pdata->out_h;
    info->info[0].dimension[3] = 1;

    return 0;
}

/**
 * Invoke sub-plugin
 */
static int crop_invoke (const GstTensorFilterProperties * prop,
    void **private_data,
    const GstTensorMemory * input,
    GstTensorMemory * output)
{
    crop_pdata *pdata = (crop_pdata *) (*private_data);

    GstTensorMetaInfo meta;
    GstTensorInfo in_tensor_info;
    gsize hsize, dsize;
    const guint8 *raw;
    guint sw, sh, sc;
    gfloat *dst;

    if (!pdata || !input || !output)
        return -EINVAL;

    if (!input[0].data || input[0].size == 0)
        return -EINVAL;

    gst_tensor_meta_info_init (&meta);
    gst_tensor_info_init (&in_tensor_info);

    /* tensor_crop output is flexible tensor: [header][payload] */
    if (!gst_tensor_meta_info_parse_header (&meta, input[0].data)) {
        g_printerr ("[crop_decode] failed to parse tensor header\n");
        return -EINVAL;
    }

    if (!gst_tensor_meta_info_convert (&meta, &in_tensor_info)) {
        g_printerr ("[crop_decode] failed to convert meta to info\n");
        return -EINVAL;
    }

    hsize = gst_tensor_meta_info_get_header_size (&meta);
    dsize = gst_tensor_meta_info_get_data_size (&meta);

    if (hsize + dsize > input[0].size) {
        g_printerr ("[crop_decode] invalid tensor size: got %zu, expected at least %zu\n",
            (size_t) input[0].size, (size_t) (hsize + dsize));
        return -EINVAL;
    }

    raw = (const guint8 *) input[0].data + hsize;

    sc = in_tensor_info.dimension[0];
    sw = in_tensor_info.dimension[1];
    sh = in_tensor_info.dimension[2];

    if (sc < 1 || sw < 1 || sh < 1) {
        g_printerr ("[crop_to_face_landmark] invalid input dims\n");
        return -EINVAL;
    }

    dst = (gfloat *) output[0].data;
    if (!dst)
        return -EINVAL;

    resize_rgb_u8_to_f32_nn (raw, sw, sh, sc, dst, pdata->out_w, pdata->out_h, pdata->out_c);

    return 0;
}

/**
 * Close sub-plugin
 */
static void crop_close (const GstTensorFilterProperties * prop,
    void **private_data)
{
    crop_pdata *pdata = (crop_pdata *) (*private_data);

    if (pdata) {
        g_print("[crop_decode] Closing: %s\n", pdata->model_path);

        g_free(pdata->model_path);
        pdata->model_path = NULL;

        g_free(pdata);
        *private_data = NULL;
    }
}

/**
 * Register sub-plugin
 */
static gchar filter_subplugin_crop[] = "crop_decode";

static GstTensorFilterFramework crop_custom = {
#ifdef GST_TENSOR_FILTER_API_VERSION_DEFINED
    .version = GST_TENSOR_FILTER_FRAMEWORK_V0,
#else
    .name = filter_subplugin_crop,
    .allow_in_place = FALSE,
    .allocate_in_invoke = FALSE,
    .run_without_model = TRUE,
    .invoke_NN = crop_invoke,
    .setInputDimension  = crop_setInputDim,
    .getOutputDimension = crop_getOutputDim,
#endif
    .open = crop_open,
    .close = crop_close,
};

void init_filter_crop (void)
{
#ifdef GST_TENSOR_FILTER_API_VERSION_DEFINED
    crop_custom.name = filter_subplugin_crop;
    crop_custom.allow_in_place = FALSE;
    crop_custom.allocate_in_invoke = FALSE;
    crop_custom.run_without_model = TRUE;
    crop_custom.invoke_NN = crop_invoke;
    crop_custom.setInputDimension = crop_setInputDim;
    crop_custom.getOutputDimension = crop_getOutputDim;
#endif

    nnstreamer_filter_probe(&crop_custom);
}

void fini_filter_crop (void)
{
    nnstreamer_filter_exit(crop_custom.name);
}