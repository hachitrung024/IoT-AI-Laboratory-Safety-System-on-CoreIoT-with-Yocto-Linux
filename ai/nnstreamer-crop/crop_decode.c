#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

#include <nnstreamer_plugin_api.h>
#include <nnstreamer_util.h>

#include <string.h>
#include <math.h>

#ifndef PACKAGE
#define PACKAGE "crop_decode"
#endif

#define OUT_W 192
#define OUT_H 192
#define OUT_C 3

typedef struct _GstCropDecode
{
  GstBaseTransform parent;
} GstCropDecode;

typedef struct _GstCropDecodeClass
{
  GstBaseTransformClass parent_class;
} GstCropDecodeClass;

G_DEFINE_TYPE (GstCropDecode, gst_crop_decode, GST_TYPE_BASE_TRANSFORM);

/* ========================= */
/* Resize */
/* ========================= */
static void
resize_nn (const guint8 * src,
           guint sw, guint sh, guint sc,
           gfloat * dst)
{
  for (guint y = 0; y < OUT_H; y++) {
    guint sy = (y * sh) / OUT_H;
    if (sy >= sh) sy = sh - 1;

    for (guint x = 0; x < OUT_W; x++) {
      guint sx = (x * sw) / OUT_W;
      if (sx >= sw) sx = sw - 1;

      guint sidx = (sy * sw + sx) * sc;
      guint didx = (y * OUT_W + x) * OUT_C;

      for (guint c = 0; c < OUT_C; c++) {
        guint8 v = src[sidx + (c < sc ? c : sc - 1)];
        dst[didx + c] = v / 255.0f;
      }
    }
  }
}

/* ========================= */
/* Transform */
/* ========================= */
static GstFlowReturn
gst_crop_decode_transform (GstBaseTransform * trans,
                           GstBuffer * inbuf,
                           GstBuffer * outbuf)
{
  GstMapInfo inmap;
  GstMemory *inmem;
  GstTensorMetaInfo meta;
  GstTensorInfo info;
  gsize hsize;
  gsize outsize;
  guint8 *dst;
  GstMemory *omem;

  if (gst_buffer_n_memory (inbuf) < 1) {
    g_printerr ("[crop_decode] input has no memory\n");
    return GST_FLOW_ERROR;
  }

  inmem = gst_buffer_peek_memory (inbuf, 0);
  if (!gst_memory_map (inmem, &inmap, GST_MAP_READ)) {
    g_printerr ("[crop_decode] map input memory failed\n");
    return GST_FLOW_ERROR;
  }

  gst_tensor_meta_info_init (&meta);
  gst_tensor_info_init (&info);

  if (!gst_tensor_meta_info_parse_header (&meta, inmap.data)) {
    g_printerr ("[crop_decode] parse header failed\n");
    gst_memory_unmap (inmem, &inmap);
    return GST_FLOW_ERROR;
  }

  if (!gst_tensor_meta_info_validate (&meta)) {
    g_printerr ("[crop_decode] meta invalid after parse\n");
    gst_memory_unmap (inmem, &inmap);
    return GST_FLOW_ERROR;
  }

  if (!gst_tensor_meta_info_convert (&meta, &info)) {
    g_printerr ("[crop_decode] convert failed\n");
    gst_memory_unmap (inmem, &inmap);
    return GST_FLOW_ERROR;
  }

  hsize = gst_tensor_meta_info_get_header_size (&meta);

  guint sc = info.dimension[0];
  guint sw = info.dimension[1];
  guint sh = info.dimension[2];

  if (sc == 0 || sw == 0 || sh == 0) {
    g_printerr ("[crop_decode] invalid input dimension\n");
    gst_memory_unmap (inmem, &inmap);
    return GST_FLOW_ERROR;
  }

  outsize = OUT_W * OUT_H * OUT_C * sizeof (gfloat);
  dst = g_malloc0 (outsize);
  if (!dst) {
    gst_memory_unmap (inmem, &inmap);
    return GST_FLOW_ERROR;
  }

  resize_nn ((const guint8 *) inmap.data + hsize, sw, sh, sc, (gfloat *) dst);

  gst_memory_unmap (inmem, &inmap);

  gst_buffer_remove_all_memory (outbuf);
  omem = gst_memory_new_wrapped (0, dst, outsize, 0, outsize, dst, g_free);
  gst_buffer_append_memory (outbuf, omem);

  gst_buffer_copy_into (outbuf, inbuf, GST_BUFFER_COPY_METADATA, 0, -1);

  return GST_FLOW_OK;
}

/* ========================= */
/* Caps */
/* ========================= */
static GstCaps *
gst_crop_decode_transform_caps (GstBaseTransform * trans,
                                GstPadDirection direction,
                                GstCaps * caps,
                                GstCaps * filter)
{
  return gst_caps_new_simple ("other/tensors",
      "num_tensors", G_TYPE_INT, 1,
      "types", G_TYPE_STRING, "float32",
      "dimensions", G_TYPE_STRING, "3:192:192:1",
      NULL);
}

static gboolean
gst_crop_decode_transform_size (GstBaseTransform * trans,
                                GstPadDirection direction,
                                GstCaps * caps,
                                gsize size,
                                GstCaps * othercaps,
                                gsize * othersize)
{
  if (direction == GST_PAD_SINK) {
    *othersize = OUT_W * OUT_H * OUT_C * sizeof (gfloat);
    return TRUE;
  }

  *othersize = size;
  return TRUE;
}

/* ========================= */
/* Class */
/* ========================= */
static void
gst_crop_decode_class_init (GstCropDecodeClass * klass)
{
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
  GstBaseTransformClass *base = GST_BASE_TRANSFORM_CLASS (klass);

  /* ========================= */
  /* PAD TEMPLATE */
  /* ========================= */

  GstCaps *sink_caps = gst_caps_new_any ();  // flexible tensor
  GstCaps *src_caps = gst_caps_new_simple ("other/tensors",
      "num_tensors", G_TYPE_INT, 1,
      "types", G_TYPE_STRING, "float32",
      "dimensions", G_TYPE_STRING, "3:192:192:1",
      NULL);

  gst_element_class_add_pad_template (
      element_class,
      gst_pad_template_new ("sink",
          GST_PAD_SINK,
          GST_PAD_ALWAYS,
          sink_caps));

  gst_element_class_add_pad_template (
      element_class,
      gst_pad_template_new ("src",
          GST_PAD_SRC,
          GST_PAD_ALWAYS,
          src_caps));

  /* ========================= */
  /* METADATA */
  /* ========================= */

  gst_element_class_set_static_metadata (
      element_class,
      "Crop Decode",
      "Filter/Tensor",
      "Decode tensor_crop output",
      "you");

  /* ========================= */
  /* TRANSFORM */
  /* ========================= */

  base->transform = gst_crop_decode_transform;
  base->transform_size = gst_crop_decode_transform_size;

  /* rất quan trọng */
  // gst_base_transform_class_set_passthrough (base, FALSE);
  base->passthrough_on_same_caps = FALSE;
}

/* ========================= */
/* Init */
/* ========================= */
static void
gst_crop_decode_init (GstCropDecode * self)
{
}

/* ========================= */
/* Plugin */
/* ========================= */
static gboolean
plugin_init (GstPlugin * plugin)
{
  return gst_element_register (plugin,
      "crop_decode",
      GST_RANK_NONE,
      gst_crop_decode_get_type ());
}

GST_PLUGIN_DEFINE (
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    cropdecode,
    "Crop decode plugin",
    plugin_init,
    "1.0",
    "MIT",
    "nnstreamer",
    "https://github.com"
)