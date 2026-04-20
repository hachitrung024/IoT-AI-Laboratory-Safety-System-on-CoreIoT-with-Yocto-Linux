#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

#include <nnstreamer_util.h>

#include <string.h>
#include <math.h>

/* ============================= */
/* CONFIG */
/* ============================= */
#define OUT_W 192
#define OUT_H 192
#define OUT_C 3

/* ============================= */
/* OBJECT STRUCT */
/* ============================= */
typedef struct _GstCropDecode
{
  GstBaseTransform parent;
} GstCropDecode;

typedef struct _GstCropDecodeClass
{
  GstBaseTransformClass parent_class;
} GstCropDecodeClass;

G_DEFINE_TYPE (GstCropDecode, gst_crop_decode, GST_TYPE_BASE_TRANSFORM);

/* ============================= */
/* PAD TEMPLATE */
/* ============================= */

/* nhận flexible tensor từ tensor_crop */
static GstStaticPadTemplate sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("other/tensors")
);

/* output tensor static cho model landmark */
static GstStaticPadTemplate src_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (
        "other/tensors, "
        "format=static, "
        "type=float32, "
        "dimension=(string)3:192:192:1"
    )
);

/* ============================= */
/* HELPER: RESIZE + NORMALIZE */
/* ============================= */
static void
resize_rgb_u8_to_f32_nn (const guint8 * src,
    guint sw, guint sh, guint sc,
    gfloat * dst,
    guint dw, guint dh, guint dc)
{
  for (guint y = 0; y < dh; y++) {
    guint sy = (guint) (((float) y * sh) / dh);
    if (sy >= sh)
      sy = sh - 1;

    for (guint x = 0; x < dw; x++) {
      guint sx = (guint) (((float) x * sw) / dw);
      if (sx >= sw)
        sx = sw - 1;

      guint src_idx = (sy * sw + sx) * sc;
      guint dst_idx = (y * dw + x) * dc;

      if (sc == 1 && dc == 3) {
        float v = src[src_idx] / 255.0f;
        dst[dst_idx + 0] = v;
        dst[dst_idx + 1] = v;
        dst[dst_idx + 2] = v;
      } else {
        for (guint c = 0; c < dc; c++) {
          guint8 sv = src[src_idx + (c < sc ? c : sc - 1)];
          dst[dst_idx + c] = sv / 255.0f;
        }
      }
    }
  }
}

/* ============================= */
/* TRANSFORM FUNCTION */
/* ============================= */
static GstFlowReturn
gst_crop_decode_transform (GstBaseTransform * base,
    GstBuffer * inbuf,
    GstBuffer * outbuf)
{
  GstMapInfo inmap, outmap;

  if (!gst_buffer_map (inbuf, &inmap, GST_MAP_READ))
    return GST_FLOW_ERROR;

  if (!gst_buffer_map (outbuf, &outmap, GST_MAP_WRITE)) {
    gst_buffer_unmap (inbuf, &inmap);
    return GST_FLOW_ERROR;
  }

  /* ============================= */
  /* PARSE FLEXIBLE TENSOR HEADER */
  /* ============================= */
  GstTensorMetaInfo meta;
  GstTensorInfo info;

  gst_tensor_meta_info_init (&meta);
  gst_tensor_info_init (&info);

  if (!gst_tensor_meta_info_parse_header (&meta, inmap.data)) {
    g_printerr ("[cropdecode] parse header failed\n");
    goto error;
  }

  if (!gst_tensor_meta_info_convert (&meta, &info)) {
    g_printerr ("[cropdecode] convert meta failed\n");
    goto error;
  }

  gsize hsize = gst_tensor_meta_info_get_header_size (&meta);
  gsize dsize = gst_tensor_meta_info_get_data_size (&meta);

  if (hsize + dsize > inmap.size) {
    g_printerr ("[cropdecode] invalid tensor size\n");
    goto error;
  }

  guint8 *raw = (guint8 *) inmap.data + hsize;

  guint sc = info.dimension[0];
  guint sw = info.dimension[1];
  guint sh = info.dimension[2];

  if (sc == 0 || sw == 0 || sh == 0) {
    g_printerr ("[cropdecode] invalid input dim\n");
    goto error;
  }

  /* ============================= */
  /* OUTPUT BUFFER */
  /* ============================= */
  gfloat *dst = (gfloat *) outmap.data;

  resize_rgb_u8_to_f32_nn (raw, sw, sh, sc,
      dst, OUT_W, OUT_H, OUT_C);

  gst_buffer_unmap (inbuf, &inmap);
  gst_buffer_unmap (outbuf, &outmap);

  return GST_FLOW_OK;

error:
  gst_buffer_unmap (inbuf, &inmap);
  gst_buffer_unmap (outbuf, &outmap);
  return GST_FLOW_ERROR;
}

/* ============================= */
/* SET CAPS */
/* ============================= */
static gboolean
gst_crop_decode_set_caps (GstBaseTransform * trans,
    GstCaps * incaps,
    GstCaps * outcaps)
{
  return TRUE;
}

/* ============================= */
/* CLASS INIT */
/* ============================= */
static void
gst_crop_decode_class_init (GstCropDecodeClass * klass)
{
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
  GstBaseTransformClass *trans_class = GST_BASE_TRANSFORM_CLASS (klass);

  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&sink_template));
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&src_template));

  gst_element_class_set_static_metadata (element_class,
      "Crop Decode",
      "Filter/Tensor",
      "Decode tensor_crop output to static tensor",
      "YourName");

  trans_class->transform = gst_crop_decode_transform;
  trans_class->set_caps = gst_crop_decode_set_caps;

  gst_base_transform_set_in_place (trans_class, FALSE);
}

/* ============================= */
/* INIT */
/* ============================= */
static void
gst_crop_decode_init (GstCropDecode * self)
{
}

/* ============================= */
/* PLUGIN INIT */
/* ============================= */
static gboolean
plugin_init (GstPlugin * plugin)
{
  return gst_element_register (plugin,
      "cropdecode",
      GST_RANK_NONE,
      gst_crop_decode_get_type ());
}

/* ============================= */
/* DEFINE PLUGIN */
/* ============================= */
GST_PLUGIN_DEFINE (
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    cropdecode,
    "Crop decode plugin",
    plugin_init,
    "1.0",
    "LGPL",
    "nnstreamer",
    "nnstreamer"
)