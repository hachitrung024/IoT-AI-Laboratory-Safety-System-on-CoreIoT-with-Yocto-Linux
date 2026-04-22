#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

#include <nnstreamer_plugin_api.h>
#include <nnstreamer_util.h>

#include <string.h>
#include <math.h>

#ifndef PACKAGE
#define PACKAGE "crop_view"
#endif

#define OUT_W 192
#define OUT_H 192
#define OUT_C 3

typedef struct _GstCropView
{
  GstBaseTransform parent;
} GstCropView;

typedef struct _GstCropViewClass
{
  GstBaseTransformClass parent_class;
} GstCropViewClass;

G_DEFINE_TYPE (GstCropView, gst_crop_view, GST_TYPE_BASE_TRANSFORM);

static GstCaps *
make_sink_caps (void)
{
  return gst_caps_from_string (
      "other/tensors, "
      "format=(string)static, "
      "num_tensors=(int)1, "
      "types=(string)float32, "
      "dimensions=(string)3:192:192:1");
}

static GstCaps *
make_src_caps (void)
{
  return gst_caps_from_string (
      "video/x-raw, "
      "format=(string)RGB, "
      "width=(int)192, "
      "height=(int)192, "
      "pixel-aspect-ratio=(fraction)1/1");
}

static inline guint8
clamp_u8_from_float (gfloat v)
{
  if (v < 0.0f)
    v = 0.0f;
  if (v > 1.0f)
    v = 1.0f;
  return (guint8) (v * 255.0f + 0.5f);
}

/* Input:  float32 CHW = [3][192][192]
 * Output: uint8  HWC  = [192][192][3]
 */
static GstFlowReturn
gst_crop_view_transform (GstBaseTransform * trans,
                         GstBuffer * inbuf,
                         GstBuffer * outbuf)
{
  GstMapInfo inmap, outmap;

  (void) trans;

  if (!gst_buffer_map (inbuf, &inmap, GST_MAP_READ)) {
    g_printerr ("[crop_view] map input failed\n");
    return GST_FLOW_ERROR;
  }

  if (!gst_buffer_map (outbuf, &outmap, GST_MAP_WRITE)) {
    g_printerr ("[crop_view] map output failed\n");
    gst_buffer_unmap (inbuf, &inmap);
    return GST_FLOW_ERROR;
  }

  const gsize expected_in = OUT_W * OUT_H * OUT_C * sizeof (gfloat);
  const gsize expected_out = OUT_W * OUT_H * OUT_C;

  if (inmap.size < expected_in || outmap.size < expected_out) {
    g_printerr ("[crop_view] invalid buffer size in=%" G_GSIZE_FORMAT
                " out=%" G_GSIZE_FORMAT "\n",
                inmap.size, outmap.size);
    gst_buffer_unmap (inbuf, &inmap);
    gst_buffer_unmap (outbuf, &outmap);
    return GST_FLOW_ERROR;
  }

  const gfloat *src = (const gfloat *) inmap.data;
  guint8 *dst = (guint8 *) outmap.data;

  /* CHW -> HWC */
  for (guint y = 0; y < OUT_H; y++) {
    for (guint x = 0; x < OUT_W; x++) {
      guint src_idx_r = 0 * (OUT_W * OUT_H) + y * OUT_W + x;
      guint src_idx_g = 1 * (OUT_W * OUT_H) + y * OUT_W + x;
      guint src_idx_b = 2 * (OUT_W * OUT_H) + y * OUT_W + x;

      guint dst_idx = (y * OUT_W + x) * OUT_C;

      dst[dst_idx + 0] = clamp_u8_from_float (src[src_idx_r]);
      dst[dst_idx + 1] = clamp_u8_from_float (src[src_idx_g]);
      dst[dst_idx + 2] = clamp_u8_from_float (src[src_idx_b]);
    }
  }

  gst_buffer_copy_into (outbuf, inbuf,
      GST_BUFFER_COPY_TIMESTAMPS | GST_BUFFER_COPY_FLAGS, 0, -1);

  gst_buffer_unmap (inbuf, &inmap);
  gst_buffer_unmap (outbuf, &outmap);
  return GST_FLOW_OK;
}

static GstCaps *
gst_crop_view_transform_caps (GstBaseTransform * trans,
                              GstPadDirection direction,
                              GstCaps * caps,
                              GstCaps * filter)
{
  GstCaps *result = NULL;
  GstCaps *sink_caps = make_sink_caps ();
  GstCaps *src_caps = make_src_caps ();

  (void) trans;

  if (direction == GST_PAD_SINK) {
    /* sink -> src */
    result = gst_caps_ref (src_caps);
  } else {
    /* src -> sink */
    result = gst_caps_ref (sink_caps);
  }

  if (caps && gst_caps_get_size (caps) > 0) {
    s = gst_caps_get_structure (caps, 0);
    v = gst_structure_get_value (s, "framerate");
    if (v) {
      gst_caps_set_value (result, "framerate", v);
      // g_print ("[DEBUG] [crop_decode] Propagated framerate to next element\n");
    }
  }

  if (filter) {
    GstCaps *intersection =
        gst_caps_intersect_full (filter, result, GST_CAPS_INTERSECT_FIRST);
    gst_caps_unref (result);
    result = intersection;
  }

  gst_caps_unref (sink_caps);
  gst_caps_unref (src_caps);
  return result;
}

static gboolean
gst_crop_view_transform_size (GstBaseTransform * trans,
                              GstPadDirection direction,
                              GstCaps * caps,
                              gsize size,
                              GstCaps * othercaps,
                              gsize * othersize)
{
  (void) trans;
  (void) caps;
  (void) othercaps;

  if (direction == GST_PAD_SINK) {
    /* input tensor -> output RGB bytes */
    *othersize = OUT_W * OUT_H * OUT_C;
  } else {
    /* reverse negotiation */
    *othersize = OUT_W * OUT_H * OUT_C * sizeof (gfloat);
  }

  (void) size;
  return TRUE;
}

static void
gst_crop_view_class_init (GstCropViewClass * klass)
{
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
  GstBaseTransformClass *base = GST_BASE_TRANSFORM_CLASS (klass);

  GstCaps *sink_caps = make_sink_caps ();
  GstCaps *src_caps = make_src_caps ();

  GstPadTemplate *sink_tmpl =
      gst_pad_template_new ("sink", GST_PAD_SINK, GST_PAD_ALWAYS, sink_caps);
  GstPadTemplate *src_tmpl =
      gst_pad_template_new ("src", GST_PAD_SRC, GST_PAD_ALWAYS, src_caps);

  gst_caps_set_simple (sink_caps, "framerate", GST_TYPE_FRACTION_RANGE, 0, 1, G_MAXINT, 1, NULL);
  gst_caps_set_simple (src_caps, "framerate", GST_TYPE_FRACTION_RANGE, 0, 1, G_MAXINT, 1, NULL);

  gst_element_class_add_pad_template (element_class, sink_tmpl);
  gst_element_class_add_pad_template (element_class, src_tmpl);

  gst_object_unref (sink_tmpl);
  gst_object_unref (src_tmpl);
  gst_caps_unref (sink_caps);
  gst_caps_unref (src_caps);

  gst_element_class_set_static_metadata (
      element_class,
      "Crop View",
      "Filter/Video",
      "Convert crop tensor to RGB video for display",
      "you");

  base->transform = gst_crop_view_transform;
  base->transform_caps = gst_crop_view_transform_caps;
  base->transform_size = gst_crop_view_transform_size;
  base->passthrough_on_same_caps = FALSE;
  base->transform_ip_on_passthrough = FALSE;
}

static void
gst_crop_view_init (GstCropView * self)
{
  (void) self;
}

static gboolean
plugin_init (GstPlugin * plugin)
{
  return gst_element_register (plugin,
      "crop_view",
      GST_RANK_NONE,
      gst_crop_view_get_type ());
}

GST_PLUGIN_DEFINE (
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    cropview,
    "Crop view plugin",
    plugin_init,
    "1.0",
    "MIT",
    "you",
    "https://example.com")