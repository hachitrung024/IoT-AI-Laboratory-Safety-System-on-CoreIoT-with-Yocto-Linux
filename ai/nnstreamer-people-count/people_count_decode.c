#include <math.h>
#include <stdio.h>
#include <string.h>
#include <glib.h>
#include <nnstreamer_plugin_api_filter.h>

#define MAX_DETECTION    10
#define SCORE_THRESHOLD  0.5f
#define IOU_THRESHOLD    0.3f
#define MIN_BOX_AREA     0.01f
#define MERGE_DIST_SQ    (0.06f * 0.06f)   /* center-distance merge, normalized */
#define SMOOTH_WINDOW    3
#define PERSON_CLASS_ID  0

/*
 * Output layout: [count, x0,y0,w0,h0, x1,y1,w1,h1, ... up to MAX_DETECTION]
 * All coordinates are in pixel space. (x,y,w,h) so Cairo rectangle() works directly.
 */
#define OUTPUT_FLOATS (1 + MAX_DETECTION * 4)

void init_filter_people_count (void) __attribute__ ((constructor));
void fini_filter_people_count (void) __attribute__ ((destructor));

typedef struct {
  gchar *model_path;
  float  width_img;
  float  height_img;
  /* temporal smoothing: circular buffer of recent per-frame counts */
  int    count_history[SMOOTH_WINDOW];
  int    history_idx;
  int    history_len;   /* number of valid entries, ramps up to SMOOTH_WINDOW */
} people_count_pdata;

static void people_count_close (const GstTensorFilterProperties *prop,
                                void **private_data);

static void
parse_custom (people_count_pdata *pdata, const char *custom)
{
  if (!custom || sscanf (custom, "%f,%f", &pdata->width_img, &pdata->height_img) != 2) {
    pdata->width_img  = 640.0f;
    pdata->height_img = 480.0f;
  }
}

static int
people_count_reopen (const GstTensorFilterProperties *prop, void **private_data)
{
  people_count_pdata *pdata = *private_data;
  if (prop->num_models > 0 && pdata->model_path &&
      strcmp (prop->model_files[0], pdata->model_path) != 0)
    return 1;
  return 0;
}

static int
people_count_open (const GstTensorFilterProperties *prop, void **private_data)
{
  if (*private_data != NULL) {
    if (people_count_reopen (prop, private_data) != 0)
      people_count_close (prop, private_data);
    else
      return 1;
  }

  people_count_pdata *pdata = g_new0 (people_count_pdata, 1);
  if (!pdata)
    return -ENOMEM;

  *private_data = pdata;
  if (prop->num_models > 0)
    pdata->model_path = g_strdup (prop->model_files[0]);

  parse_custom (pdata, prop->custom_properties);
  return 0;
}

/*
 * Input tensors from TFLite_Detection_PostProcess:
 *   [0] boxes          float32[4][MAX_DETECTION]  — (ymin, xmin, ymax, xmax) normalized
 *   [1] classes        float32[MAX_DETECTION]
 *   [2] scores         float32[MAX_DETECTION]
 *   [3] num_detections float32[1]
 *
 * custom= property carries "width,height" of the source frame.
 */
static int
people_count_getInputDim (const GstTensorFilterProperties *prop,
                          void **private_data, GstTensorsInfo *info)
{
  parse_custom ((people_count_pdata *) *private_data, prop->custom_properties);

  info->num_tensors = 4;

  /* boxes [4, MAX_DETECTION] */
  info->info[0].type         = _NNS_FLOAT32;
  info->info[0].dimension[0] = 4;
  info->info[0].dimension[1] = MAX_DETECTION;
  info->info[0].dimension[2] = 1;
  info->info[0].dimension[3] = 1;

  /* classes [MAX_DETECTION] */
  info->info[1].type         = _NNS_FLOAT32;
  info->info[1].dimension[0] = MAX_DETECTION;
  info->info[1].dimension[1] = 1;
  info->info[1].dimension[2] = 1;
  info->info[1].dimension[3] = 1;

  /* scores [MAX_DETECTION] */
  info->info[2].type         = _NNS_FLOAT32;
  info->info[2].dimension[0] = MAX_DETECTION;
  info->info[2].dimension[1] = 1;
  info->info[2].dimension[2] = 1;
  info->info[2].dimension[3] = 1;

  /* num_detections [1] */
  info->info[3].type         = _NNS_FLOAT32;
  info->info[3].dimension[0] = 1;
  info->info[3].dimension[1] = 1;
  info->info[3].dimension[2] = 1;
  info->info[3].dimension[3] = 1;

  return 0;
}

static int
people_count_getOutputDim (const GstTensorFilterProperties *prop,
                           void **private_data, GstTensorsInfo *info)
{
  (void) prop;
  (void) private_data;
  info->num_tensors          = 1;
  info->info[0].type         = _NNS_FLOAT32;
  info->info[0].dimension[0] = OUTPUT_FLOATS;
  info->info[0].dimension[1] = 1;
  info->info[0].dimension[2] = 1;
  info->info[0].dimension[3] = 1;
  return 0;
}

static float
iou_calc (float ax1, float ay1, float ax2, float ay2,
          float bx1, float by1, float bx2, float by2)
{
  float ix1  = ax1 > bx1 ? ax1 : bx1;
  float iy1  = ay1 > by1 ? ay1 : by1;
  float ix2  = ax2 < bx2 ? ax2 : bx2;
  float iy2  = ay2 < by2 ? ay2 : by2;
  float iw   = ix2 - ix1;
  float ih   = iy2 - iy1;
  if (iw <= 0.0f || ih <= 0.0f)
    return 0.0f;
  float inter  = iw * ih;
  float area_a = (ax2 - ax1) * (ay2 - ay1);
  float area_b = (bx2 - bx1) * (by2 - by1);
  float uni    = area_a + area_b - inter;
  return uni > 0.0f ? inter / uni : 0.0f;
}

static inline float
clamp01 (float v)
{
  return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

typedef struct { float score, xmin, ymin, xmax, ymax; } Candidate;

/* Insertion sort descending by score (n <= MAX_DETECTION, O(n²) is fine). */
static void
isort_desc (Candidate *arr, int n)
{
  for (int i = 1; i < n; i++) {
    Candidate key = arr[i];
    int j = i - 1;
    while (j >= 0 && arr[j].score < key.score) {
      arr[j + 1] = arr[j];
      j--;
    }
    arr[j + 1] = key;
  }
}

static int
people_count_invoke (const GstTensorFilterProperties *prop, void **private_data,
                     const GstTensorMemory *input, GstTensorMemory *output)
{
  (void) prop;
  people_count_pdata *pdata = *private_data;

  /* Re-parse custom in case open() was called without custom_properties. */
  parse_custom (pdata, prop->custom_properties);

  /* TFLite_Detection_PostProcess output tensors, in order: */
  float *boxes_ptr = (float *) input[0].data;   /* [MAX_DETECTION][4]: ymin,xmin,ymax,xmax */
  float *class_ptr = (float *) input[1].data;   /* [MAX_DETECTION] */
  float *score_ptr = (float *) input[2].data;   /* [MAX_DETECTION] */
  float *count_ptr = (float *) input[3].data;   /* [1] */
  float *out_ptr   = (float *) output[0].data;

  float W = pdata->width_img;
  float H = pdata->height_img;

  memset (out_ptr, 0, sizeof (float) * OUTPUT_FLOATS);

  int raw_count = 0;  /* final per-frame count before smoothing */

  int det = (int) count_ptr[0];
  if (det > MAX_DETECTION)
    det = MAX_DETECTION;

  if (det > 0) {
    /* --- 1. Filter: person class, score threshold, minimum box area --- */
    Candidate cands[MAX_DETECTION];
    int nc = 0;
    for (int i = 0; i < det; i++) {
      if ((int) class_ptr[i] != PERSON_CLASS_ID)
        continue;
      float sc = score_ptr[i];
      if (sc < SCORE_THRESHOLD)
        continue;
      float ymin = clamp01 (boxes_ptr[i * 4 + 0]);
      float xmin = clamp01 (boxes_ptr[i * 4 + 1]);
      float ymax = clamp01 (boxes_ptr[i * 4 + 2]);
      float xmax = clamp01 (boxes_ptr[i * 4 + 3]);
      if (xmax <= xmin || ymax <= ymin)
        continue;
      if ((xmax - xmin) * (ymax - ymin) < MIN_BOX_AREA)
        continue;
      cands[nc++] = (Candidate){ sc, xmin, ymin, xmax, ymax };
    }

    if (nc > 0) {
      isort_desc (cands, nc);

      /* --- 2. NMS (IoU-based greedy suppression) --- */
      int suppressed[MAX_DETECTION] = {0};
      int nms_kept[MAX_DETECTION];
      int nk = 0;
      for (int i = 0; i < nc; i++) {
        if (suppressed[i])
          continue;
        nms_kept[nk++] = i;
        for (int j = i + 1; j < nc; j++) {
          if (suppressed[j])
            continue;
          if (iou_calc (cands[i].xmin, cands[i].ymin, cands[i].xmax, cands[i].ymax,
                        cands[j].xmin, cands[j].ymin, cands[j].xmax, cands[j].ymax)
              >= IOU_THRESHOLD)
            suppressed[j] = 1;
        }
      }

      /*
       * --- 3. Center-distance merge ---
       * Eliminates residual near-duplicate boxes that survive NMS when one box
       * is nested/offset enough that IoU is below the threshold.
       * Mirrors GstPersonDetector.py merge_center_dist=0.06 logic.
       */
      float merge_cx[MAX_DETECTION], merge_cy[MAX_DETECTION];
      int final_kept[MAX_DETECTION];
      int nf = 0;
      for (int i = 0; i < nk; i++) {
        Candidate *c = &cands[nms_kept[i]];
        float cx = (c->xmin + c->xmax) * 0.5f;
        float cy = (c->ymin + c->ymax) * 0.5f;
        int close = 0;
        for (int j = 0; j < nf; j++) {
          float dx = cx - merge_cx[j];
          float dy = cy - merge_cy[j];
          if (dx * dx + dy * dy <= MERGE_DIST_SQ) {
            close = 1;
            break;
          }
        }
        if (!close) {
          merge_cx[nf]   = cx;
          merge_cy[nf]   = cy;
          final_kept[nf] = nms_kept[i];
          nf++;
        }
      }

      /* Write pixel-coord boxes (x, y, width, height). */
      for (int i = 0; i < nf; i++) {
        Candidate *c = &cands[final_kept[i]];
        float x1 = c->xmin * W;
        float y1 = c->ymin * H;
        out_ptr[1 + i * 4 + 0] = x1;
        out_ptr[1 + i * 4 + 1] = y1;
        out_ptr[1 + i * 4 + 2] = c->xmax * W - x1;  /* width  */
        out_ptr[1 + i * 4 + 3] = c->ymax * H - y1;  /* height */
      }

      raw_count = nf;
    }
  }

  /* --- 4. Temporal smoothing (window = SMOOTH_WINDOW frames) ---
   * Mirrors GstPersonDetector.py count_history deque + round-average.
   */
  pdata->count_history[pdata->history_idx] = raw_count;
  pdata->history_idx = (pdata->history_idx + 1) % SMOOTH_WINDOW;
  if (pdata->history_len < SMOOTH_WINDOW)
    pdata->history_len++;

  int sum = 0;
  for (int i = 0; i < pdata->history_len; i++)
    sum += pdata->count_history[i];
  int smoothed = (int) roundf ((float) sum / pdata->history_len);

  out_ptr[0] = (float) smoothed;
  return 0;
}

static void
people_count_close (const GstTensorFilterProperties *prop, void **private_data)
{
  (void) prop;
  people_count_pdata *pdata = *private_data;
  if (pdata) {
    g_free (pdata->model_path);
    g_free (pdata);
    *private_data = NULL;
  }
}

static gchar filter_subplugin_people_count[] = "people_count_decode";

static GstTensorFilterFramework people_count_custom = {
#ifdef GST_TENSOR_FILTER_API_VERSION_DEFINED
  .version = GST_TENSOR_FILTER_FRAMEWORK_V0,
#else
  .name               = filter_subplugin_people_count,
  .allow_in_place     = FALSE,
  .allocate_in_invoke = FALSE,
  .run_without_model  = TRUE,
  .invoke_NN          = people_count_invoke,
  .getInputDimension  = people_count_getInputDim,
  .getOutputDimension = people_count_getOutputDim,
#endif
  .open  = people_count_open,
  .close = people_count_close,
};

void init_filter_people_count (void)
{
#ifdef GST_TENSOR_FILTER_API_VERSION_DEFINED
  people_count_custom.name               = filter_subplugin_people_count;
  people_count_custom.allow_in_place     = FALSE;
  people_count_custom.allocate_in_invoke = FALSE;
  people_count_custom.run_without_model  = TRUE;
  people_count_custom.invoke_NN          = people_count_invoke;
  people_count_custom.getInputDimension  = people_count_getInputDim;
  people_count_custom.getOutputDimension = people_count_getOutputDim;
#endif
  nnstreamer_filter_probe (&people_count_custom);
}

void fini_filter_people_count (void)
{
  nnstreamer_filter_exit (people_count_custom.name);
}
