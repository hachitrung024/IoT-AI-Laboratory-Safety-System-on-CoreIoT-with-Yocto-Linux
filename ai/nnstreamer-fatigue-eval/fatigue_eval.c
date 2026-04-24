# include <math.h>
# include <stdio.h>
#include <string.h>
#include <glib.h>
#include <nnstreamer_plugin_api_filter.h>

#define NUM_LANDMARKS 468
#define LANDMARK_DIM 2

#define WIDTH_IMG 640
#define HEIGHT_IMG 480

/* Output:
 * [0] state
 * [1] fatigue_score
 * [2] distraction_score
 * [3] left_EAR
 * [4] right_EAR
 * [5] blink_rate_per_min
 * [6] closed_duration_ms
 * [7] MAR
 * [8] yawn_hold_ms
 * [9] head_roll_deg
 * [10] head_yaw_proxy
 * [11] head_pitch_proxy
 * [12] gaze_x_proxy
 * [13] gaze_y_proxy
 */
#define OUTPUT_DIM 14

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ===== Thresholds ===== */
#define EAR_CLOSE_THRESHOLD            0.18f
#define EAR_WARNING_THRESHOLD          0.23f
#define EAR_WARNING_HOLD_MS            500u
#define EAR_CLOSED_HOLD_MS             1500u

#define BLINK_MIN_MS                   60u
#define BLINK_MAX_MS                   700u
#define BLINK_WINDOW_MS                60000u
#define BLINK_LOW_THRESHOLD_PER_MIN    8.0f
#define BLINK_HISTORY_MAX              128

#define MAR_YAWN_THRESHOLD             0.35f
#define YAWN_HOLD_MS                   800u

#define DISTRACTION_SCORE_THRESHOLD     0.60f
#define DISTRACTION_HOLD_MS            1500u

#define HEAD_ROLL_THRESHOLD_DEG        15.0f
#define HEAD_YAW_PROXY_THRESHOLD       0.18f
#define HEAD_PITCH_PROXY_THRESHOLD     0.20f

/* ===== MediaPipe Face Mesh indices =====
 * EAR eyes left/right
 * MAR mouth
 * Head pose proxy
 */
static const int LEFT_EYE_IDX[6]  = { 33, 160, 158, 133, 153, 144 };
static const int RIGHT_EYE_IDX[6] = { 362, 385, 387, 263, 373, 380 };

/* Mouth: corners + upper/lower lip */
static const int MOUTH_LEFT  = 61;
static const int MOUTH_RIGHT = 291;
static const int MOUTH_UP    = 13;
static const int MOUTH_DOWN  = 14;

/* Head proxy points */
static const int NOSE_TIP    = 1;
static const int LEFT_EYE_OUT = 33;
static const int RIGHT_EYE_OUT = 263;

void init_filter_fatigue_eval (void) __attribute__ ((constructor));
void fini_filter_fatigue_eval (void) __attribute__ ((destructor));

typedef enum
{
  F_STATE_NORMAL = 0,
  F_STATE_WARNING = 1,
  F_STATE_TIRED = 2,
  F_STATE_DISTRACTED = 3,
  F_STATE_NO_FACE = 4
} FatigueState;

typedef struct {
  gchar *model_path;
  float width_img;
  float height_img;

  /* blink tracking */
  gboolean eyes_was_closed;
  gint64 closed_since_us;

  gint64 blink_ts_us[BLINK_HISTORY_MAX];
  guint blink_ts_count;

  /* yawn tracking */
  gboolean mouth_was_open;
  gint64 yawn_since_us;

  /* distraction tracking */
  gboolean looking_away;
  gint64 looking_away_since_us;
} fatigue_pdata;

/* ===== Helper ===== */
static inline float
clampf01 (float v)
{
  if (v < 0.0f)
    return 0.0f;
  if (v > 1.0f)
    return 1.0f;
  return v;
}

static inline float
dist2f (float x1, float y1, float x2, float y2)
{
  float dx = x1 - x2;
  float dy = y1 - y2;
  return sqrtf (dx * dx + dy * dy);
}

static inline float
degf (float rad)
{
  return rad * (180.0f / (float) M_PI);
}

static inline float
get_x (const float *lm, int idx)
{
  return lm[idx * 2 + 0];
}

static inline float
get_y (const float *lm, int idx)
{
  return lm[idx * 2 + 1];
}

static gboolean
landmarks_valid (const float *lm)
{
  /* Check some important milestones; if all are zero, consider it as not present */
  const int check_idx[] = {
    NOSE_TIP, LEFT_EYE_OUT, RIGHT_EYE_OUT, MOUTH_LEFT, MOUTH_RIGHT,
    MOUTH_UP, MOUTH_DOWN
  };

  for (guint i = 0; i < sizeof (check_idx) / sizeof (check_idx[0]); i++) {
    int idx = check_idx[i];
    if (get_x (lm, idx) != 0.0f || get_y (lm, idx) != 0.0f)
      return TRUE;
  }

  return FALSE;
}

static float
compute_ear (const float *lm, const int idx[6])
{
  // p1 —— p2 —— p3
  // |    Eyes    |
  // |            |
  // p6 —— p5 —— p4
  float x1 = get_x (lm, idx[0]);
  float y1 = get_y (lm, idx[0]);
  float x2 = get_x (lm, idx[1]);
  float y2 = get_y (lm, idx[1]);
  float x3 = get_x (lm, idx[2]);
  float y3 = get_y (lm, idx[2]);
  float x4 = get_x (lm, idx[3]);
  float y4 = get_y (lm, idx[3]);
  float x5 = get_x (lm, idx[4]);
  float y5 = get_y (lm, idx[4]);
  float x6 = get_x (lm, idx[5]);
  float y6 = get_y (lm, idx[5]);

  float vertical_1 = dist2f (x2, y2, x6, y6);
  float vertical_2 = dist2f (x3, y3, x5, y5);
  float horizontal = dist2f (x1, y1, x4, y4);

  if (horizontal < 1e-6f)
    return 0.0f;

  // EAR = ((p2 - p6) + (p3 - p5)) / (2 * (p1 - p4))
  return (vertical_1 + vertical_2) / (2.0f * horizontal);
}

static float
compute_mar (const float *lm)
{
  //         (MOUTH_UP)
  //            ●
  //            |
  //            |
  // ●-------------------●
  // LEFT              RIGHT
  //            |
  //            |
  //            ●
  //       (MOUTH_DOWN)

  float x_left  = get_x (lm, MOUTH_LEFT);
  float y_left  = get_y (lm, MOUTH_LEFT);
  float x_right = get_x (lm, MOUTH_RIGHT);
  float y_right = get_y (lm, MOUTH_RIGHT);
  float x_up    = get_x (lm, MOUTH_UP);
  float y_up    = get_y (lm, MOUTH_UP);
  float x_down  = get_x (lm, MOUTH_DOWN);
  float y_down  = get_y (lm, MOUTH_DOWN);

  float horizontal = dist2f (x_left, y_left, x_right, y_right);
  float vertical = dist2f (x_up, y_up, x_down, y_down);

  if (horizontal < 1e-6f)
    return 0.0f;

  // MAR = (MOUTH_UP - MOUTH_DOWN) / (MOUTH_LEFT - MOUTH_RIGHT)
  return vertical / horizontal;
}

/* Remove old blinks <= BLINK_WINDOW_MS */
static void
prune_blinks (fatigue_pdata *pdata, gint64 now_us)
{
  gint64 cutoff_us = now_us - ((gint64) BLINK_WINDOW_MS * 1000LL);
  guint dst = 0;

  for (guint i = 0; i < pdata->blink_ts_count; i++) {
    if (pdata->blink_ts_us[i] >= cutoff_us) {
      pdata->blink_ts_us[dst++] = pdata->blink_ts_us[i];
    }
  }

  pdata->blink_ts_count = dst;
}

static void
push_blink (fatigue_pdata *pdata, gint64 blink_ts_us)
{
  if (pdata->blink_ts_count < BLINK_HISTORY_MAX) {
    pdata->blink_ts_us[pdata->blink_ts_count++] = blink_ts_us;
  } else {
    memmove (&pdata->blink_ts_us[0], &pdata->blink_ts_us[1],
        sizeof (gint64) * (BLINK_HISTORY_MAX - 1));
    pdata->blink_ts_us[BLINK_HISTORY_MAX - 1] = blink_ts_us;
  }
}

static float
compute_eye_center_x (const float *lm)
{
      // ● (33)         ● (263)
      //  \            /
      //   \          /
      //    ●--------●
      //   (133)   (362)
  float lx = (get_x (lm, LEFT_EYE_OUT) + get_x (lm, 133)) * 0.5f;
  float rx = (get_x (lm, RIGHT_EYE_OUT) + get_x (lm, 362)) * 0.5f;
  return (lx + rx) * 0.5f;
}

static float
compute_eye_center_y (const float *lm)
{
      // ● (33)         ● (263)
      //  \            /
      //   \          /
      //    ●--------●
      //   (133)   (362)  
  float ly = (get_y (lm, LEFT_EYE_OUT) + get_y (lm, 133)) * 0.5f;
  float ry = (get_y (lm, RIGHT_EYE_OUT) + get_y (lm, 362)) * 0.5f;
  return (ly + ry) * 0.5f;
}

static float
compute_eye_distance (const float *lm)
{
//(eye left) ● ------------------------- ● (eye right)
//             ←----- eye_distance ----→  
  return dist2f (get_x (lm, LEFT_EYE_OUT), get_y (lm, LEFT_EYE_OUT),
      get_x (lm, RIGHT_EYE_OUT), get_y (lm, RIGHT_EYE_OUT));
}

/* ===== Fatigue sub-plugin ===== */
static void fatigue_eval_close (const GstTensorFilterProperties * prop,
    void **private_data);

/**
 * Check condition to reopen model.
 */
static int
fatigue_eval_reopen (const GstTensorFilterProperties * prop, void **private_data)
{
  fatigue_pdata *pdata = *private_data;

  if (prop->num_models > 0 && pdata->model_path && strcmp (prop->model_files[0], pdata->model_path) != 0) {
    return 1;
  }

  return 0;
}

/**
 * Init sub-plugin
 */
static int
fatigue_eval_open (const GstTensorFilterProperties * prop, void **private_data)
{
  fatigue_pdata *pdata;

  if (*private_data != NULL) {
    if (fatigue_eval_reopen (prop, private_data) != 0) {
      fatigue_eval_close (prop, private_data);
    } else {
      return 1; 
    }
  }

  pdata = g_new0 (fatigue_pdata, 1);
  if (pdata == NULL)
    return -ENOMEM;

  *private_data = (void *) pdata;

  if (prop->num_models > 0)
    pdata->model_path = g_strdup (prop->model_files[0]);

  pdata->width_img = WIDTH_IMG;
  pdata->height_img = HEIGHT_IMG;

  // Eyes tracking
  pdata->eyes_was_closed = FALSE;
  pdata->closed_since_us = 0;

  // Blink tracking
  pdata->blink_ts_count = 0;

  // Yawn tracking
  pdata->mouth_was_open = FALSE;
  pdata->yawn_since_us = 0;

  // Distraction tracking
  pdata->looking_away = FALSE;
  pdata->looking_away_since_us = 0;

  g_print ("[fatigue_eval] Loaded model: %s\n", pdata->model_path);

  return 0;
}

/**
 * @brief The standard tensor_filter callback for static input/output dimension.
 * @note If you want to support flexible/dynamic input/output dimension,
 *       read nnstreamer_plugin_api_filter.h and supply the
 *       setInputDimension callback.
 */
static int
fatigue_eval_getInputDim (const GstTensorFilterProperties * prop,
    void **private_data, GstTensorsInfo * info)
{
  // Input: 468 landmarks have been decoded into [x, y]
  info->num_tensors = 1;

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
fatigue_eval_getOutputDim (const GstTensorFilterProperties * prop,
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
fatigue_eval_invoke (const GstTensorFilterProperties * prop, void **private_data,
              const GstTensorMemory * input, GstTensorMemory * output)
{
  fatigue_pdata *pdata = (fatigue_pdata *) (*private_data);
  float *lm = (float *) input[0].data;
  float *out = (float *) output[0].data;

  memset (out, 0, sizeof (float) * OUTPUT_DIM);

  gint64 now_us = g_get_monotonic_time ();

  if (!landmarks_valid (lm)) {
    pdata->eyes_was_closed = FALSE;
    pdata->closed_since_us = 0;
    pdata->mouth_was_open = FALSE;
    pdata->yawn_since_us = 0;
    pdata->looking_away = FALSE;
    pdata->looking_away_since_us = 0;
    pdata->blink_ts_count = 0;

    out[0] = (float) F_STATE_NO_FACE;
    return 0;
  }

  /* ===== 1) EAR ===== */
  float left_ear = compute_ear (lm, LEFT_EYE_IDX);
  float right_ear = compute_ear (lm, RIGHT_EYE_IDX);
  float ear_avg = 0.5f * (left_ear + right_ear);

  gboolean eyes_closed = (ear_avg < EAR_CLOSE_THRESHOLD);

  if (eyes_closed) {
    if (!pdata->eyes_was_closed) {
      pdata->eyes_was_closed = TRUE;
      pdata->closed_since_us = now_us;
    }
  } else {
    if (pdata->eyes_was_closed) {
      guint closed_ms = (guint) ((now_us - pdata->closed_since_us) / 1000LL);

      /* Valid blink: closing for a short duration */
      if (closed_ms >= BLINK_MIN_MS && closed_ms <= BLINK_MAX_MS) {
        push_blink (pdata, now_us);
      }
    }

    pdata->eyes_was_closed = FALSE;
    pdata->closed_since_us = 0;
  }

  /* Calculating the time the eyes are closed */
  guint closed_ms = 0;
  if (eyes_closed && pdata->closed_since_us > 0) {
    closed_ms = (guint) ((now_us - pdata->closed_since_us) / 1000LL);
  }

  /* Remove old blinks <= BLINK_WINDOW_MS */
  prune_blinks (pdata, now_us);

  float blink_rate_per_min = 0.0f;
  if (BLINK_WINDOW_MS > 0) {
    blink_rate_per_min =
        ((float) pdata->blink_ts_count) * 60000.0f / (float) BLINK_WINDOW_MS;
  }

  /* ===== 2) MAR / yawn ===== */
  float mar = compute_mar (lm);

  gboolean mouth_open = (mar >= MAR_YAWN_THRESHOLD);
  if (mouth_open) {
    if (!pdata->mouth_was_open) {
      pdata->mouth_was_open = TRUE;
      pdata->yawn_since_us = now_us;
    }
  } else {
    pdata->mouth_was_open = FALSE;
    pdata->yawn_since_us = 0;
  }

  guint yawn_hold_ms = 0;
  if (mouth_open && pdata->yawn_since_us > 0) {
    yawn_hold_ms = (guint) ((now_us - pdata->yawn_since_us) / 1000LL);
  }

  /* ===== 3) Head pose proxy / gaze proxy =====
  * With 468 landmarks without iris, this is a proxy.
  * To achieve more accurate gaze, 478 landmarks or a separate iris model are needed.
  */
  float eye_center_x = compute_eye_center_x (lm);
  float eye_center_y = compute_eye_center_y (lm);
  float eye_dist = compute_eye_distance (lm);
  if (eye_dist < 1e-6f)
    eye_dist = 1.0f;

  float nose_x = get_x (lm, NOSE_TIP);
  float nose_y = get_y (lm, NOSE_TIP);

  float lx = get_x (lm, LEFT_EYE_OUT);
  float ly = get_y (lm, LEFT_EYE_OUT);
  float rx = get_x (lm, RIGHT_EYE_OUT);
  float ry = get_y (lm, RIGHT_EYE_OUT);

  float eye_line_dx = rx - lx;
  float eye_line_dy = ry - ly;
  float head_roll_deg = degf (atan2f (eye_line_dy, eye_line_dx));

  /* proxy: left/right offset */
  float head_yaw_proxy = (nose_x - eye_center_x) / eye_dist;

  /* proxy: bow/raise */
  float mouth_center_y =
      (get_y (lm, MOUTH_LEFT) + get_y (lm, MOUTH_RIGHT)) * 0.5f;
  float head_pitch_proxy = (mouth_center_y - eye_center_y) / eye_dist;

  /* gaze proxy: no iris, so use proxy from head orientation */
  float gaze_x_proxy = head_yaw_proxy;
  float gaze_y_proxy = head_pitch_proxy;

  /* ===== 4) Distraction tracking ===== */
  float distraction_raw =
      0.45f * clampf01 (fabsf (head_yaw_proxy) / HEAD_YAW_PROXY_THRESHOLD) +
      0.25f * clampf01 (fabsf (head_pitch_proxy) / HEAD_PITCH_PROXY_THRESHOLD) +
      0.20f * clampf01 (fabsf (head_roll_deg) / HEAD_ROLL_THRESHOLD_DEG) +
      0.10f * clampf01 (fabsf (gaze_x_proxy));

  gboolean looking_away =
      (fabsf (head_yaw_proxy) > HEAD_YAW_PROXY_THRESHOLD) ||
      (fabsf (head_pitch_proxy) > HEAD_PITCH_PROXY_THRESHOLD) ||
      (fabsf (head_roll_deg) > HEAD_ROLL_THRESHOLD_DEG) ||
      (distraction_raw > DISTRACTION_SCORE_THRESHOLD);

  if (looking_away) {
    if (!pdata->looking_away) {
      pdata->looking_away = TRUE;
      pdata->looking_away_since_us = now_us;
    }
  } else {
    pdata->looking_away = FALSE;
    pdata->looking_away_since_us = 0;
  }

  guint looking_away_ms = 0;
  if (pdata->looking_away && pdata->looking_away_since_us > 0) {
    looking_away_ms =
        (guint) ((now_us - pdata->looking_away_since_us) / 1000LL);
  }

  /* ===== 5) Fatigue score ===== */
  float eye_closed_score = 0.0f;
  if (closed_ms >= EAR_CLOSED_HOLD_MS) {
    eye_closed_score = 1.0f;
  } else if (closed_ms > EAR_WARNING_HOLD_MS) {
    eye_closed_score =
        (float) (closed_ms - EAR_WARNING_HOLD_MS) /
        (float) (EAR_CLOSED_HOLD_MS - EAR_WARNING_HOLD_MS);
  }

  float blink_low_score = 0.0f;
  if (blink_rate_per_min <= BLINK_LOW_THRESHOLD_PER_MIN) {
    blink_low_score =
        clampf01 ((BLINK_LOW_THRESHOLD_PER_MIN - blink_rate_per_min) /
        BLINK_LOW_THRESHOLD_PER_MIN);
  }

  float yawn_score = 0.0f;
  if (yawn_hold_ms >= YAWN_HOLD_MS) {
    yawn_score = 1.0f;
  } else if (mouth_open) {
    yawn_score = clampf01 (mar / (MAR_YAWN_THRESHOLD * 1.5f));
  }

  float posture_score =
      clampf01 (fabsf (head_yaw_proxy) / HEAD_YAW_PROXY_THRESHOLD) * 0.4f +
      clampf01 (fabsf (head_pitch_proxy) / HEAD_PITCH_PROXY_THRESHOLD) * 0.4f +
      clampf01 (fabsf (head_roll_deg) / HEAD_ROLL_THRESHOLD_DEG) * 0.2f;

  float fatigue_score =
      clampf01 (0.42f * eye_closed_score +
      0.22f * blink_low_score +
      0.26f * yawn_score +
      0.10f * posture_score);

  float distraction_score = clampf01 (distraction_raw);

  /* ===== 6) Final state ===== */
  guint state = F_STATE_NORMAL;

  if (closed_ms >= EAR_CLOSED_HOLD_MS || yawn_hold_ms >= YAWN_HOLD_MS) {
    state = F_STATE_TIRED;
  } else if (looking_away_ms >= DISTRACTION_HOLD_MS) {
    state = F_STATE_DISTRACTED;
  } else if (closed_ms >= EAR_WARNING_HOLD_MS ||
      blink_rate_per_min < BLINK_LOW_THRESHOLD_PER_MIN ||
      fatigue_score > 0.35f) {
    state = F_STATE_WARNING;
  } else {
    state = F_STATE_NORMAL;
  }

  // Outputs format
  out[0] = (float) state;             // fatigue state
  out[1] = fatigue_score;
  out[2] = distraction_score;
  out[3] = left_ear;
  out[4] = right_ear;
  out[5] = blink_rate_per_min;        // blink rate
  out[6] = (float) closed_ms;         // time the eyes are closed
  out[7] = mar;
  out[8] = (float) yawn_hold_ms;      // time the mouth is open
  out[9] = head_roll_deg;             // head roll degree
  out[10] = head_yaw_proxy;           // proxy: left/right offset
  out[11] = head_pitch_proxy;         // proxy: bow/raise
  out[12] = gaze_x_proxy;             // head yaw proxy
  out[13] = gaze_y_proxy;             // head pitch proxy

  return 0;
}

/**
 * Close sub-plugin
 */
static void
fatigue_eval_close (const GstTensorFilterProperties * prop, void **private_data)
{
  fatigue_pdata *pdata = (fatigue_pdata *) (*private_data);

  if (pdata) {
    g_print ("[fatigue_eval] Closing sub-plugin: %s\n", pdata->model_path);

    g_free (pdata->model_path);
    pdata->model_path = NULL;

    g_free (pdata);
    *private_data = NULL;
  }
}

/**@brief Name of this subplugin */
static gchar filter_subplugin_fatigue_eval[] = "fatigue_eval";

static GstTensorFilterFramework fatigue_eval_custom = {
#ifdef GST_TENSOR_FILTER_API_VERSION_DEFINED
  .version = GST_TENSOR_FILTER_FRAMEWORK_V0,
#else
  .name = filter_subplugin_fatigue_eval,
  .allow_in_place = FALSE,
  .allocate_in_invoke = FALSE,
  .run_without_model = TRUE,
  .invoke_NN = fatigue_eval_invoke,
  .getInputDimension = fatigue_eval_getInputDim,
  .getOutputDimension = fatigue_eval_getOutputDim,
#endif
  .open = fatigue_eval_open,
  .close = fatigue_eval_close,
};

/**@brief Initialize this object for tensor_filter subplugin runtime register */
void init_filter_fatigue_eval (void)
{
#ifdef GST_TENSOR_FILTER_API_VERSION_DEFINED
  fatigue_eval_custom.name = filter_subplugin_fatigue_eval;
  fatigue_eval_custom.allow_in_place = FALSE;
  fatigue_eval_custom.allocate_in_invoke = FALSE;
  fatigue_eval_custom.run_without_model = TRUE;
  fatigue_eval_custom.invoke_NN = fatigue_eval_invoke;
  fatigue_eval_custom.getInputDimension = fatigue_eval_getInputDim;
  fatigue_eval_custom.getOutputDimension = fatigue_eval_getOutputDim;
#endif
  nnstreamer_filter_probe (&fatigue_eval_custom);
}

/**@brief Destruct the subplugin */
void fini_filter_fatigue_eval (void)
{
  nnstreamer_filter_exit (fatigue_eval_custom.name);
}