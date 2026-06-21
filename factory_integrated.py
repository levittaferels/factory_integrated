"""
Smart Factory Predictive Maintenance — Edge AI Gateway
=======================================================
Systems Over Scripts

Datasets (place in de_final/):
  ai4i2020.csv
    https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020
  casting_product_defect/casting_data/train|test/def_front/*.jpeg
    https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product
  Lab-12-.../yolov10n.pt
  Lab-11-.../mobilenet_v1_1.0_224_quant.tflite
  Lab-13-.../sample_video.mp4  (fallback if no casting images)

Install:
  pip install numpy opencv-python ultralytics tensorflow paho-mqtt pandas

Usage:
    python factory_integrated.py                          # casting images + AI4I (first run builds video)
    python factory_integrated.py --duration 30            # same, runs longer
    python factory_integrated.py --stress                 # 1000 fps OOM test
    python factory_integrated.py --mqtt-host localhost --duration 30  # MQTT fallback demo

Silence TensorFlow startup messages (optional):
  set TF_ENABLE_ONEDNN_OPTS=0
  set TF_CPP_MIN_LOG_LEVEL=3
"""

import os, csv, time, math, json, queue, random
import bisect, argparse, threading, statistics, glob
from collections import deque
import numpy as np

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))

YOLO_MODEL   = os.path.join(BASE_DIR,
    "Lab-12-Object-Detection-The-NMS-Latency-Tax", "yolov10n.pt")
SAMPLE_VIDEO = os.path.join(BASE_DIR,
    "Lab-13-Multi-threading-Frame-Dropping-in-Codespaces", "sample_video.mp4")
MOBILENET    = os.path.join(BASE_DIR,
    "Lab-11-End-to-End-Edge-Inference", "mobilenet_v1_1.0_224_quant.tflite")
MOBILENET_LB = os.path.join(BASE_DIR,
    "Lab-11-End-to-End-Edge-Inference", "labels_mobilenet_quant_v1_224.txt")
AI4I_CSV     = os.path.join(BASE_DIR, "ai4i2020.csv")
CASTING_DIR  = os.path.join(BASE_DIR, "casting_product_defect", "casting_data")
LAB2_CSV     = os.path.join(BASE_DIR,
    "Lab-2-Data-Acquisition-and-Downsampling", "raw_sensor_data.csv")
JSONL_OUT    = os.path.join(BASE_DIR, "factory_local_cache.jsonl")


# ============================================================================
# OPTIONAL DEPENDENCIES — graceful fallback if not installed
# ============================================================================
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

HAS_TF   = False
tflite   = None
try:
    import tensorflow as tf
    tflite  = tf.lite
    HAS_TF  = True
except ImportError:
    pass

try:
    import paho.mqtt.client as mqtt
    HAS_MQTT = True
except ImportError:
    HAS_MQTT = False


# ============================================================================
# W1 — Edge I/O Throughput & Pipeline Fundamentals
#      Concept: bulk write is dramatically faster than line-by-line
#      Runs at startup to benchmark your machine's disk I/O
# ============================================================================
def w1_io_benchmark(n=20000):
    tmp  = os.path.join(BASE_DIR, "_w1_bench.csv")
    rows = [[time.time(), round(random.uniform(20, 35), 2),
             round(random.uniform(40, 90), 2)] for _ in range(n)]

    # Test 1: line-by-line (bad pattern — Lab 1 anti-pattern)
    if os.path.exists(tmp): os.remove(tmp)
    t0 = time.perf_counter()
    for r in rows:
        with open(tmp, "a", newline="") as f:
            csv.writer(f).writerow(r)
    t_line = time.perf_counter() - t0

    # Test 2: bulk (good pattern — Lab 1 best practice)
    if os.path.exists(tmp): os.remove(tmp)
    t0 = time.perf_counter()
    with open(tmp, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    t_bulk = time.perf_counter() - t0
    if os.path.exists(tmp): os.remove(tmp)

    speedup  = round(t_line / t_bulk) if t_bulk else 0
    bulk_mbs = round(n * 35 / 1e6 / t_bulk, 1) if t_bulk else 0
    line_mbs = round(n * 35 / 1e6 / t_line, 1) if t_line else 0
    print(f"[W1]  I/O benchmark  : line-by-line={line_mbs} MB/s  "
          f"bulk={bulk_mbs} MB/s  (bulk {speedup}x faster)")
    return speedup


# ============================================================================
# W2 — Data Acquisition & Downsampling
#      Concept: chunk-average a high-frequency stream to reduce data rate
#      Priority: AI4I torque column → Lab2 CSV → numpy CNC simulation
# ============================================================================
def w2_sensor_stream():
    """Generator. Yields one vibration reading per next(). Loops forever."""

    # Priority 1: AI4I 2020 real machine data (Torque [Nm] column)
    if os.path.exists(AI4I_CSV):
        try:
            import pandas as pd
            df   = pd.read_csv(AI4I_CSV)
            col  = "Torque [Nm]"
            vals = df[col].values.astype(float)
            vals = vals - vals.mean()            # center around 0
            print(f"[W2]  sensor source  : [AI4I] {col}  "
                  f"({len(vals)} readings, "
                  f"range {vals.min():.2f} to {vals.max():.2f})")
            idx = 0
            while True:
                yield float(vals[idx % len(vals)])
                idx += 1
        except Exception as e:
            print(f"[W2]  AI4I failed ({e}) -> trying Lab2 CSV")

    # Priority 2: Lab 2 raw_sensor_data.csv
    if os.path.exists(LAB2_CSV):
        print("[W2]  sensor source  : [CSV] raw_sensor_data.csv")
        while True:
            with open(LAB2_CSV) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            yield float(line) - 5.0
                        except ValueError:
                            pass

    # Priority 3: numpy CNC spindle simulation
    print("[W2]  sensor source  : [SIM] numpy CNC vibration (50 Hz spindle)")
    t = 0.0
    while True:
        v = 0.5 * math.sin(2 * math.pi * 50 * t) + np.random.normal(0, 0.1)
        yield float(v)
        t += 0.01


def w2_chunk_average(values, chunk=10):
    """Downsample list by chunk-averaging (100 Hz → 10 Hz)."""
    return [sum(values[i:i+chunk]) / chunk
            for i in range(0, len(values) - chunk + 1, chunk)]


# ============================================================================
# W3 — Anomaly Detection
#      Concept: Modified Z-Score (MAD) detects statistical outliers
#      Formula: M = 0.6745 * (value - median) / MAD
#      Threshold: |M| > 3.5 → anomaly
# ============================================================================
WINDOW_SIZE   = 10
MAD_THRESHOLD = 3.5

def w3_mad_anomaly(window: deque, value: float) -> bool:
    """Returns True if value is a MAD anomaly. Adds to window if clean."""
    if len(window) < WINDOW_SIZE:
        window.append(value)
        return False
    med = statistics.median(window)
    mad = statistics.median([abs(v - med) for v in window]) + 0.0001
    M   = 0.6745 * (value - med) / mad
    if abs(M) > MAD_THRESHOLD:
        return True                     # anomaly — do NOT update window
    window.append(value)
    return False


# ============================================================================
# W4 — Fault Tolerance & Exponential Backoff
#      Concept: retry with exponential delay + jitter; never lose data
#      Fallback: write to local JSONL if all retries fail
# ============================================================================
def w4_backoff_delays(max_retries=5, base=1.0):
    """Generator. Yields sleep duration for each retry."""
    delay = base
    for _ in range(max_retries):
        yield delay + random.uniform(0, delay * 0.2)
        delay *= 2


# ============================================================================
# W5 — Efficient Edge Transport
#      Concept: compact JSON saves bandwidth; measure real byte difference
# ============================================================================
def w5_serialize(payload: dict):
    """Returns (full_str, compact_str, bytes_saved)."""
    full    = json.dumps(payload, indent=None)
    compact = json.dumps(payload, separators=(",", ":"))
    return full, compact, len(full.encode()) - len(compact.encode())


# ============================================================================
# W7 — Visual Quality Control
#      Concept: variance of Laplacian = blur score; mean pixel = brightness
#      Rejects bad frames before wasting compute on inference
# ============================================================================
BLUR_THRESHOLD = 100.0
MIN_BRIGHTNESS = 40.0
MAX_BRIGHTNESS = 220.0

def w7_qc(frame_bgr: np.ndarray):
    """Returns (passed: bool, focus: float, brightness: float)."""
    if HAS_CV2:
        gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        focus = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    else:
        gray  = frame_bgr.mean(axis=2).astype(np.float32)
        lap   = (-4*gray[1:-1,1:-1] + gray[:-2,1:-1] + gray[2:,1:-1]
                 + gray[1:-1,:-2]   + gray[1:-1,2:])
        focus = float(lap.var())
    brightness = float(gray.mean())
    passed = (focus >= BLUR_THRESHOLD
              and MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS)
    return passed, round(focus, 1), round(brightness, 1)


# ============================================================================
# W9 — Vectorized Image Data Pipeline
#      Concept: channel-wise normalization with pure NumPy — no Python loops
#      Constants: ImageNet mean/std (exact values from Lab 9)
# ============================================================================
CH_MEAN = np.array([123.675, 116.28,  103.53],  dtype=np.float32)
CH_STD  = np.array([58.395,  57.12,   57.375],  dtype=np.float32)

def w9_normalize(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR→RGB, clamp glare >240, float32 cast, channel normalize."""
    rgb = frame_bgr[:, :, ::-1].copy()
    rgb[rgb > 240] = 240
    return (rgb.astype(np.float32) - CH_MEAN) / CH_STD


# ============================================================================
# W10 — Interpolation Benchmarking
#      Concept: INTER_LINEAR (bilinear) = best speed/quality tradeoff
#      Each resize is timed so latency appears in the live dashboard
# ============================================================================
def w10_resize(frame: np.ndarray, size: int = 320):
    """Resize to size×size using INTER_LINEAR. Returns (frame, ms)."""
    t0 = time.perf_counter()
    if HAS_CV2:
        out = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
    else:
        h, w = frame.shape[:2]
        ys   = np.linspace(0, h - 1, size).astype(int)
        xs   = np.linspace(0, w - 1, size).astype(int)
        out  = frame[ys][:, xs]
    return out, round((time.perf_counter() - t0) * 1000, 3)


# ============================================================================
# W11 — End-to-End Edge Inference (INT8 quantized MobileNet)
# ============================================================================
class W11_MobileNet:
    def __init__(self):
        self.ok = False
        if not HAS_TF:
            print("[W11] [SIM] tensorflow not installed  "
                  "(pip install tensorflow)")
            return
        if not os.path.exists(MOBILENET):
            print(f"[W11] [SIM] model not found: {MOBILENET}")
            return
        try:
            self.interp = tflite.Interpreter(model_path=MOBILENET)
            self.interp.allocate_tensors()
            self.inp    = self.interp.get_input_details()
            self.out    = self.interp.get_output_details()
            self.labels = (open(MOBILENET_LB).read().splitlines()
                           if os.path.exists(MOBILENET_LB) else [])
            self.ok     = True
            print(f"[W11] [INT8] MobileNet loaded   : "
                  f"{os.path.basename(MOBILENET)}")
        except Exception as e:
            print(f"[W11] [SIM] MobileNet load failed ({e})")

    def classify(self, frame_bgr: np.ndarray):
        """Returns (label: str, confidence: int 0-255)."""
        if not self.ok:
            return None, 0
        img = cv2.resize(frame_bgr, (224, 224)) if HAS_CV2 \
              else frame_bgr[:224, :224]
        img = img[:, :, ::-1]                   # BGR → RGB, keep uint8
        x   = np.expand_dims(img.astype(np.uint8), 0)
        self.interp.set_tensor(self.inp[0]["index"], x)
        self.interp.invoke()
        pred = np.squeeze(self.interp.get_tensor(self.out[0]["index"]))
        top  = int(np.argmax(pred))
        lbl  = self.labels[top] if top < len(self.labels) else str(top)
        return lbl, int(pred[top])


# ============================================================================
# W12 — Object Detection & The NMS Latency Tax
#      Concept: YOLOv10-N is NMS-free (latency tax eliminated vs YOLOv8)
#      Real model: yolov10n.pt from Lab-12 folder
# ============================================================================
def w12_load_detector():
    if HAS_YOLO and os.path.exists(YOLO_MODEL):
        print(f"[W12] [YOLOv10-N] loaded        : "
              f"{os.path.basename(YOLO_MODEL)}")
        return YOLO(YOLO_MODEL)
    print("[W12] [SIM] YOLO unavailable — simulated detector active")
    return None


def w12_detect(model, frame: np.ndarray, ts: float, defect_wins: list):
    """Run detection. Returns (detections: list, infer_ms: float)."""
    if model is not None:
        t0   = time.perf_counter()
        res  = model(frame, imgsz=320, verbose=False)
        ms   = round((time.perf_counter() - t0) * 1000, 2)
        dets = []
        for r in res:
            for b in r.boxes:
                dets.append({
                    "cls":  int(b.cls[0]),
                    "conf": round(float(b.conf[0]), 3),
                    "bbox": [round(float(v), 1) for v in b.xyxy[0].tolist()],
                })
        return dets, ms
    # Simulated: return a defect box during scheduled fault windows
    t0 = time.perf_counter()
    time.sleep(0.002)
    ms = round((time.perf_counter() - t0) * 1000, 2)
    if any(s <= ts <= e for s, e in defect_wins):
        return [{"cls": 0,
                 "conf": round(random.uniform(0.75, 0.95), 3),
                 "bbox": [110, 85, 185, 155],
                 "label": "defect"}], ms
    return [], ms


# ============================================================================
# W13 — Multi-threading & Frame Dropping
#      Concept: bounded queue (maxsize=1) decouples producer from consumer
#      Producer keeps only the FRESHEST frame; stale frames are evicted
#      Prevents OOM when inference is slower than the camera feed rate
# ============================================================================
# W13 IS the queue.Queue(maxsize=1) in Pipeline + the producer thread pattern


# ============================================================================
# W14 — Heterogeneous Fusion & Cloud Strategy
#      Concept: nearest-neighbor temporal join aligns two streams with
#      different rates using bisect O(log n) search
#      100 Hz scalar (sensor) <-> 5 Hz tensor (camera/YOLO)
# ============================================================================
def w14_temporal_join(scalar_buf: list, tensor_ts: float):
    """Match tensor event to closest scalar reading by timestamp.
    Returns (vibration, temperature, sync_error_ms)."""
    if not scalar_buf:
        return None, None, None
    times   = [r[0] for r in scalar_buf]
    i       = bisect.bisect_left(times, tensor_ts)
    cands   = []
    if i < len(scalar_buf): cands.append(scalar_buf[i])
    if i > 0:               cands.append(scalar_buf[i - 1])
    nearest = min(cands, key=lambda r: abs(r[0] - tensor_ts))
    sync_ms = round(abs(nearest[0] - tensor_ts) * 1000, 3)
    return nearest[1], nearest[2], sync_ms


# ============================================================================
# CAMERA SOURCE RESOLVER
# Priority: casting defect images → synthetic frames
# ============================================================================
def resolve_camera_source(force_video: bool):
    """Finds the best available camera source.
    Returns (path_or_None, label_string)."""

    # Priority 1: casting product defect images → build video once
    if not force_video:
        patterns = [
            os.path.join(CASTING_DIR, "train", "def_front", "*.jpeg"),
            os.path.join(CASTING_DIR, "train", "def_front", "*.jpg"),
            os.path.join(CASTING_DIR, "train", "def_front", "*.png"),
            os.path.join(CASTING_DIR, "test",  "def_front", "*.jpeg"),
            os.path.join(CASTING_DIR, "test",  "def_front", "*.jpg"),
            os.path.join(CASTING_DIR, "test",  "def_front", "*.png"),
        ]
        imgs = []
        for pat in patterns:
            imgs += sorted(glob.glob(pat))

        if imgs and HAS_CV2:
            out = SAMPLE_VIDEO
            if not os.path.exists(out):
                print(f"[cam] Building defect video from "
                      f"{len(imgs)} casting images...")
                first = cv2.imread(imgs[0])
                h, w  = first.shape[:2]
                wr    = cv2.VideoWriter(
                    out, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
                for path in imgs:
                    img = cv2.imread(path)
                    if img is not None:
                        wr.write(img)
                wr.release()
                print(f"[cam] Video saved: {len(imgs)} frames "
                      f"@ 10 fps = {len(imgs)//10}s")
            return out, f"[CASTING] {len(imgs)} defect images → mp4"

    # Priority 2: synthetic numpy frames
    return None, "[SIM] synthetic numpy frames"


# ============================================================================
# PIPELINE SHARED STATE
# ============================================================================
class Pipeline:
    def __init__(self, qsize: int):
        self.frame_q      = queue.Queue(maxsize=qsize)  # W13: bounded queue
        self.scalar_buf   = deque()
        self.scalar_lock  = threading.Lock()
        self.stop         = threading.Event()
        # metrics
        self.frames_in    = 0
        self.frames_drop  = 0
        self.qc_fail      = 0
        self.inferences   = 0
        self.fused        = 0
        self.anomalies    = 0
        self.last_yolo_ms = 0.0
        self.last_rsz_ms  = 0.0
        self.last_json_b  = 0
        self.last_cmpct_b = 0
        self.sync_hist    = deque(maxlen=100)
        self.mad_window   = deque(maxlen=WINDOW_SIZE)
        self.defect_wins  = []
        self.t0           = time.time()

    def avg_sync(self):
        return round(sum(self.sync_hist) / len(self.sync_hist), 3) \
               if self.sync_hist else None


# ============================================================================
# DEFECT WINDOW SCHEDULER
# Schedules correlated faults so vibration spike and visual defect
# appear at the same timestamp — the predictive maintenance story
# ============================================================================
def defect_scheduler(pipe: Pipeline):
    while not pipe.stop.is_set():
        time.sleep(random.uniform(3, 7))
        now = time.time()
        pipe.defect_wins.append((now, now + 0.5))
        pipe.defect_wins = pipe.defect_wins[-20:]


# ============================================================================
# PRODUCER 1 — Sensor thread
# W2: reads real AI4I torque data (or CSV / simulation)
# Injects correlated spike during defect windows
# ============================================================================
def sensor_producer(pipe: Pipeline):
    SCALAR_HZ = 100
    period    = 1.0 / SCALAR_HZ
    gen       = w2_sensor_stream()
    t0        = time.time()

    while not pipe.stop.is_set():
        now = time.time()
        vib = float(next(gen))

        # Correlate: inject spike when defect window is active
        if any(s <= now <= e for s, e in pipe.defect_wins):
            vib += np.random.choice([5.0, -5.0])

        # Temperature: slow thermal drift + gaussian noise
        temp = (72.0
                + 3.0 * math.sin(2 * math.pi * 0.05 * (now - t0))
                + np.random.normal(0, 0.3))

        with pipe.scalar_lock:
            pipe.scalar_buf.append((now,
                                    round(float(vib), 4),
                                    round(float(temp), 2)))
            cutoff = now - 2.0          # keep 2-second rolling window
            while pipe.scalar_buf and pipe.scalar_buf[0][0] < cutoff:
                pipe.scalar_buf.popleft()

        time.sleep(period)


# ============================================================================
# PRODUCER 2 — Camera thread
# W13: bounded queue pattern — evict stale, insert fresh
# ============================================================================
def camera_producer(pipe: Pipeline, video_path, fps: int):
    period = 1.0 / fps
    cap    = cv2.VideoCapture(video_path) if (video_path and HAS_CV2) else None

    while not pipe.stop.is_set():
        ts = time.time()
        if cap is not None:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
        else:
            frame = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

        pipe.frames_in += 1

        # W13 core: evict stale frame, insert fresh — never block
        try:
            pipe.frame_q.put_nowait((ts, frame))
        except queue.Full:
            try:
                pipe.frame_q.get_nowait()
                pipe.frames_drop += 1
            except queue.Empty:
                pass
            try:
                pipe.frame_q.put_nowait((ts, frame))
            except queue.Full:
                pipe.frames_drop += 1

        time.sleep(period)

    if cap:
        cap.release()


# ============================================================================
# CONSUMER — Inference thread
# Full chain: W7 → W10 → W9 → W12 → W11 → W14 → W3 → W5 → W4
# Throttled to 5 Hz so YOLO does not saturate the CPU
# ============================================================================
def inference_consumer(pipe: Pipeline, model, mobilenet: W11_MobileNet,
                       exfil):
    TENSOR_HZ    = 5
    infer_period = 1.0 / TENSOR_HZ
    last_infer   = 0.0

    while not pipe.stop.is_set():
        try:
            ts, frame = pipe.frame_q.get(timeout=1.0)
        except queue.Empty:
            continue

        if time.time() - last_infer < infer_period:
            time.sleep(0.001)
            continue
        last_infer = time.time()

        # W7 — QC gate: reject blurry / dark / overexposed frames
        passed, focus, bright = w7_qc(frame)
        if not passed:
            pipe.qc_fail += 1

        # W10 — Bilinear resize (timed per frame)
        frame_r, rsz_ms = w10_resize(frame, 320)
        pipe.last_rsz_ms = rsz_ms

        # W9 — Vectorized channel normalization
        _ = w9_normalize(frame_r)

        # W12 — YOLOv10-N detection
        dets, yolo_ms = w12_detect(model, frame_r, ts, pipe.defect_wins)
        pipe.last_yolo_ms = yolo_ms
        pipe.inferences  += 1

        if not dets:
            continue

        # W11 — MobileNet INT8 classification
        lbl, conf = mobilenet.classify(frame_r)

        # W14 — Nearest-neighbor temporal join
        with pipe.scalar_lock:
            buf = list(pipe.scalar_buf)
        vib, temp, sync_ms = w14_temporal_join(buf, ts)
        if vib is None:
            continue

        pipe.sync_hist.append(sync_ms)
        pipe.fused += 1

        # W3 — MAD anomaly on fused vibration value
        is_anom = w3_mad_anomaly(pipe.mad_window, vib)
        if is_anom:
            pipe.anomalies += 1

        # Fused payload: defect detection + sensor reading at same timestamp
        payload = {
            "ts":            round(ts, 4),
            "machine_id":    "CNC-07",
            "detections":    dets,
            "vibration":     vib,
            "temperature":   temp,
            "sync_error_ms": sync_ms,
            "mad_anomaly":   is_anom,
            "qc":            {"passed": passed, "focus": focus,
                              "brightness": bright},
            "classifier":    {"label": lbl, "conf": conf},
            "infer_ms":      yolo_ms,
            "resize_ms":     rsz_ms,
        }

        # W5 — Measure JSON vs compact byte size
        full, compact, _ = w5_serialize(payload)
        pipe.last_json_b  = len(full.encode())
        pipe.last_cmpct_b = len(compact.encode())

        # W4 — MQTT send; JSONL fallback on failure
        exfil.send(compact)


# ============================================================================
# EXFILTRATOR — W4 MQTT + JSONL fallback
# ============================================================================
class Exfiltrator:
    def __init__(self, host, port=1883):
        self.connected  = False
        self.client     = None
        self.mqtt_sent  = 0
        self.jsonl_sent = 0

        if HAS_MQTT and host:
            try:
                self.client = mqtt.Client()
                self.client.connect(host, port, keepalive=30)
                self.client.loop_start()
                self.connected = True
                print(f"[W4]  MQTT connected        : {host}:{port}")
            except Exception as e:
                print(f"[W4]  MQTT failed ({e})")
                print(f"[W4]  -> JSONL fallback active")
        else:
            print("[W4/W5] MQTT not configured    : JSONL fallback active")

    def send(self, line: str):
        if self.connected:
            try:
                info = self.client.publish("factory/cnc/fused", line)
                if info.rc == 0:
                    self.mqtt_sent += 1
                    return
                raise RuntimeError(f"rc={info.rc}")
            except Exception:
                self.connected = False
        # W4 fallback: local JSONL — no data ever lost
        with open(JSONL_OUT, "a") as f:
            f.write(line + "\n")
        self.jsonl_sent += 1

    def close(self):
        if self.client:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass


# ============================================================================
# LIVE METRICS DASHBOARD
# ============================================================================
def metrics_reporter(pipe: Pipeline, exfil: Exfiltrator):
    while not pipe.stop.is_set():
        time.sleep(2.0)
        el       = time.time() - pipe.t0
        fps      = pipe.inferences / el if el else 0
        drop_pct = (pipe.frames_drop / pipe.frames_in * 100
                    if pipe.frames_in else 0)
        avg      = pipe.avg_sync()
        sync_str = (f"{avg:.3f}ms (avg)" if avg is not None
                    else "pending (no detections yet)")
        saved    = (round((1 - pipe.last_cmpct_b / pipe.last_json_b) * 100)
                    if pipe.last_json_b else 0)
        print(
            f"\n--- t={el:5.1f}s "
            f"-------------------------------------------\n"
            f"  frames : produced={pipe.frames_in:6d}  "
            f"dropped={pipe.frames_drop:6d} ({drop_pct:4.1f}%)  "
            f"qc_fail={pipe.qc_fail}\n"
            f"  infer  : count={pipe.inferences:4d}  "
            f"yolo={pipe.last_yolo_ms:6.2f}ms  "
            f"resize={pipe.last_rsz_ms:5.3f}ms  fps={fps:4.1f}\n"
            f"  fusion : events={pipe.fused:4d}  "
            f"sync={sync_str:>32s}  mad_anom={pipe.anomalies}\n"
            f"  cloud  : mqtt={exfil.mqtt_sent}  "
            f"jsonl={exfil.jsonl_sent}  "
            f"payload={pipe.last_cmpct_b}B (saved {saved}%)"
        )


# ============================================================================
# MAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Smart Factory Edge AI Gateway — W1-W14 Unified")
    ap.add_argument("--real-video",  action="store_true",
                    help="force sample_video.mp4 instead of casting images")
    ap.add_argument("--mqtt-host",   default=None,
                    help="MQTT broker host (omit to use JSONL fallback)")
    ap.add_argument("--duration",    type=int, default=20,
                    help="run duration in seconds (default: 20)")
    ap.add_argument("--stress",      action="store_true",
                    help="OOM stress test: push camera to 1000 fps")
    ap.add_argument("--frame-qsize", type=int, default=1,
                    help="bounded queue size (default: 1)")
    args = ap.parse_args()

    # Banner
    print("=" * 66)
    print("   SMART FACTORY EDGE AI GATEWAY — W1-W14 UNIFIED")
    print("=" * 66)
    print("   INGEST  W1  I/O benchmark  ->  W2  sensor stream  (100 Hz)")
    print("   CLEAN   W3  MAD anomaly    ->  W10 interpolation resize")
    print("   VISION  W7  QC gate        ->  W9  normalize  ->  W13 queue")
    print("   INFER   W12 YOLOv10-N      ->  W11 MobileNet INT8")
    print("   FUSE    W14 temporal join  (5 Hz tensor <-> 100 Hz scalar)")
    print("   SHIP    W5  serialize      ->  W4  MQTT + JSONL fallback")
    print("=" * 66)

    # W1 — I/O benchmark at startup
    w1_io_benchmark()

    # Resolve camera source
    cam_fps          = 1000 if args.stress else 30
    video, cam_label = resolve_camera_source(force_video=args.real_video)
    print(f"[cam] source: {cam_label} @ {cam_fps} fps")

    # Initialise components
    pipe      = Pipeline(qsize=args.frame_qsize)
    exfil     = Exfiltrator(args.mqtt_host)
    model     = w12_load_detector()
    mobilenet = W11_MobileNet()

    print(f"\n[run] cam_fps={cam_fps}  qsize={args.frame_qsize}  "
          f"duration={args.duration}s  stress={args.stress}\n")

    threads = [
        threading.Thread(target=defect_scheduler,
                         args=(pipe,), daemon=True),
        threading.Thread(target=sensor_producer,
                         args=(pipe,), daemon=True),
        threading.Thread(target=camera_producer,
                         args=(pipe, video, cam_fps), daemon=True),
        threading.Thread(target=inference_consumer,
                         args=(pipe, model, mobilenet, exfil), daemon=True),
        threading.Thread(target=metrics_reporter,
                         args=(pipe, exfil), daemon=True),
    ]
    for t in threads:
        t.start()

    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("\n[run] interrupted by user")
    finally:
        pipe.stop.set()
        time.sleep(0.5)
        exfil.close()

        el    = time.time() - pipe.t0
        avg   = pipe.avg_sync()
        s_min = min(pipe.sync_hist, default=None)
        s_max = max(pipe.sync_hist, default=None)

        print("\n==================== FINAL METRICS ====================")
        print(f"  runtime                  : {el:.1f} s")
        print(f"  frames produced          : {pipe.frames_in}")
        print(f"  frames dropped   (W13)   : {pipe.frames_drop} "
              f"({pipe.frames_drop / max(pipe.frames_in, 1) * 100:.1f}%)")
        print(f"  qc failures      (W7)    : {pipe.qc_fail}")
        print(f"  inferences       (W12)   : {pipe.inferences}")
        print(f"  fused events     (W14)   : {pipe.fused}")
        print(f"  MAD anomalies    (W3)    : {pipe.anomalies}")
        print(f"  avg YOLO latency         : {pipe.last_yolo_ms:.2f} ms")
        print(f"  avg resize       (W10)   : {pipe.last_rsz_ms:.3f} ms")
        if avg is not None:
            print(f"  sync avg / min / max     : "
                  f"{avg:.3f} / {s_min:.3f} / {s_max:.3f} ms  (tol 100 ms)")
        else:
            print(f"  sync error               : "
                  f"N/A — no fusion events  "
                  f"(add casting images or use --real-video)")
        print(f"  payload JSON / compact   : "
              f"{pipe.last_json_b}B / {pipe.last_cmpct_b}B  (W5)")
        print(f"  mqtt published           : {exfil.mqtt_sent}")
        print(f"  jsonl fallback writes    : {exfil.jsonl_sent}")
        print("=======================================================")


if __name__ == "__main__":
    main()
