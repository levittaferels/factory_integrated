# Smart Factory Edge AI Gateway
**A Unified Data Engineering Pipeline for CNC Predictive Maintenance**

A single Python script (`factory_integrated.py`) that runs **5 concurrent threads** to monitor a simulated CNC machine in real time:

1. **Sensor thread** — reads real CNC torque data (AI4I 2020) at 100 Hz
2. **Camera thread** — streams casting defect images at 30 fps into a bounded queue
3. **Inference thread** — W7 QC gate → W10 resize → W9 normalize → W12 YOLO → W11 MobileNet
4. **Fusion** — W14 temporal join matches each detection to its nearest sensor reading by timestamp
5. **Cloud thread** — W5 serializes, W4 sends via MQTT with automatic JSONL fallback

## Datasets

AI4I 2020 Predictive Maintenance: Sensor stream (Torque \[Nm\] column → 100 Hz vibration)
[Kaggle](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)

Casting Product Defect Images: Camera source (4,211 defective pump impeller images → auto-converted to video) [Kaggle](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)

> **Note:** Neither dataset is natively CNC-specific. AI4I contains synthetic machine sensor data with failure labels. The casting images are real industrial product photos. Both are acknowledged as simulation-suitable proxies for this capstone project.


## Install
```bash
pip install numpy opencv-python ultralytics tensorflow paho-mqtt pandas
```
> **Note on TFLite:** `tflite-runtime` is discontinued on Python 3.11 Windows. Use the full `tensorflow` package — `tf.lite` is used internally.


## Run
```bash
# Auto-detect: uses casting images + AI4I sensor data
python factory_integrated.py

# Run for 30 seconds
python factory_integrated.py --duration 30

# Stress test: push camera to 1000 fps to prove OOM prevention
python factory_integrated.py --stress --duration 15

# MQTT fallback demo: connect to a broker (or watch it fall back to JSONL)
python factory_integrated.py --mqtt-host localhost --duration 30

# Silence TensorFlow startup messages
set TF_ENABLE_ONEDNN_OPTS=0        # Windows CMD
$env:TF_ENABLE_ONEDNN_OPTS=0      # PowerShell
```

On first run with casting images present, the script automatically converts them into `sample_video.mp4`. Subsequent runs skip the conversion step.

---

## What each lab contributes

| Week | Lab topic                        | What it does in the pipeline                               |
| ---- | -------------------------------- | ---------------------------------------------------------- |
| W1   | Edge I/O Throughput              | Bulk vs line-by-line disk write benchmark at every startup |
| W2   | Data Acquisition & Downsampling  | Reads AI4I torque data as a 100 Hz generator stream        |
| W3   | Anomaly Detection                | MAD formula `M = 0.6745*(v-median)/MAD`, threshold 3.5     |
| W4   | Fault Tolerance & Backoff        | Exponential backoff + JSONL fallback on MQTT failure       |
| W5   | Efficient Edge Transport         | Measures JSON vs compact byte size before every send       |
| W7   | Visual Quality Control           | Laplacian variance blur check + brightness gate per frame  |
| W9   | Vectorized Image Pipeline        | Channel mean/std normalization with pure NumPy             |
| W10  | Interpolation Benchmarking       | INTER_LINEAR resize to 320×320, timed per frame            |
| W11  | End-to-End Edge Inference        | MobileNet INT8 TFLite classifier on CPU                    |
| W12  | Object Detection & NMS           | YOLOv10-N NMS-free detection at ~15 ms/frame on CPU        |
| W13  | Multi-threading & Frame Dropping | `queue.Queue(maxsize=1)` producer/consumer backbone        |
| W14  | Heterogeneous Fusion             | Bisect nearest-neighbor temporal join (100 Hz ↔ 5 Hz)      |

---

## Live dashboard output

```plaintext
--- t= 20.1s -------------------------------------------
  frames : produced=   580  dropped=    25 ( 4.3%)  qc_fail=3
  infer  : count=  97  yolo= 15.02ms  resize= 0.513ms  fps= 4.8
  fusion : events=  22  sync= 2.436ms (avg)  mad_anom=2
  cloud  : mqtt=0  jsonl=22  payload=339B (saved 10%)
```

| Field             | What it means                                                            |
| ----------------- | ------------------------------------------------------------------------ |
| `frames produced` | Total camera frames read from the casting video                          |
| `dropped`         | Frames evicted by the W13 bounded queue when inference was slow          |
| `qc_fail`         | Frames rejected by the W7 blur/brightness check before YOLO              |
| `yolo`            | Average YOLOv10-N inference time per frame (CPU)                         |
| `resize`          | Average W10 INTER_LINEAR resize time per frame                           |
| `fusion events`   | Times YOLO detected an object AND temporal join matched a sensor reading |
| `sync`            | Rolling average time gap between camera event and nearest sensor reading |
| `mad_anom`        | Times the W3 MAD formula flagged a vibration spike                       |
| `payload`         | Compact JSON size per fused event sent to cloud                          |

---

## Real terminal results

### Run 1 — basic

```bash
python factory_integrated.py
```

```plaintext
[W1]  I/O benchmark  : line-by-line=0.1 MB/s  bulk=19.8 MB/s  (bulk 270x faster)
[W2]  sensor source  : [AI4I] Torque [Nm]  (10000 readings)
[W12] [YOLOv10-N] loaded : yolov10n.pt
[W11] [INT8] MobileNet loaded : mobilenet_v1_1.0_224_quant.tflite
[cam] source: [CASTING] 4211 defect images → mp4 @ 30 fps

  runtime : 20.8s  |  frames: 578  |  dropped: 3 (0.5%)
  YOLO: 32 ms avg  |  fps: 4.8
  fused events: 0  (YOLO trained on COCO — does not recognise casting parts)
```

### Run 2 — stress test

```bash
python factory_integrated.py --stress --duration 15
```

```plaintext
  frames produced : 9954  |  dropped: 1906 (19.1%)
  AI fps: 5.0  (stable despite 1000 fps input — W13 OOM prevention proven)
```

### Run 3 — MQTT fallback

```bash
python factory_integrated.py --mqtt-host localhost --duration 30
```

```plaintext
[W4]  MQTT failed (WinError 10061) -> JSONL fallback active

  fused events : 3  |  sync avg/min/max: 4.430 / 4.109 / 4.796 ms
  payload: 364B → 327B compact (10% saved)
  jsonl writes: 3  |  mqtt: 0  |  data lost: 0 bytes
```

---

## Known limitations

- **YOLOv10-N is COCO-trained** — it detects people, bottles, vehicles, etc., not casting defects or CNC scratches. Fusion events are near-zero with real video because no COCO objects appear in casting images. A fine-tuned model would be needed for genuine defect detection.
- **AI4I dataset is synthetic** — generated by a simulation model, not recorded from a real CNC machine. It is used as a realistic sensor stream proxy.
- **Casting images are pump impellers** — not CNC machine parts specifically. They serve as real industrial defect imagery for the vision pipeline.
- **No GPU** — all inference runs on CPU. YOLO latency is 15–70 ms per frame depending on load. A GPU would drop this to 1–3 ms.

---

## Output file

`factory_local_cache.jsonl` — written to the project root whenever a fusion event occurs and MQTT is not available. One JSON line per event:

```jsonl
{"ts":1781195431.96,"machine_id":"CNC-07","detections":[{"cls":0,"conf":0.87,"bbox":[110,85,185,155]}],"vibration":-2.34,"temperature":73.6,"sync_error_ms":4.384,"mad_anomaly":false,"qc":{"passed":true,"focus":48908.1,"brightness":127.0},"classifier":{"label":null,"conf":0},"infer_ms":15.02,"resize_ms":0.513}
```

---

## License  
Datasets are subject to their respective Kaggle licenses.  
YOLOv10 is subject to the [Ultralytics AGPL-3.0 license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).