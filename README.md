# face_verify

Dual-camera (RGB + IR) face verification for Linux PAM authentication.

Combines deep learning (ArcFace via ONNX Runtime) and classical methods
(LBP spatial histograms + template matching) in a decision-level ensemble.
Logs accept/reject decisions to syslog (`journalctl`).

## Hardware requirements

| Component | Requirement |
|---|---|
| RGB camera | Any USB or built-in webcam |
| **IR camera** | **Required** — a plain webcam is not sufficient |

The IR camera provides liveness detection (printed photos and screen replays
are rejected by checking that the IR stream differs from a flat surface).

**Compatible IR cameras:**
- Generic USB webcam with IR(Tested)

**Find your devices:**
```bash
v4l2-ctl --list-devices          # show all cameras
ffplay /dev/video0               # preview RGB
ffplay /dev/video2               # preview IR
```

## Dependencies

```bash
# Ubuntu / Debian
sudo apt install \
    build-essential cmake \
    libopencv-dev \
    libonnxruntime-dev \
    libpam-dev          # for pam_face_verify.so

# Fedora
sudo dnf install \
    cmake gcc-c++ \
    opencv-devel \
    onnxruntime-devel \
    pam-devel

# Arch
sudo pacman -S cmake opencv pam
yay -S onnxruntime
```

## Build

```bash
# 1. Download models
chmod +x scripts/download_models.sh
./scripts/download_models.sh

# 2. Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Artifacts:
#   build/face_verify          — CLI tool
#   build/pam_face_verify.so   — PAM module (if libpam-dev found)
```

## Configuration

Copy the example config and edit it:

```bash
sudo mkdir -p /etc/face_verify/{data,models} ##################
sudo cp face_verify.conf.example /etc/face_verify/face_verify.conf
sudo $EDITOR /etc/face_verify/face_verify.conf
```

The CLI searches for a config file in this order:
1. path given with `--config <path>`
2. `./face_verify.conf` in the current directory
3. `/etc/face_verify/face_verify.conf`

CLI flags always override values from the config file.

### All configuration options

| Key | Default | Description |
|---|---|---|
| `rgb_device` | `/dev/video0` | RGB camera device |
| `ir_device` | `/dev/video2` | IR camera device |
| `rgb_width` / `rgb_height` | `640` / `480` | RGB capture resolution |
| `ir_width` / `ir_height` | `640` / `360` | IR capture resolution |
| `data_dir` | `data` | Directory containing enrollment `.yml` files |
| `models_dir` | `models` | Directory containing `.onnx` model files |
| `threshold` | `0.50` | Ensemble score gate (lower = more permissive) |
| `dl_threshold` | `0.40` | Deep-learning score minimum (independent gate) |
| `dl_weight` | `0.60` | Weight of the deep-learning score in the ensemble (0.0–1.0) |
| `classical_weight` | `0.40` | Weight of the classical score in the ensemble (0.0–1.0) |
| `detect_conf_threshold` | `0.60` | Face detector minimum confidence (YuNet) |
| `detect_nms_threshold` | `0.30` | Face detector NMS IoU threshold (YuNet) |
| `num_frames` | `3` | Frame pairs to capture per attempt |
| `frame_interval_ms` | `200` | Milliseconds between frame captures |
| `liveness_min_shift` | `0.5` | Min face bbox shift (px) between frames to pass liveness. Lower if rejected despite moving. |
| `liveness_max_shift` | `80.0` | Max shift before rejecting (camera shake / different face) |
| `debug` | `false` | Log per-component scores to stderr / syslog |
| `debug_save_frames` | `false` | Save raw captured frames to disk on enroll/verify |

Dump the current effective config:
```bash
face_verify dump-config
```

## Enrollment

```bash
# Enroll your face (requires both cameras)
sudo face_verify enroll <unique-label> \
    --data-dir /etc/face_verify/data \
    --models-dir /etc/face_verify/models

# List enrolled faces
face_verify list --data-dir /etc/face_verify/data

# Remove an enrollment
sudo face_verify remove <unique-label> --data-dir /etc/face_verify/data
```

## CLI usage

```bash
face_verify enroll <unique-label>   # Enroll a new face
face_verify verify                  # Verify (exit 0 = match, 1 = rejected)
face_verify list                    # List enrolled labels
face_verify remove <unique-label>   # Delete an enrollment
face_verify dump-config             # Show effective configuration

Options:
  --config <path>        Config file (default: /etc/face_verify/face_verify.conf)
  --rgb-dev <path>       RGB camera device
  --ir-dev <path>        IR camera device
  --data-dir <path>      Enrollment data directory
  --models-dir <path>    Model directory
  --threshold <float>    Ensemble threshold
  --debug                Verbose per-component score logging
```

## PAM integration

Install the module and copy assets:

```bash
sudo cp build/pam_face_verify.so /lib/security/
sudo cp models/*.onnx /etc/face_verify/models/
sudo cp data/*.yml    /etc/face_verify/data/
sudo chmod 755 /lib/security/pam_face_verify.so
sudo chown -R root:root /etc/face_verify
```

### sudo

Create `/etc/pam.d/sudo`:

```
#%PAM-1.0

# Face verification — falls back to password on failure or no face detected
auth  sufficient  pam_face_verify.so

# Password fallback
@include common-auth
```

### su

Add one line before `@include common-auth` in `/etc/pam.d/su`:

```
auth  sufficient  pam_face_verify.so
@include common-auth
```

### Module arguments

All config-file keys are also accepted as PAM module arguments:

```
auth  sufficient  pam_face_verify.so \
    config=/etc/face_verify/face_verify.conf \
    data_dir=/etc/face_verify/data \
    models_dir=/etc/face_verify/models \
    threshold=0.55 \
    dl_threshold=0.40 \
    debug
```

Arguments override the config file. All keys from the configuration table above are accepted as module arguments (`key=value`). Boolean flags (`debug`, `debug_save_frames`) may be passed bare or as `flag=true` / `flag=false`.

## Logging (journalctl)

All accept/reject decisions are written to syslog at `LOG_AUTH` facility:

```bash
# Live tail of face verification events
journalctl -f -t face_verify

# All auth decisions (including password auth)
journalctl -f SYSLOG_FACILITY=10

# Or on older systems
grep face_verify /var/log/auth.log
```

Example output:
```
May 23 09:14:02 host sudo[1234]: face_verify: ACCEPT as XXXXXX (score=0.812)
May 23 09:14:45 host sudo[1235]: face_verify: REJECT (best score=0.321, threshold=0.50)
May 23 09:15:01 host sudo[1236]: face_verify: liveness check failed
```

Enable detailed score breakdown:
```bash
# Add to config file or as module arg
debug = true

# Then check
journalctl -f -t face_verify
```

## Tests

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

Tests cover: ensemble scoring math, liveness logic, FaceDB persistence (enroll / load / remove), config file parsing (all keys, bool variants, round-trip), and classical recognizer properties (LBP histogram correctness, similarity ordering).

> Tests do not require cameras or model files.

## Tuning thresholds

Run a verify with `--debug` to see per-component scores:

```
[ensemble] DL(rgb=0.72 ir=0.81 fused=0.77) Classical(rgb=0.61 ir=0.68 fused=0.65) Ensemble=0.72 -> ACCEPT
```

| Problem | Adjustment |
|---|---|
| Rejected too often | Lower `threshold` (e.g. 0.45) |
| Too easy to spoof | Raise `threshold` (e.g. 0.60) or `dl_threshold` |
| Liveness check fails | Lower `liveness_min_shift` (e.g. 0.5); move head slightly during verify |
| No IR face detected | Check `ir_device`, test with `ffplay /dev/video2` |
| Low DL score | Re-enroll with better lighting; ensure face fills the frame |

## License

!!!!!!!!!!!!!!!!!!! — see [LICENSE](LICENSE).
