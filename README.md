# Pix2Pix Generator Inference (ONNX Runtime) + rxing Decode on Raspberry Pi 4 vs Laptop

```
barcode crop → preprocess → Pix2Pix generator inference (ONNX) → postprocess → rxing decode → barcode text
```

---

## 1. TL;DR results
Both machines decoded the exact same text: `Stack384TCVIA1`.

| Hardware | Load model (ms) | Inference 30× – mean (ms) | Preprocess (ms) | Inference 1× (ms) | Postprocess (ms) | rxing decode (ms) | Total (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Laptop (i7-8550U) | 211 | 191 | 24.4 | 238.6 | 10.1 | 35.7 | 308.8 |
| Raspberry Pi 4 (2 GB) | 1619 | 1531 | 27.9 | 1539.0 | 21.3 | 71.4 | 1659.6 |

- **Load model**: time to create the ONNX Runtime session from `gen.onnx`. Happens once per process, so it's not part of Total.
- **Inference 30× – mean**: average `session.run` time over 30 runs, after 5 warm-up runs.
- **Preprocess, Inference 1×, Postprocess, Decode**: one end-to-end run after warm-up, with each stage timed separately.
- **Total**: preprocess + inference 1× + postprocess + decode. Model loading and saving the output image are not included.

---

## 2. Hardware specs

| | Laptop | Raspberry Pi |
|---|---|---|
| Device | Swift SF314-54G (V1.17) | Raspberry Pi 4 |
| CPU | Intel Core i7-8550U, 8 threads, up to 4.00 GHz | Broadcom BCM2711, 4× Cortex-A72 cores |

## 3. Software specs

| Component | Version / config |
|---|---|
| ONNX Runtime | 1.31.0 |
| Rust binding | `ort` crate 2.0.0-rc.14 |
| How ONNX Runtime is loaded on the Pi | `load-dynamic`, official `onnxruntime-linux-aarch64-1.31.0` library (`ORT_DYLIB_PATH`) |
| How ONNX Runtime is loaded on the laptop | default `ort` binary |
| Execution provider | `CPUExecutionProvider` |
| Session options | graph optimization Level3, `intra_op_num_threads = 4` (on both machines) |
| Decoder | `rxing` 0.9.x, `helpers::detect_in_luma` |
| Image handling | `image` 0.25, `ndarray` 0.17 |


## 4. How we measured

1. **Input**: a single `test.png` (CODE_128 barcode crop), loaded as RGB.
2. **Preprocess**: resize to 512 × 128 (CatmullRom filter), normalize with `pixel / 127.5 − 1`, build the `1×128×512×3` tensor.
3. **ONNX Runtime session**: created once (its time is what we report as *load model*).
4. **Warm-up**: 5 `session.run` calls, results thrown away.
5. **Inference benchmark**: 30 `session.run` calls + extracting the output tensor. We report the mean.
6. **End-to-end run**: one full run with a stopwatch on each stage:
   - *preprocess*: read file, resize, normalize
   - *inference*: `session.run` + extract output
   - *postprocess*: `[-1, 1] → [0, 255]`, build the RGB image
   - *decode*: convert to grayscale, then `rxing::helpers::detect_in_luma`
7. **Total**: sum of those four stages. Model loading and saving `restored.png` are not counted.

---

## 5. How to reproduce

**Setup on the Pi (Raspberry Pi OS 64-bit):**

```bash
# official ONNX Runtime library (aarch64), version 1.31.0
cd ~
tar xzf onnxruntime-linux-aarch64-1.31.0.tgz
export ORT_DYLIB_PATH=$HOME/onnxruntime-linux-aarch64-1.31.0/lib/libonnxruntime.so

# Rust and build tools
sudo apt install -y build-essential pkg-config
```

`Cargo.toml` on the Pi:

```toml
[dependencies]
ort = { version = "=2.0.0-rc.14", default-features = false, features = ["load-dynamic", "ndarray"] }
ndarray = "0.17"
image = "0.25"
rxing = "0.9"
```

`Cargo.toml` on the laptop:

```toml
ort = { version = "=2.0.0-rc.14", features = ["tls-rustls"] }
```

**Run the benchmark:**

```bash
cd inference
cargo run --release -- gen.onnx test.png
```

Put `gen.onnx` and `test.png` in the project folder (next to `Cargo.toml`).

**Tips for steadier numbers on the Pi:**

```bash
# log the state before and after the test
vcgencmd measure_temp
vcgencmd get_throttled          # 0x0 means no throttling
vcgencmd measure_clock arm

# lock the CPU clock during the test
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

---

## 6. Compatibility gotcha

On Raspberry Pi OS Bookworm (GCC 12.2), the ONNX Runtime binary that `ort-sys` rc.14
downloads by default fails to link (`undefined reference` to `__cxa_call_terminate` and
`std::__cxx11::basic_string::_M_replace_cold`, which come from a newer libstdc++).
What we did: use the `load-dynamic` feature with the official `libonnxruntime.so` from
Microsoft. The other option is moving to an OS based on Debian 13 (Trixie).

---
