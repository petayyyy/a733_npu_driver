# Gate G1 Report - Radxa Cubie A7Z

Date: 2026-06-20 local / 2026-06-19 UTC

## Board

- Host: `radxa-cubie-a7z`
- SSH: `radxa@192.168.31.76`
- OS: Debian GNU/Linux 11 (bullseye)
- Kernel: `5.15.147-21-a733`
- CPU cores: 8
- NPU device: `/dev/vipcore` present as `crw-rw-rw- 1 root root 199, 0`

## Runtime Found On Board

- VIPLite libraries: `/home/radxa/lib/libNBGlinker.so`,
  `/home/radxa/lib/libVIPhal.so`
- Existing YOLO project: `/home/radxa/yolo_shm`
- Model: `/home/radxa/yolo_shm/yolov8n_6_uint8_a733.nb`
- Model size: 2,452,448 bytes
- Single-image test binary built from existing source:
  `/home/radxa/yolo_shm/build/test_npu_a733`

## Inference Command

```bash
cd /home/radxa/yolo_shm
export LD_LIBRARY_PATH=/home/radxa/lib:$LD_LIBRARY_PATH
export VIV_VX_PROFILE=1
export VIV_VX_DEBUG_LEVEL=1
./build/test_npu_a733 -nb yolov8n_6_uint8_a733.nb -i dog.jpg
```

## Evidence

VIPLite initialized and loaded the NBG:

```text
VIPLite driver software version 2.0.3.2-AW-2024-08-30
input  0 dim 3 640 640 1
nbg name=yolov8n_6_uint8_a733.nb, size: 2452448.
create network 0: 5792 us.
prepare network: 978 us.
```

Postprocess detections from `dog.jpg`:

```text
detection num: 3
 1:  85%, [ 130,  137,  568,  419], bicycle
16:  95%, [ 131,  219,  308,  540], dog
 2:  66%, [ 466,   74,  694,  171], car
```

## Gate Result

G1 is passed: the A733 VIP9000 NPU path through `/dev/vipcore`, VIPLite
2.0.3.2, and an NBG YOLOv8n graph is working on Debian.

Final smoke-test summary with the custom YOLO command:

```text
Summary: pass=6 warn=2 fail=0
```

## Notes

- `vpm_run` was not found on the image.
- The existing `run_yolo.sh` referenced `build/test_npu_a733`, but that target
  was not present. I built it from the existing `test_main.cpp` and project
  sources.
- The existing streaming binary `yolo_shm_a733` initializes VIPLite correctly
  but expects a shared-memory video source and exits with `[SHM]: No such file
  or directory` when that source is not running.
