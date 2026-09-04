# Reconstruction data-loading optimization log

Date: 2026-09-04  
Dataset: `/zsh/cache/data/Lerobot/libero`  
Workload: 20 frames, interval 3, two 256x256 camera views, batch size 24

## Baseline: inherited LeRobot reader

The original `ReconstructionDataset` inherited `LeRobotDataset`, whose HF
dataset installed `hf_transform_to_torch`. A selected column therefore entered
the transform path and decoded image columns through PIL. On the current
machine, direct single-sample measurements were 5.09–12.33 s. A 24-sample
threaded batch with the existing eight decode threads was about 2.5 s in the
training profile.

The detailed operation measurements showed the cause:

| Operation | Time for 20 frames |
| --- | ---: |
| `rows["timestamp"]` after HF transform | 1.36 s |
| head image column conversion | 2.92 s |
| wrist image column conversion | 4.87 s |

The images are PNG bytes embedded in Parquet (`meta.video_keys == []`), so this
was image decoding and conversion overhead rather than video seeking.

## Iteration 1: direct Arrow bytes and torchvision decoding

Changed image reads to `hf_dataset.data[column][index].as_py()` and decoded
inline bytes with `torchvision.io.decode_image`. The result stays `uint8`; the
existing GPU-side input path performs the same `/255` normalization as before.
The numerical comparison against the old PIL path had maximum absolute error
`0`.

Single-sample measurements after this change were 0.19–0.55 s in the first
run, and 0.13–0.20 s after the cache was warm. The per-sample profile from a
12-sample run was:

| Detail | Mean |
| --- | ---: |
| image decode | 636.7 ms |
| resize/crop | 0.27 ms |
| view assembly | 27.9 ms |
| total sample | 667.6 ms |

The larger first-run average reflects filesystem/cache variation; it is still
well below the inherited PIL path.

## Iteration 2: remove the LeRobotDataset inheritance

`ReconstructionDataset` now loads metadata directly, creates a raw Arrow
dataset without installing the HF transform, and retains only camera columns
plus `timestamp`. The full Parquet column list is used for cache reuse, then
`select_columns` avoids rebuilding a second large Arrow cache. Video-backed
camera keys retain the existing decoder path.

Initialization measured about 2.6 s with the existing cache, and three samples
measured 0.175, 0.133, and 0.202 s. Both explicit multiview and the legacy
single-view call form were smoke-tested.

## Iteration 3: batch pipeline and profiling detail

The bounded loader now records collate time and exposes dataset timings through
the existing single `profile.json`; these metrics are under
`profile.data_detail` and are not sent to W&B. The profile window resets also
reset the data counters, so warmup samples are excluded.

The first sweep (before the independent-reader change) using 10 batches of 24
samples was:

| Decode threads | Mean batch wait |
| ---: | ---: |
| 4 | 1.14 s |
| 8 | 0.82 s |
| 12 | 0.75 s |

Eight threads remains the current configuration because twelve only improves
the measured wait marginally while increasing per-sample and collate CPU
contention. The training process used for profiling was stopped before these
changes were tested; the code takes effect on the next launch.

After the independent reader was enabled, eight threads produced batch waits
of `0.734, 0.599, 0.594, 0.599, 0.764, 0.639, 0.579, 0.599, 0.548` seconds
after the first batch, for a mean of `0.628 s`. Its profile reported 203.5 ms
image decode, 0.51 ms resize/crop, 7.66 ms assembly, and 91.1 ms collate per
batch (240 samples, 10 batches).

## Training verification after restart from step 1000

The optimized training was launched from `step-00001000.pt` in tmux session
`vrae_multiview_s_resume1000_opt`. The first profile window reported
`data_wait=55.5 ms`; subsequent windows reported `36.3 ms`, `1.1 ms`, `1.6 ms`,
and `32.2 ms` per batch while running at roughly `1.06–1.11 s/step`. The
profile detail at step 1140 reported 174.5 ms image decode, 0.57 ms
resize/crop, 3.32 ms assembly, and 37.2 ms collate. This confirms end to end
that data loading is no longer the dominant training stall.

## Current conclusion

The dominant cost was repeated PIL decoding through the HF transform, not
`F.interpolate`, GPU transfer, or video seeking. The current path removes that
overhead, skips identity resizes, keeps normalization on the GPU, and exposes
the remaining decode/assembly/collate costs for the next training run.

## Iteration 4: reduce training profiler overhead

The training profiler now wraps only `data.loading`, `stage1`, `dit.forward`,
and `dit.backward`. DIT block/module wrappers and the nested Stage1 wrappers
were removed. The report is written every 100 global steps and contains only
`average_metrics`; individual windows are no longer serialized. The running
verification reached step 1100 and produced `version: 2`, `interval_steps: 100`
with no `windows` field.
