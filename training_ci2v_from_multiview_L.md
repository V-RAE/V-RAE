[epoch-end] epoch=67 elapsed=382.52s step=7140
[epoch-start] epoch=68 step=7140 steps_per_epoch=105
ultiview_s_retry.yaml` training process group at approximately step 5840. The last observed loss was `0.761261`.
- Verified `configs/training/generation/libero_ci2v_from_multiview_L.yaml` with `--build-only`; task/dataset/chunk/transport validation passed.
- This host currently has one GPU. For `global_batch_size=16` and `gradient_accumulation_steps=4`, the resolved local micro-batch size is 4.
- Planned launch: `libero_ci2v_from_multiview_L.yaml` with output appended below and mirrored to `logs/train_ci2v_from_multiview_L.log`.
- 20:56: first L attempt failed at the first VideoDiT forward with `torch.OutOfMemoryError` on the 79.18 GiB GPU; peak allocation reached 79.13 GiB. The failure occurred with `dit.gradient_checkpointing: false` and no checkpoint was written.
- 20:57: changed `dit.gradient_checkpointing` to `true` to reduce activation memory before restart.
- 20:57: the failed attempt left a non-empty run directory, so the restarted attempt uses run name `vjepa2_1_libero_ci2v_from_multiview_l_gc` to preserve the failure artifacts.
- 21:06: checkpointed run reached the end of epoch 0 (`step=105`) without OOM. At `step=100`, loss was `2.775942`, allocated GPU memory was about 28.7 GiB, and reserved memory about 35.5 GiB.
- 21:07: run is active in epoch 1 at `step=120`, loss `2.287875`; GPU usage remains stable at about 37.1 GiB.
- 21:08: run remains active in epoch 1 at `step=140`, loss `2.120821`; no new errors observed.
- 21:10: run remains active in epoch 1 at `step=160`, loss `2.050769`; GPU memory remains stable at about 37.1 GiB.
- 21:10: process verification confirms PID `17993` is still active at `step=160`; no checkpoint has been written yet because the configured interval is 1000 steps.
- 21:11: run remains active in epoch 1 at `step=180`, loss `2.543123`; no new errors and GPU memory remains about 37.1 GiB.
- 21:11: verified the trainer process and W&B-backed run are still alive; log output is advancing and no checkpoint is expected before step 1000.
- 21:11: current verification still finds PID `17993` running at `step=180`, with approximately 37.1 GiB GPU memory in use; checkpoint directory remains empty as expected before step 1000.
- 21:12: run advanced to `epoch=1, step=200`, loss `1.966541`; profile window reports allocated GPU memory about 28.7 GiB and peak about 34.1 GiB, with no errors.
- 21:13: epoch 1 completed successfully at `step=210`; run entered epoch 2. No checkpoint yet because the configured save interval is 1000 steps.
- 21:13: verification confirms epoch 1 completion and active epoch 2 at `step=210`; GPU memory remains about 37.1 GiB with no errors.
- 21:14: run advanced to `epoch=2, step=220`, loss `2.134679`; process remains active and GPU memory is stable.
- 21:15: run advanced to `epoch=2, step=240`, loss `1.998211`; process remains active with no errors.
- 21:15: latest verification confirms PID `17993` is active at `epoch=2, step=240`; GPU usage is about 37.1 GiB and checkpoint directory is still empty before step 1000.
- 21:16: run advanced to `epoch=2, step=260`, loss `1.943248`; no errors observed and GPU memory remains stable.
- 21:17: latest check confirms PID `17993` is active at `epoch=2, step=260`; checkpoint directory remains empty before the configured 1000-step interval.
- 21:18: run advanced to `epoch=2, step=280`, loss `1.976372`; no errors observed, and the training log continues to grow.
- 21:18: process verification confirms PID `17993` remains active at `epoch=2, step=280`; GPU memory is stable and no checkpoint is expected before step 1000.
- 21:19: run advanced to `epoch=2, step=300`, loss `1.888905`; profile remains stable (allocated about 28.7 GiB, peak about 34.1 GiB) with no errors.
- 21:22: epoch 2 completed successfully at `step=315`; run entered epoch 3. Profile remains stable and no checkpoint is expected before step 1000.
- 21:20: run advanced to `epoch=3, step=320`, loss `1.880365`; process is active with stable GPU memory and no errors.
- 21:20: latest verification confirms PID `17993` remains active at `epoch=3, step=320`; checkpoint directory is still empty before step 1000.
- 21:21: run advanced to `epoch=3, step=340`, loss `1.884779`; process and GPU usage remain stable with no errors.
- 21:22: run advanced to `epoch=3, step=360`, loss `1.728921`; no errors observed and GPU memory remains stable.
- 21:24: run advanced to `epoch=3, step=400`, loss `1.562099`; profile reports allocated GPU memory about 28.7 GiB and peak about 34.1 GiB, with PID `17993` still active. No checkpoint is expected before step 1000.
- 21:25: verification confirms PID `17993` is still active after the step-400 profile; GPU usage is about 37.1 GiB, no error lines appeared, and the checkpoint directory remains empty as expected before step 1000.
- 21:26: epoch 3 completed at `step=420` with loss `1.536422`; run entered epoch 4. PID `17993` remains active, GPU usage is about 37.1 GiB, and no checkpoint is expected before step 1000.
- 21:26: follow-up verification still finds PID `17993` active in epoch 4; the log has no traceback/OOM/error after `step=420`, GPU usage remains about 37.1 GiB, and checkpoints are still pending the 1000-step interval.
- 21:26: another live-process check confirms the trainer and torchrun parents are present; epoch 4 is still computing after `step=420`, with no terminal/error output and stable GPU usage.
- 21:27: run advanced to `epoch=4, step=440`, loss `1.476651`; PID `17993` remains active, GPU usage is about 37.1 GiB, and no error output or checkpoint is present yet.
- 21:27: live verification confirms the same trainer process is active and the log has no terminal/error lines after `step=440`; checkpoint interval remains 1000 steps.
- 21:28: run advanced to `epoch=4, step=460`, loss `1.287621`; trainer PID `17993` remains active with stable GPU usage around 37.1 GiB and no errors.
- 21:29: verification confirms PID `17993` is still active at `epoch=4, step=460`; no error or termination output has appeared, GPU usage remains about 37.1 GiB, and no checkpoint is expected before step 1000.
- 21:30: run advanced to `epoch=4, step=480`, loss `1.417124`; trainer remains active with GPU usage about 37.1 GiB, no errors, and no checkpoint yet before step 1000.
- 21:30: live check confirms PID `17993` and its torchrun parents remain active at `step=480`; no termination/error output is present and checkpoint saving is still pending step 1000.
- 21:31: run advanced to `epoch=4, step=500`, loss `1.228363`; profile remains stable (allocated about 28.7 GiB, peak about 34.1 GiB), with no errors and no checkpoint before step 1000.
- 21:31: live check confirms the trainer remains active after the step-500 profile; no traceback/OOM/termination output is present and checkpoint saving remains pending step 1000.
- 21:32: verification confirms the process is still active during the next normal training window after `step=500`; no error or exit output is present, GPU usage remains about 37.1 GiB, and no checkpoint is expected before step 1000.
- 21:32: trainer PID `17993` remains active after the step-500 profile; no new error/termination lines appeared during this check, and checkpoint saving is still pending step 1000.
- 21:32: epoch 4 completed at `step=525`; the run entered epoch 5. Latest logged step was `520` with loss `1.358590`; PID `17993` remains active, GPU usage is about 37.1 GiB, and no checkpoint is expected before step 1000.
- 21:33: live check confirms epoch 5 is still running after `step=525`; trainer PID `17993` and GPU allocation remain active, with no error/termination output and no checkpoint before step 1000.
- 21:33: run advanced to `epoch=5, step=540`, loss `1.391029`; PID `17993` remains active with GPU usage about 37.1 GiB, no errors, and checkpoint saving still pending step 1000.
- 21:34: live verification confirms PID `17993` remains active after `step=540`; no traceback/OOM/termination output is present, GPU usage is stable, and checkpoint saving remains pending step 1000.
- 21:35: run advanced to `epoch=5, step=560`, loss `1.291268`; trainer PID `17993` remains active, GPU usage is about 37.1 GiB, and no checkpoint or error has appeared.
- 21:36: run advanced to `epoch=5, step=580`, loss `1.267787`; trainer remains active with GPU usage about 37.1 GiB, no errors, and checkpoint saving still pending step 1000.
- 21:37: run advanced to `epoch=5, step=600`, loss `1.315734`; profile reports allocated GPU memory about 28.7 GiB and peak about 34.1 GiB. PID `17993` remains active with no errors; checkpoint saving is still pending step 1000.
- 21:38: run advanced to `epoch=5, step=620`, loss `1.168958`; trainer PID `17993` remains active with GPU usage about 37.1 GiB, no errors, and checkpoint saving still pending step 1000.
- 21:39: epoch 5 completed at `step=630`; run entered epoch 6. Latest logged step was `620` with loss `1.168958`; PID `17993` remains active, GPU usage is about 37.1 GiB, and no checkpoint is expected before step 1000.
- 21:40: run advanced to `epoch=6, step=640`, loss `1.232628`; trainer PID `17993` remains active with GPU usage about 37.1 GiB, no errors, and checkpoint saving still pending step 1000.
- 21:41: run advanced to `epoch=6, step=660`, loss `1.201708`; trainer PID `17993` remains active with GPU usage about 37.1 GiB, no errors, and checkpoint saving still pending step 1000.
- 21:42: live verification confirms PID `17993` remains active during the next normal compute window after `step=660`; no error/termination output appeared and checkpoint saving remains pending step 1000.
- 21:42: run advanced to `epoch=6, step=680`, loss `1.175758`; trainer PID `17993` remains active with GPU usage about 37.1 GiB, no errors, and checkpoint saving still pending step 1000.
- 21:43: live verification confirms the trainer remains active after `step=680`; no error/termination output is present, GPU usage is stable around 37.1 GiB, and checkpoint saving remains pending step 1000.
- 21:44: run advanced to `epoch=6, step=700`, loss `1.103614`; profile reports allocated GPU memory about 28.7 GiB and peak about 34.1 GiB. PID `17993` remains active with no errors; checkpoint saving is still pending step 1000.
- 21:45: run advanced to `epoch=6, step=720`, loss `1.241891`; trainer PID `17993` remains active with GPU usage about 37.1 GiB, no errors, and checkpoint saving still pending step 1000.
- 21:45: epoch 6 completed at `step=735`; run entered epoch 7. Latest logged step was `720` with loss `1.241891`; PID `17993` remains active, GPU usage is about 37.1 GiB, and no checkpoint is expected before step 1000.
- 21:46: run advanced to `epoch=7, step=740`, loss `1.146844`; trainer PID `17993` remains active with GPU usage about 37.1 GiB, no errors, and checkpoint saving still pending step 1000.
- 21:47: run advanced to `epoch=7, step=760`, loss `1.207947`; trainer PID `17993` remains active with GPU usage about 37.1 GiB, no errors, and checkpoint saving still pending step 1000.
- 21:48: run advanced to `epoch=7, step=780`, loss `1.068355`; trainer PID `17993` remains active with GPU usage about 37.1 GiB, no errors, and checkpoint saving still pending step 1000.
- 21:49: live verification confirms PID `17993` remains active after `step=780`; no traceback/OOM/termination output is present, GPU usage remains about 37.1 GiB, and checkpoint saving is still pending step 1000.
- 21:49: run advanced to `epoch=7, step=800`, loss `1.145005`; profile reports allocated GPU memory about 28.7 GiB and peak about 34.1 GiB. PID `17993` remains active with no errors; checkpoint saving is still pending step 1000.
- 21:50: live verification confirms PID `17993` remains active after the step-800 profile; no traceback/OOM/termination output is present, GPU usage remains about 37.1 GiB, and checkpoint saving remains pending step 1000.
- 21:51: run advanced to `epoch=7, step=820`, loss `1.043442`; trainer PID `17993` remains active with GPU usage about 37.1 GiB, no errors, and checkpoint saving still pending step 1000.
- 21:52: epoch 7 completed at `step=840`; run entered epoch 8. Latest logged step was `840` with loss `1.196561`; PID `17993` remains active, GPU usage is about 37.1 GiB, and no checkpoint is expected before step 1000.
- 21:53: run advanced to `epoch=8, step=860`, loss `1.075567`; trainer PID `17993` remains active with GPU usage about 37.1 GiB, no errors, and checkpoint saving still pending step 1000.
- 21:54: run advanced to `epoch=8, step=880`, loss `1.109346`; trainer PID `17993` remains active with GPU usage about 37.1 GiB, no errors, and checkpoint saving still pending step 1000.
- 21:59: run advanced to `epoch=8, step=900`, loss `1.030423`; profile reports allocated GPU memory about 28.7 GiB and peak about 34.1 GiB. PID `17993` remains active with no errors; checkpoint saving is expected at step 1000.
- 22:00: run advanced to `epoch=8, step=920`, loss `1.055093`; trainer PID `17993` remains active with GPU usage about 37.1 GiB, no errors, and checkpoint saving remains expected at step 1000.
- 22:01: run advanced to `epoch=8, step=940`, loss `1.053696`; trainer PID `17993` remains active with GPU usage about 37.1 GiB, no errors, and checkpoint saving remains expected at step 1000.
- 22:02: run advanced to `epoch=9, step=960`, loss `0.999212`; trainer PID `17993` remains active with GPU usage about 37.1 GiB, no new errors, and the first checkpoint remains expected at step 1000.
- 22:05: run reached `epoch=9, step=1000`, loss `1.038965`; checkpoint `samples/step-00001000.pt` was written successfully. The trainer remains active; post-checkpoint sample video encoding temporarily uses about 40.8 GiB GPU memory, with no fatal error observed.
- 22:07: post-checkpoint sampling is still actively using the GPU (`98%` utilization, PID `17993` alive, about `40.8 GiB` allocated); no new error or termination output is present. The saved checkpoint `checkpoints/step-00001000.pt` is intact.
- 22:09: sampling completed and training resumed at `epoch=9, step=1020`, loss `0.978945`; PID `17993` remains active with GPU utilization near `100%` and no new errors. The `step-00001000.pt` checkpoint remains intact.
- 22:11: run advanced to `epoch=9, step=1040`, loss `1.072561`; trainer PID `17993` remains active with GPU memory about `40.8 GiB`, and no new errors or termination output.
- 22:13: epoch 9 completed at `step=1050`; run entered epoch 10. Trainer PID `17993` remains active, the step-1000 checkpoint is intact, and no fatal error has appeared.
- 22:15: run advanced to `epoch=10, step=1060`, loss `1.000047`; trainer PID `17993` remains active with no new errors, and GPU memory remains about `40.8 GiB`.
- 22:17: run advanced to `epoch=10, step=1080`, loss `1.086707`; trainer PID `17993` remains active, checkpoint `step-00001000.pt` is present, and no fatal errors were detected.
- 22:20: run advanced to `epoch=10, step=1100`, loss `1.098396`; profile reports allocated GPU memory about `29.0 GiB`, reserved `39.1 GiB`, peak `37.9 GiB`. Trainer PID `17993` remains active and no errors were detected.
- 22:22: run advanced to `epoch=10, step=1120`, loss `0.986295`; trainer PID `17993` remains active with GPU utilization about `91%`, no new errors, and the step-1000 checkpoint remains intact.
- 22:25: run advanced to `epoch=10, step=1140`, loss `0.984865`; trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.
- 22:27: epoch 10 completed at `step=1155`; run entered epoch 11. Trainer PID `17993` remains active, GPU memory is about `40.8 GiB`, and no fatal errors were detected.
- 22:29: run advanced to `epoch=11, step=1160`, loss `0.960983`; trainer PID `17993` remains active, no new errors were detected, and the step-1000 checkpoint remains intact.
- 22:31: run advanced to `epoch=11, step=1180`, loss `1.002442`; trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.
- 22:34: run advanced to `epoch=11, step=1200`, loss `1.040824`; profile reports allocated GPU memory about `29.0 GiB`, reserved `39.1 GiB`, peak `37.9 GiB`. Trainer PID `17993` remains active and no errors were detected.
- 22:36: run advanced to `epoch=11, step=1220`, loss `0.950405`; trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.
- 22:38: run advanced to `epoch=11, step=1240`, loss `0.909270`; trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.
- 22:40: epoch 11 completed at `step=1260`; latest logged loss was `0.945821`, and the run entered epoch 12. Trainer PID `17993` remains active with no fatal errors.
- 22:42: run advanced to `epoch=12, step=1280`, loss `0.995037`; trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.
- 22:44: run advanced to `epoch=12, step=1300`, loss `0.958845`; profile reports allocated GPU memory about `29.0 GiB`, reserved `39.1 GiB`, peak `37.9 GiB`. Trainer PID `17993` remains active and no errors were detected.
- 22:46: run advanced to `epoch=12, step=1320`, loss `0.976904`; trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.
- 22:48: run advanced to `epoch=12, step=1340`, loss `0.957906`; trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.
- 22:50: run advanced to `epoch=12, step=1360`, loss `1.025861`; trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.
- 22:52: epoch 12 completed at `step=1365`; latest logged loss was `1.025861`, and the run entered epoch 13. Trainer PID `17993` remains active with no fatal errors.
- 22:54: run advanced to `epoch=13, step=1380`, loss `0.912579`; trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.
- 22:56: run advanced to `epoch=13, step=1400`, loss `0.923682`; profile reports allocated GPU memory about `29.0 GiB`, reserved `39.1 GiB`, peak `37.9 GiB`. Trainer PID `17993` remains active and no errors were detected.
- 22:58: run advanced to `epoch=13, step=1420`, loss `0.945586`; trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.
- 23:00: run advanced through `epoch=13, step=1460`, loss `0.914778`, completed epoch 13 at `step=1470`, and entered epoch 14. Trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.
- 23:02: run advanced to `epoch=14, step=1480`, loss `0.937636`; trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.
- 23:04: live verification still finds trainer PID `17993` active at `epoch=14, step=1480`; no new error or termination output appeared, GPU memory remains about `40.8 GiB`, and the step-1000 checkpoint is intact.
- 23:06: run advanced to `epoch=14, step=1500`, loss `0.971276`; the profile window reports allocated GPU memory about `29.0 GiB`, reserved `39.1 GiB`, peak `37.9 GiB`. Trainer PID `17993` remains active with no errors, and the step-1000 checkpoint is intact.
- 23:08: live verification still finds trainer PID `17993` active after `step=1500`; no new error or termination output appeared, GPU memory remains about `40.8 GiB`, and the step-1000 checkpoint is intact.
- 23:10: live verification still finds trainer PID `17993` active after `step=1500`; the process is consuming GPU memory normally and no new error or termination output has appeared. The step-1000 checkpoint remains intact.
- 23:12: run advanced to `epoch=14, step=1520`, loss `0.925605`; trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.
- 23:14: live verification confirms trainer PID `17993` remains active after `step=1520`; no error or termination output is present, GPU memory remains about `40.8 GiB`, and the step-1000 checkpoint is intact.
- 23:16: live verification confirms trainer PID `17993` remains active after `step=1520`; no new error or termination output is present, GPU memory remains about `40.8 GiB`, and the step-1000 checkpoint is intact.
- 23:18: live verification confirms trainer PID `17993` remains active after `step=1520`; no new error or termination output is present, GPU memory remains about `40.8 GiB`, and the step-1000 checkpoint is intact.
- 23:20: run advanced to `epoch=14, step=1540`, loss `0.919933`; trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.
- 23:22: live verification confirms trainer PID `17993` remains active at `epoch=14, step=1540`; no new error or termination output is present, GPU memory remains about `40.8 GiB`, and the step-1000 checkpoint is intact.
- 23:24: live verification confirms trainer PID `17993` remains active at `epoch=14, step=1540`; no new error or termination output is present, GPU memory remains about `40.8 GiB`, and the step-1000 checkpoint is intact.
- 23:26: run advanced to `epoch=14, step=1560`, loss `0.955115`; trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.
- 23:28: live verification confirms trainer PID `17993` remains active after `step=1560`; no new error or termination output is present, GPU memory remains about `40.8 GiB`, and the step-1000 checkpoint is intact.
- 23:30: epoch 14 completed at `step=1575`; latest logged loss was `0.955115` at `step=1560`, and the run entered epoch 15. Trainer PID `17993` remains active with no fatal errors, GPU memory about `40.8 GiB`, and the step-1000 checkpoint intact.
- 23:32: run advanced to `epoch=15, step=1580`, loss `0.924605`; trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.
- 23:34: live verification confirms trainer PID `17993` remains active at `epoch=15, step=1580`; no new error or termination output is present, GPU memory remains about `40.8 GiB`, and the step-1000 checkpoint is intact.
- 23:36: live verification confirms trainer PID `17993` remains active at `epoch=15, step=1580`; no new error or termination output is present, GPU memory remains about `40.8 GiB`, and the step-1000 checkpoint is intact.
- 23:38: run advanced to `epoch=15, step=1600`, loss `0.913364`; profile reports allocated GPU memory about `29.0 GiB`, reserved `39.1 GiB`, peak `37.9 GiB`. Trainer PID `17993` remains active with no errors, and the step-1000 checkpoint remains intact.
- 23:40: live verification confirms trainer PID `17993` remains active after `step=1600`; no new error or termination output is present, GPU memory remains about `40.8 GiB`, and the step-1000 checkpoint is intact.
- 23:42: live verification confirms trainer PID `17993` remains active after `step=1600`; no new error or termination output is present, GPU memory remains about `40.8 GiB`, and the step-1000 checkpoint is intact.
- 23:44: run advanced to `epoch=15, step=1620`, loss `0.935380`; trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.
- 23:46: live verification confirms trainer PID `17993` remains active after `step=1620`; no new error or termination output is present, GPU memory remains about `40.8 GiB`, and the step-1000 checkpoint is intact.
- 23:48: live verification confirms trainer PID `17993` remains active after `step=1620`; no new error or termination output is present, GPU memory remains about `40.8 GiB`, and the step-1000 checkpoint is intact.
- 23:50: run advanced to `epoch=15, step=1640`, loss `0.900972`; trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.
- 23:52: live verification confirms trainer PID `17993` remains active at `epoch=15, step=1640`; no new error or termination output is present, GPU memory remains about `40.8 GiB`, and the step-1000 checkpoint is intact.
- 23:54: live verification confirms trainer PID `17993` remains active at `epoch=15, step=1640`; no new error or termination output is present, GPU memory remains about `40.8 GiB`, and the step-1000 checkpoint is intact.
- 23:56: run advanced to `epoch=15, step=1660`, loss `0.868003`; trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.
- 23:58: live verification confirms trainer PID `17993` remains active at `epoch=15, step=1660`; no new error or termination output is present, GPU memory remains about `40.8 GiB`, and the step-1000 checkpoint is intact.
- 00:00: epoch 15 completed at `step=1680`; latest logged loss was `0.913431`, and the run entered epoch 16. Trainer PID `17993` remains active with no fatal errors, GPU memory about `40.8 GiB`, and the step-1000 checkpoint intact.
- 00:02: live verification confirms trainer PID `17993` remains active at the start of epoch 16 (`step=1680`); no new error or termination output is present, GPU memory remains about `40.8 GiB`, and the step-1000 checkpoint is intact.
- 00:04: live verification confirms trainer PID `17993` remains active at the start of epoch 16 (`step=1680`); no new error or termination output is present, GPU memory remains about `40.8 GiB`, and the step-1000 checkpoint is intact.
- 00:06: run advanced to `epoch=16, step=1700`, loss `0.961621`; profile reports allocated GPU memory about `29.0 GiB`, reserved `39.1 GiB`, peak `37.9 GiB`. Trainer PID `17993` remains active with no errors, and the step-1000 checkpoint remains intact.
- 00:08: live verification confirms trainer PID `17993` remains active at `epoch=16, step=1700`; no new error or termination output is present, GPU memory remains about `40.8 GiB`, and the step-1000 checkpoint is intact.
- 00:10: live verification confirms trainer PID `17993` remains active at `epoch=16, step=1700`; no new error or termination output is present, GPU memory remains about `40.8 GiB`, and the step-1000 checkpoint is intact.
- 00:12: run advanced to `epoch=16, step=1720`, loss `0.835395`; trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.
- 00:14: live verification confirms trainer PID `17993` remains active at `epoch=16, step=1720`; no new error or termination output is present, GPU memory remains about `40.8 GiB`, and the step-1000 checkpoint is intact.
- 00:16: run advanced to `epoch=16, step=1740`, loss `0.924109`; trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.
- 00:18: live verification confirms trainer PID `17993` remains active at `epoch=16, step=1740`; no new error or termination output is present, GPU memory remains about `40.8 GiB`, and the step-1000 checkpoint is intact.
- 00:20: run advanced to `epoch=16, step=1760`, loss `0.952822`; trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.
- 00:22: live verification confirms trainer PID `17993` remains active at `epoch=16, step=1760`; no new error or termination output is present, GPU memory remains about `40.8 GiB`, and the step-1000 checkpoint is intact.
- 00:24: run advanced to `epoch=16, step=1780`, loss `0.904110`; trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.
- 00:26: epoch 16 completed at `step=1785`; latest logged loss was `0.904110` at `step=1780`, and the run entered epoch 17. Trainer PID `17993` remains active with no fatal errors, GPU memory about `40.8 GiB`, and the step-1000 checkpoint intact.
- 00:28: live verification confirms trainer PID `17993` remains active at the start of epoch 17 (`step=1785`); no new error or termination output is present, GPU memory remains about `40.8 GiB`, and the step-1000 checkpoint is intact.
- 00:30: run advanced to `epoch=17, step=1800`, loss `0.947476`; profile reports allocated GPU memory about `29.0 GiB`, reserved `39.1 GiB`, peak `37.9 GiB`. Trainer PID `17993` remains active with no errors, and the step-1000 checkpoint remains intact.
- 00:32: live verification confirms trainer PID `17993` remains active at `epoch=17, step=1800`; no new error or termination output is present, GPU memory remains about `40.8 GiB`, and the step-1000 checkpoint is intact.
- 00:34: run advanced to `epoch=17, step=1820`, loss `0.866943`; trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.
- 00:36: run advanced to `epoch=17, step=1840`, loss `0.855312`; trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.
- 00:38: run advanced to `epoch=17, step=1860`, loss `0.849310`; trainer PID `17993` remains active, no new errors were detected, and the step-1000 checkpoint remains intact.
- 00:40: run advanced to `epoch=17, step=1880`, loss `0.835727`; trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.
- 00:42: epoch 17 completed at `step=1890`; run entered epoch 18. Trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.
- 00:44: run advanced to `epoch=18, step=1900`, loss `0.968320`; trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.
- 00:46: live verification confirms trainer PID `17993` remains active after the step-1900 profile; no new error or termination output is present, GPU memory is about `40.8 GiB`, and the step-1000 checkpoint remains intact.
- 00:48: live verification confirms trainer PID `17993` remains active in epoch 18 after `step=1900`; no new error or termination output is present, GPU memory is about `40.8 GiB`, and the step-1000 checkpoint remains intact.
- 00:50: run advanced to `epoch=18, step=1920`, loss `0.897143`; trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.
- 00:52: live verification confirms trainer PID `17993` remains active after `step=1920`; no new error or termination output is present, GPU memory is about `40.8 GiB`, and the step-1000 checkpoint remains intact.
- 00:54: live verification confirms trainer PID `17993` remains active after `step=1920`; both `checkpoints/step-00001000.pt` and `samples/step-00001000.pt` are intact, with no new error or termination output.
- 00:56: run advanced to `epoch=18, step=1940`, loss `0.914479`; trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.
- 00:58: live verification confirms trainer PID `17993` remains active after `step=1940`; no new error or termination output is present, GPU memory is about `40.8 GiB`, and both step-1000 checkpoint and sample remain intact.
- 01:00: live verification confirms trainer PID `17993` remains active in epoch 18 after `step=1940`; no new error or termination output is present, GPU memory is about `40.8 GiB`, and the step-1000 checkpoint remains intact.
- 01:02: run advanced to `epoch=18, step=1960`, loss `0.872961`; trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.
- 01:04: live verification confirms trainer PID `17993` remains active after `step=1960`; no new error or termination output is present, GPU memory is about `40.8 GiB`, and the step-1000 checkpoint remains intact.
- 01:06: live verification confirms trainer PID `17993` remains active after `step=1960`; checkpoint files remain unchanged pending `step=2000`, with no new error or termination output.
- 01:08: run advanced to `epoch=18, step=1980`, loss `0.913512`; trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.
- 01:10: live verification confirms trainer PID `17993` remains active after `step=1980`; no new error or termination output is present, GPU memory is about `40.8 GiB`, and the run is still approaching the step-2000 save point.
- 01:12: epoch 18 completed at `step=1995`; run entered epoch 19. Trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no errors are present, and checkpoint saving is expected at `step=2000`.
- 01:14: run reached `epoch=19, step=2000`, loss `0.846224`. Sample `samples/step-00002000.pt` and checkpoint `checkpoints/step-00002000.pt` were fully written; checkpoint save took `32.05s`. Trainer PID `17993` remains active with no fatal errors.
- 01:16: post-checkpoint verification confirms trainer PID `17993` remains active; `step-00002000.pt` is fully present at `21,885,166,392` bytes, GPU memory is about `41.2 GiB`, and no new errors are present.
- 01:18: live verification confirms trainer PID `17993` remains active after the step-2000 checkpoint; both step-2000 checkpoint and sample are intact, with no new error or termination output.
- 01:20: post-checkpoint training resumed at `epoch=19, step=2020`, loss `0.874565`; trainer PID `17993` remains active with GPU memory about `41.2 GiB`, and no new errors were detected.
- 01:22: live verification confirms trainer PID `17993` remains active after `step=2020`; both step-2000 artifacts are intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 01:24: run advanced to `epoch=19, step=2040`, loss `0.840488`; trainer PID `17993` remains active with GPU memory about `41.2 GiB`, no new errors, and both step-2000 artifacts remain intact.
- 01:26: live verification confirms trainer PID `17993` remains active after `step=2040`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 01:28: run remains active at `epoch=19, step=2040` after the latest normal output; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 01:30: run advanced to `epoch=19, step=2060`, loss `0.799289`; trainer PID `17993` remains active with GPU memory about `41.2 GiB`, no new errors, and both step-2000 artifacts remain intact.
- 01:32: live verification confirms trainer PID `17993` remains active after `step=2060`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 01:34: live verification confirms trainer PID `17993` remains active after `step=2060`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 01:36: run advanced to `epoch=19, step=2080`, loss `0.829721`; trainer PID `17993` remains active with GPU memory about `41.2 GiB`, no new errors, and both step-2000 artifacts remain intact.
- 01:38: live verification confirms trainer PID `17993` remains active after `step=2080`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 01:40: epoch 19 completed at `step=2100`; run entered epoch 20. Latest loss was `0.873689` at step 2100, trainer PID `17993` remains active, GPU memory is about `41.2 GiB`, and no new errors were detected.
- 01:42: live verification confirms trainer PID `17993` remains active at the start of epoch 20 (`step=2100`); both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 01:44: live verification confirms trainer PID `17993` remains active at the start of epoch 20 (`step=2100`); both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 01:46: run advanced to `epoch=20, step=2120`, loss `0.855594`; trainer PID `17993` remains active with GPU memory about `41.2 GiB`, no new errors, and both step-2000 artifacts remain intact.
- 01:48: live verification confirms trainer PID `17993` remains active after `step=2120`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 01:50: live verification confirms trainer PID `17993` remains active in epoch 20 after `step=2120`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 01:52: run advanced to `epoch=20, step=2140`, loss `0.871636`; trainer PID `17993` remains active with GPU memory about `41.2 GiB`, no new errors, and both step-2000 artifacts remain intact.
- 01:54: live verification confirms trainer PID `17993` remains active after `step=2140`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 01:56: run advanced to `epoch=20, step=2160`, loss `0.834441`; trainer PID `17993` remains active with GPU memory about `41.2 GiB`, no new errors, and both step-2000 artifacts remain intact.
- 01:58: live verification confirms trainer PID `17993` remains active after `step=2160`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 02:00: live verification confirms trainer PID `17993` remains active in epoch 20 after `step=2160`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 02:02: epoch 20 completed at `step=2205`; latest logged loss was `0.891673` at step 2200, and the run entered epoch 21. Trainer PID `17993` remains active with GPU memory about `41.2 GiB`, no new errors, and both step-2000 artifacts remain intact.
- 02:04: live verification confirms trainer PID `17993` remains active at the start of epoch 21 (`step=2205`); both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 02:06: run advanced to `epoch=21, step=2220`, loss `0.862773`; trainer PID `17993` remains active with GPU memory about `41.2 GiB`, no new errors, and both step-2000 artifacts remain intact.
- 02:08: live verification confirms trainer PID `17993` remains active after `step=2220`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 02:10: run advanced to `epoch=21, step=2240`, loss `0.848141`; trainer PID `17993` remains active with GPU memory about `41.2 GiB`, no new errors, and both step-2000 artifacts remain intact.
- 02:12: live verification confirms trainer PID `17993` remains active after `step=2240`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 02:14: run advanced to `epoch=21, step=2260`, loss `0.970288`; trainer PID `17993` remains active with GPU memory about `41.2 GiB`, no new errors, and both step-2000 artifacts remain intact.
- 02:16: live verification confirms trainer PID `17993` remains active after `step=2260`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 02:18: live verification confirms trainer PID `17993` remains active after `step=2260`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 02:20: run advanced to `epoch=21, step=2280`, loss `0.836184`; trainer PID `17993` remains active with GPU memory about `41.2 GiB`, no new errors, and both step-2000 artifacts remain intact.
- 02:22: live verification confirms trainer PID `17993` remains active after `step=2280`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 02:24: live verification confirms trainer PID `17993` remains active after `step=2280`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 02:26: live verification confirms trainer PID `17993` remains active after `step=2280`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 02:28: run advanced to `epoch=21, step=2300`, loss `0.806491`; trainer PID `17993` remains active with GPU memory about `41.2 GiB`, no new errors, and both step-2000 artifacts remain intact.
- 02:30: epoch 21 completed at `step=2310`; latest logged loss was `0.806491` at step 2300, and the run entered epoch 22. Trainer PID `17993` remains active with GPU memory about `41.2 GiB`, no new errors, and both step-2000 artifacts remain intact.
- 02:32: run advanced to `epoch=22, step=2320`, loss `0.828512`; trainer PID `17993` remains active with GPU memory about `41.2 GiB`, no new errors, and both step-2000 artifacts remain intact.
- 02:34: live verification confirms trainer PID `17993` remains active after `step=2320`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 02:36: live verification confirms trainer PID `17993` remains active after `step=2320`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 02:38: run advanced to `epoch=22, step=2340`, loss `0.847517`; trainer PID `17993` remains active with GPU memory about `41.2 GiB`, no new errors, and both step-2000 artifacts remain intact.
- 02:40: live verification confirms trainer PID `17993` remains active after `step=2340`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 02:42: live verification confirms trainer PID `17993` remains active after `step=2340`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 02:44: live verification confirms trainer PID `17993` remains active after `step=2340`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 02:46: run advanced to `epoch=22, step=2360`, loss `0.837475`; trainer PID `17993` remains active with GPU memory about `41.2 GiB`, no new errors, no temporary checkpoint files, and both step-2000 artifacts remain intact.
- 02:48: live verification confirms trainer PID `17993` remains active after `step=2360`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 02:50: live verification confirms trainer PID `17993` remains active after `step=2360`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 02:52: run advanced to `epoch=22, step=2380`, loss `0.832505`; trainer PID `17993` remains active with GPU memory about `41.2 GiB`, no new errors, and both step-2000 artifacts remain intact.
- 02:54: live verification confirms trainer PID `17993` remains active after `step=2380`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 02:56: live verification confirms trainer PID `17993` remains active after `step=2380`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 02:58: run advanced to `epoch=22, step=2400`, loss `0.888245`; trainer PID `17993` remains active with GPU memory about `41.2 GiB`, no new errors, and both step-2000 artifacts remain intact.
- 03:00: live verification confirms trainer PID `17993` remains active after `step=2400`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 03:02: epoch 22 completed at `step=2415`; latest logged loss was `0.888245` at step 2400, and the run entered epoch 23. Trainer PID `17993` remains active; step 2420 loss is `0.878663`, GPU memory is about `41.2 GiB`, no new errors, and both step-2000 artifacts remain intact.
- 03:04: live verification confirms trainer PID `17993` remains active after `step=2420`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 03:06: live verification confirms trainer PID `17993` remains active after `step=2420`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 03:08: run advanced to `epoch=23, step=2440`, loss `0.774282`; trainer PID `17993` remains active with GPU memory about `41.2 GiB`, no new errors, and both step-2000 artifacts remain intact.
- 03:10: live verification confirms trainer PID `17993` remains active after `step=2440`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 03:12: live verification confirms trainer PID `17993` remains active after `step=2440`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 03:14: run advanced to `epoch=23, step=2460`, loss `0.887137`; trainer PID `17993` remains active with GPU memory about `41.2 GiB`, no new errors, and both step-2000 artifacts remain intact.
- 03:16: live verification confirms trainer PID `17993` remains active after `step=2460`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 03:18: live verification confirms trainer PID `17993` remains active after `step=2460`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 03:20: run advanced to `epoch=23, step=2480`, loss `0.820295`; trainer PID `17993` remains active with GPU memory about `41.2 GiB`, no new errors, and both step-2000 artifacts remain intact.
- 03:22: live verification confirms trainer PID `17993` remains active after `step=2480`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 03:24: live verification confirms trainer PID `17993` remains active after `step=2480`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 03:26: run advanced to `epoch=23, step=2500`, loss `0.847539`; trainer PID `17993` remains active with GPU memory about `41.2 GiB`, no new errors, and both step-2000 artifacts remain intact.
- 03:28: live verification confirms trainer PID `17993` remains active after `step=2500`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 03:30: live verification confirms trainer PID `17993` remains active after `step=2500`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 03:32: epoch 23 completed at `step=2520`; latest logged loss was `0.806051` at step 2520, and the run entered epoch 24. Trainer PID `17993` remains active with GPU memory about `41.2 GiB`, no new errors, and both step-2000 artifacts remain intact.
- 03:34: run advanced to `epoch=24, step=2540`, loss `0.896195`; trainer PID `17993` remains active with GPU memory about `41.2 GiB`, no new errors, and both step-2000 artifacts remain intact.
- 03:36: live verification confirms trainer PID `17993` remains active after `step=2540`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 03:38: run advanced to `epoch=24, step=2560`, loss `0.820588`; trainer PID `17993` remains active with GPU memory about `41.2 GiB`, no new errors, and both step-2000 artifacts remain intact.
- 03:40: live verification confirms trainer PID `17993` remains active after `step=2560`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 03:42: run advanced to `epoch=24, step=2600`, loss `0.781418`; trainer PID `17993` remains active with GPU memory about `41.2 GiB`, no new errors, and both step-2000 artifacts remain intact.
- 03:44: live verification confirms trainer PID `17993` remains active after `step=2600`; both step-2000 artifacts remain intact, GPU memory is `41,218 MiB`, and no new error or termination output is present.
- 03:46: run advanced to `epoch=25, step=2640`, loss `0.803613`; epoch 24 completed at `step=2625`, trainer PID `17993` remains active with GPU memory about `41.2 GiB`, and no new errors were detected.
- 03:48: run advanced to `epoch=25, step=2660`, loss `0.811533`; trainer PID `17993` remains active with GPU memory about `41.2 GiB`, no new errors, and both step-2000 artifacts remain intact.
- 03:50: live verification confirms trainer PID `17993` remains active after `step=2660`; GPU memory is `41,218 MiB`, no new error or termination output is present, and checkpoint saving remains scheduled for step 3000.
- 03:52: live verification confirms trainer PID `17993` remains active while continuing epoch 25 after `step=2660`; GPU memory remains `41,218 MiB`, no new error or termination output is present, and checkpoint saving remains scheduled for step 3000.
- 03:54: run advanced to `epoch=25, step=2680`, loss `0.899362`; trainer PID `17993` remains active with GPU memory `41,218 MiB`, no new errors, and checkpoint saving remains scheduled for step 3000.
- 03:56: live verification confirms trainer PID `17993` remains active after `step=2680`; GPU memory remains `41,218 MiB`, no new error or termination output is present, and the step-3000 checkpoint window has not yet been reached.
- 03:58: live verification confirms trainer PID `17993` remains active while continuing epoch 25 after `step=2680`; GPU memory remains `41,218 MiB`, no new error or termination output is present, and checkpoint saving remains scheduled for step 3000.
- 04:00: run advanced to `epoch=25, step=2700`, loss `0.814639`; profile output was refreshed, trainer PID `17993` remains active with GPU memory `41,218 MiB`, and no new errors were detected.
- 04:02: live verification confirms trainer PID `17993` remains active after `step=2700`; GPU memory remains `41,218 MiB`, no new error or termination output is present, and checkpoint saving remains scheduled for step 3000.
- 04:04: live verification confirms trainer PID `17993` remains active after `step=2700`; GPU memory remains `41,218 MiB`, no new error or termination output is present, and checkpoint saving remains scheduled for step 3000.
- 04:06: run advanced to `epoch=25, step=2720`, loss `0.791812`; trainer PID `17993` remains active with GPU memory `41,218 MiB`, no new errors, and checkpoint saving remains scheduled for step 3000.
- 04:08: epoch 25 completed at `step=2730`; run entered epoch 26. Trainer PID `17993` remains active with GPU memory `41,218 MiB`, no new errors, and checkpoint saving remains scheduled for step 3000.
- 04:10: live verification confirms trainer PID `17993` remains active in epoch 26 after `step=2730`; GPU memory remains `41,218 MiB`, no new error or termination output is present, and checkpoint saving remains scheduled for step 3000.
- 04:12: run advanced to `epoch=26, step=2740`, loss `0.815354`; trainer PID `17993` remains active with GPU memory `41,218 MiB`, no new errors, and checkpoint saving remains scheduled for step 3000.
- 04:14: live verification confirms trainer PID `17993` remains active after `step=2740`; GPU memory remains `41,218 MiB`, no new error or termination output is present, and checkpoint saving remains scheduled for step 3000.
- 04:16: run advanced to `epoch=26, step=2760`, loss `0.795253`; trainer PID `17993` remains active with GPU memory `41,218 MiB`, no new errors, and checkpoint saving remains scheduled for step 3000.
- 04:18: live verification confirms trainer PID `17993` remains active after `step=2760`; GPU memory remains `41,218 MiB`, no new error or termination output is present, and checkpoint saving remains scheduled for step 3000.
- 04:20: live verification confirms trainer PID `17993` remains active while continuing epoch 26 after `step=2760`; GPU memory remains `41,218 MiB`, no new error or termination output is present, and checkpoint saving remains scheduled for step 3000.
- 04:22: run advanced to `epoch=26, step=2780`, loss `0.823126`; trainer PID `17993` remains active with GPU memory `41,218 MiB`, no new errors, and checkpoint saving remains scheduled for step 3000.
- 04:24: live verification confirms trainer PID `17993` remains active after `step=2780`; GPU memory remains `41,218 MiB`, no new error or termination output is present, and checkpoint saving remains scheduled for step 3000.
- 04:26: live verification confirms trainer PID `17993` remains active after `step=2780`; GPU memory remains `41,218 MiB`, no new error or termination output is present, and checkpoint saving remains scheduled for step 3000.
- 04:28: run advanced to `epoch=26, step=2800`, loss `0.782847`; profile output was refreshed, trainer PID `17993` remains active with GPU memory `41,218 MiB`, and no new errors were detected.
- 04:30: live verification confirms trainer PID `17993` remains active after `step=2800`; GPU memory remains `41,218 MiB`, no new error or termination output is present, and checkpoint saving remains scheduled for step 3000.
- 04:32: run advanced to `epoch=26, step=2800`, loss `0.782847`; trainer PID `17993` remains active with GPU memory `41,218 MiB`, no new errors, and checkpoint saving remains scheduled for step 3000.
- 04:34: run advanced to `epoch=26, step=2820`, loss `0.773650`; trainer PID `17993` remains active with GPU memory `41,218 MiB`, no new errors, and checkpoint saving remains scheduled for step 3000.
- 04:36: live verification confirms trainer PID `17993` remains active after `step=2820`; GPU memory remains `41,218 MiB`, no new error or termination output is present, and checkpoint saving remains scheduled for step 3000.
- 04:38: epoch 26 completed at `step=2835`; run entered epoch 27. Trainer PID `17993` remains active with GPU memory `41,218 MiB`, no new errors, and checkpoint saving remains scheduled for step 3000.
- 04:40: run advanced to `epoch=27, step=2840`, loss `0.863690`; trainer PID `17993` remains active with GPU memory `41,218 MiB`, no new errors, and checkpoint saving remains scheduled for step 3000.
- 04:42: live verification confirms trainer PID `17993` remains active after `step=2840`; GPU memory remains `41,218 MiB`, no new error or termination output is present, and checkpoint saving remains scheduled for step 3000.
- 04:44: live verification confirms trainer PID `17993` remains active while continuing epoch 27 after `step=2840`; GPU memory remains `41,218 MiB`, no new error or termination output is present, and checkpoint saving remains scheduled for step 3000.
- 04:46: run advanced to `epoch=27, step=2860`, loss `0.858850`; trainer PID `17993` remains active with GPU memory `41,218 MiB`, no new errors, and checkpoint saving remains scheduled for step 3000.
- 2026-09-05 00:02: live verification confirms trainer PID `17993` remains active after `step=2860`; GPU memory remains `41,218 MiB`, no new error or termination output is present, and checkpoint saving remains scheduled for step 3000.
- 2026-09-05 00:04: run advanced to `epoch=27, step=2880`, loss `0.886059`; trainer PID `17993` remains active with GPU memory `41,218 MiB`, no new errors, and checkpoint saving remains scheduled for step 3000.
- 2026-09-05 00:06: live verification confirms trainer PID `17993` remains active after `step=2880`; GPU memory remains `41,218 MiB`, no new error or termination output is present, and checkpoint saving remains scheduled for step 3000.
- 02:26: live verification confirms trainer PID `17993` remains active after `step=2280`; both step-2000 artifacts remain intact, GPU memory is about `41.2 GiB`, and no new error or termination output is present.
- 00:16: run advanced to `epoch=16, step=1740`, loss `0.924109`; trainer PID `17993` remains active with GPU memory about `40.8 GiB`, no new errors, and the step-1000 checkpoint remains intact.

## Runtime output

[latent-stats] using existing file: /zsh/code/V-RAE/ckpts/libero_videogen/latent_stats/ci2v_shared_20f_multiview.pt
/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/timm/models/layers/__init__.py:49: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
[wandb] API key detected (WANDB_API_KEY); online logging enabled.
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from WANDB_API_KEY.
wandb: Currently logged in as: shuhangzheng20 (Aircraft) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: setting up run 69ee01d44ff64cb3ad43c560699b0228
wandb: Tracking run with wandb version 0.24.2
wandb: Run data is saved locally in /zsh/code/V-RAE/ckpts/libero_videogen/vjepa2_1_libero_ci2v_from_multiview_l/wandb/wandb/run-20260904_205538-69ee01d44ff64cb3ad43c560699b0228
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run vjepa2_1_libero_ci2v_from_multiview_l
wandb: ⭐️ View project at https://wandb.ai/Aircraft/V-RAE
wandb: 🚀 View run at https://wandb.ai/Aircraft/V-RAE/runs/69ee01d44ff64cb3ad43c560699b0228
[epoch-start] epoch=0 step=0 steps_per_epoch=105
/zsh/miniconda3/envs/vrae/lib/python3.10/contextlib.py:103: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the future, this context manager will be removed. Please see `torch.nn.attention.sdpa_kernel()` for the new context manager, with updated signature.
  self.gen = func(*args, **kwds)
/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/nvidia_cutlass_dsl/dsl_packages/cutlass/base_dsl/dsl.py:2309: UserWarning: Argument aux_data (position 17) cannot be converted to a JitArgument for function 'cutlass___call___flash_attncuteflash_fwd_sm90FlashAttentionForwardSm90_object_at__Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmem'. Its type `<class 'flash_attn.cute.utils.AuxData'>` is not supported. If the value is known at compile time, annotate it: `aux_data: Constexpr`. Otherwise, implement the `JitArgument` or `DynamicExpression` protocol for type `<class 'flash_attn.cute.utils.AuxData'>`, or register a custom argument adapter.
  exe_args, func_types, adapted_args = self.generate_mlir_function_types(
/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/nvidia_cutlass_dsl/dsl_packages/cutlass/base_dsl/dsl.py:2309: UserWarning: Argument aux_data (position 21) cannot be converted to a JitArgument for function 'cutlass___call___flash_attncuteflash_bwd_sm90FlashAttentionBackwardSm90_object_at__Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgme'. Its type `<class 'flash_attn.cute.utils.AuxData'>` is not supported. If the value is known at compile time, annotate it: `aux_data: Constexpr`. Otherwise, implement the `JitArgument` or `DynamicExpression` protocol for type `<class 'flash_attn.cute.utils.AuxData'>`, or register a custom argument adapter.
  exe_args, func_types, adapted_args = self.generate_mlir_function_types(
/zsh/miniconda3/envs/vrae/lib/python3.10/contextlib.py:103: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the future, this context manager will be removed. Please see `torch.nn.attention.sdpa_kernel()` for the new context manager, with updated signature.
  self.gen = func(*args, **kwds)
Traceback (most recent call last):
  File "/zsh/miniconda3/envs/vrae/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/zsh/miniconda3/envs/vrae/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/zsh/code/V-RAE/src/vrae/training/libero_videogen/train.py", line 147, in <module>
    main()
  File "/zsh/code/V-RAE/src/vrae/training/libero_videogen/train.py", line 137, in main
    train_class_conditional(
  File "/zsh/code/V-RAE/src/vrae/training/common/engine.py", line 1121, in train_class_conditional
    losses, loss = run_dit_forward(
  File "/zsh/code/V-RAE/src/vrae/training/common/profiler.py", line 95, in wrapped
    return function(*args, **kwargs)
  File "/zsh/code/V-RAE/src/vrae/training/common/engine.py", line 966, in run_dit_forward
    losses = transport.training_losses(
  File "/zsh/code/V-RAE/src/vrae/models/dit/transport.py", line 216, in training_losses
    output = model(noisy, time, **arguments)
  File "/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
    return forward_call(*args, **kwargs)
  File "/zsh/code/V-RAE/src/vrae/models/dit/video_dit.py", line 663, in forward
    full_tokens, base_tokens, base_sequence = self._forward_core(*prepared)
  File "/zsh/code/V-RAE/src/vrae/models/dit/video_dit.py", line 602, in _forward_core
    decoder_sequence = self._run_conditioned_block(
  File "/zsh/code/V-RAE/src/vrae/models/dit/video_dit.py", line 269, in _run_conditioned_block
    return block(
  File "/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
    return forward_call(*args, **kwargs)
  File "/zsh/code/V-RAE/src/vrae/models/dit/blocks.py", line 332, in forward
    modulate(self.norm1(value), shift_a, scale_a),
  File "/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
    return forward_call(*args, **kwargs)
  File "/zsh/code/V-RAE/src/vrae/models/dit/blocks.py", line 69, in forward
    value_float.square().mean(dim=-1, keepdim=True) + self.eps
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 96.00 MiB. GPU 0 has a total capacity of 79.18 GiB of which 41.94 MiB is free. Including non-PyTorch memory, this process has 79.13 GiB memory in use. Of the allocated memory 76.25 GiB is allocated by PyTorch, and 2.13 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
Traceback (most recent call last):
  File "/zsh/miniconda3/envs/vrae/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/zsh/miniconda3/envs/vrae/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/zsh/code/V-RAE/src/vrae/training/libero_videogen/train.py", line 147, in <module>
    main()
  File "/zsh/code/V-RAE/src/vrae/training/libero_videogen/train.py", line 137, in main
    train_class_conditional(
  File "/zsh/code/V-RAE/src/vrae/training/common/engine.py", line 1121, in train_class_conditional
    losses, loss = run_dit_forward(
  File "/zsh/code/V-RAE/src/vrae/training/common/profiler.py", line 95, in wrapped
    return function(*args, **kwargs)
  File "/zsh/code/V-RAE/src/vrae/training/common/engine.py", line 966, in run_dit_forward
    losses = transport.training_losses(
  File "/zsh/code/V-RAE/src/vrae/models/dit/transport.py", line 216, in training_losses
    output = model(noisy, time, **arguments)
  File "/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
    return forward_call(*args, **kwargs)
  File "/zsh/code/V-RAE/src/vrae/models/dit/video_dit.py", line 663, in forward
    full_tokens, base_tokens, base_sequence = self._forward_core(*prepared)
  File "/zsh/code/V-RAE/src/vrae/models/dit/video_dit.py", line 602, in _forward_core
    decoder_sequence = self._run_conditioned_block(
  File "/zsh/code/V-RAE/src/vrae/models/dit/video_dit.py", line 269, in _run_conditioned_block
    return block(
  File "/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
    return forward_call(*args, **kwargs)
  File "/zsh/code/V-RAE/src/vrae/models/dit/blocks.py", line 332, in forward
    modulate(self.norm1(value), shift_a, scale_a),
  File "/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
    return forward_call(*args, **kwargs)
  File "/zsh/code/V-RAE/src/vrae/models/dit/blocks.py", line 69, in forward
    value_float.square().mean(dim=-1, keepdim=True) + self.eps
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 96.00 MiB. GPU 0 has a total capacity of 79.18 GiB of which 41.94 MiB is free. Including non-PyTorch memory, this process has 79.13 GiB memory in use. Of the allocated memory 76.25 GiB is allocated by PyTorch, and 2.13 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
[1;34mwandb[0m: 
[1;34mwandb[0m: 🚀 View run [33mvjepa2_1_libero_ci2v_from_multiview_l[0m at: [34mhttps://wandb.ai/Aircraft/V-RAE/runs/69ee01d44ff64cb3ad43c560699b0228[0m
[1;34mwandb[0m: Find logs at: [1;35mckpts/libero_videogen/vjepa2_1_libero_ci2v_from_multiview_l/wandb/wandb/run-20260904_205538-69ee01d44ff64cb3ad43c560699b0228/logs[0m
E0904 20:56:16.612000 15765 site-packages/torch/distributed/elastic/multiprocessing/api.py:984] failed (exitcode: 1) local_rank: 0 (pid: 15893) of binary: /zsh/miniconda3/envs/vrae/bin/python3.10
Traceback (most recent call last):
  File "/zsh/miniconda3/envs/vrae/bin/torchrun", line 6, in <module>
    sys.exit(main())
  File "/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 362, in wrapper
    return f(*args, **kwargs)
  File "/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/torch/distributed/run.py", line 991, in main
    run(args)
  File "/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/torch/distributed/run.py", line 982, in run
    elastic_launch(
  File "/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 170, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 317, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
vrae.training.libero_videogen.train FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2026-09-04_20:56:16
  host      : job-a055724593dd-20260831045148-668f65bfc6-b56bd
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 15893)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
[latent-stats] using existing file: /zsh/code/V-RAE/ckpts/libero_videogen/latent_stats/ci2v_shared_20f_multiview.pt
/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/timm/models/layers/__init__.py:49: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
[wandb] API key detected (WANDB_API_KEY); online logging enabled.
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from WANDB_API_KEY.
wandb: Currently logged in as: shuhangzheng20 (Aircraft) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: setting up run 36f2b0b1c97146f8b0d92c0d57f12b86
wandb: Tracking run with wandb version 0.24.2
wandb: Run data is saved locally in /zsh/code/V-RAE/ckpts/libero_videogen/vjepa2_1_libero_ci2v_from_multiview_l_gc/wandb/wandb/run-20260904_205901-36f2b0b1c97146f8b0d92c0d57f12b86
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run vjepa2_1_libero_ci2v_from_multiview_l_gc
wandb: ⭐️ View project at https://wandb.ai/Aircraft/V-RAE
wandb: 🚀 View run at https://wandb.ai/Aircraft/V-RAE/runs/36f2b0b1c97146f8b0d92c0d57f12b86
[epoch-start] epoch=0 step=0 steps_per_epoch=105
/zsh/miniconda3/envs/vrae/lib/python3.10/contextlib.py:103: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the future, this context manager will be removed. Please see `torch.nn.attention.sdpa_kernel()` for the new context manager, with updated signature.
  self.gen = func(*args, **kwds)
/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/nvidia_cutlass_dsl/dsl_packages/cutlass/base_dsl/dsl.py:2309: UserWarning: Argument aux_data (position 17) cannot be converted to a JitArgument for function 'cutlass___call___flash_attncuteflash_fwd_sm90FlashAttentionForwardSm90_object_at__Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmem'. Its type `<class 'flash_attn.cute.utils.AuxData'>` is not supported. If the value is known at compile time, annotate it: `aux_data: Constexpr`. Otherwise, implement the `JitArgument` or `DynamicExpression` protocol for type `<class 'flash_attn.cute.utils.AuxData'>`, or register a custom argument adapter.
  exe_args, func_types, adapted_args = self.generate_mlir_function_types(
/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/nvidia_cutlass_dsl/dsl_packages/cutlass/base_dsl/dsl.py:2309: UserWarning: Argument aux_data (position 21) cannot be converted to a JitArgument for function 'cutlass___call___flash_attncuteflash_bwd_sm90FlashAttentionBackwardSm90_object_at__Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgme'. Its type `<class 'flash_attn.cute.utils.AuxData'>` is not supported. If the value is known at compile time, annotate it: `aux_data: Constexpr`. Otherwise, implement the `JitArgument` or `DynamicExpression` protocol for type `<class 'flash_attn.cute.utils.AuxData'>`, or register a custom argument adapter.
  exe_args, func_types, adapted_args = self.generate_mlir_function_types(
/zsh/miniconda3/envs/vrae/lib/python3.10/contextlib.py:103: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the future, this context manager will be removed. Please see `torch.nn.attention.sdpa_kernel()` for the new context manager, with updated signature.
  self.gen = func(*args, **kwds)
task=ci2v epoch=0 step=20 loss=2.062750 loss_full=1.031667 loss_base=1.031083 ms_per_step=4487.85
task=ci2v epoch=0 step=40 loss=2.211474 loss_full=1.106233 loss_base=1.105241 ms_per_step=3595.14
task=ci2v epoch=0 step=60 loss=2.230375 loss_full=1.116418 loss_base=1.113956 ms_per_step=3535.85
task=ci2v epoch=0 step=80 loss=2.216860 loss_full=1.107027 loss_base=1.109833 ms_per_step=3804.75
task=ci2v epoch=0 step=100 loss=2.775942 loss_full=1.372347 loss_base=1.403596 ms_per_step=3815.32
[profile] window=295.21s updates=80 data_wait=0.9ms h2d=1.4ms stage1=272.1ms dit=642.0ms update=27.8ms allocated=28.7GiB reserved=35.5GiB peak=34.1GiB
[epoch-end] epoch=0 elapsed=403.56s step=105
[epoch-start] epoch=1 step=105 steps_per_epoch=105
[2026-09-05 13:48] monitor: 新 run 完成 epoch=0（step=105，elapsed=396.92s），最后记录 loss=2.775942（loss_full=1.372347, loss_base=1.403596），已进入 epoch=1；GPU 利用率约 100%，显存约 37092 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=1 step=120 loss=2.287875 loss_full=1.132116 loss_base=1.155759 ms_per_step=3887.01
task=ci2v epoch=1 step=140 loss=2.120821 loss_full=1.042100 loss_base=1.078721 ms_per_step=3826.12
task=ci2v epoch=1 step=160 loss=2.050769 loss_full=1.005317 loss_base=1.045452 ms_per_step=3820.03
task=ci2v epoch=1 step=180 loss=2.543123 loss_full=1.220507 loss_base=1.322616 ms_per_step=3820.01
task=ci2v epoch=1 step=200 loss=1.966541 loss_full=0.983039 loss_base=0.983502 ms_per_step=3812.68
[profile] window=383.57s updates=100 data_wait=2.3ms h2d=1.7ms stage1=301.8ms dit=646.7ms update=27.8ms allocated=28.7GiB reserved=35.5GiB peak=34.1GiB
[epoch-end] epoch=1 elapsed=399.06s step=210
[epoch-start] epoch=2 step=210 steps_per_epoch=105
[2026-09-05 13:54] monitor: 新 run 完成 epoch=1（step=210，elapsed=380.81s），最后记录 loss=1.966539（loss_full=0.983038, loss_base=0.983502），已进入 epoch=2；trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=2 step=220 loss=2.134679 loss_full=1.036129 loss_base=1.098550 ms_per_step=3395.30
task=ci2v epoch=2 step=240 loss=1.998211 loss_full=0.984468 loss_base=1.013743 ms_per_step=3838.06
task=ci2v epoch=2 step=260 loss=1.943248 loss_full=0.942924 loss_base=1.000324 ms_per_step=3832.10
task=ci2v epoch=2 step=280 loss=1.976372 loss_full=0.947481 loss_base=1.028891 ms_per_step=3796.54
task=ci2v epoch=2 step=300 loss=1.888905 loss_full=0.908207 loss_base=0.980697 ms_per_step=3809.25
[profile] window=373.68s updates=100 data_wait=1.8ms h2d=1.6ms stage1=281.8ms dit=643.1ms update=27.7ms allocated=28.7GiB reserved=35.5GiB peak=34.1GiB
[epoch-end] epoch=2 elapsed=396.08s step=315
[epoch-start] epoch=3 step=315 steps_per_epoch=105
[2026-09-05 14:01] monitor: 新 run 完成 epoch=2（step=315，elapsed=391.99s），最后记录 loss=1.888921（loss_full=0.908219, loss_base=0.980702），已进入 epoch=3；trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=3 step=320 loss=1.880365 loss_full=0.898139 loss_base=0.982226 ms_per_step=3800.47
task=ci2v epoch=3 step=340 loss=1.884779 loss_full=0.893659 loss_base=0.991120 ms_per_step=3793.51
task=ci2v epoch=3 step=360 loss=1.728921 loss_full=0.799698 loss_base=0.929223 ms_per_step=3682.22
task=ci2v epoch=3 step=380 loss=1.842803 loss_full=0.814623 loss_base=1.028180 ms_per_step=3587.95
task=ci2v epoch=3 step=400 loss=1.562099 loss_full=0.695833 loss_base=0.866265 ms_per_step=3841.40
[profile] window=374.38s updates=100 data_wait=2.1ms h2d=1.6ms stage1=282.4ms dit=643.6ms update=27.7ms allocated=28.7GiB reserved=35.5GiB peak=34.1GiB
task=ci2v epoch=3 step=420 loss=1.536422 loss_full=0.668692 loss_base=0.867730 ms_per_step=3756.80
[epoch-end] epoch=3 elapsed=393.74s step=420
[epoch-start] epoch=4 step=420 steps_per_epoch=105
[2026-09-05 14:08] monitor: 新 run 完成 epoch=3（step=420，elapsed=386.35s），最后记录 loss=1.533401（loss_full=0.668495, loss_base=0.864906），已进入 epoch=4；trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=4 step=440 loss=1.476651 loss_full=0.658460 loss_base=0.818191 ms_per_step=3865.74
task=ci2v epoch=4 step=460 loss=1.287621 loss_full=0.568843 loss_base=0.718778 ms_per_step=3816.58
task=ci2v epoch=4 step=480 loss=1.417124 loss_full=0.637715 loss_base=0.779410 ms_per_step=3772.24
task=ci2v epoch=4 step=500 loss=1.228363 loss_full=0.542455 loss_base=0.685909 ms_per_step=3814.06
[profile] window=380.77s updates=100 data_wait=2.0ms h2d=1.7ms stage1=296.6ms dit=645.2ms update=27.7ms allocated=28.7GiB reserved=35.5GiB peak=34.1GiB
task=ci2v epoch=4 step=520 loss=1.358590 loss_full=0.614665 loss_base=0.743925 ms_per_step=3718.76
[epoch-end] epoch=4 elapsed=398.13s step=525
[epoch-start] epoch=5 step=525 steps_per_epoch=105
[2026-09-05 14:14] monitor: 新 run 完成 epoch=4（step=525，elapsed=378.06s），最后记录 loss=1.356135（loss_full=0.614144, loss_base=0.741991），已进入 epoch=5；trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=5 step=540 loss=1.391029 loss_full=0.609642 loss_base=0.781387 ms_per_step=3792.78
task=ci2v epoch=5 step=560 loss=1.291268 loss_full=0.591501 loss_base=0.699767 ms_per_step=3609.30
task=ci2v epoch=5 step=580 loss=1.267787 loss_full=0.590135 loss_base=0.677651 ms_per_step=3575.17
task=ci2v epoch=5 step=600 loss=1.315734 loss_full=0.608332 loss_base=0.707403 ms_per_step=3780.22
[profile] window=369.78s updates=100 data_wait=2.3ms h2d=1.5ms stage1=272.2ms dit=641.2ms update=31.3ms allocated=28.7GiB reserved=35.5GiB peak=34.1GiB
task=ci2v epoch=5 step=620 loss=1.168958 loss_full=0.551045 loss_base=0.617913 ms_per_step=3805.87
[epoch-end] epoch=5 elapsed=390.34s step=630
[epoch-start] epoch=6 step=630 steps_per_epoch=105
[2026-09-05 14:20] monitor: 新 run 完成 epoch=5（step=630，elapsed=376.64s），最后记录 loss=1.166715（loss_full=0.550461, loss_base=0.616254），已进入 epoch=6；trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=6 step=640 loss=1.232628 loss_full=0.575961 loss_base=0.656667 ms_per_step=3831.61
task=ci2v epoch=6 step=660 loss=1.201708 loss_full=0.564352 loss_base=0.637356 ms_per_step=3801.86
task=ci2v epoch=6 step=680 loss=1.175758 loss_full=0.553675 loss_base=0.622082 ms_per_step=3823.55
task=ci2v epoch=6 step=700 loss=1.103614 loss_full=0.519734 loss_base=0.583880 ms_per_step=3789.52
[profile] window=381.30s updates=100 data_wait=1.9ms h2d=1.7ms stage1=294.6ms dit=647.9ms update=30.3ms allocated=28.7GiB reserved=35.5GiB peak=34.1GiB
task=ci2v epoch=6 step=720 loss=1.241891 loss_full=0.587631 loss_base=0.654260 ms_per_step=3595.85
[epoch-end] epoch=6 elapsed=396.42s step=735
[epoch-start] epoch=7 step=735 steps_per_epoch=105
[2026-09-05 14:26] monitor: 新 run 完成 epoch=6（step=735，elapsed=375.73s），最后记录 loss=1.241845（loss_full=0.588060, loss_base=0.653785），已进入 epoch=7；首个 step=740 loss=1.146924；trainer PID=34953 存活，错误扫描无命中。
task=ci2v epoch=7 step=740 loss=1.146844 loss_full=0.539757 loss_base=0.607087 ms_per_step=3804.36
task=ci2v epoch=7 step=760 loss=1.207947 loss_full=0.576538 loss_base=0.631410 ms_per_step=3616.84
task=ci2v epoch=7 step=780 loss=1.068355 loss_full=0.514196 loss_base=0.554159 ms_per_step=3429.47
task=ci2v epoch=7 step=800 loss=1.145005 loss_full=0.539968 loss_base=0.605037 ms_per_step=3491.89
[profile] window=359.03s updates=100 data_wait=2.0ms h2d=1.3ms stage1=247.7ms dit=640.1ms update=27.7ms allocated=28.7GiB reserved=35.5GiB peak=34.1GiB
task=ci2v epoch=7 step=820 loss=1.043442 loss_full=0.501865 loss_base=0.541578 ms_per_step=3703.37
task=ci2v epoch=7 step=840 loss=1.196561 loss_full=0.565403 loss_base=0.631159 ms_per_step=3532.23
[epoch-end] epoch=7 elapsed=376.06s step=840
[epoch-start] epoch=8 step=840 steps_per_epoch=105
[2026-09-05 14:33] monitor: 新 run 完成 epoch=7（step=840，elapsed=380.08s），最后记录 loss=1.199482（loss_full=0.566522, loss_base=0.632960），已进入 epoch=8；trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=8 step=860 loss=1.075567 loss_full=0.516010 loss_base=0.559557 ms_per_step=3857.21
task=ci2v epoch=8 step=880 loss=1.109346 loss_full=0.536817 loss_base=0.572529 ms_per_step=3556.80
task=ci2v epoch=8 step=900 loss=1.030423 loss_full=0.494584 loss_base=0.535840 ms_per_step=3652.66
[profile] window=366.32s updates=100 data_wait=2.2ms h2d=1.4ms stage1=264.4ms dit=641.4ms update=27.7ms allocated=28.7GiB reserved=35.5GiB peak=34.1GiB
task=ci2v epoch=8 step=920 loss=1.055093 loss_full=0.502264 loss_base=0.552829 ms_per_step=3698.92
task=ci2v epoch=8 step=940 loss=1.053696 loss_full=0.503891 loss_base=0.549805 ms_per_step=3870.33
[epoch-end] epoch=8 elapsed=391.59s step=945
[epoch-start] epoch=9 step=945 steps_per_epoch=105
task=ci2v epoch=9 step=960 loss=0.999212 loss_full=0.479539 loss_base=0.519673 ms_per_step=3770.14
task=ci2v epoch=9 step=980 loss=1.089883 loss_full=0.527897 loss_base=0.561986 ms_per_step=3782.48
task=ci2v epoch=9 step=1000 loss=1.038965 loss_full=0.498664 loss_base=0.540301 ms_per_step=3820.76
[profile] window=379.11s updates=100 data_wait=2.4ms h2d=1.6ms stage1=287.3ms dit=650.5ms update=27.7ms allocated=28.7GiB reserved=35.5GiB peak=34.1GiB
/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/nvidia_cutlass_dsl/dsl_packages/cutlass/base_dsl/dsl.py:2309: UserWarning: Argument aux_data (position 17) cannot be converted to a JitArgument for function 'cutlass___call___flash_attncuteflash_fwd_sm90FlashAttentionForwardSm90_object_at__Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_None_0125_'. Its type `<class 'flash_attn.cute.utils.AuxData'>` is not supported. If the value is known at compile time, annotate it: `aux_data: Constexpr`. Otherwise, implement the `JitArgument` or `DynamicExpression` protocol for type `<class 'flash_attn.cute.utils.AuxData'>`, or register a custom argument adapter.
  exe_args, func_types, adapted_args = self.generate_mlir_function_types(
/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/nvidia_cutlass_dsl/dsl_packages/cutlass/base_dsl/dsl.py:2309: UserWarning: Argument aux_data (position 17) cannot be converted to a JitArgument for function 'cutlass___call___flash_attncuteflash_fwd_sm90FlashAttentionForwardSm90_object_at__Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_None_00883'. Its type `<class 'flash_attn.cute.utils.AuxData'>` is not supported. If the value is known at compile time, annotate it: `aux_data: Constexpr`. Otherwise, implement the `JitArgument` or `DynamicExpression` protocol for type `<class 'flash_attn.cute.utils.AuxData'>`, or register a custom argument adapter.
  exe_args, func_types, adapted_args = self.generate_mlir_function_types(
/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/nvidia_cutlass_dsl/dsl_packages/cutlass/base_dsl/dsl.py:2309: UserWarning: Argument aux_data (position 17) cannot be converted to a JitArgument for function 'cutlass___call___flash_attncuteflash_fwd_sm90FlashAttentionForwardSm90_object_at__Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_None_01178'. Its type `<class 'flash_attn.cute.utils.AuxData'>` is not supported. If the value is known at compile time, annotate it: `aux_data: Constexpr`. Otherwise, implement the `JitArgument` or `DynamicExpression` protocol for type `<class 'flash_attn.cute.utils.AuxData'>`, or register a custom argument adapter.
  exe_args, func_types, adapted_args = self.generate_mlir_function_types(
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=1000 elapsed=19.04s
[checkpoint] step=1000 elapsed=36.16s
/zsh/miniconda3/envs/vrae/lib/python3.10/contextlib.py:103: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the future, this context manager will be removed. Please see `torch.nn.attention.sdpa_kernel()` for the new context manager, with updated signature.
  self.gen = func(*args, **kwds)
task=ci2v epoch=9 step=1020 loss=0.978945 loss_full=0.467685 loss_base=0.511260 ms_per_step=3824.63
task=ci2v epoch=9 step=1040 loss=1.072561 loss_full=0.499292 loss_base=0.573268 ms_per_step=3693.41
[epoch-end] epoch=9 elapsed=450.04s step=1050
[epoch-start] epoch=10 step=1050 steps_per_epoch=105
task=ci2v epoch=10 step=1060 loss=1.000047 loss_full=0.475781 loss_base=0.524266 ms_per_step=3707.17
task=ci2v epoch=10 step=1080 loss=1.086707 loss_full=0.523935 loss_base=0.562773 ms_per_step=3808.05
task=ci2v epoch=10 step=1100 loss=1.098396 loss_full=0.521603 loss_base=0.576793 ms_per_step=3816.86
[profile] window=377.25s updates=100 data_wait=2.1ms h2d=1.6ms stage1=287.8ms dit=645.4ms update=27.7ms allocated=29.0GiB reserved=39.1GiB peak=37.9GiB
task=ci2v epoch=10 step=1120 loss=0.986295 loss_full=0.474556 loss_base=0.511739 ms_per_step=3827.40
task=ci2v epoch=10 step=1140 loss=0.984865 loss_full=0.474182 loss_base=0.510683 ms_per_step=3809.02
[epoch-end] epoch=10 elapsed=399.00s step=1155
[epoch-start] epoch=11 step=1155 steps_per_epoch=105
task=ci2v epoch=11 step=1160 loss=0.960983 loss_full=0.460638 loss_base=0.500345 ms_per_step=3724.06
task=ci2v epoch=11 step=1180 loss=1.002442 loss_full=0.476445 loss_base=0.525997 ms_per_step=3811.37
task=ci2v epoch=11 step=1200 loss=1.040824 loss_full=0.494052 loss_base=0.546772 ms_per_step=3800.27
[profile] window=379.80s updates=100 data_wait=2.2ms h2d=1.7ms stage1=293.0ms dit=646.3ms update=27.7ms allocated=29.0GiB reserved=39.1GiB peak=37.9GiB
task=ci2v epoch=11 step=1220 loss=0.950405 loss_full=0.458686 loss_base=0.491719 ms_per_step=3670.93
task=ci2v epoch=11 step=1240 loss=0.909270 loss_full=0.437875 loss_base=0.471395 ms_per_step=3823.33
task=ci2v epoch=11 step=1260 loss=0.945821 loss_full=0.449191 loss_base=0.496630 ms_per_step=3759.85
[epoch-end] epoch=11 elapsed=397.92s step=1260
[epoch-start] epoch=12 step=1260 steps_per_epoch=105
task=ci2v epoch=12 step=1280 loss=0.995037 loss_full=0.477448 loss_base=0.517588 ms_per_step=3791.01
task=ci2v epoch=12 step=1300 loss=0.958845 loss_full=0.455937 loss_base=0.502909 ms_per_step=3834.61
[profile] window=377.85s updates=100 data_wait=2.1ms h2d=1.6ms stage1=286.8ms dit=647.8ms update=27.8ms allocated=29.0GiB reserved=39.1GiB peak=37.9GiB
task=ci2v epoch=12 step=1320 loss=0.976904 loss_full=0.466787 loss_base=0.510117 ms_per_step=3803.33
task=ci2v epoch=12 step=1340 loss=0.957906 loss_full=0.455080 loss_base=0.502826 ms_per_step=3783.69
task=ci2v epoch=12 step=1360 loss=1.025861 loss_full=0.489895 loss_base=0.535966 ms_per_step=3801.02
[epoch-end] epoch=12 elapsed=398.90s step=1365
[epoch-start] epoch=13 step=1365 steps_per_epoch=105
task=ci2v epoch=13 step=1380 loss=0.912579 loss_full=0.436970 loss_base=0.475610 ms_per_step=3838.73
task=ci2v epoch=13 step=1400 loss=0.923682 loss_full=0.442242 loss_base=0.481440 ms_per_step=3566.60
[profile] window=376.12s updates=100 data_wait=2.4ms h2d=1.6ms stage1=285.4ms dit=644.7ms update=27.7ms allocated=29.0GiB reserved=39.1GiB peak=37.9GiB
task=ci2v epoch=13 step=1420 loss=0.945586 loss_full=0.458727 loss_base=0.486859 ms_per_step=3793.64
task=ci2v epoch=13 step=1440 loss=0.992165 loss_full=0.470080 loss_base=0.522085 ms_per_step=3642.57
task=ci2v epoch=13 step=1460 loss=0.914778 loss_full=0.442940 loss_base=0.471838 ms_per_step=3692.48
[epoch-end] epoch=13 elapsed=390.41s step=1470
[epoch-start] epoch=14 step=1470 steps_per_epoch=105
task=ci2v epoch=14 step=1480 loss=0.937636 loss_full=0.431400 loss_base=0.506235 ms_per_step=3740.52
task=ci2v epoch=14 step=1500 loss=0.971276 loss_full=0.461801 loss_base=0.509475 ms_per_step=3476.42
[profile] window=367.17s updates=100 data_wait=2.1ms h2d=1.5ms stage1=267.1ms dit=640.8ms update=27.7ms allocated=29.0GiB reserved=39.1GiB peak=37.9GiB
task=ci2v epoch=14 step=1520 loss=0.925605 loss_full=0.440908 loss_base=0.484697 ms_per_step=3712.39
task=ci2v epoch=14 step=1540 loss=0.919933 loss_full=0.438304 loss_base=0.481629 ms_per_step=3795.35
task=ci2v epoch=14 step=1560 loss=0.955115 loss_full=0.452319 loss_base=0.502795 ms_per_step=3791.31
[epoch-end] epoch=14 elapsed=385.57s step=1575
[epoch-start] epoch=15 step=1575 steps_per_epoch=105
task=ci2v epoch=15 step=1580 loss=0.924605 loss_full=0.422113 loss_base=0.502492 ms_per_step=3560.51
task=ci2v epoch=15 step=1600 loss=0.913364 loss_full=0.436284 loss_base=0.477080 ms_per_step=3787.93
[profile] window=373.21s updates=100 data_wait=1.9ms h2d=1.5ms stage1=280.0ms dit=643.7ms update=27.7ms allocated=29.0GiB reserved=39.1GiB peak=37.9GiB
task=ci2v epoch=15 step=1620 loss=0.935380 loss_full=0.446039 loss_base=0.489341 ms_per_step=3794.84
task=ci2v epoch=15 step=1640 loss=0.900972 loss_full=0.427006 loss_base=0.473966 ms_per_step=3568.61
task=ci2v epoch=15 step=1660 loss=0.868003 loss_full=0.417089 loss_base=0.450913 ms_per_step=3805.59
task=ci2v epoch=15 step=1680 loss=0.913431 loss_full=0.433357 loss_base=0.480075 ms_per_step=3763.47
[epoch-end] epoch=15 elapsed=393.52s step=1680
[epoch-start] epoch=16 step=1680 steps_per_epoch=105
task=ci2v epoch=16 step=1700 loss=0.961621 loss_full=0.461193 loss_base=0.500428 ms_per_step=3837.26
[profile] window=375.65s updates=100 data_wait=2.0ms h2d=1.6ms stage1=285.1ms dit=643.9ms update=27.7ms allocated=29.0GiB reserved=39.1GiB peak=37.9GiB
task=ci2v epoch=16 step=1720 loss=0.835395 loss_full=0.393743 loss_base=0.441652 ms_per_step=3795.76
task=ci2v epoch=16 step=1740 loss=0.924109 loss_full=0.441447 loss_base=0.482662 ms_per_step=3792.91
task=ci2v epoch=16 step=1760 loss=0.952822 loss_full=0.455834 loss_base=0.496988 ms_per_step=3767.95
task=ci2v epoch=16 step=1780 loss=0.904110 loss_full=0.422567 loss_base=0.481543 ms_per_step=3790.19
[epoch-end] epoch=16 elapsed=398.22s step=1785
[epoch-start] epoch=17 step=1785 steps_per_epoch=105
task=ci2v epoch=17 step=1800 loss=0.947476 loss_full=0.447112 loss_base=0.500363 ms_per_step=3802.76
[profile] window=379.25s updates=100 data_wait=2.3ms h2d=1.7ms stage1=294.2ms dit=643.0ms update=30.7ms allocated=29.0GiB reserved=39.1GiB peak=37.9GiB
task=ci2v epoch=17 step=1820 loss=0.866943 loss_full=0.405023 loss_base=0.461920 ms_per_step=3788.27
task=ci2v epoch=17 step=1840 loss=0.855312 loss_full=0.403373 loss_base=0.451939 ms_per_step=3792.04
task=ci2v epoch=17 step=1860 loss=0.849310 loss_full=0.410711 loss_base=0.438599 ms_per_step=3801.24
task=ci2v epoch=17 step=1880 loss=0.835727 loss_full=0.381232 loss_base=0.454495 ms_per_step=3786.06
[epoch-end] epoch=17 elapsed=398.54s step=1890
[epoch-start] epoch=18 step=1890 steps_per_epoch=105
task=ci2v epoch=18 step=1900 loss=0.968320 loss_full=0.449160 loss_base=0.519160 ms_per_step=3811.44
[profile] window=379.83s updates=100 data_wait=2.0ms h2d=1.7ms stage1=293.8ms dit=645.8ms update=27.7ms allocated=29.0GiB reserved=39.1GiB peak=37.9GiB
task=ci2v epoch=18 step=1920 loss=0.897143 loss_full=0.424833 loss_base=0.472310 ms_per_step=3795.75
task=ci2v epoch=18 step=1940 loss=0.914479 loss_full=0.434927 loss_base=0.479552 ms_per_step=3787.46
task=ci2v epoch=18 step=1960 loss=0.872961 loss_full=0.422271 loss_base=0.450690 ms_per_step=3499.21
task=ci2v epoch=18 step=1980 loss=0.913512 loss_full=0.428210 loss_base=0.485302 ms_per_step=3545.69
[epoch-end] epoch=18 elapsed=388.29s step=1995
[epoch-start] epoch=19 step=1995 steps_per_epoch=105
task=ci2v epoch=19 step=2000 loss=0.846224 loss_full=0.392275 loss_base=0.453950 ms_per_step=3795.18
[profile] window=368.73s updates=100 data_wait=2.0ms h2d=1.5ms stage1=271.2ms dit=640.8ms update=27.8ms allocated=29.0GiB reserved=39.1GiB peak=37.9GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=2000 elapsed=2.54s
[checkpoint] step=2000 elapsed=32.05s
task=ci2v epoch=19 step=2020 loss=0.874565 loss_full=0.416703 loss_base=0.457863 ms_per_step=3451.74
task=ci2v epoch=19 step=2040 loss=0.840488 loss_full=0.392418 loss_base=0.448070 ms_per_step=3785.86
task=ci2v epoch=19 step=2060 loss=0.799289 loss_full=0.378731 loss_base=0.420558 ms_per_step=3785.44
task=ci2v epoch=19 step=2080 loss=0.829721 loss_full=0.386521 loss_base=0.443200 ms_per_step=3756.48
task=ci2v epoch=19 step=2100 loss=0.873689 loss_full=0.416898 loss_base=0.456791 ms_per_step=3740.00
[profile] window=370.65s updates=100 data_wait=1.0ms h2d=1.5ms stage1=274.8ms dit=643.1ms update=27.7ms allocated=29.0GiB reserved=39.5GiB peak=38.0GiB
[epoch-end] epoch=19 elapsed=425.28s step=2100
[epoch-start] epoch=20 step=2100 steps_per_epoch=105
task=ci2v epoch=20 step=2120 loss=0.855594 loss_full=0.396508 loss_base=0.459087 ms_per_step=3855.53
task=ci2v epoch=20 step=2140 loss=0.871636 loss_full=0.412664 loss_base=0.458973 ms_per_step=3802.06
task=ci2v epoch=20 step=2160 loss=0.834441 loss_full=0.397208 loss_base=0.437234 ms_per_step=3581.38
task=ci2v epoch=20 step=2180 loss=0.833705 loss_full=0.383416 loss_base=0.450289 ms_per_step=3796.11
task=ci2v epoch=20 step=2200 loss=0.891673 loss_full=0.428415 loss_base=0.463258 ms_per_step=3788.92
[profile] window=376.74s updates=100 data_wait=2.4ms h2d=1.6ms stage1=287.5ms dit=644.0ms update=27.7ms allocated=29.0GiB reserved=39.5GiB peak=38.0GiB
[epoch-end] epoch=20 elapsed=394.92s step=2205
[epoch-start] epoch=21 step=2205 steps_per_epoch=105
task=ci2v epoch=21 step=2220 loss=0.862773 loss_full=0.411088 loss_base=0.451686 ms_per_step=3796.49
task=ci2v epoch=21 step=2240 loss=0.848141 loss_full=0.392068 loss_base=0.456073 ms_per_step=3787.02
task=ci2v epoch=21 step=2260 loss=0.970288 loss_full=0.466594 loss_base=0.503694 ms_per_step=3828.97
task=ci2v epoch=21 step=2280 loss=0.836184 loss_full=0.386073 loss_base=0.450111 ms_per_step=3808.49
task=ci2v epoch=21 step=2300 loss=0.806491 loss_full=0.371178 loss_base=0.435313 ms_per_step=3795.15
[profile] window=380.58s updates=100 data_wait=2.0ms h2d=1.7ms stage1=296.2ms dit=645.1ms update=27.7ms allocated=29.0GiB reserved=39.5GiB peak=38.0GiB
[epoch-end] epoch=21 elapsed=399.55s step=2310
[epoch-start] epoch=22 step=2310 steps_per_epoch=105
task=ci2v epoch=22 step=2320 loss=0.828512 loss_full=0.387036 loss_base=0.441475 ms_per_step=3758.30
task=ci2v epoch=22 step=2340 loss=0.847517 loss_full=0.395905 loss_base=0.451611 ms_per_step=3652.99
task=ci2v epoch=22 step=2360 loss=0.837475 loss_full=0.380000 loss_base=0.457476 ms_per_step=3787.10
task=ci2v epoch=22 step=2380 loss=0.832505 loss_full=0.391406 loss_base=0.441099 ms_per_step=3806.78
task=ci2v epoch=22 step=2400 loss=0.888245 loss_full=0.419638 loss_base=0.468607 ms_per_step=3791.97
[profile] window=376.20s updates=100 data_wait=2.1ms h2d=1.6ms stage1=287.0ms dit=643.5ms update=27.7ms allocated=29.0GiB reserved=39.5GiB peak=38.0GiB
[epoch-end] epoch=22 elapsed=395.28s step=2415
[epoch-start] epoch=23 step=2415 steps_per_epoch=105
task=ci2v epoch=23 step=2420 loss=0.878663 loss_full=0.409218 loss_base=0.469445 ms_per_step=3707.73
task=ci2v epoch=23 step=2440 loss=0.774282 loss_full=0.367906 loss_base=0.406376 ms_per_step=3694.33
task=ci2v epoch=23 step=2460 loss=0.887137 loss_full=0.424115 loss_base=0.463022 ms_per_step=3778.74
task=ci2v epoch=23 step=2480 loss=0.820295 loss_full=0.396000 loss_base=0.424295 ms_per_step=3771.65
task=ci2v epoch=23 step=2500 loss=0.847539 loss_full=0.404393 loss_base=0.443146 ms_per_step=3789.81
[profile] window=375.10s updates=100 data_wait=2.2ms h2d=1.6ms stage1=284.2ms dit=643.6ms update=27.7ms allocated=29.0GiB reserved=39.5GiB peak=38.0GiB
task=ci2v epoch=23 step=2520 loss=0.806051 loss_full=0.379205 loss_base=0.426846 ms_per_step=3758.81
[epoch-end] epoch=23 elapsed=394.65s step=2520
[epoch-start] epoch=24 step=2520 steps_per_epoch=105
task=ci2v epoch=24 step=2540 loss=0.896195 loss_full=0.426387 loss_base=0.469808 ms_per_step=3848.33
task=ci2v epoch=24 step=2560 loss=0.820588 loss_full=0.393229 loss_base=0.427359 ms_per_step=3876.97
task=ci2v epoch=24 step=2580 loss=0.823641 loss_full=0.385151 loss_base=0.438491 ms_per_step=3789.54
task=ci2v epoch=24 step=2600 loss=0.781418 loss_full=0.365348 loss_base=0.416070 ms_per_step=3766.54
[profile] window=381.06s updates=100 data_wait=2.1ms h2d=1.7ms stage1=295.8ms dit=646.6ms update=27.7ms allocated=29.0GiB reserved=39.5GiB peak=38.0GiB
task=ci2v epoch=24 step=2620 loss=0.807037 loss_full=0.376486 loss_base=0.430550 ms_per_step=3661.22
[epoch-end] epoch=24 elapsed=397.32s step=2625
[epoch-start] epoch=25 step=2625 steps_per_epoch=105
task=ci2v epoch=25 step=2640 loss=0.803613 loss_full=0.380867 loss_base=0.422746 ms_per_step=3811.18
task=ci2v epoch=25 step=2660 loss=0.811533 loss_full=0.386246 loss_base=0.425287 ms_per_step=3781.77
task=ci2v epoch=25 step=2680 loss=0.899362 loss_full=0.422849 loss_base=0.476514 ms_per_step=3791.09
task=ci2v epoch=25 step=2700 loss=0.814639 loss_full=0.382083 loss_base=0.432556 ms_per_step=3798.39
[profile] window=377.13s updates=100 data_wait=2.0ms h2d=1.6ms stage1=288.9ms dit=643.9ms update=27.7ms allocated=29.0GiB reserved=39.5GiB peak=38.0GiB
task=ci2v epoch=25 step=2720 loss=0.791812 loss_full=0.365123 loss_base=0.426689 ms_per_step=3789.14
[epoch-end] epoch=25 elapsed=398.98s step=2730
[epoch-start] epoch=26 step=2730 steps_per_epoch=105
task=ci2v epoch=26 step=2740 loss=0.815354 loss_full=0.389335 loss_base=0.426019 ms_per_step=3671.67
task=ci2v epoch=26 step=2760 loss=0.795253 loss_full=0.371653 loss_base=0.423600 ms_per_step=3751.05
task=ci2v epoch=26 step=2780 loss=0.823126 loss_full=0.389431 loss_base=0.433696 ms_per_step=3645.41
task=ci2v epoch=26 step=2800 loss=0.782847 loss_full=0.361397 loss_base=0.421450 ms_per_step=3775.51
[profile] window=372.92s updates=100 data_wait=1.7ms h2d=1.6ms stage1=279.5ms dit=643.4ms update=27.8ms allocated=29.0GiB reserved=39.5GiB peak=38.0GiB
task=ci2v epoch=26 step=2820 loss=0.773650 loss_full=0.353044 loss_base=0.420606 ms_per_step=3794.55
[epoch-end] epoch=26 elapsed=391.44s step=2835
[epoch-start] epoch=27 step=2835 steps_per_epoch=105
task=ci2v epoch=27 step=2840 loss=0.863690 loss_full=0.414183 loss_base=0.449506 ms_per_step=3793.15
task=ci2v epoch=27 step=2860 loss=0.858850 loss_full=0.394985 loss_base=0.463865 ms_per_step=3803.90
task=ci2v epoch=27 step=2880 loss=0.886059 loss_full=0.416606 loss_base=0.469453 ms_per_step=3623.09
task=ci2v epoch=27 step=2900 loss=0.787035 loss_full=0.371779 loss_base=0.415256 ms_per_step=3828.01
[profile] window=377.11s updates=100 data_wait=2.2ms h2d=1.6ms stage1=287.8ms dit=645.0ms update=27.7ms allocated=29.0GiB reserved=39.5GiB peak=38.0GiB
[2026-09-05 00:03] live-check: trainer PID 17993 still running; epoch=27 step=2900 loss=0.787035; GPU memory=41218 MiB; no fatal/error traceback; step-3000 checkpoint not present yet.
[2026-09-05 00:05] live-check: trainer PID 17993 still running; epoch=27 step=2920 loss=0.840125; GPU memory=41218 MiB; no fatal/error traceback; step-3000 checkpoint not present yet.
[2026-09-05 00:06] progress: epoch=27 completed at step=2940 (loss=0.765145); epoch=28 started; trainer remains healthy and step-3000 save is pending.
[2026-09-05 00:09] checkpoint-verified: step=3000 training loss=0.817714; sample and checkpoint completed successfully; checkpoint size=21,885,166,456 bytes; trainer PID 17993 remains active.
[2026-09-05 00:10] live-check: checkpoint step-00003000.pt is present and complete; trainer PID 17993 remains active after save; no fatal/error traceback detected.
[2026-09-05 00:10] progress: epoch=28 step=3020 loss=0.793476 after step-3000 checkpoint; trainer PID 17993 remains active and no fatal/error traceback is present.
[2026-09-05 00:11] progress: epoch=28 step=3040 loss=0.786859; trainer PID 17993 remains active, GPU memory=41218 MiB, and no fatal/error traceback is present.
[2026-09-05 00:12] progress: epoch=28 completed at step=3045; epoch=29 started; trainer PID 17993 remains active and no fatal/error traceback is present.
[2026-09-05 00:13] progress: epoch=29 step=3060 loss=0.765052; trainer PID 17993 remains active, GPU memory=41218 MiB, and no fatal/error traceback is present.
[2026-09-05 00:15] progress: epoch=29 step=3080 loss=0.825437; trainer PID 17993 remains active, GPU memory=41218 MiB, and no fatal/error traceback is present.
[2026-09-05 00:16] progress: epoch=29 step=3100 loss=0.788227; profile refreshed, trainer PID 17993 remains active, GPU memory=41218 MiB, and no fatal/error traceback is present.
[2026-09-05 00:17] live-check: epoch=29 step=3100 loss=0.788227 remains the latest output; trainer PID 17993 is still running with GPU memory=41218 MiB; no fatal/error traceback detected.
[2026-09-05 00:18] progress: epoch=29 step=3120 loss=0.780711; trainer PID 17993 remains active, GPU memory=41218 MiB, and no fatal/error traceback is present.
[2026-09-05 00:19] progress: epoch=29 step=3140 loss=0.774040; trainer PID 17993 remains active, GPU memory=41218 MiB, and no fatal/error traceback is present.
[2026-09-05 00:20] progress: epoch=29 completed at step=3150; epoch=30 started; trainer PID 17993 remains active and no fatal/error traceback is present.
[2026-09-05 00:21] live-check: epoch=30 has started at step=3150; trainer PID 17993 remains active with GPU memory=41218 MiB; no fatal/error traceback detected.
[2026-09-05 00:22] progress: epoch=30 step=3160 loss=0.811715; trainer PID 17993 remains active, GPU memory=41218 MiB, and no fatal/error traceback is present.
[2026-09-05 00:23] progress: epoch=30 step=3180 loss=0.749395; trainer PID 17993 remains active, GPU memory=41218 MiB, and no fatal/error traceback is present.
[2026-09-05 00:24] progress: epoch=30 step=3200 loss=0.711321; trainer PID 17993 remains active, GPU memory=41218 MiB, and no fatal/error traceback is present.
[2026-09-05 00:25] progress: epoch=30 step=3200 remains the latest verified output; profile.json refreshed; trainer PID 17993 remains active, GPU memory=41218 MiB, and no fatal/error traceback is present.
[2026-09-05 00:26] progress: epoch=30 step=3220 loss=0.808564; trainer PID 17993 remains active, GPU memory=41218 MiB, and no fatal/error traceback is present.
task=ci2v epoch=27 step=2920 loss=0.840125 loss_full=0.385330 loss_base=0.454795 ms_per_step=3796.07
task=ci2v epoch=27 step=2940 loss=0.765145 loss_full=0.345951 loss_base=0.419195 ms_per_step=3733.24
[epoch-end] epoch=27 elapsed=396.39s step=2940
[epoch-start] epoch=28 step=2940 steps_per_epoch=105
task=ci2v epoch=28 step=2960 loss=0.818200 loss_full=0.379312 loss_base=0.438887 ms_per_step=3884.84
task=ci2v epoch=28 step=2980 loss=0.761423 loss_full=0.338970 loss_base=0.422452 ms_per_step=3593.58
task=ci2v epoch=28 step=3000 loss=0.817714 loss_full=0.382864 loss_base=0.434850 ms_per_step=3786.40
[profile] window=376.15s updates=100 data_wait=2.2ms h2d=1.6ms stage1=287.0ms dit=643.2ms update=27.7ms allocated=29.0GiB reserved=39.5GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=3000 elapsed=3.02s
[checkpoint] step=3000 elapsed=37.53s
task=ci2v epoch=28 step=3020 loss=0.793476 loss_full=0.376533 loss_base=0.416943 ms_per_step=3694.00
task=ci2v epoch=28 step=3040 loss=0.786859 loss_full=0.377436 loss_base=0.409423 ms_per_step=3807.51
[epoch-end] epoch=28 elapsed=434.90s step=3045
[epoch-start] epoch=29 step=3045 steps_per_epoch=105
task=ci2v epoch=29 step=3060 loss=0.765052 loss_full=0.361827 loss_base=0.403225 ms_per_step=3833.85
task=ci2v epoch=29 step=3080 loss=0.825437 loss_full=0.389502 loss_base=0.435935 ms_per_step=3793.88
task=ci2v epoch=29 step=3100 loss=0.788227 loss_full=0.353414 loss_base=0.434813 ms_per_step=3796.44
[profile] window=378.78s updates=100 data_wait=2.1ms h2d=1.6ms stage1=290.2ms dit=646.0ms update=27.7ms allocated=29.0GiB reserved=39.5GiB peak=38.0GiB
task=ci2v epoch=29 step=3120 loss=0.780711 loss_full=0.367884 loss_base=0.412827 ms_per_step=3794.32
task=ci2v epoch=29 step=3140 loss=0.774040 loss_full=0.365269 loss_base=0.408771 ms_per_step=3794.17
[epoch-end] epoch=29 elapsed=398.72s step=3150
[epoch-start] epoch=30 step=3150 steps_per_epoch=105
task=ci2v epoch=30 step=3160 loss=0.811715 loss_full=0.384782 loss_base=0.426932 ms_per_step=3791.58
task=ci2v epoch=30 step=3180 loss=0.749395 loss_full=0.346726 loss_base=0.402670 ms_per_step=3420.33
task=ci2v epoch=30 step=3200 loss=0.711321 loss_full=0.332696 loss_base=0.378625 ms_per_step=3785.25
[profile] window=371.97s updates=100 data_wait=2.0ms h2d=1.6ms stage1=278.0ms dit=642.1ms update=27.7ms allocated=29.0GiB reserved=39.5GiB peak=38.0GiB
task=ci2v epoch=30 step=3220 loss=0.808564 loss_full=0.378790 loss_base=0.429774 ms_per_step=3785.54
task=ci2v epoch=30 step=3240 loss=0.726650 loss_full=0.326570 loss_base=0.400080 ms_per_step=3776.22
[2026-09-05 00:24] monitor: trainer pid=17993 alive; epoch=30 step=3240 loss=0.726650; GPU memory=41218 MiB; latest checkpoint remains step-00003000.pt; no fatal/error output detected.
[epoch-end] epoch=30 elapsed=390.64s step=3255
[epoch-start] epoch=31 step=3255 steps_per_epoch=105
task=ci2v epoch=31 step=3260 loss=0.753413 loss_full=0.345393 loss_base=0.408021 ms_per_step=3802.10
[2026-09-05 00:25] monitor: trainer pid=17993 alive; epoch=31 step=3260 loss=0.753413; log mtime=00:25:36; no fatal/error output detected.
task=ci2v epoch=31 step=3280 loss=0.746613 loss_full=0.342931 loss_base=0.403682 ms_per_step=3788.96
[2026-09-05 00:26] monitor: trainer pid=17993 alive; epoch=31 step=3280 loss=0.746613; log mtime=00:26:51; no fatal/error output detected.
[2026-09-05 00:27] monitor: trainer pid=17993 alive; latest output remains epoch=31 step=3280; GPU memory=41218 MiB; output interval is consistent with ~3.8 s/step; no fatal/error output detected.
task=ci2v epoch=31 step=3300 loss=0.810769 loss_full=0.382241 loss_base=0.428528 ms_per_step=3784.53
[2026-09-05 00:28] monitor: trainer pid=17993 alive; epoch=31 step=3300 loss=0.810769; GPU memory=41218 MiB; latest checkpoint remains step-00003000.pt; no fatal/error output detected.
[2026-09-05 00:28] monitor: recheck confirms trainer pid=17993 alive; latest emitted step=3300; log stream remains active and no fatal/error output detected.
[2026-09-05 00:28] monitor: trainer pid=17993 alive; epoch=31 step=3300 loss=0.810769; profile.json updated; GPU memory=41218 MiB; no fatal/error output detected.
[profile] window=378.99s updates=100 data_wait=2.0ms h2d=1.7ms stage1=293.1ms dit=644.2ms update=27.7ms allocated=29.0GiB reserved=39.5GiB peak=38.0GiB
task=ci2v epoch=31 step=3320 loss=0.809947 loss_full=0.371063 loss_base=0.438884 ms_per_step=3783.94
[2026-09-05 00:30] monitor: trainer pid=17993 alive; epoch=31 step=3320 loss=0.809947; GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[2026-09-05 00:29] monitor: trainer pid=17993 alive; epoch=31 step=3320 loss=0.809947; GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[2026-09-05 00:29] monitor: trainer pid=17993 alive; epoch=31 step=3320 loss=0.809947; GPU memory=41218 MiB; log stream active; no fatal/error output detected.
task=ci2v epoch=31 step=3340 loss=0.837741 loss_full=0.398816 loss_base=0.438925 ms_per_step=3667.06
[2026-09-05 00:31] monitor: trainer pid=17993 alive; epoch=31 step=3340 loss=0.837741; GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[2026-09-05 00:30] monitor: trainer pid=17993 alive; epoch=31 step=3340 loss=0.837741; GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[2026-09-05 00:30] monitor: trainer pid=17993 alive; epoch=31 step=3340 loss=0.837741; GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
task=ci2v epoch=31 step=3360 loss=0.729661 loss_full=0.333896 loss_base=0.395764 ms_per_step=3520.59
[epoch-end] epoch=31 elapsed=391.54s step=3360
[epoch-start] epoch=32 step=3360 steps_per_epoch=105
[2026-09-05 00:32] monitor: trainer pid=17993 alive; epoch=32 active from step=3360; GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[2026-09-05 00:32] monitor: trainer pid=17993 alive; epoch=31 ended at step=3360 (loss=0.729661), epoch=32 started; GPU memory=41218 MiB; no fatal/error output detected.
[2026-09-05 00:31] monitor: trainer pid=17993 alive; epoch=31 completed at step=3360 (latest loss=0.729661), epoch=32 started; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=32 step=3380 loss=0.749920 loss_full=0.355569 loss_base=0.394351 ms_per_step=3713.28
[2026-09-05 00:33] monitor: trainer pid=17993 alive; epoch=32 step=3380 loss=0.749920; GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[2026-09-05 00:33] monitor: trainer pid=17993 alive; epoch=32 step=3380 loss=0.749920; GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[2026-09-05 00:33] monitor: trainer pid=17993 alive; epoch=32 step=3380 loss=0.749920; GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
task=ci2v epoch=32 step=3400 loss=0.789638 loss_full=0.376409 loss_base=0.413229 ms_per_step=3793.19
[2026-09-05 00:34] monitor: trainer pid=17993 alive; epoch=32 step=3400 loss=0.789638; GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[2026-09-05 00:34] monitor: trainer pid=17993 alive; epoch=32 step=3400 loss=0.789638; GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[2026-09-05 00:34] monitor: trainer pid=17993 alive; epoch=32 step=3400 loss=0.789638; profile.json updated; GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[profile] window=369.82s updates=100 data_wait=2.1ms h2d=1.5ms stage1=272.7ms dit=641.9ms update=27.7ms allocated=29.0GiB reserved=39.5GiB peak=38.0GiB
task=ci2v epoch=32 step=3420 loss=0.803452 loss_full=0.382157 loss_base=0.421294 ms_per_step=3776.86
[2026-09-05 00:36] monitor: trainer pid=17993 alive; epoch=32 step=3420 loss=0.803452; GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[2026-09-05 00:35] monitor: trainer pid=17993 alive; epoch=32 step=3420 loss=0.803452; GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[2026-09-05 00:35] monitor: trainer pid=17993 alive; epoch=32 step=3420 loss=0.803452; GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
task=ci2v epoch=32 step=3440 loss=0.817008 loss_full=0.390595 loss_base=0.426413 ms_per_step=3797.93
[2026-09-05 00:37] monitor: trainer pid=17993 alive; epoch=32 step=3440 loss=0.817008; GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[2026-09-05 00:37] monitor: trainer pid=17993 alive; epoch=32 step=3440 loss=0.817008; GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[2026-09-05 00:36] monitor: trainer pid=17993 alive; epoch=32 step=3440 loss=0.817008; GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
task=ci2v epoch=32 step=3460 loss=0.765114 loss_full=0.354586 loss_base=0.410528 ms_per_step=3791.64
[epoch-end] epoch=32 elapsed=395.55s step=3465
[epoch-start] epoch=33 step=3465 steps_per_epoch=105
[2026-09-05 00:38] monitor: trainer pid=17993 alive; epoch=32 completed at step=3465 (last loss=0.765114), epoch=33 started; GPU memory=41218 MiB; no fatal/error output detected.
[2026-09-05 00:38] monitor: trainer pid=17993 alive; epoch=32 ended at step=3465 (latest loss=0.765114), epoch=33 started; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=33 step=3480 loss=0.706731 loss_full=0.330864 loss_base=0.375867 ms_per_step=3799.36
[2026-09-05 00:39] monitor: trainer pid=17993 alive; epoch=33 step=3480 loss=0.706731; GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[2026-09-05 00:39] monitor: trainer pid=17993 alive; epoch=33 step=3480 loss=0.706731; GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[2026-09-05 00:39] monitor: trainer pid=17993 alive; epoch=33 step=3480 loss=0.706731; GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
task=ci2v epoch=33 step=3500 loss=0.790608 loss_full=0.372174 loss_base=0.418434 ms_per_step=3771.06
[2026-09-05 00:41] monitor: trainer pid=17993 alive; epoch=33 step=3500 loss=0.790608; GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[2026-09-05 00:40] monitor: trainer pid=17993 alive; epoch=33 step=3500 loss=0.790608; GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[2026-09-05 00:40] monitor: trainer pid=17993 alive; epoch=33 step=3500 loss=0.790608; profile.json updated; GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[profile] window=379.00s updates=100 data_wait=2.0ms h2d=1.7ms stage1=292.9ms dit=644.4ms update=27.7ms allocated=29.0GiB reserved=39.5GiB peak=38.0GiB
task=ci2v epoch=33 step=3520 loss=0.719083 loss_full=0.339803 loss_base=0.379280 ms_per_step=3793.99
task=ci2v epoch=33 step=3540 loss=0.748964 loss_full=0.342313 loss_base=0.406651 ms_per_step=3791.56
[2026-09-05 00:43] monitor: trainer pid=17993 alive; epoch=33 step=3540 loss=0.748964 (loss_full=0.342313, loss_base=0.406651); GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
task=ci2v epoch=33 step=3560 loss=0.777302 loss_full=0.361434 loss_base=0.415868 ms_per_step=3773.90
[epoch-end] epoch=33 elapsed=398.49s step=3570
[epoch-start] epoch=34 step=3570 steps_per_epoch=105
[2026-09-05 00:45] monitor: trainer pid=17993 alive; epoch=34 started at step=3570 after epoch=33 (latest loss=0.777302 at step=3560); GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
task=ci2v epoch=34 step=3580 loss=0.840933 loss_full=0.397084 loss_base=0.443848 ms_per_step=3799.80
[2026-09-05 00:45] monitor: trainer pid=17993 alive; epoch=34 step=3580 loss=0.840933 (loss_full=0.397084, loss_base=0.443848); GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
task=ci2v epoch=34 step=3600 loss=0.743687 loss_full=0.344080 loss_base=0.399607 ms_per_step=3778.70
[2026-09-05 00:47] monitor: trainer pid=17993 alive; epoch=34 step=3600 loss=0.743687 (loss_full=0.344080, loss_base=0.399607); GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[profile] window=379.09s updates=100 data_wait=2.0ms h2d=1.7ms stage1=293.3ms dit=644.3ms update=27.7ms allocated=29.0GiB reserved=39.5GiB peak=38.0GiB
[2026-09-05 00:47] monitor: trainer pid=17993 still alive after profile window; GPU memory=41218 MiB; latest training output remains step=3600; no fatal/error output detected; no step-4000 artifacts yet.
task=ci2v epoch=34 step=3620 loss=0.803505 loss_full=0.380564 loss_base=0.422941 ms_per_step=3786.76
[2026-09-05 00:49] monitor: trainer pid=17993 alive; epoch=34 step=3620 loss=0.803505 (loss_full=0.380564, loss_base=0.422941); GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
task=ci2v epoch=34 step=3640 loss=0.773175 loss_full=0.349918 loss_base=0.423256 ms_per_step=3777.49
[2026-09-05 00:49] monitor: trainer pid=17993 alive; epoch=34 step=3640 loss=0.773175 (loss_full=0.349918, loss_base=0.423256); GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
task=ci2v epoch=34 step=3660 loss=0.812139 loss_full=0.383384 loss_base=0.428755 ms_per_step=3774.95
[2026-09-05 00:51] monitor: trainer pid=17993 alive; epoch=34 step=3660 loss=0.812139 (loss_full=0.383384, loss_base=0.428755); GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[epoch-end] epoch=34 elapsed=397.55s step=3675
[epoch-start] epoch=35 step=3675 steps_per_epoch=105
[2026-09-05 00:51] monitor: trainer pid=17993 alive; epoch=34 ended at step=3675 and epoch=35 started; latest logged loss=0.812139 at step=3660; GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
task=ci2v epoch=35 step=3680 loss=0.714680 loss_full=0.334039 loss_base=0.380640 ms_per_step=3729.78
[2026-09-05 00:52] monitor: trainer pid=17993 alive; epoch=35 step=3680 loss=0.714680 (loss_full=0.334039, loss_base=0.380640); GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
task=ci2v epoch=35 step=3700 loss=0.678526 loss_full=0.302452 loss_base=0.376074 ms_per_step=3706.69
[2026-09-05 00:53] monitor: trainer pid=17993 alive; epoch=35 step=3700 loss=0.678526 (loss_full=0.302452, loss_base=0.376074); GPU memory=41218 MiB; profile.json updated; no fatal/error output detected; checkpoint remains step-00003000.pt.
[2026-09-05 00:53] monitor: trainer pid=17993 alive; epoch=35 step=3700 loss=0.678526 (loss_full=0.302452, loss_base=0.376074); GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[profile] window=375.78s updates=100 data_wait=1.8ms h2d=1.6ms stage1=286.2ms dit=643.5ms update=27.7ms allocated=29.0GiB reserved=39.5GiB peak=38.0GiB
task=ci2v epoch=35 step=3720 loss=0.800948 loss_full=0.370205 loss_base=0.430743 ms_per_step=3786.73
[2026-09-05 00:54] monitor: trainer pid=17993 alive; epoch=35 step=3720 loss=0.800948 (loss_full=0.370205, loss_base=0.430743); GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[2026-09-05 00:54] monitor: trainer pid=17993 alive; epoch=35 step=3720 loss=0.800948 (loss_full=0.370205, loss_base=0.430743); GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
task=ci2v epoch=35 step=3740 loss=0.751161 loss_full=0.345067 loss_base=0.406094 ms_per_step=3824.64
[2026-09-05 00:56] monitor: trainer pid=17993 alive; epoch=35 step=3740 loss=0.751161 (loss_full=0.345067, loss_base=0.406094); GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[2026-09-05 00:56] monitor: trainer pid=17993 alive; epoch=35 step=3740 loss=0.751161 (loss_full=0.345067, loss_base=0.406094); GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
task=ci2v epoch=35 step=3760 loss=0.718179 loss_full=0.320450 loss_base=0.397729 ms_per_step=3791.28
[2026-09-05 00:57] monitor: trainer pid=17993 alive; epoch=35 step=3760 loss=0.718179 (loss_full=0.320450, loss_base=0.397729); GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[2026-09-05 00:57] monitor: trainer pid=17993 alive; epoch=35 step=3760 loss=0.718179 (loss_full=0.320450, loss_base=0.397729); GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
task=ci2v epoch=35 step=3780 loss=0.853271 loss_full=0.390541 loss_base=0.462731 ms_per_step=3734.66
[epoch-end] epoch=35 elapsed=395.96s step=3780
[epoch-start] epoch=36 step=3780 steps_per_epoch=105
[2026-09-05 00:58] monitor: trainer pid=17993 alive; epoch=35 ended at step=3780 and epoch=36 started; latest loss=0.853271 at step=3780; GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
task=ci2v epoch=36 step=3800 loss=0.777935 loss_full=0.360854 loss_base=0.417081 ms_per_step=3858.88
[2026-09-05 00:59] monitor: trainer pid=17993 alive; epoch=36 step=3800 loss=0.777935 (loss_full=0.360854, loss_base=0.417081); GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[profile] window=380.17s updates=100 data_wait=2.1ms h2d=1.7ms stage1=294.4ms dit=645.7ms update=27.7ms allocated=29.0GiB reserved=39.5GiB peak=38.0GiB
task=ci2v epoch=36 step=3820 loss=0.709096 loss_full=0.330269 loss_base=0.378827 ms_per_step=3735.46
[2026-09-05 01:00] monitor: trainer pid=17993 alive; epoch=36 step=3820 loss=0.709096 (loss_full=0.330269, loss_base=0.378827); GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[2026-09-05 01:00] monitor: trainer pid=17993 alive; epoch=36 step=3820 loss=0.709096 (loss_full=0.330269, loss_base=0.378827); GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
task=ci2v epoch=36 step=3840 loss=0.755091 loss_full=0.355349 loss_base=0.399742 ms_per_step=3720.84
[2026-09-05 01:02] monitor: trainer pid=17993 alive; epoch=36 step=3840 loss=0.755091 (loss_full=0.355349, loss_base=0.399742); GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
task=ci2v epoch=36 step=3860 loss=0.752681 loss_full=0.345410 loss_base=0.407272 ms_per_step=3779.65
[2026-09-05 01:03] monitor: trainer pid=17993 alive; epoch=36 step=3860 loss=0.752681 (loss_full=0.345410, loss_base=0.407272); GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[2026-09-05 01:03] monitor: trainer pid=17993 alive; epoch=36 step=3860 loss=0.752681 (loss_full=0.345410, loss_base=0.407272); GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[2026-09-05 01:03] monitor: trainer pid=17993 alive; epoch=36 step=3860 loss=0.752681 (loss_full=0.345410, loss_base=0.407272); GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
task=ci2v epoch=36 step=3880 loss=0.678201 loss_full=0.299530 loss_base=0.378671 ms_per_step=3794.78
[epoch-end] epoch=36 elapsed=396.21s step=3885
[epoch-start] epoch=37 step=3885 steps_per_epoch=105
task=ci2v epoch=37 step=3900 loss=0.701375 loss_full=0.322111 loss_base=0.379265 ms_per_step=3737.99
[profile] window=375.63s updates=100 data_wait=2.1ms h2d=1.6ms stage1=283.8ms dit=645.3ms update=27.7ms allocated=29.0GiB reserved=39.5GiB peak=38.0GiB
[2026-09-05 01:06] monitor: trainer pid=17993 alive; epoch=37 step=3900 loss=0.701375 (loss_full=0.322111, loss_base=0.379265); GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[2026-09-05 01:07] monitor: trainer pid=17993 alive; epoch=37 step=3920 loss=0.730818 (loss_full=0.339587, loss_base=0.391230); GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[2026-09-05 01:08] monitor: trainer pid=17993 alive; epoch=37 step=3940 loss=0.767033 (loss_full=0.353674, loss_base=0.413359); GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[2026-09-05 01:09] monitor: trainer pid=17993 alive; epoch=37 step=3960 loss=0.806443 (loss_full=0.380087, loss_base=0.426356); GPU memory=41218 MiB; no fatal/error output detected; checkpoint remains step-00003000.pt.
[2026-09-05 01:12] monitor: trainer pid=17993 alive; epoch=38 step=4000 loss=0.818275 (loss_full=0.379765, loss_base=0.438511); sample step-00004000.pt and checkpoint step-00004000.pt saved successfully; GPU memory=41218 MiB; no fatal/error output detected.
[2026-09-05 01:14] monitor: trainer pid=17993 alive; epoch=38 step=4020 loss=0.782886 (loss_full=0.370765, loss_base=0.412121); step-00004000 sample/checkpoint remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=37 step=3920 loss=0.730818 loss_full=0.339587 loss_base=0.391230 ms_per_step=3811.03
task=ci2v epoch=37 step=3940 loss=0.767033 loss_full=0.353674 loss_base=0.413359 ms_per_step=3795.95
task=ci2v epoch=37 step=3960 loss=0.806443 loss_full=0.380087 loss_base=0.426356 ms_per_step=3803.71
task=ci2v epoch=37 step=3980 loss=0.746278 loss_full=0.343346 loss_base=0.402932 ms_per_step=3721.51
[epoch-end] epoch=37 elapsed=392.53s step=3990
[epoch-start] epoch=38 step=3990 steps_per_epoch=105
task=ci2v epoch=38 step=4000 loss=0.818275 loss_full=0.379765 loss_base=0.438511 ms_per_step=3617.48
[profile] window=375.25s updates=100 data_wait=1.8ms h2d=1.6ms stage1=280.8ms dit=648.0ms update=27.7ms allocated=29.0GiB reserved=39.5GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=4000 elapsed=2.78s
[checkpoint] step=4000 elapsed=42.29s
task=ci2v epoch=38 step=4020 loss=0.782886 loss_full=0.370765 loss_base=0.412121 ms_per_step=3846.53
[2026-09-05 01:14] monitor: trainer pid=17993 alive; epoch=38 step=4020 loss=0.782886 (loss_full=0.370765, loss_base=0.412121); step-00004000 sample/checkpoint remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=38 step=4040 loss=0.665443 loss_full=0.306899 loss_base=0.358545 ms_per_step=3794.97
[2026-09-05 01:16] monitor: trainer pid=17993 alive; epoch=38 step=4060 loss=0.770011 (loss_full=0.355580, loss_base=0.414431); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
[2026-09-05 01:15] monitor: trainer pid=17993 alive; epoch=38 step=4040 loss=0.665443 (loss_full=0.306899, loss_base=0.358545); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=38 step=4060 loss=0.770011 loss_full=0.355580 loss_base=0.414431 ms_per_step=3796.65
task=ci2v epoch=38 step=4080 loss=0.835562 loss_full=0.402942 loss_base=0.432620 ms_per_step=3853.05
[2026-09-05 01:18] monitor: trainer pid=17993 alive; epoch=38 step=4080 loss=0.835562 (loss_full=0.402942, loss_base=0.432620); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
[epoch-end] epoch=38 elapsed=446.79s step=4095
[epoch-start] epoch=39 step=4095 steps_per_epoch=105
task=ci2v epoch=39 step=4100 loss=0.777326 loss_full=0.357577 loss_base=0.419749 ms_per_step=3814.83
[2026-09-05 01:20] monitor: trainer pid=17993 alive; epoch=39 step=4100 loss=0.777326 (loss_full=0.357577, loss_base=0.419749); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
[2026-09-05 01:19] monitor: trainer pid=17993 alive; epoch=39 step=4100 loss=0.777326 (loss_full=0.357577, loss_base=0.419749); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
[profile] window=382.39s updates=100 data_wait=2.3ms h2d=1.9ms stage1=298.1ms dit=647.5ms update=27.7ms allocated=29.0GiB reserved=39.5GiB peak=38.0GiB
task=ci2v epoch=39 step=4120 loss=0.855202 loss_full=0.408094 loss_base=0.447109 ms_per_step=3813.27
[2026-09-05 01:21] monitor: trainer pid=17993 alive; epoch=39 step=4120 loss=0.855202 (loss_full=0.408094, loss_base=0.447109); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
[2026-09-05 01:20] monitor: trainer pid=17993 alive; epoch=39 step=4120 loss=0.855202 (loss_full=0.408094, loss_base=0.447109); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=39 step=4140 loss=0.737705 loss_full=0.342710 loss_base=0.394994 ms_per_step=3407.10
[2026-09-05 01:22] monitor: trainer pid=17993 alive; epoch=39 step=4140 loss=0.737705 (loss_full=0.342710, loss_base=0.394994); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=39 step=4160 loss=0.680993 loss_full=0.311581 loss_base=0.369411 ms_per_step=3708.26
[2026-09-05 01:22] monitor: trainer pid=17993 alive; epoch=39 step=4160 loss=0.680993 (loss_full=0.311581, loss_base=0.369411); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=39 step=4180 loss=0.796049 loss_full=0.377870 loss_base=0.418179 ms_per_step=3798.48
[2026-09-05 01:24] monitor: trainer pid=17993 alive; epoch=39 step=4180 loss=0.796049 (loss_full=0.377870, loss_base=0.418179); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=39 step=4200 loss=0.714804 loss_full=0.319591 loss_base=0.395213 ms_per_step=3743.02
[profile] window=369.66s updates=100 data_wait=1.1ms h2d=1.5ms stage1=272.4ms dit=643.2ms update=27.7ms allocated=29.0GiB reserved=39.5GiB peak=38.0GiB
[epoch-end] epoch=39 elapsed=390.18s step=4200
[epoch-start] epoch=40 step=4200 steps_per_epoch=105
[2026-09-05 01:25] monitor: trainer pid=17993 alive; epoch=40 started at step=4200 after epoch=39 completion; latest logged loss at step=4200 was 0.714804; checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=40 step=4220 loss=0.719208 loss_full=0.321195 loss_base=0.398013 ms_per_step=3844.98
[2026-09-05 01:27] monitor: trainer pid=17993 alive; epoch=40 step=4220 loss=0.719208 (loss_full=0.321195, loss_base=0.398013); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=40 step=4240 loss=0.720921 loss_full=0.331734 loss_base=0.389187 ms_per_step=3791.20
[2026-09-05 01:28] monitor: trainer pid=17993 alive; epoch=40 step=4240 loss=0.720921 (loss_full=0.331734, loss_base=0.389187); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=40 step=4260 loss=0.775160 loss_full=0.364531 loss_base=0.410629 ms_per_step=3782.64
[2026-09-05 01:30] monitor: trainer pid=17993 alive; epoch=40 step=4260 loss=0.775160 (loss_full=0.364531, loss_base=0.410629); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=40 step=4280 loss=0.769057 loss_full=0.364102 loss_base=0.404954 ms_per_step=3861.00
[2026-09-05 01:30] monitor: trainer pid=17993 alive; epoch=40 step=4280 loss=0.769057 (loss_full=0.364102, loss_base=0.404954); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=40 step=4300 loss=0.688585 loss_full=0.315542 loss_base=0.373043 ms_per_step=3776.94
[profile] window=381.39s updates=100 data_wait=2.1ms h2d=1.7ms stage1=296.0ms dit=647.3ms update=27.7ms allocated=29.0GiB reserved=39.5GiB peak=38.0GiB
[epoch-end] epoch=40 elapsed=399.43s step=4305
[epoch-start] epoch=41 step=4305 steps_per_epoch=105
[2026-09-05 01:32] monitor: trainer pid=17993 alive; epoch=41 started at step=4305 after epoch=40 completion; latest logged loss at step=4300 was 0.688585; checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=41 step=4320 loss=0.708175 loss_full=0.315650 loss_base=0.392525 ms_per_step=3788.18
[2026-09-05 01:33] monitor: trainer pid=17993 alive; epoch=41 step=4320 loss=0.708175 (loss_full=0.315650, loss_base=0.392525); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=41 step=4340 loss=0.714799 loss_full=0.327382 loss_base=0.387417 ms_per_step=3648.29
[2026-09-05 01:34] monitor: trainer pid=17993 alive; epoch=41 step=4340 loss=0.714799 (loss_full=0.327382, loss_base=0.387417); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=41 step=4360 loss=0.791257 loss_full=0.370255 loss_base=0.421002 ms_per_step=3785.79
[2026-09-05 01:35] monitor: trainer pid=17993 alive; epoch=41 step=4360 loss=0.791257 (loss_full=0.370255, loss_base=0.421002); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=41 step=4380 loss=0.728306 loss_full=0.342045 loss_base=0.386260 ms_per_step=3784.56
[2026-09-05 01:37] monitor: trainer pid=17993 alive; epoch=41 step=4380 loss=0.728306 (loss_full=0.342045, loss_base=0.386260); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=41 step=4400 loss=0.650914 loss_full=0.286339 loss_base=0.364575 ms_per_step=3759.99
[profile] window=375.60s updates=100 data_wait=2.0ms h2d=1.6ms stage1=284.9ms dit=644.1ms update=27.7ms allocated=29.0GiB reserved=39.5GiB peak=38.0GiB
[epoch-end] epoch=41 elapsed=391.50s step=4410
[epoch-start] epoch=42 step=4410 steps_per_epoch=105
[2026-09-05 01:38] monitor: trainer pid=17993 alive; epoch=42 started at step=4410 after epoch=41 completion; latest logged loss at step=4400 was 0.650914; checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=42 step=4420 loss=0.737902 loss_full=0.338607 loss_base=0.399294 ms_per_step=3630.71
[2026-09-05 01:39] monitor: trainer pid=17993 alive; epoch=42 step=4420 loss=0.737902 (loss_full=0.338607, loss_base=0.399294); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=42 step=4440 loss=0.720376 loss_full=0.322765 loss_base=0.397612 ms_per_step=3786.94
[2026-09-05 01:40] monitor: trainer pid=17993 alive; epoch=42 step=4440 loss=0.720376 (loss_full=0.322765, loss_base=0.397612); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=42 step=4460 loss=0.715835 loss_full=0.327778 loss_base=0.388057 ms_per_step=3706.56
[2026-09-05 01:42] monitor: trainer pid=17993 alive; epoch=42 step=4460 loss=0.715835 (loss_full=0.327778, loss_base=0.388057); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=42 step=4480 loss=0.802988 loss_full=0.376878 loss_base=0.426110 ms_per_step=3736.68
[2026-09-05 01:43] monitor: trainer pid=17993 alive; epoch=42 step=4480 loss=0.802988 (loss_full=0.376878, loss_base=0.426110); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=42 step=4500 loss=0.764804 loss_full=0.355805 loss_base=0.408999 ms_per_step=3665.97
[2026-09-05 01:44] monitor: trainer pid=17993 alive; epoch=42 step=4500 loss=0.764804 (loss_full=0.355805, loss_base=0.408999); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
[2026-09-05 01:44] monitor: trainer pid=17993 alive; epoch=42 step=4500 loss=0.764804 (loss_full=0.355805, loss_base=0.408999); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
[profile] window=370.80s updates=100 data_wait=1.9ms h2d=1.5ms stage1=275.1ms dit=642.1ms update=27.7ms allocated=29.0GiB reserved=39.5GiB peak=38.0GiB
[epoch-end] epoch=42 elapsed=393.02s step=4515
[epoch-start] epoch=43 step=4515 steps_per_epoch=105
[2026-09-05 01:45] monitor: trainer pid=17993 alive; epoch=43 started at step=4515 after epoch=42 completion; latest logged loss at step=4500 was 0.764804; checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=43 step=4520 loss=0.734970 loss_full=0.338170 loss_base=0.396800 ms_per_step=3800.08
[2026-09-05 01:45] monitor: trainer pid=17993 alive; epoch=43 step=4520 loss=0.734970 (loss_full=0.338170, loss_base=0.396800); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=43 step=4540 loss=0.659125 loss_full=0.304532 loss_base=0.354593 ms_per_step=3790.83
[2026-09-05 01:47] monitor: trainer pid=17993 alive; epoch=43 step=4540 loss=0.659125 (loss_full=0.304532, loss_base=0.354593); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=43 step=4560 loss=0.800330 loss_full=0.370235 loss_base=0.430095 ms_per_step=3782.73
[2026-09-05 01:47] monitor: trainer pid=17993 alive; epoch=43 step=4560 loss=0.800330 (loss_full=0.370235, loss_base=0.430095); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=43 step=4580 loss=0.768388 loss_full=0.359227 loss_base=0.409161 ms_per_step=3827.28
task=ci2v epoch=43 step=4600 loss=0.720871 loss_full=0.324892 loss_base=0.395979 ms_per_step=3695.47
\n[2026-09-05 01:50] monitor: trainer pid=17993 alive; epoch=43 step=4600 loss=0.720871 (loss_full=0.324892, loss_base=0.395979); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
[profile] window=378.19s updates=100 data_wait=2.1ms h2d=1.6ms stage1=290.1ms dit=645.2ms update=27.7ms allocated=29.0GiB reserved=39.5GiB peak=38.0GiB
task=ci2v epoch=43 step=4620 loss=0.669932 loss_full=0.299949 loss_base=0.369983 ms_per_step=3749.02
[epoch-end] epoch=43 elapsed=397.44s step=4620
[epoch-start] epoch=44 step=4620 steps_per_epoch=105
[2026-09-05 01:51] monitor: trainer pid=17993 alive; epoch=44 started at step=4620 after epoch=43 completion; latest loss at step=4620 was 0.669932 (loss_full=0.299949, loss_base=0.369983); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=44 step=4640 loss=0.743850 loss_full=0.342480 loss_base=0.401371 ms_per_step=3620.63
[2026-09-05 01:52] monitor: trainer pid=17993 alive; epoch=44 step=4640 loss=0.743850 (loss_full=0.342480, loss_base=0.401371); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=44 step=4660 loss=0.667651 loss_full=0.305815 loss_base=0.361836 ms_per_step=3784.95
[2026-09-05 01:54] monitor: trainer pid=17993 alive; epoch=44 step=4660 loss=0.667651 (loss_full=0.305815, loss_base=0.361836); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=44 step=4680 loss=0.744571 loss_full=0.339345 loss_base=0.405226 ms_per_step=3744.27
[2026-09-05 01:56] monitor: trainer pid=17993 alive; epoch=44 step=4680 loss=0.744571 (loss_full=0.339345, loss_base=0.405226); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=44 step=4700 loss=0.769572 loss_full=0.354793 loss_base=0.414780 ms_per_step=3780.11
[2026-09-05 01:56] monitor: trainer pid=17993 alive; epoch=44 step=4700 loss=0.769572 (loss_full=0.354793, loss_base=0.414780); profile window updated; checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
[profile] window=373.85s updates=100 data_wait=1.8ms h2d=1.6ms stage1=282.1ms dit=643.0ms update=27.7ms allocated=29.0GiB reserved=39.5GiB peak=38.0GiB
task=ci2v epoch=44 step=4720 loss=0.721619 loss_full=0.332174 loss_base=0.389445 ms_per_step=3802.48
[epoch-end] epoch=44 elapsed=392.97s step=4725
[epoch-start] epoch=45 step=4725 steps_per_epoch=105
[2026-09-05 01:58] monitor: trainer pid=17993 alive; epoch=45 started at step=4725 after epoch=44 completion; latest logged loss at step=4720 was 0.721619 (loss_full=0.332174, loss_base=0.389445); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=45 step=4740 loss=0.703894 loss_full=0.320837 loss_base=0.383057 ms_per_step=3793.49
[2026-09-05 01:59] monitor: trainer pid=17993 alive; epoch=45 step=4740 loss=0.703894 (loss_full=0.320837, loss_base=0.383057); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=45 step=4760 loss=0.661600 loss_full=0.293168 loss_base=0.368432 ms_per_step=3792.97
[2026-09-05 02:00] monitor: trainer pid=17993 alive; epoch=45 step=4760 loss=0.661600 (loss_full=0.293168, loss_base=0.368432); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=45 step=4780 loss=0.640134 loss_full=0.281023 loss_base=0.359111 ms_per_step=3774.73
[2026-09-05 02:02] monitor: trainer pid=17993 alive; epoch=45 step=4780 loss=0.640134 (loss_full=0.281023, loss_base=0.359111); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=45 step=4800 loss=0.742342 loss_full=0.347251 loss_base=0.395091 ms_per_step=3785.52
[2026-09-05 02:02] monitor: trainer pid=17993 alive; epoch=45 step=4800 loss=0.742342 (loss_full=0.347251, loss_base=0.395091); profile window updated; checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
[profile] window=379.25s updates=100 data_wait=2.0ms h2d=1.7ms stage1=293.6ms dit=644.5ms update=27.7ms allocated=29.0GiB reserved=39.5GiB peak=38.0GiB
[2026-09-05 02:03] monitor: trainer pid=17993 alive; epoch=45 step=4800 loss=0.742342 (loss_full=0.347251, loss_base=0.395091); profile window updated; checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=45 step=4820 loss=0.668850 loss_full=0.316365 loss_base=0.352484 ms_per_step=3803.64
[epoch-end] epoch=45 elapsed=398.77s step=4830
[epoch-start] epoch=46 step=4830 steps_per_epoch=105
[2026-09-05 02:05] monitor: trainer pid=17993 alive; epoch=46 started at step=4830 after epoch=45 completion; latest logged loss at step=4820 was 0.668850 (loss_full=0.316365, loss_base=0.352484); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=46 step=4840 loss=0.760512 loss_full=0.356306 loss_base=0.404206 ms_per_step=3808.33
task=ci2v epoch=46 step=4860 loss=0.704479 loss_full=0.327894 loss_base=0.376585 ms_per_step=3804.78
[2026-09-05 02:07] monitor: trainer pid=17993 alive; epoch=46 step=4860 loss=0.704479 (loss_full=0.327894, loss_base=0.376585); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; log mtime advanced; no fatal/error output detected.
task=ci2v epoch=46 step=4880 loss=0.789792 loss_full=0.364464 loss_base=0.425327 ms_per_step=3800.47
[2026-09-05 02:10] monitor: trainer pid=17993 alive; epoch=46 step=4880 loss=0.789792 (loss_full=0.364464, loss_base=0.425327); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
[2026-09-05 02:09] monitor: trainer pid=17993 alive; epoch=46 step=4880 loss=0.789792 (loss_full=0.364464, loss_base=0.425327); log mtime advanced; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=46 step=4900 loss=0.761640 loss_full=0.356111 loss_base=0.405529 ms_per_step=3800.01
[profile] window=380.59s updates=100 data_wait=2.1ms h2d=1.7ms stage1=293.4ms dit=648.0ms update=27.7ms allocated=29.0GiB reserved=39.5GiB peak=38.0GiB
[2026-09-05 02:10] monitor: trainer pid=17993 alive; epoch=46 step=4900 loss=0.761640 (loss_full=0.356111, loss_base=0.405529); profile window updated; checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
[2026-09-05 02:11] monitor: trainer pid=17993 alive; epoch=46 step=4900 loss=0.761640 (loss_full=0.356111, loss_base=0.405529); log mtime advanced; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=46 step=4920 loss=0.644486 loss_full=0.297164 loss_base=0.347322 ms_per_step=3786.70
[2026-09-05 02:12] monitor: trainer pid=17993 alive; epoch=46 step=4920 loss=0.644486 (loss_full=0.297164, loss_base=0.347322); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; log mtime advanced; no fatal/error output detected.
[epoch-end] epoch=46 elapsed=397.73s step=4935
[epoch-start] epoch=47 step=4935 steps_per_epoch=105
[2026-09-05 02:13] monitor: trainer pid=17993 alive; epoch=47 started at step=4935 after epoch=46 completion; latest logged loss at step=4920 was 0.644486 (loss_full=0.297164, loss_base=0.347322); GPU memory=41218 MiB; checkpoint step-00004000.pt and sample remain present; no fatal/error output detected.
task=ci2v epoch=47 step=4940 loss=0.727647 loss_full=0.327633 loss_base=0.400014 ms_per_step=3735.69
[2026-09-05 02:14] monitor: trainer pid=17993 alive; epoch=47 step=4940 loss=0.727647 (loss_full=0.327633, loss_base=0.400014); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; log mtime advanced; no fatal/error output detected.
task=ci2v epoch=47 step=4960 loss=0.714537 loss_full=0.326018 loss_base=0.388519 ms_per_step=3802.97
[2026-09-05 02:15] monitor: trainer pid=17993 alive; epoch=47 step=4960 loss=0.714537 (loss_full=0.326018, loss_base=0.388519); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; log mtime advanced; no fatal/error output detected.
task=ci2v epoch=47 step=4980 loss=0.728116 loss_full=0.329040 loss_base=0.399076 ms_per_step=3788.99
[2026-09-05 02:16] monitor: trainer pid=17993 alive; epoch=47 step=4980 loss=0.728116 (loss_full=0.329040, loss_base=0.399076); checkpoint step-00004000.pt and sample remain present; GPU memory=41218 MiB; no fatal/error output detected.
task=ci2v epoch=47 step=5000 loss=0.711899 loss_full=0.322253 loss_base=0.389646 ms_per_step=3807.18
[profile] window=378.68s updates=100 data_wait=2.1ms h2d=1.7ms stage1=291.2ms dit=645.6ms update=27.7ms allocated=29.0GiB reserved=39.5GiB peak=38.0GiB
[2026-09-05 02:17] monitor: trainer pid=17993 alive; epoch=47 step=5000 loss=0.711899 (loss_full=0.322253, loss_base=0.389646); checkpoint step-00005000.pt and sample were saved successfully; GPU memory=41410 MiB; no fatal/error output detected.
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=5000 elapsed=2.76s
[checkpoint] step=5000 elapsed=34.28s
[2026-09-05 02:18] monitor: trainer pid=17993 alive; epoch=47 step=5000; checkpoint and sample step-00005000.pt verified on disk after save; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=47 step=5020 loss=0.774549 loss_full=0.361638 loss_base=0.412910 ms_per_step=3709.20
[2026-09-05 02:19] monitor: trainer pid=17993 alive; epoch=47 step=5020 loss=0.774549 (loss_full=0.361638, loss_base=0.412910); checkpoint step-00005000.pt and sample remain present; GPU memory=41410 MiB; log mtime advanced; no fatal/error output detected.
task=ci2v epoch=47 step=5040 loss=0.675030 loss_full=0.301614 loss_base=0.373416 ms_per_step=3752.83
[epoch-end] epoch=47 elapsed=434.82s step=5040
[epoch-start] epoch=48 step=5040 steps_per_epoch=105
[2026-09-05 02:19] monitor: trainer pid=17993 alive; epoch=48 started at step=5040 after epoch=47 completion; latest logged loss at step=5040 was 0.675030 (loss_full=0.301614, loss_base=0.373416); checkpoint step-00005000.pt and sample remain present; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=48 step=5060 loss=0.758361 loss_full=0.345753 loss_base=0.412608 ms_per_step=3851.77
[2026-09-05 02:20] monitor: trainer pid=17993 alive; epoch=48 step=5060 loss=0.758361 (loss_full=0.345753, loss_base=0.412608); checkpoint step-00005000.pt and sample remain present; GPU memory=41410 MiB; log mtime advanced; no fatal/error output detected.
task=ci2v epoch=48 step=5080 loss=0.857467 loss_full=0.401341 loss_base=0.456125 ms_per_step=3785.46
[2026-09-05 02:22] monitor: trainer pid=17993 alive; epoch=48 step=5080 loss=0.857467 (loss_full=0.401341, loss_base=0.456125); checkpoint step-00005000.pt and sample remain present; GPU memory=41410 MiB; log mtime advanced; no fatal/error output detected.
task=ci2v epoch=48 step=5100 loss=0.725083 loss_full=0.330970 loss_base=0.394113 ms_per_step=3766.22
[profile] window=377.57s updates=100 data_wait=2.1ms h2d=1.7ms stage1=288.6ms dit=645.0ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 02:23] monitor: trainer pid=17993 alive; epoch=48 step=5100 loss=0.725083 (loss_full=0.330970, loss_base=0.394113); profile window updated; checkpoint step-00005000.pt and sample remain present; GPU memory=41410 MiB; log mtime advanced; no fatal/error output detected.
task=ci2v epoch=48 step=5120 loss=0.681003 loss_full=0.308748 loss_base=0.372255 ms_per_step=3805.55
[2026-09-05 02:24] monitor: trainer pid=17993 alive; epoch=48 step=5120 loss=0.681003 (loss_full=0.308748, loss_base=0.372255); checkpoint step-00005000.pt and sample remain present; GPU memory=41410 MiB; log mtime advanced; no fatal/error output detected.
task=ci2v epoch=48 step=5140 loss=0.670643 loss_full=0.302122 loss_base=0.368521 ms_per_step=3820.20
[epoch-end] epoch=48 elapsed=398.73s step=5145
[epoch-start] epoch=49 step=5145 steps_per_epoch=105
task=ci2v epoch=49 step=5160 loss=0.731613 loss_full=0.328994 loss_base=0.402619 ms_per_step=3631.01
[2026-09-05 02:26] monitor: trainer pid=17993 alive; epoch=49 step=5160 loss=0.731613 (loss_full=0.328994, loss_base=0.402619); checkpoint step-00005000.pt and sample remain present; GPU memory=41410 MiB; log mtime advanced; no fatal/error output detected.
[2026-09-05 02:27] monitor: trainer pid=17993 remains alive in epoch=49 at step=5160; no new periodic step line during the 30s check (normal logging cadence); GPU process remains active; no fatal/error output detected.
[2026-09-05 02:27] monitor: trainer pid=17993 alive; epoch=49 advanced to step=5180 loss=0.771000 (loss_full=0.356604, loss_base=0.414396); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 02:28] monitor: trainer pid=17993 alive; epoch=49 step=5180 remains latest observed; GPU process active at 41410 MiB; checkpoint step-00005000.pt and sample present; no fatal/error output detected.
[2026-09-05 02:28] monitor: trainer pid=17993 alive; epoch=49 advanced to step=5200 loss=0.747153 (loss_full=0.352913, loss_base=0.394240); profile window updated; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 02:29] monitor: trainer pid=17993 alive; epoch=49 step=5200 remains latest observed; profile window updated; GPU memory=41410 MiB; checkpoint step-00005000.pt and sample present; no fatal/error output detected.
[2026-09-05 02:29] monitor: trainer pid=17993 alive; epoch=49 step=5200 remains latest observed; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 02:30] monitor: trainer pid=17993 alive; epoch=49 advanced to step=5220 loss=0.624554 (loss_full=0.281404, loss_base=0.343150); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 02:30] monitor: trainer pid=17993 remains alive; epoch=49 step=5220 latest observed; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 02:32] monitor: trainer pid=17993 alive; epoch=49 reached step=5250 and epoch=50 started; latest logged loss at step=5240 is 0.676815 (loss_full=0.314726, loss_base=0.362089); GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=49 step=5180 loss=0.771000 loss_full=0.356604 loss_base=0.414396 ms_per_step=3794.71
task=ci2v epoch=49 step=5200 loss=0.747153 loss_full=0.352913 loss_base=0.394240 ms_per_step=3774.67
[profile] window=376.78s updates=100 data_wait=2.2ms h2d=1.6ms stage1=287.1ms dit=644.8ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=49 step=5220 loss=0.624554 loss_full=0.281404 loss_base=0.343150 ms_per_step=3820.81
task=ci2v epoch=49 step=5240 loss=0.676815 loss_full=0.314726 loss_base=0.362089 ms_per_step=3782.33
[epoch-end] epoch=49 elapsed=396.08s step=5250
[epoch-start] epoch=50 step=5250 steps_per_epoch=105
task=ci2v epoch=50 step=5260 loss=0.681579 loss_full=0.311209 loss_base=0.370370 ms_per_step=3828.62
[2026-09-05 02:33] monitor: trainer pid=17993 alive; epoch=50 step=5260 remains latest observed; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 02:33] monitor: trainer pid=17993 alive; epoch=50 step=5260 loss=0.681579 (loss_full=0.311209, loss_base=0.370370); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 02:33] monitor: trainer pid=17993 alive; epoch=50 advanced to step=5260 loss=0.681579 (loss_full=0.311209, loss_base=0.370370); GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=50 step=5280 loss=0.746367 loss_full=0.338945 loss_base=0.407422 ms_per_step=3807.17
[2026-09-05 02:34] monitor: trainer pid=17993 alive; epoch=50 advanced to step=5280 loss=0.746367 (loss_full=0.338945, loss_base=0.407422); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 02:34] monitor: trainer pid=17993 alive; epoch=50 step=5280 remains latest observed; GPU process active at 41410 MiB; no fatal/error output detected.
task=ci2v epoch=50 step=5300 loss=0.695271 loss_full=0.320266 loss_base=0.375005 ms_per_step=3796.44
[2026-09-05 02:35] monitor: trainer pid=17993 alive; epoch=50 step=5300 loss=0.695271 (loss_full=0.320266, loss_base=0.375005); profile window updated; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 02:35] monitor: trainer pid=17993 remains alive; epoch=50 step=5300 latest observed; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 02:36] monitor: trainer pid=17993 alive; epoch=50 step=5300 remains latest observed; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 02:37] monitor: trainer pid=17993 alive; epoch=50 advanced to step=5320 loss=0.735694 (loss_full=0.341919, loss_base=0.393775); GPU memory=41410 MiB; no fatal/error output detected.
[profile] window=380.96s updates=100 data_wait=2.2ms h2d=1.7ms stage1=295.8ms dit=646.3ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=50 step=5320 loss=0.735694 loss_full=0.341919 loss_base=0.393775 ms_per_step=3791.22
[2026-09-05 02:36] monitor: trainer pid=17993 alive; epoch=50 advanced to step=5320 loss=0.735694 (loss_full=0.341919, loss_base=0.393775); GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=50 step=5340 loss=0.733615 loss_full=0.333001 loss_base=0.400614 ms_per_step=3798.66
[2026-09-05 02:37] monitor: trainer pid=17993 alive; epoch=50 advanced to step=5340 loss=0.733615 (loss_full=0.333001, loss_base=0.400614); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 02:38] monitor: trainer pid=17993 remains alive; epoch=50 step=5340 latest observed; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 02:38] monitor: trainer pid=17993 alive; epoch=50 completed at step=5355 and epoch=51 started; latest loss at step=5340 is 0.733615 (loss_full=0.333001, loss_base=0.400614); GPU memory=41410 MiB; no fatal/error output detected.
[epoch-end] epoch=50 elapsed=399.15s step=5355
[epoch-start] epoch=51 step=5355 steps_per_epoch=105
task=ci2v epoch=51 step=5360 loss=0.640971 loss_full=0.294318 loss_base=0.346653 ms_per_step=3809.48
[2026-09-05 02:39] monitor: trainer pid=17993 alive; epoch=51 advanced to step=5360 loss=0.640971 (loss_full=0.294318, loss_base=0.346653); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 02:39] monitor: trainer pid=17993 remains alive; epoch=51 step=5360 latest observed; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=51 step=5380 loss=0.766447 loss_full=0.360254 loss_base=0.406193 ms_per_step=3771.52
[2026-09-05 02:40] monitor: trainer pid=17993 alive; epoch=51 advanced to step=5380 loss=0.766447 (loss_full=0.360254, loss_base=0.406193); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 02:41] monitor: trainer pid=17993 remains alive; epoch=51 step=5380 latest observed; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=51 step=5400 loss=0.762489 loss_full=0.351901 loss_base=0.410587 ms_per_step=3789.76
[2026-09-05 02:41] monitor: trainer pid=17993 alive; epoch=51 advanced to step=5400 loss=0.762489 (loss_full=0.351901, loss_base=0.410587); profile window updated; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 02:42] monitor: trainer pid=17993 remains alive; epoch=51 step=5400 latest observed; GPU memory=41410 MiB; no fatal/error output detected.
[profile] window=379.47s updates=100 data_wait=2.1ms h2d=1.7ms stage1=293.5ms dit=645.1ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=51 step=5420 loss=0.728763 loss_full=0.323592 loss_base=0.405170 ms_per_step=3714.82
task=ci2v epoch=51 step=5440 loss=0.680972 loss_full=0.310972 loss_base=0.370001 ms_per_step=3565.04
[2026-09-05 02:43] monitor: trainer pid=17993 alive; epoch=51 advanced through step=5440 (latest loss=0.680972, loss_full=0.310972, loss_base=0.370001); GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=51 step=5460 loss=0.754218 loss_full=0.351263 loss_base=0.402956 ms_per_step=3639.35
[epoch-end] epoch=51 elapsed=390.45s step=5460
[epoch-start] epoch=52 step=5460 steps_per_epoch=105
[2026-09-05 02:45] monitor: trainer pid=17993 alive; epoch=51 completed at step=5460 and epoch=52 started; latest loss=0.754218 (loss_full=0.351263, loss_base=0.402956); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 02:46] monitor: trainer pid=17993 remains alive in epoch=52 after step=5460; GPU memory=41410 MiB; no fatal/error output detected (next periodic log not yet due).
task=ci2v epoch=52 step=5480 loss=0.743706 loss_full=0.339470 loss_base=0.404236 ms_per_step=3705.19
[2026-09-05 02:46] monitor: trainer pid=17993 alive; epoch=52 advanced to step=5480 loss=0.743706 (loss_full=0.339470, loss_base=0.404236); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 02:46] monitor: checkpoint scan confirms latest checkpoint/sample remain step=00005000; trainer pid=17993 is active at GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 02:47] monitor: trainer pid=17993 remains alive in epoch=52; latest observed step=5480, GPU memory=41410 MiB; no fatal/error output detected (periodic logging interval still in progress).
task=ci2v epoch=52 step=5500 loss=0.788286 loss_full=0.366247 loss_base=0.422039 ms_per_step=3794.41
[2026-09-05 02:47] monitor: trainer pid=17993 alive; epoch=52 advanced to step=5500 loss=0.788286 (loss_full=0.366247, loss_base=0.422039); profile window updated; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 02:48] monitor: trainer pid=17993 remains alive after step=5500; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint remains step=00005000.
[2026-09-05 02:48] monitor: trainer pid=17993 remains active; latest observed epoch=52 step=5500, GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 02:48] monitor: trainer pid=17993 remains alive (Rsl); epoch=52 step=5500 remains latest observed; GPU memory=41410 MiB; no fatal/error output detected.
[profile] window=368.64s updates=100 data_wait=1.8ms h2d=1.5ms stage1=270.6ms dit=641.4ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=52 step=5520 loss=0.711466 loss_full=0.334591 loss_base=0.376875 ms_per_step=3789.37
[2026-09-05 02:49] monitor: trainer pid=17993 alive; epoch=52 advanced to step=5520 loss=0.711466 (loss_full=0.334591, loss_base=0.376875); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 02:49] monitor: checkpoint scan confirms latest checkpoint/sample remain step=00005000; trainer pid=17993 active at GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 02:50] monitor: trainer pid=17993 remains alive in epoch=52; latest observed step=5520, GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=52 step=5540 loss=0.704828 loss_full=0.319187 loss_base=0.385640 ms_per_step=3773.28
[2026-09-05 02:50] monitor: trainer pid=17993 alive; epoch=52 advanced to step=5540 loss=0.704828 (loss_full=0.319187, loss_base=0.385640); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 02:50] monitor: trainer pid=17993 remains active; latest epoch=52 step=5540, GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint remains step=00005000.
[2026-09-05 02:51] monitor: trainer pid=17993 remains alive in epoch=52; latest observed step=5540, GPU memory=41410 MiB; no fatal/error output detected (next periodic log pending).
task=ci2v epoch=52 step=5560 loss=0.671993 loss_full=0.299369 loss_base=0.372624 ms_per_step=3794.21
[epoch-end] epoch=52 elapsed=395.21s step=5565
[epoch-start] epoch=53 step=5565 steps_per_epoch=105
[2026-09-05 02:51] monitor: trainer pid=17993 alive; epoch=52 completed at step=5565 and epoch=53 started; latest loss at step=5560 is 0.671993 (loss_full=0.299369, loss_base=0.372624); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 02:52] monitor: trainer pid=17993 remains alive in epoch=53 after step=5565; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint remains step=00005000.
[2026-09-05 02:52] monitor: trainer pid=17993 remains active (Rsl); epoch=53 step=5565 is latest observed; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=53 step=5580 loss=0.740314 loss_full=0.334979 loss_base=0.405335 ms_per_step=3813.53
[2026-09-05 02:53] monitor: trainer pid=17993 alive; epoch=53 advanced to step=5580 loss=0.740314 (loss_full=0.334979, loss_base=0.405335); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 02:53] monitor: trainer pid=17993 remains active; epoch=53 step=5580 latest observed; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint remains step=00005000.
task=ci2v epoch=53 step=5600 loss=0.743309 loss_full=0.341914 loss_base=0.401395 ms_per_step=3794.71
[2026-09-05 02:53] monitor: trainer pid=17993 alive; epoch=53 advanced to step=5600 loss=0.743309 (loss_full=0.341914, loss_base=0.401395); profile window updated; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 02:54] monitor: trainer pid=17993 remains alive in epoch=53 after step=5600; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint remains step=00005000.
[2026-09-05 02:54] monitor: trainer pid=17993 remains active; latest observed epoch=53 step=5600, GPU memory=41410 MiB; no fatal/error output detected.
[profile] window=379.56s updates=100 data_wait=2.5ms h2d=1.7ms stage1=292.6ms dit=645.8ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=53 step=5620 loss=0.762754 loss_full=0.347684 loss_base=0.415070 ms_per_step=3794.59
[2026-09-05 02:55] monitor: trainer pid=17993 alive; epoch=53 advanced to step=5620 loss=0.762754 (loss_full=0.347684, loss_base=0.415070); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 02:55] monitor: trainer pid=17993 remains active; latest epoch=53 step=5620, GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint remains step=00005000.
[2026-09-05 02:56] monitor: trainer pid=17993 remains alive in epoch=53; latest observed step=5620, GPU memory=41410 MiB; no fatal/error output detected (next periodic log pending).
task=ci2v epoch=53 step=5640 loss=0.699359 loss_full=0.312509 loss_base=0.386851 ms_per_step=3803.73
[2026-09-05 02:56] monitor: trainer pid=17993 alive; epoch=53 advanced to step=5640 loss=0.699359 (loss_full=0.312509, loss_base=0.386851); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 02:56] monitor: trainer pid=17993 remains active; latest epoch=53 step=5640, GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint remains step=00005000.
[2026-09-05 02:57] monitor: trainer pid=17993 remains alive in epoch=53; latest observed step=5640, GPU memory=41410 MiB; no fatal/error output detected (next periodic log pending).
task=ci2v epoch=53 step=5660 loss=0.740070 loss_full=0.338627 loss_base=0.401443 ms_per_step=3801.88
[2026-09-05 02:57] monitor: trainer pid=17993 alive; epoch=53 advanced to step=5660 loss=0.740070 (loss_full=0.338627, loss_base=0.401443); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 02:58] monitor: trainer pid=17993 remains active in epoch=53; latest observed step=5660, GPU memory=41410 MiB; no fatal/error output detected (next periodic log pending).
[epoch-end] epoch=53 elapsed=399.76s step=5670
[epoch-start] epoch=54 step=5670 steps_per_epoch=105
[2026-09-05 02:58] monitor: trainer pid=17993 alive; epoch=53 completed at step=5670 and epoch=54 started; latest loss at step=5660 is 0.740070 (loss_full=0.338627, loss_base=0.401443); GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=54 step=5680 loss=0.726285 loss_full=0.340081 loss_base=0.386204 ms_per_step=3791.68
[2026-09-05 02:59] monitor: trainer pid=17993 alive; epoch=54 advanced to step=5680 loss=0.726285 (loss_full=0.340081, loss_base=0.386204); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:00] monitor: trainer pid=17993 remains active in epoch=54; latest observed step=5680, GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 02:59] monitor: trainer pid=17993 alive; epoch=54 advanced to step=5680 loss=0.726285 (loss_full=0.340081, loss_base=0.386204); GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=54 step=5700 loss=0.722193 loss_full=0.338518 loss_base=0.383675 ms_per_step=3806.05
[2026-09-05 03:00] monitor: trainer pid=17993 alive; epoch=54 advanced to step=5700 loss=0.722193 (loss_full=0.338518, loss_base=0.383675); profile window updated; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:01] monitor: trainer pid=17993 remains active; latest epoch=54 step=5700, GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint remains step=00005000.
[2026-09-05 03:01] monitor: trainer pid=17993 remains alive in epoch=54; latest observed step=5700, GPU memory=41410 MiB; no fatal/error output detected (next periodic log pending).
[profile] window=380.22s updates=100 data_wait=1.9ms h2d=1.7ms stage1=292.8ms dit=647.8ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=54 step=5720 loss=0.724241 loss_full=0.327947 loss_base=0.396294 ms_per_step=3835.56
[2026-09-05 03:01] monitor: trainer pid=17993 alive; epoch=54 advanced to step=5720 loss=0.724241 (loss_full=0.327947, loss_base=0.396294); profile window updated; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:02] monitor: trainer pid=17993 remains active; latest epoch=54 step=5720, GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint remains step=00005000.
[2026-09-05 03:03] monitor: trainer pid=17993 alive; epoch=54 advanced to step=5740 loss=0.732438 (loss_full=0.333707, loss_base=0.398730); GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=54 step=5740 loss=0.732438 loss_full=0.333707 loss_base=0.398730 ms_per_step=3792.96
[2026-09-05 03:03] monitor: trainer pid=17993 remains active; latest epoch=54 step=5740, GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint remains step=00005000.
[2026-09-05 03:02] monitor: trainer pid=17993 alive; epoch=54 advanced to step=5740 loss=0.732438 (loss_full=0.333707, loss_base=0.398730); GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=54 step=5760 loss=0.742190 loss_full=0.331766 loss_base=0.410424 ms_per_step=3778.63
[2026-09-05 03:04] monitor: trainer pid=17993 alive; epoch=54 advanced to step=5760 loss=0.742190 (loss_full=0.331766, loss_base=0.410424); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:04] monitor: trainer pid=17993 remains active; latest epoch=54 step=5760, GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint remains step=00005000.
[epoch-end] epoch=54 elapsed=399.25s step=5775
[epoch-start] epoch=55 step=5775 steps_per_epoch=105
[2026-09-05 03:06] monitor: trainer pid=17993 alive; epoch=55 advanced to step=5800 (latest loss=0.637410, loss_full=0.285461, loss_base=0.351949); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint remains step=00005000.
[2026-09-05 03:07] monitor: trainer pid=17993 remains active in epoch=55; latest observed step=5800, GPU memory=41410 MiB; no fatal/error output detected (next periodic log pending).
task=ci2v epoch=55 step=5780 loss=0.610069 loss_full=0.270838 loss_base=0.339231 ms_per_step=3655.89
task=ci2v epoch=55 step=5800 loss=0.637410 loss_full=0.285461 loss_base=0.351949 ms_per_step=3510.56
[profile] window=371.74s updates=100 data_wait=1.8ms h2d=1.5ms stage1=274.9ms dit=645.1ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=55 step=5820 loss=0.641390 loss_full=0.282481 loss_base=0.358909 ms_per_step=3789.68
[2026-09-05 03:08] monitor: trainer pid=17993 alive; epoch=55 advanced to step=5820 loss=0.641390 (loss_full=0.282481, loss_base=0.358909); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint remains step=00005000.
task=ci2v epoch=55 step=5840 loss=0.724760 loss_full=0.336159 loss_base=0.388601 ms_per_step=3573.88
[2026-09-05 03:08] monitor: trainer pid=17993 alive; epoch=55 advanced to step=5840 loss=0.724760 (loss_full=0.336159, loss_base=0.388601); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint remains step=00005000.
[2026-09-05 03:09] monitor: trainer pid=17993 remains active in epoch=55 after step=5840; GPU memory=41410 MiB; no fatal/error output detected (next periodic log pending).
task=ci2v epoch=55 step=5860 loss=0.699433 loss_full=0.315175 loss_base=0.384258 ms_per_step=3604.34
[2026-09-05 03:10] monitor: trainer pid=17993 alive; epoch=55 advanced to step=5860 loss=0.699433 (loss_full=0.315175, loss_base=0.384258); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint remains step=00005000.
[2026-09-05 03:11] monitor: trainer pid=17993 remains active in epoch=55 after step=5860; GPU memory=41410 MiB; no fatal/error output detected (next periodic log pending).
task=ci2v epoch=55 step=5880 loss=0.749203 loss_full=0.350242 loss_base=0.398961 ms_per_step=3664.27
[epoch-end] epoch=55 elapsed=380.67s step=5880
[epoch-start] epoch=56 step=5880 steps_per_epoch=105
[2026-09-05 03:11] monitor: trainer pid=17993 alive; epoch=55 completed at step=5880 and epoch=56 started; latest loss at step=5880 is 0.749203 (loss_full=0.350242, loss_base=0.398961); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:12] monitor: trainer pid=17993 remains active in epoch=56; latest observed step=5880, GPU memory=41410 MiB; no fatal/error output detected (next periodic log pending).
task=ci2v epoch=56 step=5900 loss=0.764416 loss_full=0.359120 loss_base=0.405297 ms_per_step=3871.34
[2026-09-05 03:12] monitor: trainer pid=17993 alive; epoch=56 advanced to step=5900 loss=0.764416 (loss_full=0.359120, loss_base=0.405297); profile window updated; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint remains step=00005000.
[2026-09-05 03:13] monitor: trainer pid=17993 remains active in epoch=56 after step=5900; GPU memory=41410 MiB; no fatal/error output detected (next periodic log pending).
[profile] window=370.33s updates=100 data_wait=2.1ms h2d=1.5ms stage1=272.8ms dit=641.6ms update=33.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=56 step=5920 loss=0.690046 loss_full=0.309947 loss_base=0.380099 ms_per_step=3804.53
[2026-09-05 03:14] monitor: trainer pid=17993 alive; epoch=56 advanced to step=5920 loss=0.690046 (loss_full=0.309947, loss_base=0.380099); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint remains step=00005000.
task=ci2v epoch=56 step=5940 loss=0.600065 loss_full=0.263977 loss_base=0.336088 ms_per_step=3788.05
[2026-09-05 03:15] monitor: trainer pid=17993 alive; epoch=56 advanced to step=5940 loss=0.600065 (loss_full=0.263977, loss_base=0.336088); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint remains step=00005000.
[2026-09-05 03:16] monitor: trainer pid=17993 remains active in epoch=56 after step=5940; GPU memory=41410 MiB; no fatal/error output detected (next periodic log pending).
task=ci2v epoch=56 step=5960 loss=0.785310 loss_full=0.373115 loss_base=0.412194 ms_per_step=3809.44
[2026-09-05 03:16] monitor: trainer pid=17993 alive; epoch=56 advanced to step=5960 loss=0.785310 (loss_full=0.373115, loss_base=0.412194); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint remains step=00005000.
task=ci2v epoch=56 step=5980 loss=0.681129 loss_full=0.307513 loss_base=0.373616 ms_per_step=3794.81
[2026-09-05 03:17] monitor: trainer pid=17993 alive; epoch=56 advanced to step=5980 loss=0.681129 (loss_full=0.307513, loss_base=0.373616); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint remains step=00005000.
[epoch-end] epoch=56 elapsed=400.02s step=5985
[epoch-start] epoch=57 step=5985 steps_per_epoch=105
[2026-09-05 03:18] monitor: trainer pid=17993 alive; epoch=56 completed at step=5985 and epoch=57 started; latest loss at step=5980 is 0.681129 (loss_full=0.307513, loss_base=0.373616); GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=57 step=6000 loss=0.664867 loss_full=0.313415 loss_base=0.351451 ms_per_step=3808.47
[profile] window=380.35s updates=100 data_wait=2.3ms h2d=1.7ms stage1=295.8ms dit=644.9ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=6000 elapsed=2.81s
[2026-09-05 03:19] monitor: trainer pid=17993 alive; epoch=57 advanced to step=6000 loss=0.664867 (loss_full=0.313415, loss_base=0.351451); sample step-00006000.pt written; checkpoint step-00006000.pt is currently being written (temporary file present); GPU memory=41410 MiB; no fatal/error output detected.
[checkpoint] step=6000 elapsed=36.56s
[2026-09-05 03:19] monitor: checkpoint step-00006000.pt completed successfully (21,885,166,392 bytes); trainer pid=17993 remains alive in epoch=57; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:21] monitor: trainer pid=17993 remains active; epoch=57 advanced to step=6020 loss=0.698788 (loss_full=0.314884, loss_base=0.383904); checkpoint step-00006000.pt remains finalized; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:22] monitor: trainer pid=17993 remains active; epoch=57 advanced to step=6040 loss=0.662586 (loss_full=0.298633, loss_base=0.363953); checkpoint step-00006000.pt remains finalized; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:23] monitor: trainer pid=17993 remains active in epoch=57 after step=6040; current logging interval is still within the observed ~3.8s/step cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:23] monitor: trainer pid=17993 remains active; epoch=57 advanced to step=6060 loss=0.766349 (loss_full=0.362276, loss_base=0.404073); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:24] monitor: trainer pid=17993 remains active; epoch=57 advanced to step=6060 (latest runtime record) with loss=0.766349; checkpoint step-00006000.pt remains finalized; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:25] monitor: trainer pid=17993 remains active; epoch=57 advanced to step=6080 loss=0.605013 (loss_full=0.269598, loss_base=0.335414); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:25] monitor: trainer pid=17993 completed epoch=57 at step=6090 (elapsed=432.91s) and started epoch=58 with 105 steps; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:25] monitor: trainer pid=17993 remains active in epoch=58 at step=6090; no new periodic loss record yet (normal interval); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:26] monitor: trainer pid=17993 remains active; epoch=58 advanced to step=6100 loss=0.757103 (loss_full=0.359172, loss_base=0.397931); profile.json updated; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:26] monitor: trainer pid=17993 remains active in epoch=58 after step=6100; current interval is within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:26] monitor: trainer pid=17993 remains active in epoch=58 after step=6100; no newer periodic record yet, within the observed ~3.8s/step cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:27] monitor: trainer pid=17993 remains active; epoch=58 advanced to step=6120 loss=0.677785 (loss_full=0.303075, loss_base=0.374710); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:28] monitor: trainer pid=17993 remains active; epoch=58 advanced to step=6140 loss=0.708669 (loss_full=0.316863, loss_base=0.391806); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:29] monitor: trainer pid=17993 remains active; epoch=58 advanced to step=6140 (latest runtime record) with loss=0.708669; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:29] monitor: trainer pid=17993 remains active in epoch=58 after step=6140; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:30] monitor: trainer pid=17993 remains active; epoch=58 advanced to step=6160 loss=0.671234 (loss_full=0.308224, loss_base=0.363010); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:30] monitor: trainer pid=17993 remains active in epoch=58 after step=6160; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:30] monitor: trainer pid=17993 remains active in epoch=58 after step=6160; no newer periodic record yet, still within normal ~3.8s/step cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:31] monitor: trainer pid=17993 remains active; epoch=58 advanced to step=6180 loss=0.718206 (loss_full=0.324991, loss_base=0.393215); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:31] monitor: trainer pid=17993 remains active in epoch=58 after step=6180; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:32] monitor: trainer pid=17993 completed epoch=58 at step=6195 (elapsed=392.13s) and started epoch=59 with 105 steps; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:32] monitor: trainer pid=17993 remains active; epoch=59 advanced to step=6200 loss=0.794262 (loss_full=0.363894, loss_base=0.430367); profile.json updated; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:33] monitor: trainer pid=17993 remains active; epoch=59 advanced to step=6220 loss=0.695638 (loss_full=0.310243, loss_base=0.385395); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:33] monitor: trainer pid=17993 remains active in epoch=59 after step=6200; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:34] monitor: trainer pid=17993 remains active in epoch=59 after step=6220; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:35] monitor: trainer pid=17993 remains active; epoch=59 advanced to step=6240 loss=0.689144 (loss_full=0.319744, loss_base=0.369400); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:35] monitor: trainer pid=17993 remains active in epoch=59 after step=6240; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:36] monitor: trainer pid=17993 remains active; epoch=59 advanced to step=6260 loss=0.801340 (loss_full=0.360213, loss_base=0.441127); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:37] monitor: trainer pid=17993 remains active in epoch=59 after step=6260; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:37] monitor: trainer pid=17993 remains active; epoch=59 advanced to step=6280 loss=0.741341 (loss_full=0.338673, loss_base=0.402668); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:38] monitor: trainer pid=17993 remains active in epoch=59 after step=6280; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:39] monitor: trainer pid=17993 completed epoch=59 at step=6300 (elapsed=398.91s) and started epoch=60 with 105 steps; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:39] monitor: trainer pid=17993 remains active in epoch=60 at step=6300; no fatal/error output detected; GPU memory=41410 MiB.
[2026-09-05 03:39] monitor: trainer pid=17993 remains active in epoch=60 at step=6300; no new loss record yet, normal epoch-start interval; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:40] monitor: trainer pid=17993 remains active; epoch=60 advanced to step=6320 loss=0.678968 (loss_full=0.313542, loss_base=0.365426); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:40] monitor: trainer pid=17993 remains active in epoch=60 after step=6320; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:41] monitor: trainer pid=17993 remains active; epoch=60 advanced to step=6340 loss=0.673445 (loss_full=0.298206, loss_base=0.375239); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:41] monitor: trainer pid=17993 remains active in epoch=60 after step=6320; no newer periodic record yet, still within normal ~3.8s/step cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:38] monitor: trainer pid=17993 remains active in epoch=59 after step=6280; no newer periodic record yet, still within normal ~3.8s/step cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:37] monitor: trainer pid=17993 remains active in epoch=59 after step=6260; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:36] monitor: trainer pid=17993 remains active in epoch=59 after step=6240; no newer periodic record yet, still within normal ~3.8s/step cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:34] monitor: trainer pid=17993 remains active in epoch=59 after step=6220; no newer periodic record yet, still within normal ~3.8s/step cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:35] monitor: trainer pid=17993 remains active in epoch=59 after step=6220; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:31] monitor: trainer pid=17993 remains active in epoch=58 after step=6180; no newer periodic record yet, still within normal ~3.8s/step cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:27] monitor: trainer pid=17993 remains active in epoch=58 after step=6120; no newer periodic record yet, within normal step cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:28] monitor: trainer pid=17993 remains active in epoch=58 after step=6120; no newer periodic record yet, still within normal ~3.8s/step logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:27] monitor: trainer pid=17993 remains active; epoch=58 advanced to step=6120 loss=0.677785 (loss_full=0.303075, loss_base=0.374710); GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=57 step=6020 loss=0.698788 loss_full=0.314884 loss_base=0.383904 ms_per_step=3786.91
task=ci2v epoch=57 step=6040 loss=0.662586 loss_full=0.298633 loss_base=0.363953 ms_per_step=3814.38
task=ci2v epoch=57 step=6060 loss=0.766349 loss_full=0.362276 loss_base=0.404073 ms_per_step=3601.91
task=ci2v epoch=57 step=6080 loss=0.605013 loss_full=0.269598 loss_base=0.335414 ms_per_step=3724.59
[epoch-end] epoch=57 elapsed=432.91s step=6090
[epoch-start] epoch=58 step=6090 steps_per_epoch=105
task=ci2v epoch=58 step=6100 loss=0.757103 loss_full=0.359172 loss_base=0.397931 ms_per_step=3675.27
[profile] window=372.32s updates=100 data_wait=2.2ms h2d=1.5ms stage1=277.6ms dit=643.3ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=58 step=6120 loss=0.677785 loss_full=0.303075 loss_base=0.374710 ms_per_step=3561.91
task=ci2v epoch=58 step=6140 loss=0.708669 loss_full=0.316863 loss_base=0.391806 ms_per_step=3790.19
task=ci2v epoch=58 step=6160 loss=0.671234 loss_full=0.308224 loss_base=0.363010 ms_per_step=3784.06
task=ci2v epoch=58 step=6180 loss=0.718206 loss_full=0.324991 loss_base=0.393215 ms_per_step=3800.26
[epoch-end] epoch=58 elapsed=392.13s step=6195
[epoch-start] epoch=59 step=6195 steps_per_epoch=105
task=ci2v epoch=59 step=6200 loss=0.794262 loss_full=0.363894 loss_base=0.430367 ms_per_step=3806.10
[profile] window=375.12s updates=100 data_wait=2.3ms h2d=1.6ms stage1=284.3ms dit=643.4ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=59 step=6220 loss=0.695638 loss_full=0.310243 loss_base=0.385395 ms_per_step=3788.86
task=ci2v epoch=59 step=6240 loss=0.689144 loss_full=0.319744 loss_base=0.369400 ms_per_step=3802.20
task=ci2v epoch=59 step=6260 loss=0.801340 loss_full=0.360213 loss_base=0.441127 ms_per_step=3818.07
task=ci2v epoch=59 step=6280 loss=0.741341 loss_full=0.338673 loss_base=0.402668 ms_per_step=3775.32
task=ci2v epoch=59 step=6300 loss=0.767319 loss_full=0.355245 loss_base=0.412073 ms_per_step=3743.80
[profile] window=378.83s updates=100 data_wait=1.0ms h2d=1.7ms stage1=292.3ms dit=646.1ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[epoch-end] epoch=59 elapsed=398.91s step=6300
[epoch-start] epoch=60 step=6300 steps_per_epoch=105
task=ci2v epoch=60 step=6320 loss=0.678968 loss_full=0.313542 loss_base=0.365426 ms_per_step=3618.31
task=ci2v epoch=60 step=6340 loss=0.673445 loss_full=0.298206 loss_base=0.375239 ms_per_step=3787.65
task=ci2v epoch=60 step=6360 loss=0.691104 loss_full=0.317122 loss_base=0.373982 ms_per_step=3781.93
[2026-09-05 03:42] monitor: trainer pid=17993 remains active; epoch=60 advanced to step=6360 loss=0.691104 (loss_full=0.317122, loss_base=0.373982); GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=60 step=6380 loss=0.660587 loss_full=0.285128 loss_base=0.375459 ms_per_step=3783.49
[2026-09-05 03:43] monitor: trainer pid=17993 remains active in epoch=60 after step=6360; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:43] monitor: trainer pid=17993 remains active; epoch=60 advanced to step=6380 loss=0.660587 (loss_full=0.285128, loss_base=0.375459); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:44] monitor: trainer pid=17993 remains active in epoch=60 after step=6380; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:44] monitor: trainer pid=17993 remains active in epoch=60 after step=6380; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=60 step=6400 loss=0.697755 loss_full=0.311966 loss_base=0.385789 ms_per_step=3627.44
[profile] window=372.23s updates=100 data_wait=2.3ms h2d=1.5ms stage1=275.3ms dit=645.1ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 03:44] monitor: trainer pid=17993 remains active; epoch=60 advanced to step=6400 loss=0.697755 (loss_full=0.311966, loss_base=0.385789); GPU memory=41410 MiB; no fatal/error output detected.
[epoch-end] epoch=60 elapsed=390.52s step=6405
[epoch-start] epoch=61 step=6405 steps_per_epoch=105
[2026-09-05 03:45] monitor: trainer pid=17993 completed epoch=60 at step=6405 (elapsed=390.52s) and started epoch=61 with 105 steps; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:45] monitor: trainer pid=17993 remains active in epoch=61 at step=6405; no periodic loss record yet, normal epoch-start interval; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=61 step=6420 loss=0.630454 loss_full=0.283498 loss_base=0.346956 ms_per_step=3792.79
[2026-09-05 03:45] monitor: trainer pid=17993 remains active; epoch=61 advanced to step=6420 loss=0.630454 (loss_full=0.283498, loss_base=0.346956); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:46] monitor: trainer pid=17993 remains active in epoch=61 after step=6420; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:46] monitor: trainer pid=17993 remains active in epoch=61 after step=6420; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=61 step=6440 loss=0.650721 loss_full=0.287614 loss_base=0.363107 ms_per_step=3620.47
[2026-09-05 03:46] monitor: trainer pid=17993 remains active in epoch=61 after step=6420; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:47] monitor: trainer pid=17993 remains active; epoch=61 advanced to step=6440 loss=0.650721 (loss_full=0.287614, loss_base=0.363107); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:47] monitor: trainer pid=17993 remains active in epoch=61 after step=6440; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:48] monitor: trainer pid=17993 remains active in epoch=61 after step=6440; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=61 step=6460 loss=0.631630 loss_full=0.274446 loss_base=0.357184 ms_per_step=3708.34
[2026-09-05 03:48] monitor: trainer pid=17993 remains active; epoch=61 advanced to step=6460 loss=0.631630 (loss_full=0.274446, loss_base=0.357184); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:48] monitor: trainer pid=17993 remains active in epoch=61 after step=6460; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:49] monitor: trainer pid=17993 remains active in epoch=61 after step=6460; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=61 step=6480 loss=0.674355 loss_full=0.307428 loss_base=0.366927 ms_per_step=3784.44
[2026-09-05 03:49] monitor: trainer pid=17993 remains active; epoch=61 advanced to step=6480 loss=0.674355 (loss_full=0.307428, loss_base=0.366927); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:50] monitor: trainer pid=17993 remains active in epoch=61 after step=6480; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:50] monitor: trainer pid=17993 remains active in epoch=61 after step=6480; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=61 step=6500 loss=0.678197 loss_full=0.315660 loss_base=0.362537 ms_per_step=3779.03
[profile] window=373.97s updates=100 data_wait=1.9ms h2d=1.6ms stage1=281.6ms dit=643.3ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 03:50] monitor: trainer pid=17993 remains active; epoch=61 advanced to step=6500 loss=0.678197 (loss_full=0.315660, loss_base=0.362537); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:51] monitor: trainer pid=17993 remains active in epoch=61 after step=6500; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[epoch-end] epoch=61 elapsed=392.94s step=6510
[epoch-start] epoch=62 step=6510 steps_per_epoch=105
[2026-09-05 03:51] monitor: trainer pid=17993 completed epoch=61 at step=6510 (elapsed=392.94s) and started epoch=62 with 105 steps; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=62 step=6520 loss=0.632500 loss_full=0.281195 loss_base=0.351305 ms_per_step=3787.69
[2026-09-05 03:52] monitor: trainer pid=17993 remains active in epoch=62 at step=6510; no newer periodic loss record yet, within normal epoch-start interval; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:52] monitor: trainer pid=17993 remains active; epoch=62 advanced to step=6520 loss=0.632500 (loss_full=0.281195, loss_base=0.351305); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:52] monitor: trainer pid=17993 remains active in epoch=62 after step=6520; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=62 step=6540 loss=0.638203 loss_full=0.281002 loss_base=0.357200 ms_per_step=3779.49
[2026-09-05 03:52] monitor: trainer pid=17993 remains active in epoch=62 after step=6520; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:53] monitor: trainer pid=17993 remains active; epoch=62 advanced to step=6540 loss=0.638203 (loss_full=0.281002, loss_base=0.357200); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:54] monitor: trainer pid=17993 remains active in epoch=62 after step=6540; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=62 step=6560 loss=0.720879 loss_full=0.340096 loss_base=0.380784 ms_per_step=3784.72
[2026-09-05 03:54] monitor: trainer pid=17993 remains active in epoch=62 after step=6540; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:54] monitor: trainer pid=17993 remains active; epoch=62 advanced to step=6560 loss=0.720879 (loss_full=0.340096, loss_base=0.380784); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:54] monitor: trainer pid=17993 remains active in epoch=62 after step=6560; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:55] monitor: trainer pid=17993 remains active in epoch=62 after step=6560; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=62 step=6580 loss=0.663575 loss_full=0.299903 loss_base=0.363672 ms_per_step=3786.10
[2026-09-05 03:56] monitor: trainer pid=17993 remains active; epoch=62 advanced to step=6580 loss=0.663575 (loss_full=0.299903, loss_base=0.363672); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:56] monitor: trainer pid=17993 remains active in epoch=62 after step=6580; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=62 step=6600 loss=0.661890 loss_full=0.301323 loss_base=0.360567 ms_per_step=3730.47
[profile] window=377.63s updates=100 data_wait=2.1ms h2d=1.7ms stage1=290.0ms dit=644.0ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 03:56] monitor: trainer pid=17993 remains active in epoch=62 after step=6580; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:57] monitor: trainer pid=17993 remains active; epoch=62 advanced to step=6600 loss=0.661890 (loss_full=0.301323, loss_base=0.360567); GPU memory=41410 MiB; no fatal/error output detected.
[epoch-end] epoch=62 elapsed=396.44s step=6615
[epoch-start] epoch=63 step=6615 steps_per_epoch=105
[2026-09-05 03:57] monitor: trainer pid=17993 remains active in epoch=62 after step=6600; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=63 step=6620 loss=0.656453 loss_full=0.295002 loss_base=0.361451 ms_per_step=3785.05
[2026-09-05 03:58] monitor: trainer pid=17993 completed epoch=62 at step=6615 (elapsed=396.44s), started epoch=63, and advanced to step=6620 loss=0.656453 (loss_full=0.295002, loss_base=0.361451); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:58] monitor: trainer pid=17993 remains active in epoch=63 after step=6620; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 03:59] monitor: trainer pid=17993 remains active in epoch=63 after step=6620; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=63 step=6640 loss=0.709459 loss_full=0.332765 loss_base=0.376694 ms_per_step=3787.39
[2026-09-05 03:59] monitor: trainer pid=17993 remains active; epoch=63 advanced to step=6640 loss=0.709459 (loss_full=0.332765, loss_base=0.376694); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:00] monitor: trainer pid=17993 remains active in epoch=63 after step=6640; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:00] monitor: trainer pid=17993 remains active in epoch=63 after step=6640; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=63 step=6660 loss=0.749676 loss_full=0.342549 loss_base=0.407128 ms_per_step=3710.39
[2026-09-05 04:00] monitor: trainer pid=17993 remains active; epoch=63 advanced to step=6660 loss=0.749676 (loss_full=0.342549, loss_base=0.407128); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:00] monitor: trainer pid=17993 remains active in epoch=63 after step=6660; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:01] monitor: trainer pid=17993 remains active in epoch=63 after step=6660; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=63 step=6680 loss=0.705611 loss_full=0.326586 loss_base=0.379025 ms_per_step=3785.99
[2026-09-05 04:02] monitor: trainer pid=17993 remains active; epoch=63 advanced to step=6680 loss=0.705611 (loss_full=0.326586, loss_base=0.379025); GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=63 step=6700 loss=0.681455 loss_full=0.309366 loss_base=0.372089 ms_per_step=3783.99
[2026-09-05 04:04] monitor: trainer pid=17993 remains active; epoch=63 advanced to step=6700 loss=0.681455 (loss_full=0.309366, loss_base=0.372089); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:04] monitor: trainer pid=17993 remains active in epoch=63 after step=6700; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=63 step=6720 loss=0.695839 loss_full=0.315307 loss_base=0.380532 ms_per_step=3658.83
[epoch-end] epoch=63 elapsed=394.87s step=6720
[epoch-start] epoch=64 step=6720 steps_per_epoch=105
[2026-09-05 04:05] monitor: trainer pid=17993 remains active; epoch=63 completed at step=6720 (elapsed=394.87s), epoch=64 started; latest loss=0.695839 (loss_full=0.315307, loss_base=0.380532); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:05] monitor: trainer pid=17993 remains active in epoch=64 after step=6720; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=64 step=6740 loss=0.590965 loss_full=0.255226 loss_base=0.335739 ms_per_step=3534.58
[2026-09-05 04:06] monitor: trainer pid=17993 remains active in epoch=64 after step=6740; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=64 step=6760 loss=0.679518 loss_full=0.317744 loss_base=0.361774 ms_per_step=3795.13
[2026-09-05 04:08] monitor: trainer pid=17993 remains active; epoch=64 advanced to step=6760 loss=0.679518 (loss_full=0.317744, loss_base=0.361774); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:08] monitor: trainer pid=17993 remains active in epoch=64 after step=6760; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=64 step=6780 loss=0.620962 loss_full=0.282226 loss_base=0.338736 ms_per_step=3776.28
[2026-09-05 04:09] monitor: trainer pid=17993 remains active in epoch=64 after step=6780; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:10] monitor: trainer pid=17993 remains active in epoch=64 after step=6800; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=64 step=6820 loss=0.775192 loss_full=0.336296 loss_base=0.438896 ms_per_step=3816.39
[epoch-end] epoch=64 elapsed=388.35s step=6825
[epoch-start] epoch=65 step=6825 steps_per_epoch=105
[2026-09-05 04:11] monitor: trainer pid=17993 remains active; epoch=64 completed at step=6825 (elapsed=388.35s), epoch=65 started; latest logged loss=0.775192 (loss_full=0.336296, loss_base=0.438896); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:11] monitor: trainer pid=17993 remains active in epoch=65 after step=6825; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=65 step=6840 loss=0.692591 loss_full=0.320001 loss_base=0.372590 ms_per_step=3835.45
[2026-09-05 04:12] monitor: trainer pid=17993 remains active; epoch=65 advanced to step=6840 loss=0.692591 (loss_full=0.320001, loss_base=0.372590); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:12] monitor: trainer pid=17993 remains active in epoch=65 after step=6840; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=65 step=6860 loss=0.660567 loss_full=0.295758 loss_base=0.364809 ms_per_step=3742.65
[2026-09-05 04:13] monitor: trainer pid=17993 remains active; epoch=65 advanced to step=6860 loss=0.660567 (loss_full=0.295758, loss_base=0.364809); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:13] monitor: trainer pid=17993 remains active in epoch=65 after step=6860; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:14] monitor: trainer pid=17993 remains active in epoch=65 after step=6860; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=65 step=6880 loss=0.667011 loss_full=0.305247 loss_base=0.361764 ms_per_step=3776.33
[2026-09-05 04:15] monitor: trainer pid=17993 remains active; epoch=65 advanced to step=6880 loss=0.667011 (loss_full=0.305247, loss_base=0.361764); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:15] monitor: trainer pid=17993 remains active in epoch=65 after step=6880; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=65 step=6900 loss=0.668447 loss_full=0.302731 loss_base=0.365716 ms_per_step=3683.67
[profile] window=377.35s updates=100 data_wait=1.9ms h2d=1.7ms stage1=288.8ms dit=644.5ms update=28.2ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 04:16] monitor: trainer pid=17993 remains active; epoch=65 advanced to step=6900 loss=0.668447 (loss_full=0.302731, loss_base=0.365716); profile window healthy, GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:16] monitor: trainer pid=17993 remains active in epoch=65 after step=6900; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=65 step=6920 loss=0.561358 loss_full=0.242620 loss_base=0.318738 ms_per_step=3496.87
[profile] window=377.35s updates=100 data_wait=1.9ms h2d=1.7ms stage1=288.8ms dit=644.5ms update=28.2ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 04:17] monitor: trainer pid=17993 remains active; epoch=65 advanced to step=6920 loss=0.561358 (loss_full=0.242620, loss_base=0.318738); profile window healthy, GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=65 step=6920 loss=0.561358 loss_full=0.242620 loss_base=0.318738 ms_per_step=3496.87
[epoch-end] epoch=65 elapsed=390.43s step=6930
[epoch-start] epoch=66 step=6930 steps_per_epoch=105
[2026-09-05 04:18] monitor: trainer pid=17993 remains active; epoch=65 completed at step=6930 (elapsed=390.43s), epoch=66 started; latest loss=0.561358 (loss_full=0.242620, loss_base=0.318738); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:18] monitor: trainer pid=17993 remains active in epoch=66 after step=6930; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=66 step=6940 loss=0.677634 loss_full=0.307283 loss_base=0.370352 ms_per_step=3808.28
[2026-09-05 04:19] monitor: trainer pid=17993 remains active; epoch=66 advanced to step=6940 loss=0.677634 (loss_full=0.307283, loss_base=0.370352); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:19] monitor: trainer pid=17993 remains active in epoch=66 after step=6940; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=66 step=6960 loss=0.702336 loss_full=0.317435 loss_base=0.384901 ms_per_step=3681.13
[2026-09-05 04:20] monitor: trainer pid=17993 remains active; epoch=66 advanced to step=6960 loss=0.702336 (loss_full=0.317435, loss_base=0.384901); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:20] monitor: trainer pid=17993 remains active in epoch=66 after step=6960; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=66 step=6980 loss=0.679153 loss_full=0.303854 loss_base=0.375300 ms_per_step=3773.99
[2026-09-05 04:21] monitor: trainer pid=17993 remains active; epoch=66 advanced to step=6980 loss=0.679153 (loss_full=0.303854, loss_base=0.375300); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:21] monitor: trainer pid=17993 remains active in epoch=66 after step=6980; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:21] monitor: trainer pid=17993 remains active in epoch=66 after step=6980; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:20] monitor: trainer pid=17993 remains active in epoch=66 after step=6960; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:16] monitor: trainer pid=17993 remains active in epoch=65 after step=6900; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:15] monitor: trainer pid=17993 remains active in epoch=65 after step=6880; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:12] monitor: trainer pid=17993 remains active in epoch=65 after step=6840; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=64 step=6820 loss=0.775192 loss_full=0.336296 loss_base=0.438896 ms_per_step=3816.39
[2026-09-05 04:11] monitor: trainer pid=17993 remains active; epoch=64 advanced to step=6820 loss=0.775192 (loss_full=0.336296, loss_base=0.438896); GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=64 step=6800 loss=0.646658 loss_full=0.296331 loss_base=0.350327 ms_per_step=3589.77
[profile] window=367.35s updates=100 data_wait=1.8ms h2d=1.5ms stage1=267.0ms dit=641.9ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 04:10] monitor: trainer pid=17993 remains active; epoch=64 advanced to step=6800 loss=0.646658 (loss_full=0.296331, loss_base=0.350327); profile window healthy, GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:09] monitor: trainer pid=17993 remains active in epoch=64 after step=6780; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:08] monitor: trainer pid=17993 remains active; epoch=64 advanced to step=6780 loss=0.620962 (loss_full=0.282226, loss_base=0.338736); GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=64 step=6760 loss=0.679518 loss_full=0.317744 loss_base=0.361774 ms_per_step=3795.13
[2026-09-05 04:07] monitor: trainer pid=17993 remains active; epoch=64 advanced to step=6760 loss=0.679518 (loss_full=0.317744, loss_base=0.361774); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:06] monitor: trainer pid=17993 remains active; epoch=64 advanced to step=6740 loss=0.590965 (loss_full=0.255226, loss_base=0.335739); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:06] monitor: trainer pid=17993 remains active in epoch=64 after step=6740; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[profile] window=377.33s updates=100 data_wait=1.9ms h2d=1.7ms stage1=287.7ms dit=645.7ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=63 step=6720 loss=0.695839 loss_full=0.315307 loss_base=0.380532 ms_per_step=3658.83
[epoch-end] epoch=63 elapsed=394.87s step=6720
[epoch-start] epoch=64 step=6720 steps_per_epoch=105
task=ci2v epoch=64 step=6740 loss=0.590965 loss_full=0.255226 loss_base=0.335739 ms_per_step=3534.58
task=ci2v epoch=64 step=6760 loss=0.679518 loss_full=0.317744 loss_base=0.361774 ms_per_step=3795.13
task=ci2v epoch=64 step=6780 loss=0.620962 loss_full=0.282226 loss_base=0.338736 ms_per_step=3776.28
task=ci2v epoch=64 step=6800 loss=0.646658 loss_full=0.296331 loss_base=0.350327 ms_per_step=3589.77
[profile] window=367.35s updates=100 data_wait=1.8ms h2d=1.5ms stage1=267.0ms dit=641.9ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=64 step=6820 loss=0.775192 loss_full=0.336296 loss_base=0.438896 ms_per_step=3816.39
[epoch-end] epoch=64 elapsed=388.35s step=6825
[epoch-start] epoch=65 step=6825 steps_per_epoch=105
task=ci2v epoch=65 step=6840 loss=0.692591 loss_full=0.320001 loss_base=0.372590 ms_per_step=3835.45
task=ci2v epoch=65 step=6860 loss=0.660567 loss_full=0.295758 loss_base=0.364809 ms_per_step=3742.65
task=ci2v epoch=65 step=6880 loss=0.667011 loss_full=0.305247 loss_base=0.361764 ms_per_step=3776.33
task=ci2v epoch=65 step=6900 loss=0.668447 loss_full=0.302731 loss_base=0.365716 ms_per_step=3683.67
[profile] window=377.35s updates=100 data_wait=1.9ms h2d=1.7ms stage1=288.8ms dit=644.5ms update=28.2ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=65 step=6920 loss=0.561358 loss_full=0.242620 loss_base=0.318738 ms_per_step=3496.87
[epoch-end] epoch=65 elapsed=390.43s step=6930
[epoch-start] epoch=66 step=6930 steps_per_epoch=105
task=ci2v epoch=66 step=6940 loss=0.677634 loss_full=0.307283 loss_base=0.370352 ms_per_step=3808.28
task=ci2v epoch=66 step=6960 loss=0.702336 loss_full=0.317435 loss_base=0.384901 ms_per_step=3681.13
task=ci2v epoch=66 step=6980 loss=0.679153 loss_full=0.303854 loss_base=0.375300 ms_per_step=3773.99
task=ci2v epoch=66 step=7000 loss=0.592173 loss_full=0.260407 loss_base=0.331765 ms_per_step=3792.07
[profile] window=371.31s updates=100 data_wait=2.2ms h2d=1.5ms stage1=276.2ms dit=642.2ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=7000 elapsed=2.88s
[checkpoint] step=7000 elapsed=18.86s
[2026-09-05 04:22] monitor: trainer pid=17993 remains active; step=7000 loss=0.592173 (loss_full=0.260407, loss_base=0.331765), sample and checkpoint completed (checkpoint elapsed=18.86s); final file step-00007000.pt present, GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:23] monitor: trainer pid=17993 remains active after step=7000 checkpoint; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=66 step=7020 loss=0.708473 loss_full=0.334138 loss_base=0.374335 ms_per_step=3535.53
[2026-09-05 04:24] monitor: trainer pid=17993 remains active; epoch=66 advanced to step=7020 loss=0.708473 (loss_full=0.334138, loss_base=0.374335); GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=66 step=7020 loss=0.708473 loss_full=0.334138 loss_base=0.374335 ms_per_step=3535.53
[epoch-end] epoch=66 elapsed=412.38s step=7035
[epoch-start] epoch=67 step=7035 steps_per_epoch=105
[2026-09-05 04:25] monitor: trainer pid=17993 remains active; epoch=66 completed at step=7035 (elapsed=412.38s), epoch=67 started; latest loss=0.708473 (loss_full=0.334138, loss_base=0.374335); GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=67 step=7040 loss=0.631022 loss_full=0.277195 loss_base=0.353827 ms_per_step=3779.30
[2026-09-05 04:25] monitor: trainer pid=17993 remains active; epoch=67 advanced to step=7040 loss=0.631022 (loss_full=0.277195, loss_base=0.353827); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:25] monitor: trainer pid=17993 remains active in epoch=67 after step=7040; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:26] monitor: trainer pid=17993 remains active in epoch=67 after step=7040; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=66 step=7020 loss=0.708473 loss_full=0.334138 loss_base=0.374335 ms_per_step=3535.53
[epoch-end] epoch=66 elapsed=412.38s step=7035
[epoch-start] epoch=67 step=7035 steps_per_epoch=105
task=ci2v epoch=67 step=7040 loss=0.631022 loss_full=0.277195 loss_base=0.353827 ms_per_step=3779.30
task=ci2v epoch=67 step=7060 loss=0.681915 loss_full=0.297227 loss_base=0.384688 ms_per_step=3791.66
task=ci2v epoch=67 step=7080 loss=0.702695 loss_full=0.317064 loss_base=0.385632 ms_per_step=3787.65
task=ci2v epoch=67 step=7060 loss=0.681915 loss_full=0.297227 loss_base=0.384688 ms_per_step=3791.66
[2026-09-05 04:27] monitor: trainer pid=17993 remains active; epoch=67 advanced to step=7060 loss=0.681915 (loss_full=0.297227, loss_base=0.384688); GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=67 step=7080 loss=0.702695 loss_full=0.317064 loss_base=0.385632 ms_per_step=3787.65
[2026-09-05 04:27] monitor: trainer pid=17993 remains active; epoch=67 advanced to step=7080 loss=0.702695 (loss_full=0.317064, loss_base=0.385632); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:28] monitor: trainer pid=17993 remains active in epoch=67 after step=7080; no newer periodic record yet, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:28] monitor: trainer pid=17993 remains active in epoch=67 after step=7080; 52s since last periodic record, consistent with ~20-step logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=67 step=7100 loss=0.645272 loss_full=0.295344 loss_base=0.349928 ms_per_step=3777.41
[profile] window=373.69s updates=100 data_wait=2.1ms h2d=1.6ms stage1=279.8ms dit=644.0ms update=29.2ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=67 step=7100 loss=0.645272 loss_full=0.295344 loss_base=0.349928 ms_per_step=3777.41
[2026-09-05 04:28] monitor: trainer pid=17993 remains active; epoch=67 advanced to step=7100 loss=0.645272 (loss_full=0.295344, loss_base=0.349928); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:29] monitor: trainer pid=17993 remains active; latest periodic record epoch=67 step=7100 loss=0.645272; GPU memory=41410 MiB; no fatal/error output detected; checkpoint step=8000 not reached.
[2026-09-05 04:29] monitor: trainer pid=17993 remains active in epoch=67 at step=7100; current interval is below the normal periodic logging window; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=67 step=7120 loss=0.685694 loss_full=0.308071 loss_base=0.377623 ms_per_step=3436.43
[2026-09-05 04:29] monitor: trainer pid=17993 remains active in epoch=67 at step=7100; 40s since last periodic record, within normal ~20-step logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=67 step=7120 loss=0.685694 loss_full=0.308071 loss_base=0.377623 ms_per_step=3436.43
[2026-09-05 04:30] monitor: trainer pid=17993 remains active; epoch=67 advanced to step=7120 loss=0.685694 (loss_full=0.308071, loss_base=0.377623); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:30] monitor: trainer pid=17993 remains active; latest record epoch=67 step=7120 loss=0.685694; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
task=ci2v epoch=67 step=7140 loss=0.668170 loss_full=0.289425 loss_base=0.378745 ms_per_step=3310.42
[2026-09-05 04:30] monitor: trainer pid=17993 remains active in epoch=67 at step=7120; 24s since last periodic record, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=67 step=7140 loss=0.668170 loss_full=0.289425 loss_base=0.378745 ms_per_step=3310.42
[epoch-end] epoch=67 elapsed=382.52s step=7140
[epoch-start] epoch=68 step=7140 steps_per_epoch=105
[2026-09-05 04:31] monitor: trainer pid=17993 remains active; epoch=67 completed at step=7140 and epoch=68 started; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
[2026-09-05 04:31] monitor: trainer pid=17993 remains active; epoch=68 has started at step=7140; no periodic loss record yet (normal cadence); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:31] monitor: trainer pid=17993 remains active in epoch=68 at step=7140; no periodic loss record yet, within normal epoch-start cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=68 step=7160 loss=0.724632 loss_full=0.333865 loss_base=0.390766 ms_per_step=3790.57
task=ci2v epoch=68 step=7160 loss=0.724632 loss_full=0.333865 loss_base=0.390766 ms_per_step=3790.57
[2026-09-05 04:32] monitor: trainer pid=17993 remains active; epoch=68 advanced to step=7160 loss=0.724632 (loss_full=0.333865, loss_base=0.390766); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:32] monitor: trainer pid=17993 remains active; epoch=68 latest periodic record step=7160 loss=0.724632; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
[2026-09-05 04:32] monitor: trainer pid=17993 remains active in epoch=68 at step=7160; 20s since last periodic record, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=68 step=7180 loss=0.684873 loss_full=0.309285 loss_base=0.375587 ms_per_step=3783.99
[2026-09-05 04:33] monitor: trainer pid=17993 remains active (CPU 302%, GPU memory=41410 MiB) in epoch=68 at step=7160; last log mtime 04:32:07, consistent with one ~20-step interval; no fatal/error output detected.
task=ci2v epoch=68 step=7180 loss=0.684873 loss_full=0.309285 loss_base=0.375587 ms_per_step=3783.99
[2026-09-05 04:33] monitor: trainer pid=17993 remains active; epoch=68 advanced to step=7180 loss=0.684873 (loss_full=0.309285, loss_base=0.375587); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:34] monitor: trainer pid=17993 remains active; epoch=68 latest periodic record step=7180 loss=0.684873; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
[2026-09-05 04:34] monitor: trainer pid=17993 remains active; epoch=68 latest periodic record step=7180 loss=0.684873; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
task=ci2v epoch=68 step=7200 loss=0.627592 loss_full=0.287825 loss_base=0.339767 ms_per_step=3827.39
[profile] window=363.24s updates=100 data_wait=1.7ms h2d=1.4ms stage1=256.4ms dit=642.4ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=68 step=7200 loss=0.627592 loss_full=0.287825 loss_base=0.339767 ms_per_step=3827.39
[2026-09-05 04:34] monitor: trainer pid=17993 remains active; epoch=68 advanced to step=7200 loss=0.627592 (loss_full=0.287825, loss_base=0.339767); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:35] monitor: trainer pid=17993 remains active; epoch=68 latest periodic record step=7200 loss=0.627592; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
[2026-09-05 04:35] monitor: trainer pid=17993 remains active in epoch=68 at step=7200; 46s since last periodic record, within normal ~20-step logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:35] monitor: trainer pid=17993 remains active in epoch=68 at step=7200; 38s since last periodic record, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=68 step=7220 loss=0.721037 loss_full=0.335654 loss_base=0.385383 ms_per_step=3803.33
task=ci2v epoch=68 step=7220 loss=0.721037 loss_full=0.335654 loss_base=0.385383 ms_per_step=3803.33
[2026-09-05 04:36] monitor: trainer pid=17993 remains active; epoch=68 advanced to step=7220 loss=0.721037 (loss_full=0.335654, loss_base=0.385383); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:36] monitor: trainer pid=17993 remains active; epoch=68 latest periodic record step=7220 loss=0.721037; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
[2026-09-05 04:36] monitor: trainer pid=17993 remains active in epoch=68 at step=7220; 40s since last periodic record, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=68 step=7240 loss=0.801643 loss_full=0.361335 loss_base=0.440308 ms_per_step=3814.92
[epoch-end] epoch=68 elapsed=399.05s step=7245
[epoch-start] epoch=69 step=7245 steps_per_epoch=105
task=ci2v epoch=68 step=7240 loss=0.801643 loss_full=0.361335 loss_base=0.440308 ms_per_step=3814.92
[2026-09-05 04:37] monitor: trainer pid=17993 remains active; epoch=68 advanced to step=7240 loss=0.801643 (loss_full=0.361335, loss_base=0.440308); profile window completed normally; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=68 step=7240 loss=0.801643 loss_full=0.361335 loss_base=0.440308 ms_per_step=3814.92
[epoch-end] epoch=68 elapsed=399.05s step=7245
[epoch-start] epoch=69 step=7245 steps_per_epoch=105
[2026-09-05 04:37] monitor: trainer pid=17993 remains active; epoch=68 completed at step=7245 and epoch=69 started; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
[2026-09-05 04:38] monitor: trainer pid=17993 remains active; epoch=69 started at step=7245 and is computing; no periodic loss record yet, within normal cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=69 step=7260 loss=0.701376 loss_full=0.318590 loss_base=0.382785 ms_per_step=3663.82
task=ci2v epoch=69 step=7260 loss=0.701376 loss_full=0.318590 loss_base=0.382785 ms_per_step=3663.82
[2026-09-05 04:38] monitor: trainer pid=17993 remains active; epoch=69 advanced to step=7260 loss=0.701376 (loss_full=0.318590, loss_base=0.382785); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:39] monitor: trainer pid=17993 remains active; epoch=69 latest periodic record step=7260 loss=0.701376; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
[2026-09-05 04:39] monitor: trainer pid=17993 remains active in epoch=69 at step=7260; 37s since last periodic record, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=69 step=7280 loss=0.629399 loss_full=0.275739 loss_base=0.353660 ms_per_step=3788.32
task=ci2v epoch=69 step=7280 loss=0.629399 loss_full=0.275739 loss_base=0.353660 ms_per_step=3788.32
[2026-09-05 04:39] monitor: trainer pid=17993 remains active; epoch=69 advanced to step=7280 loss=0.629399 (loss_full=0.275739, loss_base=0.353660); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:40] monitor: trainer pid=17993 remains active; epoch=69 latest periodic record step=7280 loss=0.629399; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
[2026-09-05 04:40] monitor: trainer pid=17993 remains active in epoch=69 at step=7280; 52s since last periodic record, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=69 step=7300 loss=0.644392 loss_full=0.294437 loss_base=0.349955 ms_per_step=3788.84
[profile] window=377.52s updates=100 data_wait=2.4ms h2d=1.6ms stage1=287.7ms dit=645.7ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 04:41] monitor: trainer pid=17993 remains active in epoch=69 at step=7280; 45s since last periodic record, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=69 step=7300 loss=0.644392 loss_full=0.294437 loss_base=0.349955 ms_per_step=3788.84
[2026-09-05 04:41] monitor: trainer pid=17993 remains active; epoch=69 advanced to step=7300 loss=0.644392 (loss_full=0.294437, loss_base=0.349955); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:41] monitor: trainer pid=17993 remains active; epoch=69 latest periodic record step=7300 loss=0.644392; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
task=ci2v epoch=69 step=7320 loss=0.690764 loss_full=0.319289 loss_base=0.371475 ms_per_step=3781.90
[2026-09-05 04:42] monitor: trainer pid=17993 remains active in epoch=69 at step=7300; 39s since last periodic record, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=69 step=7320 loss=0.690764 loss_full=0.319289 loss_base=0.371475 ms_per_step=3781.90
[2026-09-05 04:42] monitor: trainer pid=17993 remains active; epoch=69 advanced to step=7320 loss=0.690764 (loss_full=0.319289, loss_base=0.371475); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:43] monitor: trainer pid=17993 remains active; epoch=69 latest periodic record step=7320 loss=0.690764; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
task=ci2v epoch=69 step=7340 loss=0.643407 loss_full=0.287084 loss_base=0.356323 ms_per_step=3797.19
[2026-09-05 04:43] monitor: trainer pid=17993 remains active in epoch=69 at step=7320; 46s since last periodic record, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=69 step=7340 loss=0.643407 loss_full=0.287084 loss_base=0.356323 ms_per_step=3797.19
[2026-09-05 04:43] monitor: trainer pid=17993 remains active; epoch=69 advanced to step=7340 loss=0.643407 (loss_full=0.287084, loss_base=0.356323); GPU memory=41410 MiB; no fatal/error output detected.
[epoch-end] epoch=69 elapsed=393.37s step=7350
[epoch-start] epoch=70 step=7350 steps_per_epoch=105
task=ci2v epoch=69 step=7340 loss=0.643407 loss_full=0.287084 loss_base=0.356323 ms_per_step=3797.19
[epoch-end] epoch=69 elapsed=393.37s step=7350
[epoch-start] epoch=70 step=7350 steps_per_epoch=105
[2026-09-05 04:44] monitor: trainer pid=17993 remains active; epoch=69 completed at step=7350 and epoch=70 started; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
task=ci2v epoch=70 step=7360 loss=0.722597 loss_full=0.340747 loss_base=0.381850 ms_per_step=3513.74
task=ci2v epoch=70 step=7360 loss=0.722597 loss_full=0.340747 loss_base=0.381850 ms_per_step=3513.74
[2026-09-05 04:44] monitor: trainer pid=17993 remains active; epoch=70 advanced to step=7360 loss=0.722597 (loss_full=0.340747, loss_base=0.381850); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:45] monitor: trainer pid=17993 remains active; epoch=70 latest periodic record step=7360 loss=0.722597; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
[2026-09-05 04:45] monitor: trainer pid=17993 remains active in epoch=70 at step=7360; 45s since last periodic record, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=70 step=7380 loss=0.656465 loss_full=0.301877 loss_base=0.354589 ms_per_step=3791.70
task=ci2v epoch=70 step=7380 loss=0.656465 loss_full=0.301877 loss_base=0.354589 ms_per_step=3791.70
[2026-09-05 04:45] monitor: trainer pid=17993 remains active; epoch=70 advanced to step=7380 loss=0.656465 (loss_full=0.301877, loss_base=0.354589); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:46] monitor: trainer pid=17993 remains active; epoch=70 latest periodic record step=7380 loss=0.656465; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
[2026-09-05 04:46] monitor: trainer pid=17993 remains active in epoch=70 at step=7380; 39s since last periodic record, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=70 step=7400 loss=0.603493 loss_full=0.272016 loss_base=0.331477 ms_per_step=3764.35
[profile] window=373.24s updates=100 data_wait=1.7ms h2d=1.6ms stage1=281.4ms dit=642.3ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 04:47] monitor: trainer pid=17993 remains active in epoch=70 at step=7380; 39s since last periodic record, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=70 step=7400 loss=0.603493 loss_full=0.272016 loss_base=0.331477 ms_per_step=3764.35
[2026-09-05 04:47] monitor: trainer pid=17993 remains active; epoch=70 advanced to step=7400 loss=0.603493 (loss_full=0.272016, loss_base=0.331477); GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 04:48] monitor: trainer pid=17993 remains active; epoch=70 latest periodic record step=7400 loss=0.603493; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
task=ci2v epoch=70 step=7420 loss=0.713940 loss_full=0.330727 loss_base=0.383213 ms_per_step=3781.09
[2026-09-05 04:48] monitor: trainer pid=17993 remains active in epoch=70 at step=7400; 44s since last periodic record, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=70 step=7420 loss=0.713940 loss_full=0.330727 loss_base=0.383213 ms_per_step=3781.09
[2026-09-05 04:48] monitor: trainer pid=17993 remains active; epoch=70 advanced to step=7420 loss=0.713940 (loss_full=0.330727, loss_base=0.383213); GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=70 step=7440 loss=0.608306 loss_full=0.265445 loss_base=0.342861 ms_per_step=3493.11
[2026-09-05 04:50] monitor: trainer pid=17993 remains active; epoch=70 advanced to step=7440 loss=0.608306 (loss_full=0.265445, loss_base=0.342861); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
[epoch-end] epoch=70 elapsed=388.04s step=7455
[epoch-start] epoch=71 step=7455 steps_per_epoch=105
task=ci2v epoch=71 step=7460 loss=0.666124 loss_full=0.301089 loss_base=0.365035 ms_per_step=3776.17
[2026-09-05 04:51] monitor: trainer pid=17993 remains active; epoch=71 advanced to step=7460 loss=0.666124 (loss_full=0.301089, loss_base=0.365035); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
[2026-09-05 04:51] monitor: trainer pid=17993 remains active in epoch=71 at step=7460; 25s since last periodic record, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
task=ci2v epoch=71 step=7480 loss=0.655796 loss_full=0.289264 loss_base=0.366532 ms_per_step=3774.18
[2026-09-05 04:52] monitor: trainer pid=17993 remains active; epoch=71 advanced to step=7480 loss=0.655796 (loss_full=0.289264, loss_base=0.366532); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
[2026-09-05 04:52] monitor: trainer pid=17993 remains active in epoch=71 at step=7480; 34s since last periodic record, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
[2026-09-05 04:53] monitor: trainer pid=17993 remains active in epoch=71 at step=7480; 20s since last periodic record, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
task=ci2v epoch=71 step=7500 loss=0.655854 loss_full=0.289431 loss_base=0.366423 ms_per_step=3812.93
[2026-09-05 04:53] monitor: trainer pid=17993 remains active; epoch=71 advanced to step=7500 loss=0.655854 (loss_full=0.289431, loss_base=0.366423); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
[2026-09-05 04:53] monitor: trainer pid=17993 remains active in epoch=71 at step=7500; 26s since last periodic record, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
[2026-09-05 04:54] monitor: trainer pid=17993 remains active in epoch=71 at step=7500; 50s since last periodic record, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
[profile] window=373.01s updates=100 data_wait=2.0ms h2d=1.6ms stage1=278.3ms dit=644.3ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=71 step=7520 loss=0.700772 loss_full=0.318340 loss_base=0.382432 ms_per_step=3781.80
[2026-09-05 04:54] monitor: trainer pid=17993 remains active; epoch=71 advanced to step=7520 loss=0.700772 (loss_full=0.318340, loss_base=0.382432); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
[2026-09-05 04:55] monitor: trainer pid=17993 remains active in epoch=71 at step=7520; 50s since last periodic record, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
task=ci2v epoch=71 step=7540 loss=0.712645 loss_full=0.313337 loss_base=0.399308 ms_per_step=3829.68
[2026-09-05 04:56] monitor: trainer pid=17993 remains active; epoch=71 advanced to step=7540 loss=0.712645 (loss_full=0.313337, loss_base=0.399308); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
task=ci2v epoch=71 step=7560 loss=0.545923 loss_full=0.228194 loss_base=0.317728 ms_per_step=3626.75
[epoch-end] epoch=71 elapsed=396.90s step=7560
[epoch-start] epoch=72 step=7560 steps_per_epoch=105
[2026-09-05 04:57] monitor: trainer pid=17993 completed epoch=71 and started epoch=72 at step=7560; final epoch loss=0.545923; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
[2026-09-05 04:58] monitor: trainer pid=17993 remains active in epoch=72 at step=7560; 47s since last periodic record, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
task=ci2v epoch=72 step=7580 loss=0.673540 loss_full=0.305683 loss_base=0.367858 ms_per_step=3868.97
[2026-09-05 04:58] monitor: trainer pid=17993 remains active; epoch=72 advanced to step=7580 loss=0.673540 (loss_full=0.305683, loss_base=0.367858); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
[2026-09-05 04:59] monitor: trainer pid=17993 remains active in epoch=72 at step=7580; 26s since last periodic record, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
task=ci2v epoch=72 step=7600 loss=0.622493 loss_full=0.276944 loss_base=0.345549 ms_per_step=3784.47
[2026-09-05 04:59] monitor: trainer pid=17993 remains active; epoch=72 advanced to step=7600 loss=0.622493 (loss_full=0.276944, loss_base=0.345549); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
[2026-09-05 05:00] monitor: trainer pid=17993 remains active in epoch=72 at step=7600; 34s since last periodic record, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
[profile] window=378.10s updates=100 data_wait=2.2ms h2d=1.6ms stage1=288.9ms dit=646.1ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=72 step=7620 loss=0.705416 loss_full=0.316979 loss_base=0.388437 ms_per_step=3794.76
[2026-09-05 05:01] monitor: trainer pid=17993 remains active; epoch=72 advanced to step=7620 loss=0.705416 (loss_full=0.316979, loss_base=0.388437); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
[2026-09-05 05:02] monitor: trainer pid=17993 remains active in epoch=72 at step=7620; 48s since last periodic record, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
task=ci2v epoch=72 step=7640 loss=0.683812 loss_full=0.307671 loss_base=0.376141 ms_per_step=3820.30
[2026-09-05 05:02] monitor: trainer pid=17993 remains active; epoch=72 advanced to step=7640 loss=0.683812 (loss_full=0.307671, loss_base=0.376141); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
task=ci2v epoch=72 step=7660 loss=0.661107 loss_full=0.296966 loss_base=0.364141 ms_per_step=3812.03
[2026-09-05 05:03] monitor: trainer pid=17993 remains active; epoch=72 advanced to step=7660 loss=0.661107 (loss_full=0.296966, loss_base=0.364141); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
[epoch-end] epoch=72 elapsed=399.75s step=7665
[epoch-start] epoch=73 step=7665 steps_per_epoch=105
[2026-09-05 05:04] monitor: trainer pid=17993 completed epoch=72 and started epoch=73 at step=7665; final epoch loss=0.661107; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
task=ci2v epoch=73 step=7680 loss=0.703867 loss_full=0.320700 loss_base=0.383167 ms_per_step=3800.72
[2026-09-05 05:04] monitor: trainer pid=17993 remains active; epoch=73 advanced to step=7680 loss=0.703867 (loss_full=0.320700, loss_base=0.383167); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
[2026-09-05 05:05] monitor: trainer pid=17993 remains active in epoch=73 at step=7680; 35s since last periodic record, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
task=ci2v epoch=73 step=7700 loss=0.643603 loss_full=0.276923 loss_base=0.366680 ms_per_step=3679.81
[2026-09-05 05:06] monitor: trainer pid=17993 remains active; epoch=73 advanced to step=7700 loss=0.643603 (loss_full=0.276923, loss_base=0.366680); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
[profile] window=378.41s updates=100 data_wait=2.0ms h2d=1.6ms stage1=289.8ms dit=646.2ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 05:06] monitor: trainer pid=17993 remains active; epoch=73 advanced to step=7700 loss=0.643603 (loss_full=0.276923, loss_base=0.366680); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
task=ci2v epoch=73 step=7720 loss=0.741146 loss_full=0.325607 loss_base=0.415539 ms_per_step=3780.53
[2026-09-05 05:07] monitor: trainer pid=17993 remains active; epoch=73 advanced to step=7720 loss=0.741146 (loss_full=0.325607, loss_base=0.415539); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
task=ci2v epoch=73 step=7740 loss=0.792278 loss_full=0.367972 loss_base=0.424306 ms_per_step=3588.75
[2026-09-05 05:08] monitor: trainer pid=17993 remains active; epoch=73 advanced to step=7740 loss=0.792278 (loss_full=0.367972, loss_base=0.424306); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
[2026-09-05 05:09] monitor: trainer pid=17993 remains active in epoch=73 at step=7740; 31s since last periodic record, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
task=ci2v epoch=73 step=7760 loss=0.678781 loss_full=0.316113 loss_base=0.362668 ms_per_step=3799.31
[2026-09-05 05:09] monitor: trainer pid=17993 remains active; epoch=73 advanced to step=7760 loss=0.678781 (loss_full=0.316113, loss_base=0.362668); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
[epoch-end] epoch=73 elapsed=392.82s step=7770
[epoch-start] epoch=74 step=7770 steps_per_epoch=105
[2026-09-05 05:10] monitor: trainer pid=17993 completed epoch=73 and started epoch=74 at step=7770; final epoch loss=0.678781; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
task=ci2v epoch=74 step=7780 loss=0.582027 loss_full=0.254321 loss_base=0.327706 ms_per_step=3816.99
[2026-09-05 05:11] monitor: trainer pid=17993 remains active; epoch=74 advanced to step=7780 loss=0.582027 (loss_full=0.254321, loss_base=0.327706); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
[2026-09-05 05:11] monitor: trainer pid=17993 remains active in epoch=74 at step=7780; 31s since last periodic record, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
task=ci2v epoch=74 step=7800 loss=0.668771 loss_full=0.308165 loss_base=0.360606 ms_per_step=3680.89
[profile] window=373.59s updates=100 data_wait=2.0ms h2d=1.6ms stage1=279.4ms dit=644.7ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 05:13] monitor: trainer pid=17993 remains active; epoch=74 advanced to step=7800 loss=0.668771 (loss_full/loss_base profile follows); GPU memory=41410 MiB; allocated=29.0GiB reserved=39.7GiB peak=38.0GiB; no fatal/error output detected; latest checkpoint step=7000.
task=ci2v epoch=74 step=7820 loss=0.617588 loss_full=0.272489 loss_base=0.345100 ms_per_step=3694.34
[2026-09-05 05:14] monitor: trainer pid=17993 remains active; epoch=74 advanced to step=7820 loss=0.617588 (loss_full=0.272489, loss_base=0.345100); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
task=ci2v epoch=74 step=7840 loss=0.626299 loss_full=0.279605 loss_base=0.346694 ms_per_step=3784.76
[2026-09-05 05:15] monitor: trainer pid=17993 remains active; epoch=74 advanced to step=7840 loss=0.626299 (loss_full=0.279605, loss_base=0.346694); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
task=ci2v epoch=74 step=7860 loss=0.623057 loss_full=0.275056 loss_base=0.348000 ms_per_step=3811.99
[2026-09-05 05:16] monitor: trainer pid=17993 remains active; epoch=74 advanced to step=7860 loss=0.623057 (loss_full=0.275056, loss_base=0.348000); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
[epoch-end] epoch=74 elapsed=395.13s step=7875
[epoch-start] epoch=75 step=7875 steps_per_epoch=105
[2026-09-05 05:17] monitor: trainer pid=17993 completed epoch=74 and started epoch=75 at step=7875; final epoch loss=0.623057; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
task=ci2v epoch=75 step=7880 loss=0.636976 loss_full=0.284246 loss_base=0.352730 ms_per_step=3818.36
[2026-09-05 05:17] monitor: trainer pid=17993 remains active; epoch=75 advanced to step=7880 loss=0.636976 (loss_full=0.284246, loss_base=0.352730); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
task=ci2v epoch=75 step=7900 loss=0.618738 loss_full=0.282010 loss_base=0.336728 ms_per_step=3708.23
[2026-09-05 05:18] monitor: trainer pid=17993 remains active; epoch=75 advanced to step=7900 loss=0.618738 (loss_full=0.282010, loss_base=0.336728); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
task=ci2v epoch=75 step=7880 loss=0.636976 loss_full=0.284246 loss_base=0.352730 ms_per_step=3818.36
[2026-09-05 05:19] monitor: trainer pid=17993 remains active; latest authoritative periodic record remains epoch=75 step=7900 loss=0.618738; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
[2026-09-05 05:19] monitor: trainer pid=17993 remains active at epoch=75 step=7900; 32s since last periodic record, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
task=ci2v epoch=75 step=7920 loss=0.583490 loss_full=0.253664 loss_base=0.329826 ms_per_step=3698.88
[2026-09-05 05:20] monitor: trainer pid=17993 remains active; epoch=75 advanced to step=7920 loss=0.583490 (loss_full=0.253664, loss_base=0.329826); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
task=ci2v epoch=75 step=7940 loss=0.605807 loss_full=0.268814 loss_base=0.336992 ms_per_step=3798.74
[2026-09-05 05:21] monitor: trainer pid=17993 remains active; epoch=75 advanced to step=7940 loss=0.605807 (loss_full=0.268814, loss_base=0.336992); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
[2026-09-05 05:21] monitor: trainer pid=17993 remains active; latest periodic record epoch=75 step=7940 loss=0.605807; 34s since previous periodic record, within normal cadence; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
task=ci2v epoch=75 step=7960 loss=0.615238 loss_full=0.274825 loss_base=0.340413 ms_per_step=3742.83
[2026-09-05 05:22] monitor: trainer pid=17993 remains active; epoch=75 advanced to step=7960 loss=0.615238 (loss_full=0.274825, loss_base=0.340413); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
task=ci2v epoch=75 step=7980 loss=0.691968 loss_full=0.317281 loss_base=0.374687 ms_per_step=3721.83
[epoch-end] epoch=75 elapsed=393.99s step=7980
[epoch-start] epoch=76 step=7980 steps_per_epoch=105
[2026-09-05 05:23] monitor: trainer pid=17993 completed epoch=75 and started epoch=76 at step=7980; final epoch loss=0.691968; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=7000.
task=ci2v epoch=76 step=8000 loss=0.638523 loss_full=0.286269 loss_base=0.352254 ms_per_step=3890.10
[sample] step=8000 elapsed=2.80s
[checkpoint] step=8000 elapsed=41.20s
[2026-09-05 05:25] monitor: trainer pid=17993 remains active; step=8000 checkpoint and sample completed successfully; checkpoint file is finalized; GPU memory=41410 MiB; no fatal/error output detected.
[2026-09-05 05:26] monitor: trainer pid=17993 remains active after step=8000 checkpoint; checkpoint and sample files remain present; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=76 step=8020 loss=0.589584 loss_full=0.256582 loss_base=0.333002 ms_per_step=3801.95
[2026-09-05 05:27] monitor: trainer pid=17993 remains active; epoch=76 advanced to step=8020 loss=0.589584 (loss_full=0.256582, loss_base=0.333002); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=8000.
[2026-09-05 05:27] monitor: trainer pid=17993 remains active after step=8020; 39s since last periodic record, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=8000.
task=ci2v epoch=76 step=8040 loss=0.685600 loss_full=0.313199 loss_base=0.372401 ms_per_step=3796.31
[2026-09-05 05:28] monitor: trainer pid=17993 remains active; epoch=76 advanced to step=8040 loss=0.685600 (loss_full=0.313199, loss_base=0.372401); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=8000.
task=ci2v epoch=76 step=8060 loss=0.675282 loss_full=0.308422 loss_base=0.366860 ms_per_step=3782.41
[2026-09-05 05:29] monitor: trainer pid=17993 remains active; epoch=76 advanced to step=8060 loss=0.675282 (loss_full=0.308422, loss_base=0.366860); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=8000.
[2026-09-05 05:30] monitor: trainer pid=17993 remains active at epoch=76 step=8060; 27s since last periodic record, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=8000.
task=ci2v epoch=76 step=8080 loss=0.642762 loss_full=0.289479 loss_base=0.353283 ms_per_step=3828.31
[epoch-end] epoch=76 elapsed=444.10s step=8085
[epoch-start] epoch=77 step=8085 steps_per_epoch=105
[2026-09-05 05:31] monitor: trainer pid=17993 completed epoch=76 and started epoch=77 at step=8085; final epoch loss=0.642762; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=8000.
task=ci2v epoch=77 step=8100 loss=0.641805 loss_full=0.285165 loss_base=0.356640 ms_per_step=3577.11
[2026-09-05 05:31] monitor: trainer pid=17993 remains active; epoch=77 advanced to step=8100 loss=0.641805 (loss_full=0.285165, loss_base=0.356640); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=8000.
[2026-09-05 05:32] monitor: trainer pid=17993 remains active at epoch=77 step=8100; 21s since last periodic record, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=8000.
task=ci2v epoch=77 step=8120 loss=0.596175 loss_full=0.263240 loss_base=0.332934 ms_per_step=3776.42
[2026-09-05 05:33] monitor: trainer pid=17993 remains active; epoch=77 advanced to step=8120 loss=0.596175 (loss_full=0.263240, loss_base=0.332934); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=8000.
[2026-09-05 05:33] monitor: trainer pid=17993 remains active at epoch=77 step=8120; 39s since last periodic record, within normal logging cadence; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=8000.
task=ci2v epoch=77 step=8140 loss=0.679546 loss_full=0.309933 loss_base=0.369613 ms_per_step=3771.74
[2026-09-05 05:34] monitor: trainer pid=17993 remains active; epoch=77 advanced to step=8140 loss=0.679546 (loss_full=0.309933, loss_base=0.369613); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=8000.
task=ci2v epoch=77 step=8160 loss=0.676145 loss_full=0.310243 loss_base=0.365901 ms_per_step=3803.52
[2026-09-05 05:36] monitor: trainer pid=17993 remains active; epoch=77 advanced to step=8160 loss=0.676145 (loss_full=0.310243, loss_base=0.365901); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=8000.
task=ci2v epoch=77 step=8160 loss=0.676145 loss_full=0.310243 loss_base=0.365901 ms_per_step=3803.52
[2026-09-05 05:36] monitor: trainer pid=17993 remains active; latest periodic record remains epoch=77 step=8160 loss=0.676145; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=8000.
task=ci2v epoch=77 step=8180 loss=0.556220 loss_full=0.238197 loss_base=0.318023 ms_per_step=3776.72
[2026-09-05 05:37] monitor: trainer pid=17993 remains active; epoch=77 advanced to step=8180 loss=0.556220 (loss_full=0.238197, loss_base=0.318023); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=8000.
task=ci2v epoch=77 step=8180 loss=0.556220 loss_full=0.238197 loss_base=0.318023 ms_per_step=3776.72
[epoch-end] epoch=77 elapsed=394.05s step=8190
[epoch-start] epoch=78 step=8190 steps_per_epoch=105
[2026-09-05 05:38] monitor: trainer pid=17993 completed epoch=77 and started epoch=78 at step=8190; final epoch loss=0.556220; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=8000.
task=ci2v epoch=78 step=8200 loss=0.575143 loss_full=0.243621 loss_base=0.331522 ms_per_step=3824.11
[2026-09-05 05:38] monitor: trainer pid=17993 remains active; epoch=78 advanced to step=8200 loss=0.575143 (loss_full=0.243621, loss_base=0.331522); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=8000.
task=ci2v epoch=78 step=8200 loss=0.575143 loss_full=0.243621 loss_base=0.331522 ms_per_step=3824.11
[2026-09-05 05:38] monitor: trainer pid=17993 remains active; epoch=78 advanced to step=8200 loss=0.575143 (loss_full=0.243621, loss_base=0.331522); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=8000.
task=ci2v epoch=76 step=8020 loss=0.589584 loss_full=0.256582 loss_base=0.333002 ms_per_step=3801.95
[2026-09-05 05:26] monitor: trainer pid=17993 remains active; epoch=76 advanced to step=8020 loss=0.589584 (loss_full=0.256582, loss_base=0.333002); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint step=8000.
task=ci2v epoch=74 step=7820 loss=0.617588 loss_full=0.272489 loss_base=0.345100 ms_per_step=3694.34
task=ci2v epoch=74 step=7840 loss=0.626299 loss_full=0.279605 loss_base=0.346694 ms_per_step=3784.76
task=ci2v epoch=74 step=7860 loss=0.623057 loss_full=0.275056 loss_base=0.348000 ms_per_step=3811.99
[epoch-end] epoch=74 elapsed=395.13s step=7875
[epoch-start] epoch=75 step=7875 steps_per_epoch=105
task=ci2v epoch=75 step=7880 loss=0.636976 loss_full=0.284246 loss_base=0.352730 ms_per_step=3818.36
task=ci2v epoch=75 step=7900 loss=0.618738 loss_full=0.282010 loss_base=0.336728 ms_per_step=3708.23
[profile] window=376.60s updates=100 data_wait=1.9ms h2d=1.6ms stage1=285.5ms dit=646.3ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=75 step=7920 loss=0.583490 loss_full=0.253664 loss_base=0.329826 ms_per_step=3698.88
task=ci2v epoch=75 step=7940 loss=0.605807 loss_full=0.268814 loss_base=0.336992 ms_per_step=3798.74
task=ci2v epoch=75 step=7960 loss=0.615238 loss_full=0.274825 loss_base=0.340413 ms_per_step=3742.83
task=ci2v epoch=75 step=7980 loss=0.691968 loss_full=0.317281 loss_base=0.374687 ms_per_step=3721.83
[epoch-end] epoch=75 elapsed=393.99s step=7980
[epoch-start] epoch=76 step=7980 steps_per_epoch=105
task=ci2v epoch=76 step=8000 loss=0.638523 loss_full=0.286269 loss_base=0.352254 ms_per_step=3890.10
[profile] window=377.30s updates=100 data_wait=2.1ms h2d=1.6ms stage1=284.8ms dit=648.3ms update=27.8ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=8000 elapsed=2.80s
[checkpoint] step=8000 elapsed=41.20s
task=ci2v epoch=76 step=8020 loss=0.589584 loss_full=0.256582 loss_base=0.333002 ms_per_step=3801.95
task=ci2v epoch=76 step=8040 loss=0.685600 loss_full=0.313199 loss_base=0.372401 ms_per_step=3796.31
task=ci2v epoch=76 step=8060 loss=0.675282 loss_full=0.308422 loss_base=0.366860 ms_per_step=3782.41
task=ci2v epoch=76 step=8080 loss=0.642762 loss_full=0.289479 loss_base=0.353283 ms_per_step=3828.31
[epoch-end] epoch=76 elapsed=444.10s step=8085
[epoch-start] epoch=77 step=8085 steps_per_epoch=105
task=ci2v epoch=77 step=8100 loss=0.641805 loss_full=0.285165 loss_base=0.356640 ms_per_step=3577.11
[profile] window=375.97s updates=100 data_wait=1.9ms h2d=1.6ms stage1=285.1ms dit=645.0ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=77 step=8120 loss=0.596175 loss_full=0.263240 loss_base=0.332934 ms_per_step=3776.42
task=ci2v epoch=77 step=8140 loss=0.679546 loss_full=0.309933 loss_base=0.369613 ms_per_step=3771.74
task=ci2v epoch=77 step=8160 loss=0.676145 loss_full=0.310243 loss_base=0.365901 ms_per_step=3803.52
task=ci2v epoch=77 step=8180 loss=0.556220 loss_full=0.238197 loss_base=0.318023 ms_per_step=3776.72
[epoch-end] epoch=77 elapsed=394.05s step=8190
[epoch-start] epoch=78 step=8190 steps_per_epoch=105
task=ci2v epoch=78 step=8200 loss=0.575143 loss_full=0.243621 loss_base=0.331522 ms_per_step=3824.11
[profile] window=379.30s updates=100 data_wait=2.2ms h2d=1.7ms stage1=291.2ms dit=647.0ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=78 step=8220 loss=0.657473 loss_full=0.297084 loss_base=0.360389 ms_per_step=3789.68
[2026-09-05 05:40] monitor: trainer pid=17993 remains active; epoch=78 advanced to step=8220 loss=0.657473 (loss_full=0.297084, loss_base=0.360389, ms_per_step=3789.68); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
task=ci2v epoch=78 step=8240 loss=0.624587 loss_full=0.274960 loss_base=0.349627 ms_per_step=3787.79
[2026-09-05 05:41] monitor: trainer pid=17993 remains active; epoch=78 advanced to step=8240 loss=0.624587 (loss_full=0.274960, loss_base=0.349627, ms_per_step=3787.79); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
task=ci2v epoch=78 step=8260 loss=0.614768 loss_full=0.263603 loss_base=0.351165 ms_per_step=3791.68
[2026-09-05 05:42] monitor: trainer pid=17993 remains active; epoch=78 advanced to step=8260 loss=0.614768 (loss_full=0.263603, loss_base=0.351165, ms_per_step=3791.68); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
task=ci2v epoch=78 step=8280 loss=0.632455 loss_full=0.280252 loss_base=0.352203 ms_per_step=3785.87
[2026-09-05 05:43] monitor: trainer pid=17993 remains active; epoch=78 advanced to step=8280 loss=0.632455 (loss_full=0.280252, loss_base=0.352203, ms_per_step=3785.87); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
[epoch-end] epoch=78 elapsed=397.16s step=8295
[epoch-start] epoch=79 step=8295 steps_per_epoch=105
[2026-09-05 05:44] monitor: trainer pid=17993 completed epoch=78 (elapsed=397.16s, step=8295) and started epoch=79; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
task=ci2v epoch=79 step=8300 loss=0.642369 loss_full=0.298122 loss_base=0.344247 ms_per_step=3725.12
[2026-09-05 05:45] monitor: trainer pid=17993 remains active; epoch=79 advanced to step=8300 loss=0.642369 (loss_full=0.298122, loss_base=0.344247, ms_per_step=3725.12); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
[profile] window=377.87s updates=100 data_wait=2.2ms h2d=1.7ms stage1=288.7ms dit=645.8ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 05:45] monitor: trainer pid=17993 remains active in epoch=79 at step=8300; profile window completed normally; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
task=ci2v epoch=79 step=8320 loss=0.663411 loss_full=0.301343 loss_base=0.362068 ms_per_step=3790.65
[2026-09-05 05:45] monitor: trainer pid=17993 remains active; epoch=79 advanced to step=8320 loss=0.663411 (loss_full=0.301343, loss_base=0.362068, ms_per_step=3790.65); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
task=ci2v epoch=79 step=8340 loss=0.708178 loss_full=0.321868 loss_base=0.386310 ms_per_step=3692.66
[2026-09-05 05:47] monitor: trainer pid=17993 remains active; epoch=79 advanced to step=8340 loss=0.708178 (loss_full=0.321868, loss_base=0.386310, ms_per_step=3692.66); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
task=ci2v epoch=79 step=8360 loss=0.620855 loss_full=0.269789 loss_base=0.351066 ms_per_step=3491.08
[2026-09-05 05:48] monitor: trainer pid=17993 remains active; epoch=79 at step=8360 loss=0.620855 (loss_full=0.269789, loss_base=0.351066); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
[2026-09-05 05:48] monitor: trainer pid=17993 remains active; epoch=79 advanced to step=8360 loss=0.620855 (loss_full=0.269789, loss_base=0.351066, ms_per_step=3491.08); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
task=ci2v epoch=79 step=8380 loss=0.679212 loss_full=0.309546 loss_base=0.369667 ms_per_step=3803.82
[2026-09-05 05:50] monitor: trainer pid=17993 remains active; epoch=79 at step=8380 loss=0.679212 (loss_full=0.309546, loss_base=0.369667); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
[2026-09-05 05:49] monitor: trainer pid=17993 remains active; epoch=79 advanced to step=8380 loss=0.679212 (loss_full=0.309546, loss_base=0.369667, ms_per_step=3803.82); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
task=ci2v epoch=79 step=8400 loss=0.626748 loss_full=0.280418 loss_base=0.346330 ms_per_step=3780.61
[profile] window=371.43s updates=100 data_wait=1.1ms h2d=1.5ms stage1=275.2ms dit=644.7ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[epoch-end] epoch=79 elapsed=391.56s step=8400
[epoch-start] epoch=80 step=8400 steps_per_epoch=105
[2026-09-05 05:51] monitor: trainer pid=17993 remains active at epoch=80 step=8400; epoch transition completed normally; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
[2026-09-05 05:51] monitor: trainer pid=17993 completed epoch=79 (elapsed=391.56s, step=8400) and started epoch=80; latest checkpoint remains step-00008000.pt (next periodic save at step 9000); GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=80 step=8420 loss=0.655306 loss_full=0.291174 loss_base=0.364132 ms_per_step=3858.98
[2026-09-05 05:52] monitor: trainer pid=17993 remains active; epoch=80 at step=8420 loss=0.655306 (loss_full=0.291174, loss_base=0.364132, ms_per_step=3858.98); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
[2026-09-05 05:52] monitor: trainer pid=17993 remains active; epoch=80 advanced to step=8420 loss=0.655306 (loss_full=0.291174, loss_base=0.364132, ms_per_step=3858.98); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
task=ci2v epoch=80 step=8440 loss=0.583745 loss_full=0.252805 loss_base=0.330940 ms_per_step=3793.41
[2026-09-05 05:54] monitor: trainer pid=17993 remains active; epoch=80 advanced to step=8440 loss=0.583745 (loss_full=0.252805, loss_base=0.330940, ms_per_step=3793.41); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
[2026-09-05 05:53] monitor: trainer pid=17993 remains active; epoch=80 advanced to step=8440 loss=0.583745 (loss_full=0.252805, loss_base=0.330940, ms_per_step=3793.41); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
task=ci2v epoch=80 step=8460 loss=0.707084 loss_full=0.328281 loss_base=0.378802 ms_per_step=3708.55
[2026-09-05 05:55] monitor: trainer pid=17993 remains active; epoch=80 at step=8460 loss=0.707084 (loss_full=0.328281, loss_base=0.378802); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
[2026-09-05 05:54] monitor: trainer pid=17993 remains active; epoch=80 advanced to step=8460 loss=0.707084 (loss_full=0.328281, loss_base=0.378802, ms_per_step=3708.55); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
task=ci2v epoch=80 step=8480 loss=0.660378 loss_full=0.287407 loss_base=0.372971 ms_per_step=3642.24
[2026-09-05 05:56] monitor: trainer pid=17993 remains active; epoch=80 at step=8480 loss=0.660378 (loss_full=0.287407, loss_base=0.372971); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
[2026-09-05 05:56] monitor: trainer pid=17993 remains active; epoch=80 advanced to step=8480 loss=0.660378 (loss_full=0.287407, loss_base=0.372971, ms_per_step=3642.24); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
task=ci2v epoch=80 step=8500 loss=0.740948 loss_full=0.347413 loss_base=0.393534 ms_per_step=3735.33
[profile] window=375.03s updates=100 data_wait=1.9ms h2d=1.6ms stage1=282.9ms dit=643.9ms update=30.5ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 05:57] monitor: trainer pid=17993 remains active; epoch=80 advanced to step=8500 loss=0.740948 (loss_full=0.347413, loss_base=0.393534, ms_per_step=3735.33); profile completed normally; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
[epoch-end] epoch=80 elapsed=391.68s step=8505
[epoch-start] epoch=81 step=8505 steps_per_epoch=105
[2026-09-05 05:58] monitor: trainer pid=17993 remains active at epoch=81 step=8505; epoch=80 completed normally; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
[2026-09-05 05:57] monitor: trainer pid=17993 completed epoch=80 (elapsed=391.68s, step=8505) and started epoch=81; latest checkpoint remains step-00008000.pt; GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=81 step=8520 loss=0.681634 loss_full=0.317123 loss_base=0.364511 ms_per_step=3403.01
[2026-09-05 05:59] monitor: trainer pid=17993 remains active; epoch=81 at step=8520 loss=0.681634 (loss_full=0.317123, loss_base=0.364511); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
[2026-09-05 05:58] monitor: trainer pid=17993 remains active; epoch=81 advanced to step=8520 loss=0.681634 (loss_full=0.317123, loss_base=0.364511, ms_per_step=3403.01); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
[2026-09-05 05:58] monitor: trainer pid=17993 remains active; epoch=81 advanced to step=8520 loss=0.681634 (loss_full=0.317123, loss_base=0.364511, ms_per_step=3403.01); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
task=ci2v epoch=81 step=8540 loss=0.604974 loss_full=0.271545 loss_base=0.333429 ms_per_step=3760.95
[2026-09-05 06:00] monitor: trainer pid=17993 remains active; epoch=81 at step=8540 loss=0.604974 (loss_full=0.271545, loss_base=0.333429); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
[2026-09-05 05:59] monitor: trainer pid=17993 remains active; epoch=81 advanced to step=8540 loss=0.604974 (loss_full=0.271545, loss_base=0.333429, ms_per_step=3760.95); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
task=ci2v epoch=81 step=8560 loss=0.630987 loss_full=0.270656 loss_base=0.360331 ms_per_step=3749.04
[2026-09-05 06:01] monitor: trainer pid=17993 remains active; epoch=81 advanced to step=8560 loss=0.630987 (loss_full=0.270656, loss_base=0.360331, ms_per_step=3749.04); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
task=ci2v epoch=81 step=8580 loss=0.679199 loss_full=0.314377 loss_base=0.364822 ms_per_step=3806.97
[2026-09-05 06:02] monitor: trainer pid=17993 remains active; epoch=81 at step=8580 loss=0.679199 (loss_full=0.314377, loss_base=0.364822); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
[2026-09-05 06:02] monitor: trainer pid=17993 remains active; epoch=81 at step=8580 loss=0.679199 (loss_full=0.314377, loss_base=0.364822); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
[2026-09-05 06:01] monitor: trainer pid=17993 remains active; epoch=81 advanced to step=8580 loss=0.679199 (loss_full=0.314377, loss_base=0.364822, ms_per_step=3806.97); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
task=ci2v epoch=81 step=8600 loss=0.617393 loss_full=0.279079 loss_base=0.338315 ms_per_step=3569.02
[profile] window=366.04s updates=100 data_wait=2.1ms h2d=1.4ms stage1=262.0ms dit=643.3ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 06:03] monitor: trainer pid=17993 remains active; epoch=81 advanced to step=8600 loss=0.617393 (loss_full=0.279079, loss_base=0.338315, ms_per_step=3569.02); profile completed normally; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt (about 400 steps to step 9000).
[epoch-end] epoch=81 elapsed=385.97s step=8610
[epoch-start] epoch=82 step=8610 steps_per_epoch=105
[2026-09-05 06:04] monitor: trainer pid=17993 completed epoch=81 (elapsed=385.97s, step=8610) and started epoch=82; latest checkpoint remains step-00008000.pt (next periodic save at step 9000); GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=82 step=8620 loss=0.654598 loss_full=0.292582 loss_base=0.362015 ms_per_step=3622.11
[2026-09-05 06:05] monitor: trainer pid=17993 remains active; epoch=82 at step=8620 loss=0.654598 (loss_full=0.292582, loss_base=0.362015); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
[2026-09-05 06:04] monitor: trainer pid=17993 remains active; epoch=82 at step=8620 loss=0.654598 (loss_full=0.292582, loss_base=0.362015); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
[2026-09-05 06:04] monitor: trainer pid=17993 remains active; epoch=82 advanced to step=8620 loss=0.654598 (loss_full=0.292582, loss_base=0.362015, ms_per_step=3622.11); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
task=ci2v epoch=82 step=8640 loss=0.603536 loss_full=0.262899 loss_base=0.340638 ms_per_step=3799.01
[2026-09-05 06:06] monitor: trainer pid=17993 remains active; epoch=82 at step=8640 loss=0.603536 (loss_full=0.262899, loss_base=0.340638); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
[2026-09-05 06:06] monitor: trainer pid=17993 remains active; epoch=82 advanced to step=8640 loss=0.603536 (loss_full=0.262899, loss_base=0.340638, ms_per_step=3799.01); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
[2026-09-05 06:05] monitor: trainer pid=17993 remains active; epoch=82 advanced to step=8640 loss=0.603536 (loss_full=0.262899, loss_base=0.340638, ms_per_step=3799.01); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
task=ci2v epoch=82 step=8660 loss=0.649030 loss_full=0.288413 loss_base=0.360617 ms_per_step=3788.56
[2026-09-05 06:07] monitor: trainer pid=17993 remains active; epoch=82 at step=8660 loss=0.649030 (loss_full=0.288413, loss_base=0.360617); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
[2026-09-05 06:07] monitor: trainer pid=17993 remains active; epoch=82 at step=8660 loss=0.649030 (loss_full=0.288413, loss_base=0.360617); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
[2026-09-05 06:07] monitor: trainer pid=17993 remains active; epoch=82 advanced to step=8660 loss=0.649030 (loss_full=0.288413, loss_base=0.360617, ms_per_step=3788.56); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
task=ci2v epoch=82 step=8680 loss=0.669209 loss_full=0.301945 loss_base=0.367264 ms_per_step=3777.85
[2026-09-05 06:08] monitor: trainer pid=17993 remains active; epoch=82 advanced to step=8680 loss=0.669209 (loss_full=0.301945, loss_base=0.367264, ms_per_step=3777.85); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
[2026-09-05 06:08] monitor: trainer pid=17993 remains active; epoch=82 advanced to step=8680 loss=0.669209 (loss_full=0.301945, loss_base=0.367264, ms_per_step=3777.85); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
task=ci2v epoch=82 step=8700 loss=0.727056 loss_full=0.330158 loss_base=0.396898 ms_per_step=3791.72
[profile] window=375.84s updates=100 data_wait=2.1ms h2d=1.6ms stage1=285.6ms dit=644.0ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 06:09] monitor: trainer pid=17993 remains active; epoch=82 at step=8700 loss=0.727056 (loss_full=0.330158, loss_base=0.396898); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
[2026-09-05 06:09] monitor: trainer pid=17993 remains active; epoch=82 advanced to step=8700 loss=0.727056 (loss_full=0.330158, loss_base=0.396898, ms_per_step=3791.72); profile completed normally; GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt (about 300 steps to step 9000).
[epoch-end] epoch=82 elapsed=393.46s step=8715
[epoch-start] epoch=83 step=8715 steps_per_epoch=105
[2026-09-05 06:10] monitor: trainer pid=17993 completed epoch=82 (elapsed=393.46s, step=8715) and started epoch=83; latest checkpoint remains step-00008000.pt (about 285 steps to step 9000); GPU memory=41410 MiB; no fatal/error output detected.
task=ci2v epoch=83 step=8720 loss=0.615768 loss_full=0.273107 loss_base=0.342660 ms_per_step=3660.60
[2026-09-05 06:11] monitor: trainer pid=17993 remains active; epoch=83 at step=8720 loss=0.615768 (loss_full=0.273107, loss_base=0.342660); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
[2026-09-05 06:11] monitor: trainer pid=17993 remains active; epoch=83 advanced to step=8720 loss=0.615768 (loss_full=0.273107, loss_base=0.342660, ms_per_step=3660.60); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
[2026-09-05 06:10] monitor: trainer pid=17993 remains active; epoch=83 advanced to step=8720 loss=0.615768 (loss_full=0.273107, loss_base=0.342660, ms_per_step=3660.60); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
task=ci2v epoch=83 step=8740 loss=0.700564 loss_full=0.325705 loss_base=0.374860 ms_per_step=3778.65
[2026-09-05 06:12] monitor: trainer pid=17993 remains active; epoch=83 advanced to step=8740 loss=0.700564 (loss_full=0.325705, loss_base=0.374860, ms_per_step=3778.65); GPU memory=41410 MiB; no fatal/error output detected; latest checkpoint=step-00008000.pt.
task=ci2v epoch=83 step=8760 loss=0.697057 loss_full=0.308086 loss_base=0.388971 ms_per_step=3515.06
[2026-09-05 06:13] monitor: 训练仍在运行，epoch=83 step=8760，最新 loss=0.697057（loss_full=0.308086, loss_base=0.388971）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=83 step=8780 loss=0.581133 loss_full=0.246007 loss_base=0.335126 ms_per_step=3748.65
[2026-09-05 06:14] monitor: 训练仍在运行，epoch=83 step=8780，最新 loss=0.581133（loss_full=0.246007, loss_base=0.335126）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=83 step=8800 loss=0.743979 loss_full=0.349299 loss_base=0.394680 ms_per_step=3762.57
[profile] window=369.58s updates=100 data_wait=1.6ms h2d=1.5ms stage1=272.0ms dit=642.7ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 06:15] monitor: 训练仍在运行，epoch=83 step=8800，最新 loss=0.743979（loss_full=0.349299, loss_base=0.394680）；profile window=369.58s，显存 allocated=29.0GiB/reserved=39.7GiB/peak=38.0GiB；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=83 step=8820 loss=0.634425 loss_full=0.282255 loss_base=0.352170 ms_per_step=3320.14
[epoch-end] epoch=83 elapsed=382.22s step=8820
[epoch-start] epoch=84 step=8820 steps_per_epoch=105
[2026-09-05 06:16] monitor: epoch=83 已完成（elapsed=382.22s，step=8820），训练已进入 epoch=84；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=84 step=8840 loss=0.690594 loss_full=0.317245 loss_base=0.373349 ms_per_step=3633.26
[2026-09-05 06:18] monitor: epoch=84 step=8840，最新 loss=0.690594（loss_full=0.317245, loss_base=0.373349）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=84 step=8860 loss=0.695708 loss_full=0.313678 loss_base=0.382030 ms_per_step=3570.74
[2026-09-05 06:19] monitor: epoch=84 step=8860，最新 loss=0.695708（loss_full=0.313678, loss_base=0.382030）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=84 step=8880 loss=0.739722 loss_full=0.339305 loss_base=0.400418 ms_per_step=3716.10
[2026-09-05 06:20] monitor: epoch=84 step=8880，最新 loss=0.739722（loss_full=0.339305, loss_base=0.400418）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=84 step=8900 loss=0.629495 loss_full=0.284739 loss_base=0.344756 ms_per_step=3809.97
[profile] window=361.27s updates=100 data_wait=1.5ms h2d=1.3ms stage1=252.3ms dit=641.8ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 06:22] monitor: 训练进程仍存活（PID=17993），当前日志最后 step=8900，GPU 显存约 41410 MiB；最新 checkpoint 为 step-00008000.pt；错误扫描无命中，等待后续 step 9000 checkpoint。
[2026-09-05 06:22] monitor: epoch=84 step=8900，最新 loss=0.629495（loss_full=0.284739, loss_base=0.344756）；profile peak=38.0GiB，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=84 step=8920 loss=0.724921 loss_full=0.337641 loss_base=0.387280 ms_per_step=3785.22
[epoch-end] epoch=84 elapsed=388.53s step=8925
[epoch-start] epoch=85 step=8925 steps_per_epoch=105
[2026-09-05 06:23] monitor: epoch=84 已完成（elapsed=388.53s，step=8925），训练进入 epoch=85；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=85 step=8940 loss=0.668915 loss_full=0.296173 loss_base=0.372742 ms_per_step=3729.80
[2026-09-05 06:24] monitor: epoch=85 step=8940，最新 loss=0.668915（loss_full=0.296173, loss_base=0.372742）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=85 step=8960 loss=0.654595 loss_full=0.287320 loss_base=0.367274 ms_per_step=3698.78
[2026-09-05 06:25] monitor: epoch=85 step=8960，最新 loss=0.654595（loss_full=0.287320, loss_base=0.367274）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=85 step=8980 loss=0.709442 loss_full=0.320040 loss_base=0.389402 ms_per_step=3438.91
[2026-09-05 06:26] monitor: epoch=85 step=8980，最新 loss=0.709442（loss_full=0.320040, loss_base=0.389402）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=85 step=9000 loss=0.666812 loss_full=0.293314 loss_base=0.373497 ms_per_step=3662.89
[profile] window=366.58s updates=100 data_wait=2.0ms h2d=1.4ms stage1=264.1ms dit=642.5ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 06:28] monitor: epoch=85 step=9000，最新 loss=0.666812（loss_full=0.293314, loss_base=0.373497）；profile peak=38.0GiB，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中，正在复核 step-00009000 checkpoint 落盘。
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=9000 elapsed=2.92s
[checkpoint] step=9000 elapsed=35.47s
[2026-09-05 06:28] monitor: checkpoint 已生成：ckpts/libero_videogen/vjepa2_1_libero_ci2v_from_multiview_l_gc/checkpoints/step-00009000.pt（约 21.9 GB）；trainer PID=17993 仍存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=85 step=9020 loss=0.695197 loss_full=0.316815 loss_base=0.378381 ms_per_step=3803.61
[2026-09-05 06:30] monitor: epoch=85 step=9020，最新 loss=0.695197（loss_full=0.316815, loss_base=0.378381）；trainer PID=17993 存活，GPU 显存约 41410 MiB；checkpoint step-00009000.pt 已存在；错误扫描无命中。
[epoch-end] epoch=85 elapsed=424.38s step=9030
[epoch-start] epoch=86 step=9030 steps_per_epoch=105
[2026-09-05 06:30] monitor: epoch=85 已完成（elapsed=424.38s，step=9030），训练进入 epoch=86；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=86 step=9040 loss=0.630967 loss_full=0.281416 loss_base=0.349551 ms_per_step=3792.93
[2026-09-05 06:31] monitor: epoch=86 step=9040，最新 loss=0.630967（loss_full=0.281416, loss_base=0.349551）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=86 step=9060 loss=0.614281 loss_full=0.270014 loss_base=0.344266 ms_per_step=3791.88
[2026-09-05 06:32] monitor: epoch=86 step=9060，最新 loss=0.614281（loss_full=0.270014, loss_base=0.344266）；step=9000 sample/checkpoint 已完成，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=86 step=9080 loss=0.676771 loss_full=0.313439 loss_base=0.363332 ms_per_step=3659.97
[2026-09-05 06:34] monitor: epoch=86 step=9080，最新 loss=0.676771（loss_full=0.313439, loss_base=0.363332）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=86 step=9100 loss=0.751258 loss_full=0.358098 loss_base=0.393159 ms_per_step=3665.32
[profile] window=374.54s updates=100 data_wait=2.1ms h2d=1.6ms stage1=281.8ms dit=644.6ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 06:35] monitor: epoch=86 step=9100，最新 loss=0.751258（loss_full=0.358098, loss_base=0.393159）；profile peak=38.0GiB，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=86 step=9120 loss=0.616482 loss_full=0.268345 loss_base=0.348137 ms_per_step=3703.00
[2026-09-05 06:36] monitor: epoch=86 step=9120，最新 loss=0.616482（loss_full=0.268345, loss_base=0.348137）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[epoch-end] epoch=86 elapsed=392.59s step=9135
[epoch-start] epoch=87 step=9135 steps_per_epoch=105
[2026-09-05 06:37] monitor: epoch=86 已完成（elapsed=392.59s，step=9135），训练进入 epoch=87；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=87 step=9140 loss=0.749218 loss_full=0.349490 loss_base=0.399728 ms_per_step=3725.33
[2026-09-05 06:37] monitor: epoch=87 step=9140，最新 loss=0.749218（loss_full=0.349490, loss_base=0.399728）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=87 step=9160 loss=0.643346 loss_full=0.294865 loss_base=0.348481 ms_per_step=3740.00
[2026-09-05 06:39] monitor: epoch=87 step=9160，最新 loss=0.643346（loss_full=0.294865, loss_base=0.348481）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=87 step=9180 loss=0.630250 loss_full=0.282983 loss_base=0.347267 ms_per_step=3714.03
[2026-09-05 06:40] monitor: epoch=87 step=9180，最新 loss=0.630250（loss_full=0.282983, loss_base=0.347267）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=87 step=9200 loss=0.660637 loss_full=0.301693 loss_base=0.358944 ms_per_step=3791.48
[profile] window=373.74s updates=100 data_wait=1.8ms h2d=1.6ms stage1=280.2ms dit=644.5ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 06:42] monitor: epoch=87 step=9200，最新 loss=0.660637（loss_full=0.301693, loss_base=0.358944）；profile peak=38.0GiB，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=87 step=9220 loss=0.595599 loss_full=0.261075 loss_base=0.334523 ms_per_step=3564.86
[2026-09-05 06:42] monitor: epoch=87 step=9220，最新 loss=0.595599（loss_full=0.261075, loss_base=0.334523）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=87 step=9240 loss=0.613840 loss_full=0.268141 loss_base=0.345698 ms_per_step=3680.01
[epoch-end] epoch=87 elapsed=388.05s step=9240
[epoch-start] epoch=88 step=9240 steps_per_epoch=105
[2026-09-05 06:44] monitor: epoch=87 已完成（elapsed=388.05s，step=9240），训练进入 epoch=88；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 06:43] monitor: epoch=87 已完成（elapsed=388.05s，step=9240），训练进入 epoch=88；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=88 step=9260 loss=0.758467 loss_full=0.347057 loss_base=0.411411 ms_per_step=3840.40
[2026-09-05 06:44] monitor: epoch=88 step=9260，最新 loss=0.758467（loss_full=0.347057, loss_base=0.411411）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=88 step=9280 loss=0.652708 loss_full=0.295200 loss_base=0.357508 ms_per_step=3725.53
[2026-09-05 06:46] monitor: epoch=88 step=9280，最新 loss=0.652708（loss_full=0.295200, loss_base=0.357508）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=88 step=9300 loss=0.703220 loss_full=0.307439 loss_base=0.395782 ms_per_step=3717.60
[profile] window=370.83s updates=100 data_wait=2.0ms h2d=1.5ms stage1=273.3ms dit=643.9ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 06:47] monitor: epoch=88 step=9300，最新 loss=0.703220（loss_full=0.307439, loss_base=0.395782）；profile peak=38.0GiB，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=88 step=9320 loss=0.647656 loss_full=0.286987 loss_base=0.360668 ms_per_step=3653.66
[2026-09-05 06:48] monitor: epoch=88 step=9320，最新 loss=0.647656（loss_full=0.286987, loss_base=0.360668）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=88 step=9340 loss=0.646363 loss_full=0.285482 loss_base=0.360881 ms_per_step=3393.10
[2026-09-05 06:49] monitor: epoch=88 step=9340，最新 loss=0.646363（loss_full=0.285482, loss_base=0.360881）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[epoch-end] epoch=88 elapsed=385.04s step=9345
[epoch-start] epoch=89 step=9345 steps_per_epoch=105
[2026-09-05 06:50] monitor: epoch=88 已完成（elapsed=385.04s，step=9345），训练进入 epoch=89；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=89 step=9360 loss=0.657315 loss_full=0.299666 loss_base=0.357649 ms_per_step=3795.41
[2026-09-05 06:51] monitor: epoch=89 step=9360，最新 loss=0.657315（loss_full=0.299666, loss_base=0.357649）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=89 step=9380 loss=0.669158 loss_full=0.304461 loss_base=0.364697 ms_per_step=3798.86
[2026-09-05 06:52] monitor: epoch=89 step=9380，最新 loss=0.669158（loss_full=0.304461, loss_base=0.364697）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=89 step=9400 loss=0.686093 loss_full=0.315579 loss_base=0.370514 ms_per_step=3739.61
[profile] window=367.88s updates=100 data_wait=2.1ms h2d=1.5ms stage1=266.9ms dit=642.9ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 06:53] monitor: epoch=89 step=9400，最新 loss=0.686093（loss_full=0.315579, loss_base=0.370514）；profile peak=38.0GiB，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=89 step=9420 loss=0.599270 loss_full=0.267087 loss_base=0.332182 ms_per_step=3560.50
[2026-09-05 06:54] monitor: epoch=89 step=9420，最新 loss=0.599270（loss_full=0.267087, loss_base=0.332182）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=89 step=9440 loss=0.649855 loss_full=0.293329 loss_base=0.356525 ms_per_step=3797.37
[2026-09-05 06:56] monitor: epoch=89 step=9440，最新 loss=0.649855（loss_full=0.293329, loss_base=0.356525）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[epoch-end] epoch=89 elapsed=393.07s step=9450
[epoch-start] epoch=90 step=9450 steps_per_epoch=105
[2026-09-05 06:57] monitor: epoch=89 已完成（elapsed=393.07s，step=9450），训练进入 epoch=90；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=90 step=9460 loss=0.667511 loss_full=0.299514 loss_base=0.367997 ms_per_step=3805.32
[2026-09-05 06:57] monitor: epoch=90 step=9460，最新 loss=0.667511（loss_full=0.299514, loss_base=0.367997）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=90 step=9480 loss=0.638453 loss_full=0.281206 loss_base=0.357247 ms_per_step=3718.44
[2026-09-05 06:58] monitor: epoch=90 step=9480，最新 loss=0.638453（loss_full=0.281206, loss_base=0.357247）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=90 step=9500 loss=0.554409 loss_full=0.234756 loss_base=0.319653 ms_per_step=3809.26
[profile] window=374.08s updates=100 data_wait=2.0ms h2d=1.6ms stage1=279.7ms dit=645.7ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 07:00] monitor: epoch=90 step=9500，最新 loss=0.554409（loss_full=0.234756, loss_base=0.319653）；profile peak=38.0GiB，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=90 step=9520 loss=0.682948 loss_full=0.305482 loss_base=0.377466 ms_per_step=3794.70
[2026-09-05 07:01] monitor: epoch=90 step=9520，最新 loss=0.682948（loss_full=0.305482, loss_base=0.377466）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=90 step=9540 loss=0.634884 loss_full=0.266351 loss_base=0.368533 ms_per_step=3722.47
[2026-09-05 07:02] monitor: epoch=90 step=9540，最新 loss=0.634884（loss_full=0.266351, loss_base=0.368533）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[epoch-end] epoch=90 elapsed=396.66s step=9555
[epoch-start] epoch=91 step=9555 steps_per_epoch=105
[2026-09-05 07:03] monitor: epoch=90 已完成（elapsed=396.66s，step=9555），训练进入 epoch=91；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=91 step=9560 loss=0.662715 loss_full=0.309935 loss_base=0.352780 ms_per_step=3818.69
[2026-09-05 07:03] monitor: epoch=91 step=9560，最新 loss=0.662715（loss_full=0.309935, loss_base=0.352780）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=91 step=9580 loss=0.607119 loss_full=0.272247 loss_base=0.334872 ms_per_step=3789.97
[2026-09-05 07:05] monitor: epoch=91 step=9580，最新 loss=0.607119（loss_full=0.272247, loss_base=0.334872）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=91 step=9600 loss=0.695168 loss_full=0.327309 loss_base=0.367859 ms_per_step=3669.55
[profile] window=376.17s updates=100 data_wait=2.0ms h2d=1.6ms stage1=285.9ms dit=644.5ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 07:06] monitor: epoch=91 step=9600，最新 loss=0.695168（loss_full=0.327309, loss_base=0.367859）；profile peak=38.0GiB，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=91 step=9620 loss=0.701708 loss_full=0.328264 loss_base=0.373444 ms_per_step=3723.06
[2026-09-05 07:07] monitor: epoch=91 step=9620，最新 loss=0.701708（loss_full=0.328264, loss_base=0.373444）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=91 step=9640 loss=0.540832 loss_full=0.231093 loss_base=0.309740 ms_per_step=3812.70
[2026-09-05 07:08] monitor: epoch=91 step=9640，最新 loss=0.540832（loss_full=0.231093, loss_base=0.309740）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=91 step=9660 loss=0.641688 loss_full=0.289160 loss_base=0.352529 ms_per_step=3749.24
[epoch-end] epoch=91 elapsed=395.46s step=9660
[epoch-start] epoch=92 step=9660 steps_per_epoch=105
[2026-09-05 07:10] monitor: epoch=91 已完成（elapsed=395.46s，step=9660），训练进入 epoch=92；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 07:10] monitor: epoch=91 已完成（elapsed=395.46s，step=9660），训练进入 epoch=92；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=92 step=9680 loss=0.653397 loss_full=0.299643 loss_base=0.353754 ms_per_step=3833.41
[2026-09-05 07:11] monitor: epoch=92 step=9680，最新 loss=0.653397（loss_full=0.299643, loss_base=0.353754）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=92 step=9700 loss=0.591337 loss_full=0.263083 loss_base=0.328254 ms_per_step=3753.87
[profile] window=377.71s updates=100 data_wait=2.1ms h2d=1.7ms stage1=288.0ms dit=645.8ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 07:12] monitor: epoch=92 step=9700，最新 loss=0.591337（loss_full=0.263083, loss_base=0.328254）；profile peak=38.0GiB，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=92 step=9720 loss=0.618904 loss_full=0.275073 loss_base=0.343830 ms_per_step=3718.35
[2026-09-05 07:13] monitor: epoch=92 step=9720，最新 loss=0.618904（loss_full=0.275073, loss_base=0.343830）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=92 step=9740 loss=0.590245 loss_full=0.264839 loss_base=0.325406 ms_per_step=3782.07
[2026-09-05 07:15] monitor: epoch=92 step=9740，最新 loss=0.590245（loss_full=0.264839, loss_base=0.325406）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=92 step=9760 loss=0.617048 loss_full=0.260932 loss_base=0.356115 ms_per_step=3572.08
[epoch-end] epoch=92 elapsed=390.07s step=9765
[epoch-start] epoch=93 step=9765 steps_per_epoch=105
task=ci2v epoch=93 step=9780 loss=0.614599 loss_full=0.256773 loss_base=0.357826 ms_per_step=3566.43
[2026-09-05 07:16] monitor: epoch=92 已完成（elapsed=390.07s，step=9765），训练进入 epoch=93；当前 step=9780，最新 loss=0.614599（loss_full=0.256773, loss_base=0.357826）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=93 step=9800 loss=0.578653 loss_full=0.259054 loss_base=0.319599 ms_per_step=3787.36
[profile] window=368.78s updates=100 data_wait=1.7ms h2d=1.5ms stage1=271.2ms dit=641.5ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 07:18] monitor: epoch=93 step=9800，最新 loss=0.578653（loss_full=0.259054, loss_base=0.319599）；profile peak=38.0GiB，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=93 step=9820 loss=0.749980 loss_full=0.346761 loss_base=0.403219 ms_per_step=3740.33
[2026-09-05 07:19] monitor: epoch=93 step=9820，最新 loss=0.749980（loss_full=0.346761, loss_base=0.403219）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=93 step=9840 loss=0.771600 loss_full=0.352010 loss_base=0.419590 ms_per_step=3739.16
[2026-09-05 07:20] monitor: epoch=93 step=9840，最新 loss=0.771600（loss_full=0.352010, loss_base=0.419590）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=93 step=9860 loss=0.592738 loss_full=0.266451 loss_base=0.326286 ms_per_step=3788.34
[2026-09-05 07:21] monitor: epoch=93 step=9860，最新 loss=0.592738（loss_full=0.266451, loss_base=0.326286）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[epoch-end] epoch=93 elapsed=393.54s step=9870
[epoch-start] epoch=94 step=9870 steps_per_epoch=105
[2026-09-05 07:22] monitor: epoch=93 已完成（elapsed=393.54s，step=9870），训练进入 epoch=94；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=94 step=9880 loss=0.902298 loss_full=0.364806 loss_base=0.537492 ms_per_step=3799.31
[2026-09-05 07:22] monitor: epoch=94 step=9880，最新 loss=0.902298（loss_full=0.364806, loss_base=0.537492）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=94 step=9900 loss=0.698916 loss_full=0.321763 loss_base=0.377154 ms_per_step=3626.83
[profile] window=374.13s updates=100 data_wait=2.0ms h2d=1.6ms stage1=281.8ms dit=643.4ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 07:24] monitor: epoch=94 step=9900，最新 loss=0.698916（loss_full=0.321763, loss_base=0.377154）；profile peak=38.0GiB，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=94 step=9920 loss=0.533594 loss_full=0.234707 loss_base=0.298887 ms_per_step=3602.46
[2026-09-05 07:25] monitor: epoch=94 step=9920，最新 loss=0.533594（loss_full=0.234707, loss_base=0.298887）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=94 step=9940 loss=0.634280 loss_full=0.286039 loss_base=0.348241 ms_per_step=3775.48
[2026-09-05 07:26] monitor: epoch=94 step=9940，最新 loss=0.634280（loss_full=0.286039, loss_base=0.348241）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=94 step=9960 loss=0.615627 loss_full=0.280766 loss_base=0.334862 ms_per_step=3804.56
[2026-09-05 07:28] monitor: epoch=94 step=9960，最新 loss=0.615627（loss_full=0.280766, loss_base=0.334862）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[epoch-end] epoch=94 elapsed=391.09s step=9975
[epoch-start] epoch=95 step=9975 steps_per_epoch=105
[2026-09-05 07:29] monitor: epoch=94 已完成（elapsed=391.09s，step=9975），训练进入 epoch=95；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=95 step=9980 loss=0.656519 loss_full=0.298089 loss_base=0.358430 ms_per_step=3794.08
task=ci2v epoch=95 step=10000 loss=0.646520 loss_full=0.287569 loss_base=0.358951 ms_per_step=3706.03
[profile] window=373.92s updates=100 data_wait=2.2ms h2d=1.6ms stage1=280.5ms dit=644.3ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=10000 elapsed=2.83s
[checkpoint] step=10000 elapsed=38.79s
[2026-09-05 07:31] monitor: epoch=95 step=10000，最新 loss=0.646520（loss_full=0.287569, loss_base=0.358951）；sample 与 checkpoint 均已完成（checkpoint elapsed=38.79s，文件 step-00010000.pt）；profile peak=38.0GiB，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=95 step=10020 loss=0.676473 loss_full=0.305313 loss_base=0.371161 ms_per_step=3783.54
[2026-09-05 07:32] monitor: epoch=95 step=10020，最新 loss=0.676473（loss_full=0.305313, loss_base=0.371161）；已从 step=10000 checkpoint 继续训练，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=95 step=10040 loss=0.644850 loss_full=0.279003 loss_base=0.365846 ms_per_step=3797.97
[2026-09-05 07:34] monitor: epoch=95 step=10040，最新 loss=0.644850（loss_full=0.279003, loss_base=0.365846）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=95 step=10060 loss=0.631641 loss_full=0.289976 loss_base=0.341665 ms_per_step=3462.22
[2026-09-05 07:35] monitor: epoch=95 step=10060，最新 loss=0.631641（loss_full=0.289976, loss_base=0.341665）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=95 step=10080 loss=0.598450 loss_full=0.261830 loss_base=0.336619 ms_per_step=3735.61
[epoch-end] epoch=95 elapsed=431.87s step=10080
[epoch-start] epoch=96 step=10080 steps_per_epoch=105
[2026-09-05 07:36] monitor: epoch=95 已完成（elapsed=431.87s，step=10080），训练进入 epoch=96；最新 loss=0.598450（loss_full=0.261830, loss_base=0.336619）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=96 step=10100 loss=0.680814 loss_full=0.302228 loss_base=0.378586 ms_per_step=3637.27
[profile] window=368.59s updates=100 data_wait=2.0ms h2d=1.5ms stage1=269.8ms dit=641.8ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 07:37] monitor: epoch=96 step=10100，最新 loss=0.680814（loss_full=0.302228, loss_base=0.378586）；profile peak=38.0GiB，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=96 step=10120 loss=0.670129 loss_full=0.305419 loss_base=0.364711 ms_per_step=3565.60
[2026-09-05 07:38] monitor: epoch=96 step=10120，最新 loss=0.670129（loss_full=0.305419, loss_base=0.364711）；profile peak=38.0GiB，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=96 step=10140 loss=0.622687 loss_full=0.274259 loss_base=0.348428 ms_per_step=3779.60
[2026-09-05 07:39] monitor: epoch=96 step=10140，最新 loss=0.622687（loss_full=0.274259, loss_base=0.348428）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=96 step=10160 loss=0.764670 loss_full=0.357619 loss_base=0.407050 ms_per_step=3776.73
[2026-09-05 07:40] monitor: epoch=96 step=10160，最新 loss=0.764670（loss_full=0.357619, loss_base=0.407050）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=96 step=10180 loss=0.644074 loss_full=0.294684 loss_base=0.349389 ms_per_step=3813.14
[epoch-end] epoch=96 elapsed=389.60s step=10185
[epoch-start] epoch=97 step=10185 steps_per_epoch=105
[2026-09-05 07:42] monitor: epoch=96 已完成（elapsed=389.60s，step=10185），训练进入 epoch=97；最新 loss=0.644074（loss_full=0.294684, loss_base=0.349389）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=97 step=10200 loss=0.636528 loss_full=0.279997 loss_base=0.356531 ms_per_step=3684.24
[profile] window=372.64s updates=100 data_wait=2.1ms h2d=1.6ms stage1=277.6ms dit=644.1ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 07:44] monitor: epoch=97 step=10200，最新 loss=0.636528（loss_full=0.279997, loss_base=0.356531）；profile peak=38.0GiB，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=97 step=10220 loss=0.655277 loss_full=0.288307 loss_base=0.366970 ms_per_step=3618.38
[2026-09-05 07:45] monitor: epoch=97 step=10220，最新 loss=0.655277（loss_full=0.288307, loss_base=0.366970）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=97 step=10240 loss=0.612154 loss_full=0.272351 loss_base=0.339803 ms_per_step=3765.72
[2026-09-05 07:46] monitor: epoch=97 step=10240，最新 loss=0.612154（loss_full=0.272351, loss_base=0.339803）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=97 step=10260 loss=0.614988 loss_full=0.275301 loss_base=0.339687 ms_per_step=3659.02
[2026-09-05 07:47] monitor: epoch=97 step=10260，最新 loss=0.614988（loss_full=0.275301, loss_base=0.339687）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=97 step=10280 loss=0.645831 loss_full=0.292083 loss_base=0.353748 ms_per_step=3779.89
[2026-09-05 07:48] monitor: epoch=97 step=10280，最新 loss=0.645831（loss_full=0.292083, loss_base=0.353748）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[epoch-end] epoch=97 elapsed=389.95s step=10290
[epoch-start] epoch=98 step=10290 steps_per_epoch=105
[2026-09-05 07:49] monitor: epoch=97 已完成（elapsed=389.95s，step=10290），训练进入 epoch=98；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=98 step=10300 loss=0.585240 loss_full=0.261953 loss_base=0.323287 ms_per_step=3810.13
[profile] window=372.93s updates=100 data_wait=2.0ms h2d=1.6ms stage1=278.8ms dit=643.7ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 07:50] monitor: epoch=98 step=10300，最新 loss=0.585240（loss_full=0.261953, loss_base=0.323287）；profile peak=38.0GiB，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=98 step=10320 loss=0.579403 loss_full=0.256394 loss_base=0.323009 ms_per_step=3783.26
[2026-09-05 07:51] monitor: epoch=98 step=10320，最新 loss=0.579403（loss_full=0.256394, loss_base=0.323009）；profile peak=38.0GiB，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=98 step=10340 loss=0.658616 loss_full=0.300783 loss_base=0.357833 ms_per_step=3793.87
[2026-09-05 07:52] monitor: epoch=98 step=10340，最新 loss=0.658616（loss_full=0.300783, loss_base=0.357833）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=98 step=10360 loss=0.661121 loss_full=0.290938 loss_base=0.370183 ms_per_step=3783.26
[2026-09-05 07:53] monitor: epoch=98 step=10360，最新 loss=0.661121（loss_full=0.290938, loss_base=0.370183）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=98 step=10380 loss=0.650072 loss_full=0.299537 loss_base=0.350535 ms_per_step=3705.66
[2026-09-05 07:55] monitor: epoch=98 step=10380，最新 loss=0.650072（loss_full=0.299537, loss_base=0.350535）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[epoch-end] epoch=98 elapsed=396.45s step=10395
[epoch-start] epoch=99 step=10395 steps_per_epoch=105
[2026-09-05 07:55] monitor: epoch=98 已完成（elapsed=396.45s，step=10395），训练进入 epoch=99；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=99 step=10400 loss=0.635367 loss_full=0.285213 loss_base=0.350153 ms_per_step=3788.13
[profile] window=377.34s updates=100 data_wait=2.0ms h2d=1.6ms stage1=289.2ms dit=643.9ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 07:56] monitor: epoch=99 step=10400，最新 loss=0.635367（loss_full=0.285213, loss_base=0.350153）；profile peak=38.0GiB，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=99 step=10420 loss=0.656811 loss_full=0.294355 loss_base=0.362456 ms_per_step=3696.14
[2026-09-05 07:57] monitor: epoch=99 step=10420，最新 loss=0.656811（loss_full=0.294355, loss_base=0.362456）；profile peak=38.0GiB，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=99 step=10440 loss=0.523359 loss_full=0.223468 loss_base=0.299891 ms_per_step=3716.60
[2026-09-05 07:59] monitor: epoch=99 step=10440，最新 loss=0.523359（loss_full=0.223468, loss_base=0.299891）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=99 step=10460 loss=0.772867 loss_full=0.352673 loss_base=0.420194 ms_per_step=3704.57
[2026-09-05 07:59] monitor: epoch=99 step=10460，最新 loss=0.772867（loss_full=0.352673, loss_base=0.420194）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=99 step=10480 loss=0.678216 loss_full=0.298334 loss_base=0.379882 ms_per_step=3454.16
[2026-09-05 08:00] monitor: epoch=99 step=10480，最新 loss=0.678216（loss_full=0.298334, loss_base=0.379882）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=99 step=10500 loss=0.700320 loss_full=0.323787 loss_base=0.376533 ms_per_step=3606.04
[profile] window=363.82s updates=100 data_wait=0.9ms h2d=1.4ms stage1=260.0ms dit=641.1ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[epoch-end] epoch=99 elapsed=383.96s step=10500
[epoch-start] epoch=100 step=10500 steps_per_epoch=105
[2026-09-05 08:02] monitor: epoch=99 已完成（elapsed=383.96s，step=10500），训练进入 epoch=100；profile peak=38.0GiB，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=100 step=10520 loss=0.649408 loss_full=0.298116 loss_base=0.351292 ms_per_step=3497.96
[2026-09-05 08:03] monitor: epoch=100 step=10520，最新 loss=0.649408（loss_full=0.298116, loss_base=0.351292）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=100 step=10540 loss=0.655432 loss_full=0.288385 loss_base=0.367047 ms_per_step=3782.18
[2026-09-05 08:04] monitor: epoch=100 step=10540，最新 loss=0.655432（loss_full=0.288385, loss_base=0.367047）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=100 step=10560 loss=0.601152 loss_full=0.268934 loss_base=0.332217 ms_per_step=3732.18
[2026-09-05 08:06] monitor: epoch=100 step=10560，最新 loss=0.601152（loss_full=0.268934, loss_base=0.332217）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=100 step=10580 loss=0.564453 loss_full=0.248782 loss_base=0.315671 ms_per_step=3719.20
[2026-09-05 08:07] monitor: epoch=100 step=10580，最新 loss=0.564453（loss_full=0.248782, loss_base=0.315671）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=100 step=10600 loss=0.583821 loss_full=0.249804 loss_base=0.334017 ms_per_step=3770.63
[profile] window=370.30s updates=100 data_wait=1.7ms h2d=1.5ms stage1=273.1ms dit=643.2ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[epoch-end] epoch=100 elapsed=388.19s step=10605
[epoch-start] epoch=101 step=10605 steps_per_epoch=105
[2026-09-05 08:08] monitor: epoch=100 已完成（elapsed=388.19s，step=10605），训练进入 epoch=101；最新 loss=0.583821（loss_full=0.249804, loss_base=0.334017）；profile peak=38.0GiB，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=101 step=10620 loss=0.633144 loss_full=0.272926 loss_base=0.360218 ms_per_step=3791.10
[2026-09-05 08:10] monitor: epoch=101 step=10620，最新 loss=0.633144（loss_full=0.272926, loss_base=0.360218）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=101 step=10640 loss=0.697502 loss_full=0.322823 loss_base=0.374679 ms_per_step=3777.48
[2026-09-05 08:10] monitor: epoch=101 step=10640，最新 loss=0.697502（loss_full=0.322823, loss_base=0.374679）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=101 step=10660 loss=0.697867 loss_full=0.321741 loss_base=0.376127 ms_per_step=3722.34
[2026-09-05 08:11] monitor: epoch=101 step=10660，最新 loss=0.697867（loss_full=0.321741, loss_base=0.376127）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=101 step=10680 loss=0.499902 loss_full=0.206199 loss_base=0.293703 ms_per_step=3771.99
[2026-09-05 08:13] monitor: epoch=101 step=10680，最新 loss=0.499902（loss_full=0.206199, loss_base=0.293703）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=101 step=10700 loss=0.628130 loss_full=0.283357 loss_base=0.344773 ms_per_step=3689.79
[profile] window=375.32s updates=100 data_wait=2.2ms h2d=1.6ms stage1=285.9ms dit=642.4ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 08:14] monitor: epoch=101 step=10700，最新 loss=0.628130（loss_full=0.283357, loss_base=0.344773）；profile peak=38.0GiB，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[epoch-end] epoch=101 elapsed=392.79s step=10710
[epoch-start] epoch=102 step=10710 steps_per_epoch=105
[2026-09-05 08:15] monitor: epoch=101 已完成（elapsed=392.79s，step=10710），训练进入 epoch=102；最新 loss=0.628130（loss_full=0.283357, loss_base=0.344773）；profile peak=38.0GiB，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=102 step=10720 loss=0.573588 loss_full=0.250405 loss_base=0.323183 ms_per_step=3624.69
[2026-09-05 08:15] monitor: epoch=102 step=10720，最新 loss=0.573588（loss_full=0.250405, loss_base=0.323183）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=102 step=10740 loss=0.589639 loss_full=0.250992 loss_base=0.338647 ms_per_step=3795.66
[2026-09-05 08:17] monitor: epoch=102 step=10740，最新 loss=0.589639（loss_full=0.250992, loss_base=0.338647）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=102 step=10760 loss=0.596265 loss_full=0.261498 loss_base=0.334767 ms_per_step=3787.83
[2026-09-05 08:18] monitor: epoch=102 step=10760，最新 loss=0.596265（loss_full=0.261498, loss_base=0.334767）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=102 step=10780 loss=0.626723 loss_full=0.280418 loss_base=0.346304 ms_per_step=3773.07
[2026-09-05 08:19] monitor: epoch=102 step=10780，最新 loss=0.626723（loss_full=0.280418, loss_base=0.346304）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=102 step=10800 loss=0.616110 loss_full=0.271185 loss_base=0.344925 ms_per_step=3796.46
[profile] window=375.81s updates=100 data_wait=2.1ms h2d=1.6ms stage1=284.1ms dit=645.4ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 08:21] monitor: epoch=102 step=10800，最新 loss=0.616110（loss_full=0.271185, loss_base=0.344925）；profile peak=38.0GiB，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[epoch-end] epoch=102 elapsed=392.50s step=10815
[epoch-start] epoch=103 step=10815 steps_per_epoch=105
[2026-09-05 08:21] monitor: epoch=102 已完成（elapsed=392.50s，step=10815），训练进入 epoch=103；最新 loss=0.616110（loss_full=0.271185, loss_base=0.344925）；profile peak=38.0GiB，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=103 step=10820 loss=0.607938 loss_full=0.271582 loss_base=0.336356 ms_per_step=3546.74
[2026-09-05 08:22] monitor: epoch=103 step=10820，最新 loss=0.607938（loss_full=0.271582, loss_base=0.336356）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=103 step=10840 loss=0.513360 loss_full=0.218304 loss_base=0.295056 ms_per_step=3311.93
[2026-09-05 08:23] monitor: epoch=103 step=10840，最新 loss=0.513360（loss_full=0.218304, loss_base=0.295056）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=103 step=10860 loss=0.569463 loss_full=0.246228 loss_base=0.323236 ms_per_step=3569.74
- [2026-09-05 08:24] monitor: epoch=103 step=10860，最新 loss=0.569463（loss_full=0.246228, loss_base=0.323236）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=103 step=10880 loss=0.588822 loss_full=0.242001 loss_base=0.346821 ms_per_step=3724.61
- [2026-09-05 08:25] monitor: epoch=103 step=10880，最新 loss=0.588822（loss_full=0.242001, loss_base=0.346821）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=103 step=10900 loss=0.654088 loss_full=0.291396 loss_base=0.362692 ms_per_step=3522.52
[2026-09-05 08:26] monitor: epoch=103 step=10900，最新 loss=0.654088（loss_full=0.291396, loss_base=0.362692）；profile peak=38.0GiB，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=103 step=10880 loss=0.588822 loss_full=0.242001 loss_base=0.346821 ms_per_step=3724.61
task=ci2v epoch=103 step=10900 loss=0.654088 loss_full=0.291396 loss_base=0.362692 ms_per_step=3522.52
[profile] window=353.78s updates=100 data_wait=2.0ms h2d=1.2ms stage1=235.3ms dit=639.6ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=103 step=10920 loss=0.647763 loss_full=0.284344 loss_base=0.363418 ms_per_step=3738.04
[2026-09-05 08:28] monitor: epoch=103 已完成（elapsed=377.04s，step=10920），训练进入 epoch=104；最新 loss=0.647763（loss_full=0.284344, loss_base=0.363418）；profile peak=38.0GiB，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=104 step=10940 loss=0.542719 loss_full=0.233892 loss_base=0.308827 ms_per_step=3658.19
[2026-09-05 08:29] monitor: epoch=104 step=10940，最新 loss=0.542719（loss_full=0.233892, loss_base=0.308827）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=104 step=10960 loss=0.602831 loss_full=0.251667 loss_base=0.351164 ms_per_step=3493.57
[2026-09-05 08:30] monitor: epoch=104 step=10960，最新 loss=0.602831（loss_full=0.251667, loss_base=0.351164）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=104 step=10980 loss=0.739031 loss_full=0.342592 loss_base=0.396439 ms_per_step=3567.99
[2026-09-05 08:31] monitor: epoch=104 step=10980，最新 loss=0.739031（loss_full=0.342592, loss_base=0.396439）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=104 step=11000 loss=0.729574 loss_full=0.333560 loss_base=0.396014 ms_per_step=3770.36
[profile] window=364.83s updates=100 data_wait=1.9ms h2d=1.4ms stage1=260.5ms dit=641.6ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[sample] step=11000 elapsed=2.80s
[checkpoint] step=11000 elapsed=31.05s
[2026-09-05 08:33] monitor: epoch=104 step=11000，最新 loss=0.729574（loss_full=0.333560, loss_base=0.396014）；checkpoint=step-00011000.pt 已生成，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=104 step=11020 loss=0.623607 loss_full=0.270509 loss_base=0.353098 ms_per_step=3666.06
[2026-09-05 08:34] monitor: epoch=104 step=11020，最新 loss=0.623607（loss_full=0.270509, loss_base=0.353098）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[epoch-end] epoch=104 elapsed=415.35s step=11025
[epoch-start] epoch=105 step=11025 steps_per_epoch=105
[2026-09-05 08:35] monitor: epoch=104 已完成（elapsed=415.35s，step=11025），训练进入 epoch=105；最新 loss=0.623607（loss_full=0.270509, loss_base=0.353098）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=105 step=11040 loss=0.565014 loss_full=0.248910 loss_base=0.316104 ms_per_step=3790.94
[2026-09-05 08:35] monitor: epoch=105 step=11040，最新 loss=0.565014（loss_full=0.248910, loss_base=0.316104）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=105 step=11060 loss=0.654454 loss_full=0.299376 loss_base=0.355078 ms_per_step=3647.27
[2026-09-05 08:37] monitor: epoch=105 step=11060，最新 loss=0.654454（loss_full=0.299376, loss_base=0.355078）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=105 step=11080 loss=0.527725 loss_full=0.220942 loss_base=0.306784 ms_per_step=3666.05
[2026-09-05 08:38] monitor: epoch=105 step=11080，最新 loss=0.527725（loss_full=0.220942, loss_base=0.306784）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=105 step=11100 loss=0.620176 loss_full=0.287721 loss_base=0.332455 ms_per_step=3792.63
[profile] window=371.52s updates=100 data_wait=2.1ms h2d=1.6ms stage1=275.1ms dit=643.8ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 08:39] monitor: epoch=105 step=11100，最新 loss=0.620176（loss_full=0.287721, loss_base=0.332455）；profile peak=38.0GiB，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=105 step=11120 loss=0.669598 loss_full=0.299681 loss_base=0.369918 ms_per_step=3356.79
[2026-09-05 08:40] monitor: epoch=105 step=11120，最新 loss=0.669598（loss_full=0.299681, loss_base=0.369918）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[epoch-end] epoch=105 elapsed=380.44s step=11130
[epoch-start] epoch=106 step=11130 steps_per_epoch=105
[2026-09-05 08:41] monitor: epoch=105 已完成（elapsed=380.44s，step=11130），训练进入 epoch=106；最新 loss=0.669598（loss_full=0.299681, loss_base=0.369918）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=106 step=11140 loss=0.605720 loss_full=0.267636 loss_base=0.338084 ms_per_step=3356.22
[2026-09-05 08:41] monitor: epoch=106 step=11140，最新 loss=0.605720（loss_full=0.267636, loss_base=0.338084）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=106 step=11160 loss=0.629415 loss_full=0.273594 loss_base=0.355821 ms_per_step=3619.64
[2026-09-05 08:43] monitor: epoch=106 step=11160，最新 loss=0.629415（loss_full=0.273594, loss_base=0.355821）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=106 step=11180 loss=0.593327 loss_full=0.247840 loss_base=0.345487 ms_per_step=3743.87
[2026-09-05 08:44] monitor: epoch=106 step=11180，最新 loss=0.593327（loss_full=0.247840, loss_base=0.345487）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=106 step=11200 loss=0.589361 loss_full=0.252356 loss_base=0.337005 ms_per_step=3602.41
[profile] window=353.84s updates=100 data_wait=1.6ms h2d=1.2ms stage1=235.3ms dit=640.4ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 08:45] monitor: epoch=106 step=11200，最新 loss=0.589361（loss_full=0.252356, loss_base=0.337005）；profile peak=38.0GiB，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=106 step=11220 loss=0.608461 loss_full=0.271141 loss_base=0.337319 ms_per_step=3801.80
[2026-09-05 08:46] monitor: epoch=106 step=11220，最新 loss=0.608461（loss_full=0.271141, loss_base=0.337319）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[epoch-end] epoch=106 elapsed=385.73s step=11235
[epoch-start] epoch=107 step=11235 steps_per_epoch=105
task=ci2v epoch=107 step=11240 loss=0.680200 loss_full=0.303406 loss_base=0.376794 ms_per_step=3818.44
[2026-09-05 08:48] monitor: epoch=106 已完成（elapsed=385.73s，step=11235），训练进入 epoch=107；最新 loss=0.680200（loss_full=0.303406, loss_base=0.376794）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=107 step=11260 loss=0.610864 loss_full=0.256474 loss_base=0.354389 ms_per_step=3586.51
[2026-09-05 08:49] monitor: epoch=107 step=11260，最新 loss=0.610864（loss_full=0.256474, loss_base=0.354389）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=107 step=11280 loss=0.586329 loss_full=0.257086 loss_base=0.329243 ms_per_step=3779.74
[2026-09-05 08:50] monitor: epoch=107 step=11280，最新 loss=0.586329（loss_full=0.257086, loss_base=0.329243）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=107 step=11300 loss=0.621954 loss_full=0.278737 loss_base=0.343217 ms_per_step=3788.47
[profile] window=375.75s updates=100 data_wait=2.0ms h2d=1.6ms stage1=286.2ms dit=643.2ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 08:51] monitor: epoch=107 step=11300，最新 loss=0.621954（loss_full=0.278737, loss_base=0.343217）；profile peak=38.0GiB，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=107 step=11320 loss=0.625175 loss_full=0.287692 loss_base=0.337483 ms_per_step=3806.54
[2026-09-05 08:53] monitor: epoch=107 step=11320，最新 loss=0.625175（loss_full=0.287692, loss_base=0.337483）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=107 step=11340 loss=0.693995 loss_full=0.315421 loss_base=0.378574 ms_per_step=3779.91
[epoch-end] epoch=107 elapsed=395.73s step=11340
[epoch-start] epoch=108 step=11340 steps_per_epoch=105
[2026-09-05 08:54] monitor: epoch=107 已完成（elapsed=395.73s，step=11340），训练进入 epoch=108；最新 loss=0.693995（loss_full=0.315421, loss_base=0.378574）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=103 step=10920 loss=0.647763 loss_full=0.284344 loss_base=0.363418 ms_per_step=3738.04
[epoch-end] epoch=103 elapsed=377.04s step=10920
[epoch-start] epoch=104 step=10920 steps_per_epoch=105
task=ci2v epoch=104 step=10940 loss=0.542719 loss_full=0.233892 loss_base=0.308827 ms_per_step=3658.19
task=ci2v epoch=104 step=10960 loss=0.602831 loss_full=0.251667 loss_base=0.351164 ms_per_step=3493.57
task=ci2v epoch=104 step=10980 loss=0.739031 loss_full=0.342592 loss_base=0.396439 ms_per_step=3567.99
task=ci2v epoch=104 step=11000 loss=0.729574 loss_full=0.333560 loss_base=0.396014 ms_per_step=3770.36
[profile] window=364.83s updates=100 data_wait=1.9ms h2d=1.4ms stage1=260.5ms dit=641.6ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=11000 elapsed=2.80s
[checkpoint] step=11000 elapsed=31.05s
task=ci2v epoch=104 step=11020 loss=0.623607 loss_full=0.270509 loss_base=0.353098 ms_per_step=3666.06
[epoch-end] epoch=104 elapsed=415.35s step=11025
[epoch-start] epoch=105 step=11025 steps_per_epoch=105
task=ci2v epoch=105 step=11040 loss=0.565014 loss_full=0.248910 loss_base=0.316104 ms_per_step=3790.94
task=ci2v epoch=105 step=11060 loss=0.654454 loss_full=0.299376 loss_base=0.355078 ms_per_step=3647.27
task=ci2v epoch=105 step=11080 loss=0.527725 loss_full=0.220942 loss_base=0.306784 ms_per_step=3666.05
task=ci2v epoch=105 step=11100 loss=0.620176 loss_full=0.287721 loss_base=0.332455 ms_per_step=3792.63
[profile] window=371.52s updates=100 data_wait=2.1ms h2d=1.6ms stage1=275.1ms dit=643.8ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=105 step=11120 loss=0.669598 loss_full=0.299681 loss_base=0.369918 ms_per_step=3356.79
[epoch-end] epoch=105 elapsed=380.44s step=11130
[epoch-start] epoch=106 step=11130 steps_per_epoch=105
task=ci2v epoch=106 step=11140 loss=0.605720 loss_full=0.267636 loss_base=0.338084 ms_per_step=3356.22
task=ci2v epoch=106 step=11160 loss=0.629415 loss_full=0.273594 loss_base=0.355821 ms_per_step=3619.64
task=ci2v epoch=106 step=11180 loss=0.593327 loss_full=0.247840 loss_base=0.345487 ms_per_step=3743.87
task=ci2v epoch=106 step=11200 loss=0.589361 loss_full=0.252356 loss_base=0.337005 ms_per_step=3602.41
[profile] window=353.84s updates=100 data_wait=1.6ms h2d=1.2ms stage1=235.3ms dit=640.4ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=106 step=11220 loss=0.608461 loss_full=0.271141 loss_base=0.337319 ms_per_step=3801.80
[epoch-end] epoch=106 elapsed=385.73s step=11235
[epoch-start] epoch=107 step=11235 steps_per_epoch=105
task=ci2v epoch=107 step=11240 loss=0.680200 loss_full=0.303406 loss_base=0.376794 ms_per_step=3818.44
task=ci2v epoch=107 step=11260 loss=0.610864 loss_full=0.256474 loss_base=0.354389 ms_per_step=3586.51
task=ci2v epoch=107 step=11280 loss=0.586329 loss_full=0.257086 loss_base=0.329243 ms_per_step=3779.74
task=ci2v epoch=107 step=11300 loss=0.621954 loss_full=0.278737 loss_base=0.343217 ms_per_step=3788.47
[profile] window=375.75s updates=100 data_wait=2.0ms h2d=1.6ms stage1=286.2ms dit=643.2ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=107 step=11320 loss=0.625175 loss_full=0.287692 loss_base=0.337483 ms_per_step=3806.54
task=ci2v epoch=107 step=11340 loss=0.693995 loss_full=0.315421 loss_base=0.378574 ms_per_step=3779.91
[epoch-end] epoch=107 elapsed=395.73s step=11340
[epoch-start] epoch=108 step=11340 steps_per_epoch=105
task=ci2v epoch=108 step=11360 loss=0.593881 loss_full=0.257745 loss_base=0.336136 ms_per_step=3870.10
[2026-09-05 08:55] monitor: epoch=108 step=11360，最新 loss=0.593881（loss_full=0.257745, loss_base=0.336136）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=108 step=11380 loss=0.578584 loss_full=0.249031 loss_base=0.329553 ms_per_step=3355.64
[2026-09-05 08:56] monitor: epoch=108 step=11380，最新 loss=0.578584（loss_full=0.249031, loss_base=0.329553）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=108 step=11400 loss=0.571950 loss_full=0.244710 loss_base=0.327239 ms_per_step=3327.54
[profile] window=363.06s updates=100 data_wait=2.4ms h2d=1.4ms stage1=256.0ms dit=641.7ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 08:57] monitor: epoch=108 step=11400，最新 loss=0.571950（loss_full=0.244710, loss_base=0.327239）；profile window=363.06s，GPU allocated=29.0GiB/reserved=39.7GiB/peak=38.0GiB；trainer PID=17993 存活，错误扫描无命中。
[2026-09-05 08:58] monitor: trainer PID=17993 仍处于运行态（CPU 约 302%，GPU 显存约 41410 MiB）；日志暂未产生新 step，未发现错误，继续等待下一条训练输出。
task=ci2v epoch=108 step=11420 loss=0.565810 loss_full=0.236676 loss_base=0.329134 ms_per_step=3812.23
[2026-09-05 08:59] monitor: epoch=108 step=11420，最新 loss=0.565810（loss_full=0.236676, loss_base=0.329134）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=108 step=11380 loss=0.578584 loss_full=0.249031 loss_base=0.329553 ms_per_step=3355.64
task=ci2v epoch=108 step=11400 loss=0.571950 loss_full=0.244710 loss_base=0.327239 ms_per_step=3327.54
[profile] window=363.06s updates=100 data_wait=2.4ms h2d=1.4ms stage1=256.0ms dit=641.7ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=108 step=11420 loss=0.565810 loss_full=0.236676 loss_base=0.329134 ms_per_step=3812.23
task=ci2v epoch=108 step=11440 loss=0.490721 loss_full=0.201567 loss_base=0.289154 ms_per_step=3800.93
[epoch-end] epoch=108 elapsed=381.87s step=11445
[epoch-start] epoch=109 step=11445 steps_per_epoch=105
[2026-09-05 09:00] monitor: epoch=108 已完成（elapsed=381.87s，step=11445），训练进入 epoch=109；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=109 step=11460 loss=0.639841 loss_full=0.281445 loss_base=0.358396 ms_per_step=3851.15
[2026-09-05 09:01] monitor: epoch=109 step=11460，最新 loss=0.639841（loss_full=0.281445, loss_base=0.358396）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=109 step=11480 loss=0.545754 loss_full=0.232329 loss_base=0.313425 ms_per_step=3791.84
[2026-09-05 09:03] monitor: epoch=109 step=11480，最新 loss=0.545754（loss_full=0.232329, loss_base=0.313425）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=109 step=11500 loss=0.606744 loss_full=0.273864 loss_base=0.332880 ms_per_step=3848.28
[profile] window=382.34s updates=100 data_wait=2.1ms h2d=1.7ms stage1=296.6ms dit=649.3ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 09:04] monitor: epoch=109 step=11500，最新 loss=0.606744（loss_full=0.273864, loss_base=0.332880）；profile window=382.34s，GPU allocated=29.0GiB/reserved=39.7GiB/peak=38.0GiB；trainer PID=17993 存活，错误扫描无命中。
task=ci2v epoch=109 step=11520 loss=0.633522 loss_full=0.288797 loss_base=0.344725 ms_per_step=3777.10
[2026-09-05 09:05] monitor: epoch=109 step=11520，最新 loss=0.633522（loss_full=0.288797, loss_base=0.344725）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=109 step=11540 loss=0.627721 loss_full=0.281182 loss_base=0.346539 ms_per_step=3695.04
[2026-09-05 09:06] monitor: epoch=109 step=11540，最新 loss=0.627721（loss_full=0.281182, loss_base=0.346539）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=109 step=11480 loss=0.545754 loss_full=0.232329 loss_base=0.313425 ms_per_step=3791.84
task=ci2v epoch=109 step=11500 loss=0.606744 loss_full=0.273864 loss_base=0.332880 ms_per_step=3848.28
[profile] window=382.34s updates=100 data_wait=2.1ms h2d=1.7ms stage1=296.6ms dit=649.3ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=109 step=11520 loss=0.633522 loss_full=0.288797 loss_base=0.344725 ms_per_step=3777.10
task=ci2v epoch=109 step=11540 loss=0.627721 loss_full=0.281182 loss_base=0.346539 ms_per_step=3695.04
[epoch-end] epoch=109 elapsed=398.64s step=11550
[epoch-start] epoch=110 step=11550 steps_per_epoch=105
[2026-09-05 09:07] monitor: epoch=109 已完成（elapsed=398.64s，step=11550），训练进入 epoch=110；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=110 step=11560 loss=0.590329 loss_full=0.250898 loss_base=0.339431 ms_per_step=3840.55
[2026-09-05 09:08] monitor: epoch=110 step=11560，最新 loss=0.590329（loss_full=0.250898, loss_base=0.339431）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=110 step=11580 loss=0.597747 loss_full=0.257681 loss_base=0.340066 ms_per_step=3826.29
[2026-09-05 09:09] monitor: epoch=110 step=11580，最新 loss=0.597747（loss_full=0.257681, loss_base=0.340066）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=110 step=11600 loss=0.703597 loss_full=0.322253 loss_base=0.381345 ms_per_step=3824.15
[profile] window=379.52s updates=100 data_wait=2.1ms h2d=1.6ms stage1=288.2ms dit=650.5ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 09:10] monitor: epoch=110 step=11600，最新 loss=0.703597（loss_full=0.322253, loss_base=0.381345）；profile window=379.52s，GPU allocated=29.0GiB/reserved=39.7GiB/peak=38.0GiB；trainer PID=17993 存活，错误扫描无命中。
task=ci2v epoch=110 step=11620 loss=0.549342 loss_full=0.234278 loss_base=0.315063 ms_per_step=3842.55
[2026-09-05 09:11] monitor: epoch=110 step=11620，最新 loss=0.549342（loss_full=0.234278, loss_base=0.315063）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=110 step=11640 loss=0.638359 loss_full=0.288645 loss_base=0.349714 ms_per_step=3756.56
[2026-09-05 09:13] monitor: epoch=110 step=11640，最新 loss=0.638359（loss_full=0.288645, loss_base=0.349714）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 09:14] monitor: epoch=110 已完成（elapsed=400.96s，step=11655），训练进入 epoch=111；最新 loss=0.627705（loss_full=0.266005, loss_base=0.361701）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=110 step=11560 loss=0.590329 loss_full=0.250898 loss_base=0.339431 ms_per_step=3840.55
task=ci2v epoch=110 step=11580 loss=0.597747 loss_full=0.257681 loss_base=0.340066 ms_per_step=3826.29
task=ci2v epoch=110 step=11600 loss=0.703597 loss_full=0.322253 loss_base=0.381345 ms_per_step=3824.15
[profile] window=379.52s updates=100 data_wait=2.1ms h2d=1.6ms stage1=288.2ms dit=650.5ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=110 step=11620 loss=0.549342 loss_full=0.234278 loss_base=0.315063 ms_per_step=3842.55
task=ci2v epoch=110 step=11640 loss=0.638359 loss_full=0.288645 loss_base=0.349714 ms_per_step=3756.56
[epoch-end] epoch=110 elapsed=400.96s step=11655
[epoch-start] epoch=111 step=11655 steps_per_epoch=105
task=ci2v epoch=111 step=11660 loss=0.627705 loss_full=0.266005 loss_base=0.361701 ms_per_step=3808.67
task=ci2v epoch=111 step=11680 loss=0.592237 loss_full=0.256352 loss_base=0.335885 ms_per_step=3729.02
[2026-09-05 09:15] monitor: epoch=111 step=11680，最新 loss=0.592237（loss_full=0.256352, loss_base=0.335885）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=111 step=11700 loss=0.641925 loss_full=0.289375 loss_base=0.352549 ms_per_step=3791.73
[profile] window=378.82s updates=100 data_wait=2.0ms h2d=1.6ms stage1=288.9ms dit=648.1ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 09:16] monitor: epoch=111 step=11700，最新 loss=0.641925（loss_full=0.289375, loss_base=0.352549）；profile window=378.82s，GPU allocated=29.0GiB/reserved=39.7GiB/peak=38.0GiB；trainer PID=17993 存活，错误扫描无命中。
task=ci2v epoch=111 step=11720 loss=0.705551 loss_full=0.333957 loss_base=0.371594 ms_per_step=3829.12
[2026-09-05 09:18] monitor: epoch=111 step=11720，最新 loss=0.705551（loss_full=0.333957, loss_base=0.371594）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=111 step=11740 loss=0.688185 loss_full=0.307924 loss_base=0.380261 ms_per_step=3861.75
[2026-09-05 09:19] monitor: epoch=111 step=11740，最新 loss=0.688185（loss_full=0.307924, loss_base=0.380261）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=111 step=11700 loss=0.641925 loss_full=0.289375 loss_base=0.352549 ms_per_step=3791.73
[profile] window=378.82s updates=100 data_wait=2.0ms h2d=1.6ms stage1=288.9ms dit=648.1ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=111 step=11720 loss=0.705551 loss_full=0.333957 loss_base=0.371594 ms_per_step=3829.12
task=ci2v epoch=111 step=11740 loss=0.688185 loss_full=0.307924 loss_base=0.380261 ms_per_step=3861.75
task=ci2v epoch=111 step=11760 loss=0.579166 loss_full=0.260596 loss_base=0.318570 ms_per_step=3553.08
[epoch-end] epoch=111 elapsed=395.90s step=11760
[epoch-start] epoch=112 step=11760 steps_per_epoch=105
[2026-09-05 09:20] monitor: epoch=111 已完成（elapsed=395.90s，step=11760），训练进入 epoch=112；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=112 step=11780 loss=0.709115 loss_full=0.325054 loss_base=0.384061 ms_per_step=3876.41
[2026-09-05 09:21] monitor: epoch=112 step=11780，最新 loss=0.709115（loss_full=0.325054, loss_base=0.384061）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=112 step=11800 loss=0.612951 loss_full=0.268613 loss_base=0.344338 ms_per_step=3809.26
[profile] window=378.84s updates=100 data_wait=2.2ms h2d=1.6ms stage1=287.7ms dit=648.6ms update=30.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 09:23] monitor: epoch=112 step=11800，最新 loss=0.612951（loss_full=0.268613, loss_base=0.344338）；profile window=378.84s，GPU allocated=29.0GiB/reserved=39.7GiB/peak=38.0GiB；trainer PID=17993 存活，错误扫描无命中。
task=ci2v epoch=112 step=11820 loss=0.590449 loss_full=0.265639 loss_base=0.324809 ms_per_step=3763.59
[2026-09-05 09:24] monitor: epoch=112 step=11820，最新 loss=0.590449（loss_full=0.265639, loss_base=0.324809）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=112 step=11840 loss=0.654139 loss_full=0.291978 loss_base=0.362161 ms_per_step=3794.43
[2026-09-05 09:25] monitor: epoch=112 step=11840，最新 loss=0.654139（loss_full=0.291978, loss_base=0.362161）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=112 step=11860 loss=0.590812 loss_full=0.257701 loss_base=0.333112 ms_per_step=3819.18
[2026-09-05 09:26] monitor: epoch=112 step=11860，最新 loss=0.590812（loss_full=0.257701, loss_base=0.333112）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=112 step=11780 loss=0.709115 loss_full=0.325054 loss_base=0.384061 ms_per_step=3876.41
task=ci2v epoch=112 step=11800 loss=0.612951 loss_full=0.268613 loss_base=0.344338 ms_per_step=3809.26
[profile] window=378.84s updates=100 data_wait=2.2ms h2d=1.6ms stage1=287.7ms dit=648.6ms update=30.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=112 step=11820 loss=0.590449 loss_full=0.265639 loss_base=0.324809 ms_per_step=3763.59
task=ci2v epoch=112 step=11840 loss=0.654139 loss_full=0.291978 loss_base=0.362161 ms_per_step=3794.43
task=ci2v epoch=112 step=11860 loss=0.590812 loss_full=0.257701 loss_base=0.333112 ms_per_step=3819.18
[epoch-end] epoch=112 elapsed=399.86s step=11865
[epoch-start] epoch=113 step=11865 steps_per_epoch=105
[2026-09-05 09:27] monitor: epoch=112 已完成（elapsed=399.86s，step=11865），训练进入 epoch=113；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=113 step=11880 loss=0.675865 loss_full=0.316928 loss_base=0.358937 ms_per_step=3831.31
[2026-09-05 09:28] monitor: epoch=113 step=11880，最新 loss=0.675865（loss_full=0.316928, loss_base=0.358937）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=113 step=11900 loss=0.549419 loss_full=0.238886 loss_base=0.310534 ms_per_step=3776.28
[profile] window=379.94s updates=100 data_wait=2.0ms h2d=1.7ms stage1=293.5ms dit=646.4ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 09:29] monitor: epoch=113 step=11900，最新 loss=0.549419（loss_full=0.238886, loss_base=0.310534）；profile window=379.94s，GPU allocated=29.0GiB/reserved=39.7GiB/peak=38.0GiB；trainer PID=17993 存活，错误扫描无命中。
task=ci2v epoch=113 step=11920 loss=0.645948 loss_full=0.294195 loss_base=0.351753 ms_per_step=3799.98
[2026-09-05 09:30] monitor: epoch=113 step=11920，最新 loss=0.645948（loss_full=0.294195, loss_base=0.351753）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=113 step=11940 loss=0.711279 loss_full=0.329484 loss_base=0.381795 ms_per_step=3813.79
[2026-09-05 09:32] monitor: epoch=113 step=11940，最新 loss=0.711279（loss_full=0.329484, loss_base=0.381795）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=113 step=11960 loss=0.708009 loss_full=0.317245 loss_base=0.390764 ms_per_step=3681.64
[2026-09-05 09:33] monitor: epoch=113 step=11960，最新 loss=0.708009（loss_full=0.317245, loss_base=0.390764）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=113 step=11880 loss=0.675865 loss_full=0.316928 loss_base=0.358937 ms_per_step=3831.31
task=ci2v epoch=113 step=11900 loss=0.549419 loss_full=0.238886 loss_base=0.310534 ms_per_step=3776.28
[profile] window=379.94s updates=100 data_wait=2.0ms h2d=1.7ms stage1=293.5ms dit=646.4ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=113 step=11920 loss=0.645948 loss_full=0.294195 loss_base=0.351753 ms_per_step=3799.98
task=ci2v epoch=113 step=11940 loss=0.711279 loss_full=0.329484 loss_base=0.381795 ms_per_step=3813.79
task=ci2v epoch=113 step=11960 loss=0.708009 loss_full=0.317245 loss_base=0.390764 ms_per_step=3681.64
[epoch-end] epoch=113 elapsed=397.25s step=11970
[epoch-start] epoch=114 step=11970 steps_per_epoch=105
[2026-09-05 09:33] monitor: epoch=113 已完成（elapsed=397.25s，step=11970），训练进入 epoch=114；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=114 step=11980 loss=0.616387 loss_full=0.274993 loss_base=0.341394 ms_per_step=3750.52
[2026-09-05 09:34] monitor: epoch=114 step=11980，最新 loss=0.616387（loss_full=0.274993, loss_base=0.341394）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=114 step=11980 loss=0.616387 loss_full=0.274993 loss_base=0.341394 ms_per_step=3750.52
task=ci2v epoch=114 step=12000 loss=0.572040 loss_full=0.250161 loss_base=0.321879 ms_per_step=3728.72
[profile] window=375.75s updates=100 data_wait=1.6ms h2d=1.6ms stage1=283.7ms dit=646.3ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=12000 elapsed=2.83s
[checkpoint] step=12000 elapsed=31.90s
[2026-09-05 09:36] monitor: step=12000，最新 loss=0.572040（loss_full=0.250161, loss_base=0.321879）；采样与检查点均完成（step-00012000.pt，约 21.9 GB）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=114 step=12020 loss=0.691549 loss_full=0.317268 loss_base=0.374281 ms_per_step=3804.75
[2026-09-05 09:37] monitor: epoch=114 step=12020，最新 loss=0.691549（loss_full=0.317268, loss_base=0.374281）；step=12000 检查点后训练已恢复，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=114 step=12040 loss=0.503774 loss_full=0.222509 loss_base=0.281265 ms_per_step=3812.42
[2026-09-05 09:39] monitor: epoch=114 step=12040，最新 loss=0.503774（loss_full=0.222509, loss_base=0.281265）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=114 step=12060 loss=0.680740 loss_full=0.307722 loss_base=0.373017 ms_per_step=3659.04
[2026-09-05 09:40] monitor: epoch=114 step=12060，最新 loss=0.680740（loss_full=0.307722, loss_base=0.373017）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=114 step=12020 loss=0.691549 loss_full=0.317268 loss_base=0.374281 ms_per_step=3804.75
task=ci2v epoch=114 step=12040 loss=0.503774 loss_full=0.222509 loss_base=0.281265 ms_per_step=3812.42
task=ci2v epoch=114 step=12060 loss=0.680740 loss_full=0.307722 loss_base=0.373017 ms_per_step=3659.04
[epoch-end] epoch=114 elapsed=428.81s step=12075
[epoch-start] epoch=115 step=12075 steps_per_epoch=105
[2026-09-05 09:41] monitor: epoch=114 已完成（elapsed=428.81s，step=12075），训练进入 epoch=115；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=115 step=12080 loss=0.638010 loss_full=0.280906 loss_base=0.357104 ms_per_step=3789.44
[2026-09-05 09:41] monitor: epoch=115 step=12080，最新 loss=0.638010（loss_full=0.280906, loss_base=0.357104）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=115 step=12100 loss=0.579375 loss_full=0.254446 loss_base=0.324929 ms_per_step=3688.12
[profile] window=375.34s updates=100 data_wait=2.2ms h2d=1.6ms stage1=283.4ms dit=645.0ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 09:42] monitor: epoch=115 step=12100，最新 loss=0.579375（loss_full=0.254446, loss_base=0.324929）；profile window=375.34s，GPU allocated=29.0GiB/reserved=39.7GiB/peak=38.0GiB；trainer PID=17993 存活，错误扫描无命中。
task=ci2v epoch=115 step=12120 loss=0.649902 loss_full=0.296071 loss_base=0.353830 ms_per_step=3803.80
[2026-09-05 09:43] monitor: epoch=115 step=12120，最新 loss=0.649902（loss_full=0.296071, loss_base=0.353830）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=115 step=12140 loss=0.677712 loss_full=0.312131 loss_base=0.365580 ms_per_step=3664.73
[2026-09-05 09:45] monitor: epoch=115 step=12140，最新 loss=0.677712（loss_full=0.312131, loss_base=0.365580）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=115 step=12160 loss=0.569701 loss_full=0.250659 loss_base=0.319042 ms_per_step=3665.05
[2026-09-05 09:46] monitor: epoch=115 step=12160，最新 loss=0.569701（loss_full=0.250659, loss_base=0.319042）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=115 step=12080 loss=0.638010 loss_full=0.280906 loss_base=0.357104 ms_per_step=3789.44
task=ci2v epoch=115 step=12100 loss=0.579375 loss_full=0.254446 loss_base=0.324929 ms_per_step=3688.12
[profile] window=375.34s updates=100 data_wait=2.2ms h2d=1.6ms stage1=283.4ms dit=645.0ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=115 step=12120 loss=0.649902 loss_full=0.296071 loss_base=0.353830 ms_per_step=3803.80
task=ci2v epoch=115 step=12140 loss=0.677712 loss_full=0.312131 loss_base=0.365580 ms_per_step=3664.73
task=ci2v epoch=115 step=12160 loss=0.569701 loss_full=0.250659 loss_base=0.319042 ms_per_step=3665.05
task=ci2v epoch=115 step=12180 loss=0.688470 loss_full=0.315585 loss_base=0.372885 ms_per_step=3575.63
[epoch-end] epoch=115 elapsed=388.55s step=12180
[epoch-start] epoch=116 step=12180 steps_per_epoch=105
[2026-09-05 09:47] monitor: epoch=115 已完成（elapsed=388.55s，step=12180），训练进入 epoch=116；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=116 step=12200 loss=0.598510 loss_full=0.259995 loss_base=0.338515 ms_per_step=3728.19
[profile] window=369.00s updates=100 data_wait=1.7ms h2d=1.5ms stage1=270.7ms dit=642.3ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 09:48] monitor: epoch=116 step=12200，最新 loss=0.598510（loss_full=0.259995, loss_base=0.338515）；profile window=369.00s，GPU allocated=29.0GiB/reserved=39.7GiB/peak=38.0GiB；trainer PID=17993 存活，错误扫描无命中。
task=ci2v epoch=116 step=12220 loss=0.591322 loss_full=0.257704 loss_base=0.333619 ms_per_step=3676.49
[2026-09-05 09:50] monitor: epoch=116 step=12220，最新 loss=0.591322（loss_full=0.257704, loss_base=0.333619）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=116 step=12220 loss=0.591322 loss_full=0.257704 loss_base=0.333619 ms_per_step=3676.49
task=ci2v epoch=116 step=12240 loss=0.558180 loss_full=0.239775 loss_base=0.318406 ms_per_step=3573.11
task=ci2v epoch=116 step=12260 loss=0.659396 loss_full=0.297218 loss_base=0.362177 ms_per_step=3692.11
[2026-09-05 09:54] monitor: epoch=116 已完成（elapsed=388.50s，step=12285），训练进入 epoch=117；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 09:52] monitor: epoch=116 step=12260，最新 loss=0.659396（loss_full=0.297218, loss_base=0.362177）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=116 step=12280 loss=0.598209 loss_full=0.268310 loss_base=0.329899 ms_per_step=3822.55
[epoch-end] epoch=116 elapsed=388.50s step=12285
[epoch-start] epoch=117 step=12285 steps_per_epoch=105
task=ci2v epoch=117 step=12300 loss=0.619487 loss_full=0.278119 loss_base=0.341368 ms_per_step=3820.37
[profile] window=371.95s updates=100 data_wait=2.1ms h2d=1.5ms stage1=277.1ms dit=642.9ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 09:55] monitor: epoch=117 step=12300，最新 loss=0.619487（loss_full=0.278119, loss_base=0.341368）；profile window=371.95s，GPU allocated=29.0GiB/reserved=39.7GiB/peak=38.0GiB；trainer PID=17993 存活，错误扫描无命中。
task=ci2v epoch=117 step=12320 loss=0.629296 loss_full=0.285543 loss_base=0.343753 ms_per_step=3683.05
[2026-09-05 09:56] monitor: epoch=117 step=12320，最新 loss=0.629296（loss_full=0.285543, loss_base=0.343753）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=117 step=12340 loss=0.576846 loss_full=0.253707 loss_base=0.323139 ms_per_step=3589.03
[2026-09-05 09:58] monitor: epoch=117 step=12340，最新 loss=0.576846（loss_full=0.253707, loss_base=0.323139）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=117 step=12360 loss=0.704065 loss_full=0.317606 loss_base=0.386459 ms_per_step=3639.67
[2026-09-05 09:58] monitor: epoch=117 step=12360，最新 loss=0.704065（loss_full=0.317606, loss_base=0.386459）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=117 step=12380 loss=0.605896 loss_full=0.264970 loss_base=0.340926 ms_per_step=3838.62
[2026-09-05 09:59] monitor: epoch=117 step=12380，最新 loss=0.605896（loss_full=0.264970, loss_base=0.340926）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[epoch-end] epoch=117 elapsed=390.58s step=12390
[epoch-start] epoch=118 step=12390 steps_per_epoch=105
[2026-09-05 10:00] monitor: epoch=117 已完成（elapsed=390.58s，step=12390），训练进入 epoch=118；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=118 step=12400 loss=0.696101 loss_full=0.325783 loss_base=0.370318 ms_per_step=3813.60
[profile] window=371.53s updates=100 data_wait=1.9ms h2d=1.5ms stage1=272.6ms dit=646.3ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 10:02] monitor: epoch=118 step=12400，最新 loss=0.696101（loss_full=0.325783, loss_base=0.370318）；profile window=371.53s，GPU allocated=29.0GiB/reserved=39.7GiB/peak=38.0GiB；trainer PID=17993 存活，错误扫描无命中。
task=ci2v epoch=118 step=12420 loss=0.633570 loss_full=0.281593 loss_base=0.351976 ms_per_step=3769.30
[2026-09-05 10:02] monitor: epoch=118 step=12420，最新 loss=0.633570（loss_full=0.281593, loss_base=0.351976）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=118 step=12440 loss=0.608030 loss_full=0.264467 loss_base=0.343562 ms_per_step=3802.12
[2026-09-05 10:04] monitor: epoch=118 step=12440，最新 loss=0.608030（loss_full=0.264467, loss_base=0.343562）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=118 step=12460 loss=0.632756 loss_full=0.282306 loss_base=0.350450 ms_per_step=3795.26
[2026-09-05 10:04] monitor: epoch=118 step=12460，最新 loss=0.632756（loss_full=0.282306, loss_base=0.350450）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=118 step=12480 loss=0.542746 loss_full=0.230361 loss_base=0.312386 ms_per_step=3648.13
[2026-09-05 10:06] monitor: epoch=118 step=12480，最新 loss=0.542746（loss_full=0.230361, loss_base=0.312386）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[epoch-end] epoch=118 elapsed=391.89s step=12495
[epoch-start] epoch=119 step=12495 steps_per_epoch=105
[2026-09-05 10:07] monitor: epoch=118 已完成（elapsed=391.89s，step=12495），训练进入 epoch=119；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=119 step=12500 loss=0.613408 loss_full=0.270063 loss_base=0.343344 ms_per_step=3601.26
[profile] window=372.59s updates=100 data_wait=2.2ms h2d=1.5ms stage1=276.6ms dit=644.9ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 10:07] monitor: epoch=119 step=12500，最新 loss=0.613408（loss_full=0.270063, loss_base=0.343344）；profile window=372.59s，GPU allocated=29.0GiB/reserved=39.7GiB/peak=38.0GiB；trainer PID=17993 存活，错误扫描无命中。
task=ci2v epoch=119 step=12520 loss=0.568656 loss_full=0.236264 loss_base=0.332392 ms_per_step=3885.92
[2026-09-05 10:09] monitor: epoch=119 step=12520，最新 loss=0.568656（loss_full=0.236264, loss_base=0.332392）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=119 step=12540 loss=0.587446 loss_full=0.264890 loss_base=0.322556 ms_per_step=3900.82
[2026-09-05 10:09] monitor: epoch=119 step=12540，最新 loss=0.587446（loss_full=0.264890, loss_base=0.322556）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=119 step=12560 loss=0.631810 loss_full=0.285549 loss_base=0.346261 ms_per_step=3788.61
[2026-09-05 10:11] monitor: epoch=119 step=12560，最新 loss=0.631810（loss_full=0.285549, loss_base=0.346261）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=119 step=12580 loss=0.602332 loss_full=0.276893 loss_base=0.325439 ms_per_step=3726.75
[2026-09-05 10:12] monitor: epoch=119 step=12580，最新 loss=0.602332（loss_full=0.276893, loss_base=0.325439）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=119 step=12600 loss=0.586541 loss_full=0.262491 loss_base=0.324050 ms_per_step=3757.16
[profile] window=381.45s updates=100 data_wait=1.4ms h2d=1.9ms stage1=290.2ms dit=654.4ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[epoch-end] epoch=119 elapsed=401.72s step=12600
[epoch-start] epoch=120 step=12600 steps_per_epoch=105
[2026-09-05 10:14] monitor: epoch=119 已完成（elapsed=401.72s，step=12600），训练进入 epoch=120；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=120 step=12620 loss=0.623403 loss_full=0.276294 loss_base=0.347110 ms_per_step=3744.50
[2026-09-05 10:15] monitor: epoch=120 step=12620，最新 loss=0.623403（loss_full=0.276294, loss_base=0.347110）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=120 step=12640 loss=0.567230 loss_full=0.238446 loss_base=0.328784 ms_per_step=3865.24
[2026-09-05 10:16] monitor: epoch=120 step=12640，最新 loss=0.567230（loss_full=0.238446, loss_base=0.328784）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=120 step=12660 loss=0.651831 loss_full=0.297242 loss_base=0.354589 ms_per_step=3814.48
[2026-09-05 10:17] monitor: epoch=120 step=12660，最新 loss=0.651831（loss_full=0.297242, loss_base=0.354589）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=120 step=12680 loss=0.699506 loss_full=0.324488 loss_base=0.375017 ms_per_step=3836.20
[2026-09-05 10:18] monitor: epoch=120 step=12680，最新 loss=0.699506（loss_full=0.324488, loss_base=0.375017）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=120 step=12700 loss=0.603595 loss_full=0.265969 loss_base=0.337626 ms_per_step=3833.41
[profile] window=382.13s updates=100 data_wait=2.1ms h2d=1.6ms stage1=292.5ms dit=652.4ms update=28.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 10:20] monitor: epoch=120 step=12700，最新 loss=0.603595（loss_full=0.265969, loss_base=0.337626）；profile window=382.13s，GPU allocated=29.0GiB/reserved=39.7GiB/peak=38.0GiB；trainer PID=17993 存活，错误扫描无命中。
[epoch-end] epoch=120 elapsed=400.22s step=12705
[epoch-start] epoch=121 step=12705 steps_per_epoch=105
[2026-09-05 10:20] monitor: epoch=120 已完成（elapsed=400.22s，step=12705），训练进入 epoch=121；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=121 step=12720 loss=0.559079 loss_full=0.235592 loss_base=0.323487 ms_per_step=3725.56
[2026-09-05 10:21] monitor: epoch=121 step=12720，最新 loss=0.559079（loss_full=0.235592, loss_base=0.323487）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=121 step=12740 loss=0.640563 loss_full=0.289308 loss_base=0.351254 ms_per_step=3833.35
[2026-09-05 10:22] monitor: epoch=121 step=12740，最新 loss=0.640563（loss_full=0.289308, loss_base=0.351254）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=121 step=12760 loss=0.579343 loss_full=0.247097 loss_base=0.332247 ms_per_step=3830.66
[2026-09-05 10:23] monitor: epoch=121 step=12760，最新 loss=0.579343（loss_full=0.247097, loss_base=0.332247）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=121 step=12780 loss=0.704741 loss_full=0.314464 loss_base=0.390277 ms_per_step=3844.91
[2026-09-05 10:25] monitor: epoch=121 step=12780，最新 loss=0.704741（loss_full=0.314464, loss_base=0.390277）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=121 step=12800 loss=0.572431 loss_full=0.248512 loss_base=0.323919 ms_per_step=3836.98
[profile] window=381.67s updates=100 data_wait=2.3ms h2d=1.6ms stage1=290.6ms dit=653.5ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 10:26] monitor: epoch=121 step=12800，最新 loss=0.572431（loss_full=0.248512, loss_base=0.323919）；profile window=381.67s，GPU allocated=29.0GiB/reserved=39.7GiB/peak=38.0GiB；trainer PID=17993 存活，错误扫描无命中。
[epoch-end] epoch=121 elapsed=400.73s step=12810
[epoch-start] epoch=122 step=12810 steps_per_epoch=105
[2026-09-05 10:27] monitor: epoch=121 已完成（elapsed=400.73s，step=12810），训练进入 epoch=122；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=122 step=12820 loss=0.511816 loss_full=0.214611 loss_base=0.297205 ms_per_step=3807.45
[2026-09-05 10:27] monitor: epoch=122 step=12820，最新 loss=0.511816（loss_full=0.214611, loss_base=0.297205）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=122 step=12840 loss=0.559165 loss_full=0.241316 loss_base=0.317848 ms_per_step=3796.36
[2026-09-05 10:29] monitor: epoch=122 step=12840，最新 loss=0.559165（loss_full=0.241316, loss_base=0.317848）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:30] monitor: epoch=122 step=12860，最新 loss=0.568402（loss_full=0.243476, loss_base=0.324926）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:31] monitor: 复核 epoch=122 step=12860；trainer PID=17993 仍存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:31] monitor: epoch=122 step=12880，最新 loss=0.593470（loss_full=0.264630, loss_base=0.328840）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:31] monitor: 复核 epoch=122 step=12880；trainer PID=17993 仍存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:32] monitor: 复核 epoch=122 step=12880；trainer PID=17993 仍存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中，等待下一个训练日志点。
[2026-09-05 10:32] monitor: epoch=122 step=12900，最新 loss=0.686146（loss_full=0.298894, loss_base=0.387252）；profile 窗口已完成；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:33] monitor: epoch=122 已结束（step=12915，elapsed=398.15s），已进入 epoch=123；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:34] monitor: epoch=123 step=12920，最新 loss=0.559379（loss_full=0.240903, loss_base=0.318476）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:34] monitor: 复核 epoch=123 step=12920；trainer PID=17993 仍存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:35] monitor: epoch=123 step=12940，最新 loss=0.638228（loss_full=0.282814, loss_base=0.355414）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:34] monitor: 再次复核 epoch=123 step=12920；trainer PID=17993 与 torchrun 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:33] monitor: 复核 epoch=122 step=12900；trainer PID=17993 与 torchrun 存活，CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=122 step=12860 loss=0.568402 loss_full=0.243476 loss_base=0.324926 ms_per_step=3811.15
task=ci2v epoch=122 step=12880 loss=0.593470 loss_full=0.264630 loss_base=0.328840 ms_per_step=3761.13
task=ci2v epoch=122 step=12900 loss=0.686146 loss_full=0.298894 loss_base=0.387252 ms_per_step=3808.64
[profile] window=379.94s updates=100 data_wait=2.2ms h2d=1.7ms stage1=294.3ms dit=645.5ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[epoch-end] epoch=122 elapsed=398.15s step=12915
[epoch-start] epoch=123 step=12915 steps_per_epoch=105
task=ci2v epoch=123 step=12920 loss=0.559379 loss_full=0.240903 loss_base=0.318476 ms_per_step=3745.95
task=ci2v epoch=123 step=12940 loss=0.638228 loss_full=0.282814 loss_base=0.355414 ms_per_step=3679.09
[2026-09-05 10:36] monitor: epoch=123 step=12960，最新 loss=0.636627（loss_full=0.288027, loss_base=0.348600）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:35] monitor: 复核 epoch=123 step=12940；trainer PID=17993 仍存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=123 step=12960 loss=0.636627 loss_full=0.288027 loss_base=0.348600 ms_per_step=3451.38
[2026-09-05 10:37] monitor: 复核 epoch=123 step=12960；trainer PID=17993 仍存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=123 step=12980 loss=0.569135 loss_full=0.249744 loss_base=0.319391 ms_per_step=3816.31
[2026-09-05 10:37] monitor: epoch=123 step=12980，最新 loss=0.569135（loss_full=0.249744, loss_base=0.319391）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:38] monitor: 复核 epoch=123 step=12980；trainer PID=17993 仍存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=123 step=13000 loss=0.571369 loss_full=0.243644 loss_base=0.327725 ms_per_step=3815.72
[profile] window=370.43s updates=100 data_wait=2.1ms h2d=1.5ms stage1=269.8ms dit=646.4ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=13000 elapsed=2.83s
[2026-09-05 10:39] monitor: epoch=123 step=13000，最新 loss=0.571369（loss_full=0.243644, loss_base=0.327725）；profile 已完成并进入 sample/W&B 视频编码（正常）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[checkpoint] step=13000 elapsed=36.15s
[2026-09-05 10:39] monitor: sample 完成并成功保存 checkpoint step=13000（elapsed=36.15s）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:40] monitor: checkpoint step=13000 后复核；trainer PID=17993 与 torchrun 仍存活，CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中，等待后续 step 输出。
task=ci2v epoch=123 step=13020 loss=0.622152 loss_full=0.270314 loss_base=0.351838 ms_per_step=3764.04
[epoch-end] epoch=123 elapsed=429.82s step=13020
[epoch-start] epoch=124 step=13020 steps_per_epoch=105
[2026-09-05 10:41] monitor: epoch=123 已结束（elapsed=429.82s, step=13020），已进入 epoch=124；最新 loss=0.622152（loss_full=0.270314, loss_base=0.351838）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:41] monitor: 复核 epoch=124 step=13020；trainer PID=17993 仍存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:42] monitor: epoch=124 step=13040，最新 loss=0.573482（loss_full=0.261144, loss_base=0.312338）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:42] monitor: 复核 epoch=124 step=13040；trainer PID=17993 仍存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=124 step=13040 loss=0.573482 loss_full=0.261144 loss_base=0.312338 ms_per_step=3843.73
task=ci2v epoch=124 step=13060 loss=0.579723 loss_full=0.244564 loss_base=0.335159 ms_per_step=3470.47
[2026-09-05 10:43] monitor: epoch=124 step=13060，最新 loss=0.579723（loss_full=0.244564, loss_base=0.335159）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:43] monitor: 复核 epoch=124 step=13060；trainer PID=17993 仍存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:44] monitor: 再次复核 epoch=124 step=13060；trainer PID=17993 仍存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=124 step=13080 loss=0.557096 loss_full=0.241524 loss_base=0.315572 ms_per_step=3381.85
[2026-09-05 10:44] monitor: epoch=124 step=13080，最新 loss=0.557096（loss_full=0.241524, loss_base=0.315572）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:45] monitor: 复核 epoch=124 step=13080；trainer PID=17993 仍存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=124 step=13100 loss=0.660055 loss_full=0.299338 loss_base=0.360717 ms_per_step=3489.50
[profile] window=359.26s updates=100 data_wait=2.0ms h2d=1.3ms stage1=248.3ms dit=640.1ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 10:45] monitor: epoch=124 step=13100，最新 loss=0.660055（loss_full=0.299338, loss_base=0.360717）；profile 窗口已完成；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=124 step=13120 loss=0.618120 loss_full=0.269206 loss_base=0.348914 ms_per_step=3735.61
[epoch-end] epoch=124 elapsed=376.64s step=13125
[epoch-start] epoch=125 step=13125 steps_per_epoch=105
[2026-09-05 10:47] monitor: epoch=124 已结束（elapsed=376.64s, step=13125），已进入 epoch=125；最新 loss=0.618120（loss_full=0.269206, loss_base=0.348914）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:47] monitor: 复核 epoch=125 起始 step=13125；trainer PID=17993 仍存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=125 step=13140 loss=0.574017 loss_full=0.245948 loss_base=0.328069 ms_per_step=3794.11
[2026-09-05 10:47] monitor: epoch=125 step=13140，最新 loss=0.574017（loss_full=0.245948, loss_base=0.328069）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:48] monitor: 复核 epoch=125 step=13140；trainer PID=17993 仍存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:48] monitor: 再次复核 epoch=125 step=13140；trainer PID=17993 仍存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=125 step=13160 loss=0.607022 loss_full=0.263807 loss_base=0.343215 ms_per_step=3741.56
[2026-09-05 10:49] monitor: epoch=125 step=13160，最新 loss=0.607022（loss_full=0.263807, loss_base=0.343215）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:49] monitor: 复核 epoch=125 step=13160；trainer PID=17993 仍存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=125 step=13180 loss=0.640359 loss_full=0.285906 loss_base=0.354453 ms_per_step=3506.42
[2026-09-05 10:50] monitor: epoch=125 step=13180，最新 loss=0.640359（loss_full=0.285906, loss_base=0.354453）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:50] monitor: 复核 epoch=125 step=13180；trainer PID=17993 仍存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=125 step=13200 loss=0.566729 loss_full=0.244140 loss_base=0.322589 ms_per_step=3503.28
[profile] window=365.88s updates=100 data_wait=1.9ms h2d=1.4ms stage1=263.2ms dit=641.9ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 10:51] monitor: epoch=125 step=13200，最新 loss=0.566729（loss_full=0.244140, loss_base=0.322589）；profile 窗口已完成；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:52] monitor: 复核 epoch=125 step=13200；trainer PID=17993 仍存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:52] monitor: 再次复核 epoch=125 step=13200；trainer PID=17993 仍存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=125 step=13220 loss=0.646949 loss_full=0.286118 loss_base=0.360831 ms_per_step=3740.54
[2026-09-05 10:53] monitor: epoch=125 step=13220，最新 loss=0.646949（loss_full=0.286118, loss_base=0.360831）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[epoch-end] epoch=125 elapsed=385.17s step=13230
[epoch-start] epoch=126 step=13230 steps_per_epoch=105
[2026-09-05 10:53] monitor: epoch=125 已结束（elapsed=385.17s, step=13230），已进入 epoch=126；最新 loss=0.646949（loss_full=0.286118, loss_base=0.360831）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=126 step=13240 loss=0.577135 loss_full=0.257596 loss_base=0.319539 ms_per_step=3811.81
[2026-09-05 10:54] monitor: epoch=126 step=13240，最新 loss=0.577135（loss_full=0.257596, loss_base=0.319539）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:54] monitor: 复核 epoch=126 step=13240；trainer PID=17993 仍存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:55] monitor: 再次复核 epoch=126 step=13240；trainer PID=17993 仍存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=126 step=13260 loss=0.657424 loss_full=0.294225 loss_base=0.363199 ms_per_step=3775.32
[2026-09-05 10:55] monitor: epoch=126 step=13260，最新 loss=0.657424（loss_full=0.294225, loss_base=0.363199）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:56] monitor: 复核 epoch=126 step=13260；trainer PID=17993 仍存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=126 step=13280 loss=0.498470 loss_full=0.200701 loss_base=0.297769 ms_per_step=3707.95
[2026-09-05 10:56] monitor: epoch=126 step=13280，最新 loss=0.498470（loss_full=0.200701, loss_base=0.297769）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:57] monitor: 复核 epoch=126 step=13280；trainer PID=17993 仍存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:57] monitor: 再次复核 epoch=126 step=13280；trainer PID=17993 仍存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=126 step=13300 loss=0.627238 loss_full=0.286104 loss_base=0.341133 ms_per_step=3466.47
[profile] window=370.29s updates=100 data_wait=2.0ms h2d=1.5ms stage1=272.5ms dit=643.2ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 10:58] monitor: epoch=126 step=13300，最新 loss=0.627238（loss_full=0.286104, loss_base=0.341133）；profile 窗口已完成；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:58] monitor: 复核 epoch=126 step=13300；trainer PID=17993 仍存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=126 step=13320 loss=0.657259 loss_full=0.303604 loss_base=0.353654 ms_per_step=3474.23
[2026-09-05 10:59] monitor: epoch=126 step=13320，最新 loss=0.657259（loss_full=0.303604, loss_base=0.353654）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 10:59] monitor: 复核 epoch=126 step=13320；trainer PID=17993 仍存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[epoch-end] epoch=126 elapsed=382.80s step=13335
[epoch-start] epoch=127 step=13335 steps_per_epoch=105
[2026-09-05 11:00] monitor: epoch=126 已结束（elapsed=382.80s, step=13335），已进入 epoch=127；最新 loss=0.603123（loss_full=0.261135, loss_base=0.341988）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:00] monitor: 复核 epoch=127 step=13340；trainer PID=17993 仍存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=127 step=13340 loss=0.603123 loss_full=0.261135 loss_base=0.341988 ms_per_step=3639.95
task=ci2v epoch=127 step=13360 loss=0.650058 loss_full=0.299590 loss_base=0.350468 ms_per_step=3753.94
[2026-09-05 11:01] monitor: epoch=127 step=13360，最新 loss=0.650058（loss_full=0.299590, loss_base=0.350468）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:01] monitor: 复核 epoch=127 step=13360；trainer PID=17993 仍存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:02] monitor: 再次复核 epoch=127 step=13360；trainer PID=17993 仍存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=127 step=13380 loss=0.647129 loss_full=0.287006 loss_base=0.360123 ms_per_step=3723.90
[2026-09-05 11:03] monitor: epoch=127 step=13380，最新 loss=0.647129（loss_full=0.287006, loss_base=0.360123）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:03] monitor: 复核 epoch=127 step=13380；trainer PID=17993 仍存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=127 step=13400 loss=0.593910 loss_full=0.266524 loss_base=0.327386 ms_per_step=3794.18
[profile] window=367.99s updates=100 data_wait=1.7ms h2d=1.5ms stage1=267.0ms dit=643.9ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 11:04] monitor: epoch=127 step=13400，最新 loss=0.593910（loss_full=0.266524, loss_base=0.327386）；profile 窗口已完成；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:04] monitor: 复核 epoch=127 step=13400；trainer PID=17993 仍存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=127 step=13420 loss=0.566877 loss_full=0.247955 loss_base=0.318922 ms_per_step=3548.80
[2026-09-05 11:05] monitor: epoch=127 step=13420，最新 loss=0.566877（loss_full=0.247955, loss_base=0.318922）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=127 step=13440 loss=0.664186 loss_full=0.289195 loss_base=0.374991 ms_per_step=3655.69
[epoch-end] epoch=127 elapsed=388.11s step=13440
[epoch-start] epoch=128 step=13440 steps_per_epoch=105
[2026-09-05 11:06] monitor: epoch=127 已完成（step=13440，elapsed=388.11s），已进入 epoch=128；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:07] monitor: 复核 epoch=128 已启动（step=13440）；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；尚未到下一条 step 汇报点，错误扫描无命中。
[2026-09-05 11:08] monitor: epoch=128 step=13460，最新 loss=0.562154（loss_full=0.244727, loss_base=0.317427）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:08] monitor: epoch=128 step=13480，最新 loss=0.629505（loss_full=0.271611, loss_base=0.357894）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:09] monitor: 复核 epoch=128 step=13480 后仍在正常计算；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；尚未到下一条 step 汇报点，错误扫描无命中。
[2026-09-05 11:10] monitor: epoch=128 step=13500，最新 loss=0.624115（loss_full=0.279525, loss_base=0.344590）；profile 窗口完成；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:10] monitor: 复核 epoch=128 step=13500 后仍在正常计算；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；checkpoint 最新为 step-13000，错误扫描无命中。
[2026-09-05 11:11] monitor: epoch=128 step=13520，最新 loss=0.566629（loss_full=0.248014, loss_base=0.318615）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:12] monitor: 复核 epoch=128 step=13520 后仍在正常计算；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:12] monitor: epoch=128 step=13540，最新 loss=0.493535（loss_full=0.209676, loss_base=0.283859）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:12] monitor: epoch=128 已完成（step=13545，elapsed=386.31s），已进入 epoch=129；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:13] monitor: 复核 epoch=129 已启动（step=13545）；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；尚未到首个 step 汇报点，错误扫描无命中。
[2026-09-05 11:13] monitor: epoch=129 step=13560，最新 loss=0.622076（loss_full=0.274422, loss_base=0.347654）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:14] monitor: 复核 epoch=129 step=13560 后仍在正常计算；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:15] monitor: epoch=129 step=13580，最新 loss=0.605197（loss_full=0.271170, loss_base=0.334028）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:15] monitor: 复核 epoch=129 step=13580 后仍在正常计算；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；checkpoint 最新为 step-13000，错误扫描无命中。
[2026-09-05 11:16] monitor: epoch=129 step=13600，最新 loss=0.628393（loss_full=0.281312, loss_base=0.347082）；profile 窗口完成；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:16] monitor: 复核 epoch=129 step=13600 后仍在正常计算；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:17] monitor: 复核 epoch=129 step=13600 后仍在正常计算；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:17] monitor: epoch=129 step=13620，最新 loss=0.596250（loss_full=0.260157, loss_base=0.336093）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:18] monitor: 复核 epoch=129 step=13620 后仍在正常计算；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；checkpoint 最新为 step-13000，错误扫描无命中。
[2026-09-05 11:18] monitor: 再次复核 epoch=129 step=13620；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:19] monitor: epoch=129 step=13640，最新 loss=0.581830（loss_full=0.247137, loss_base=0.334694）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:19] monitor: epoch=129 已完成（step=13650，elapsed=396.04s），已进入 epoch=130；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:19] monitor: 复核 epoch=130 已启动（step=13650）；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；尚未到首个 step 汇报点，错误扫描无命中。
[2026-09-05 11:20] monitor: epoch=130 step=13660，最新 loss=0.588845（loss_full=0.249188, loss_base=0.339658）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:20] monitor: 复核 epoch=130 step=13660 后仍在正常计算；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:21] monitor: epoch=130 step=13680，最新 loss=0.545253（loss_full=0.238320, loss_base=0.306933）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:22] monitor: 复核 epoch=130 step=13680 后仍在正常计算；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:22] monitor: epoch=130 step=13700，最新 loss=0.594673（loss_full=0.258209, loss_base=0.336463）；profile 窗口完成；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:22] monitor: 复核 epoch=130 step=13700 后仍在正常计算；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:23] monitor: 再次复核 epoch=130 step=13700；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:23] monitor: epoch=130 step=13720，最新 loss=0.552295（loss_full=0.231990, loss_base=0.320305）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:24] monitor: 复核 epoch=130 step=13720 后仍在正常计算；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:24] monitor: 再次复核 epoch=130 step=13720；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:24] monitor: epoch=130 step=13740，最新 loss=0.530909（loss_full=0.223679, loss_base=0.307230）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:25] monitor: epoch=130 已完成（step=13755，elapsed=389.87s），已进入 epoch=131；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:26] monitor: epoch=131 step=13760，最新 loss=0.571863（loss_full=0.242338, loss_base=0.329524）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:26] monitor: 复核 epoch=131 step=13760 后仍在正常计算；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:27] monitor: 再次复核 epoch=131 step=13760；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:27] monitor: epoch=131 step=13780，最新 loss=0.643026（loss_full=0.291969, loss_base=0.351057）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:28] monitor: epoch=131 step=13800，最新 loss=0.565818（loss_full=0.236700, loss_base=0.329117）；profile 窗口完成；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:29] monitor: 复核 epoch=131 step=13800 后仍在正常计算；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:29] monitor: 再次复核 epoch=131 step=13800；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:30] monitor: epoch=131 step=13820，最新 loss=0.550206（loss_full=0.239500, loss_base=0.310706）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:30] monitor: 复核 epoch=131 step=13820 后仍在正常计算；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:31] monitor: 再次复核 epoch=131 step=13820；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:31] monitor: epoch=131 step=13840，最新 loss=0.528784（loss_full=0.227513, loss_base=0.301271）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:32] monitor: epoch=131 已完成（step=13860，elapsed=396.44s），已进入 epoch=132；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:33] monitor: 复核 epoch=132 已启动（step=13860）；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；尚未到首个 step 汇报点，错误扫描无命中。
[2026-09-05 11:33] monitor: epoch=132 step=13880，最新 loss=0.622179（loss_full=0.271792, loss_base=0.350387）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:34] monitor: 复核 epoch=132 step=13880 后仍在正常计算；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:34] monitor: 再次复核 epoch=132 step=13880；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:35] monitor: epoch=132 step=13900，最新 loss=0.615335（loss_full=0.267453, loss_base=0.347881）；profile 窗口完成；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:35] monitor: 复核 epoch=132 step=13900 后仍在正常计算；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:36] monitor: epoch=132 step=13920，最新 loss=0.642484（loss_full=0.286783, loss_base=0.355700）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:37] monitor: epoch=132 step=13940，最新 loss=0.645914（loss_full=0.288830, loss_base=0.357084）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:38] monitor: epoch=132 step=13960，最新 loss=0.658768（loss_full=0.301850, loss_base=0.356918）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:39] monitor: epoch=132 已完成（step=13965，elapsed=389.94s），已进入 epoch=133；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:39] monitor: 复核 epoch=133 已启动（step=13965）；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；尚未到首个 step 汇报点，错误扫描无命中。
[2026-09-05 11:40] monitor: epoch=133 step=13980，最新 loss=0.543817（loss_full=0.234088, loss_base=0.309729）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:40] monitor: 复核 epoch=133 step=13980 后仍在正常计算；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:41] monitor: 再次复核 epoch=133 step=13980；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:41] monitor: epoch=133 step=14000，最新 loss=0.560969（loss_full=0.250178, loss_base=0.310791）；profile 完成并触发 sample 视频编码，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:42] monitor: step=14000 sample 已完成且 checkpoint 已保存（elapsed=38.70s，step-14000.pt）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:42] monitor: checkpoint 保存后复核 trainer PID=17993 仍存活且 CPU 约 301%，GPU 显存约 41410 MiB；等待下一条 step 汇报，错误扫描无命中。
[2026-09-05 11:43] monitor: epoch=133 step=14020，checkpoint 后训练已恢复，最新 loss=0.571467（loss_full=0.245523, loss_base=0.325944）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:43] monitor: 复核 epoch=133 step=14020 后仍在正常计算；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；等待下一条 step 汇报，错误扫描无命中。
[2026-09-05 11:44] monitor: epoch=133 step=14040，最新 loss=0.616923（loss_full=0.278844, loss_base=0.338079）；checkpoint 后运行正常，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:44] monitor: 复核 epoch=133 step=14040 后仍在正常计算；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；等待下一条 step 汇报，错误扫描无命中。
[2026-09-05 11:45] monitor: 再次复核 epoch=133 step=14040；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；等待下一条 step 汇报，错误扫描无命中。
[2026-09-05 11:45] monitor: epoch=133 step=14060，最新 loss=0.561107（loss_full=0.237423, loss_base=0.323684）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:38] monitor: 复核 epoch=132 step=13940 后仍在正常计算；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:36] monitor: 复核 epoch=132 step=13920 后仍在正常计算；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:37] monitor: 再次复核 epoch=132 step=13920；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:32] monitor: 复核 epoch=131 step=13840 后仍在正常计算；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:28] monitor: 复核 epoch=131 step=13780 后仍在正常计算；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[2026-09-05 11:25] monitor: 复核 epoch=130 step=13740 后仍在正常计算；trainer PID=17993 存活且 CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=128 step=13460 loss=0.562154 loss_full=0.244727 loss_base=0.317427 ms_per_step=3848.08
task=ci2v epoch=128 step=13480 loss=0.629505 loss_full=0.271611 loss_base=0.357894 ms_per_step=3792.61
task=ci2v epoch=128 step=13500 loss=0.624115 loss_full=0.279525 loss_base=0.344590 ms_per_step=3795.37
[profile] window=373.08s updates=100 data_wait=2.0ms h2d=1.6ms stage1=280.0ms dit=642.7ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=128 step=13520 loss=0.566629 loss_full=0.248014 loss_base=0.318615 ms_per_step=3524.48
task=ci2v epoch=128 step=13540 loss=0.493535 loss_full=0.209676 loss_base=0.283859 ms_per_step=3506.24
[epoch-end] epoch=128 elapsed=386.31s step=13545
[epoch-start] epoch=129 step=13545 steps_per_epoch=105
task=ci2v epoch=129 step=13560 loss=0.622076 loss_full=0.274422 loss_base=0.347654 ms_per_step=3728.57
task=ci2v epoch=129 step=13580 loss=0.605197 loss_full=0.271170 loss_base=0.334028 ms_per_step=3792.31
task=ci2v epoch=129 step=13600 loss=0.628393 loss_full=0.281312 loss_base=0.347082 ms_per_step=3736.31
[profile] window=366.01s updates=100 data_wait=2.1ms h2d=1.4ms stage1=263.4ms dit=642.0ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=129 step=13620 loss=0.596250 loss_full=0.260157 loss_base=0.336093 ms_per_step=3657.80
task=ci2v epoch=129 step=13640 loss=0.581830 loss_full=0.247137 loss_base=0.334694 ms_per_step=3813.06
[epoch-end] epoch=129 elapsed=396.04s step=13650
[epoch-start] epoch=130 step=13650 steps_per_epoch=105
task=ci2v epoch=130 step=13660 loss=0.588845 loss_full=0.249188 loss_base=0.339658 ms_per_step=3793.41
task=ci2v epoch=130 step=13680 loss=0.545253 loss_full=0.238320 loss_base=0.306933 ms_per_step=3473.89
task=ci2v epoch=130 step=13700 loss=0.594673 loss_full=0.258209 loss_base=0.336463 ms_per_step=3687.22
[profile] window=368.77s updates=100 data_wait=1.9ms h2d=1.5ms stage1=269.1ms dit=643.1ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=130 step=13720 loss=0.552295 loss_full=0.231990 loss_base=0.320305 ms_per_step=3803.18
task=ci2v epoch=130 step=13740 loss=0.530909 loss_full=0.223679 loss_base=0.307230 ms_per_step=3807.17
[epoch-end] epoch=130 elapsed=389.87s step=13755
[epoch-start] epoch=131 step=13755 steps_per_epoch=105
task=ci2v epoch=131 step=13760 loss=0.571863 loss_full=0.242338 loss_base=0.329524 ms_per_step=3818.16
task=ci2v epoch=131 step=13780 loss=0.643026 loss_full=0.291969 loss_base=0.351057 ms_per_step=3806.99
task=ci2v epoch=131 step=13800 loss=0.565818 loss_full=0.236700 loss_base=0.329117 ms_per_step=3720.41
[profile] window=379.37s updates=100 data_wait=2.3ms h2d=1.7ms stage1=291.1ms dit=645.9ms update=30.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=131 step=13820 loss=0.550206 loss_full=0.239500 loss_base=0.310706 ms_per_step=3825.77
task=ci2v epoch=131 step=13840 loss=0.528784 loss_full=0.227513 loss_base=0.301271 ms_per_step=3781.46
task=ci2v epoch=131 step=13860 loss=0.520372 loss_full=0.220580 loss_base=0.299791 ms_per_step=3645.42
[epoch-end] epoch=131 elapsed=396.44s step=13860
[epoch-start] epoch=132 step=13860 steps_per_epoch=105
task=ci2v epoch=132 step=13880 loss=0.622179 loss_full=0.271792 loss_base=0.350387 ms_per_step=3822.44
task=ci2v epoch=132 step=13900 loss=0.615335 loss_full=0.267453 loss_base=0.347881 ms_per_step=3789.18
[profile] window=377.54s updates=100 data_wait=1.8ms h2d=1.6ms stage1=284.0ms dit=650.3ms update=27.8ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=132 step=13920 loss=0.642484 loss_full=0.286783 loss_base=0.355700 ms_per_step=3556.67
task=ci2v epoch=132 step=13940 loss=0.645914 loss_full=0.288830 loss_base=0.357084 ms_per_step=3588.06
task=ci2v epoch=132 step=13960 loss=0.658768 loss_full=0.301850 loss_base=0.356918 ms_per_step=3815.80
[epoch-end] epoch=132 elapsed=389.94s step=13965
[epoch-start] epoch=133 step=13965 steps_per_epoch=105
task=ci2v epoch=133 step=13980 loss=0.543817 loss_full=0.234088 loss_base=0.309729 ms_per_step=3838.13
task=ci2v epoch=133 step=14000 loss=0.560969 loss_full=0.250178 loss_base=0.310791 ms_per_step=3802.35
[profile] window=372.28s updates=100 data_wait=2.0ms h2d=1.5ms stage1=273.8ms dit=647.0ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=14000 elapsed=2.81s
[checkpoint] step=14000 elapsed=38.70s
task=ci2v epoch=133 step=14020 loss=0.571467 loss_full=0.245523 loss_base=0.325944 ms_per_step=3809.22
task=ci2v epoch=133 step=14040 loss=0.616923 loss_full=0.278844 loss_base=0.338079 ms_per_step=3462.72
task=ci2v epoch=133 step=14060 loss=0.561107 loss_full=0.237423 loss_base=0.323684 ms_per_step=3327.99
[epoch-end] epoch=133 elapsed=424.12s step=14070
[epoch-start] epoch=134 step=14070 steps_per_epoch=105
task=ci2v epoch=134 step=14080 loss=0.622934 loss_full=0.278653 loss_base=0.344281 ms_per_step=3483.22
[2026-09-05 11:47] monitor: epoch=133 完成（step=14070，elapsed=424.12s），已进入 epoch=134；最新 step=14080 loss=0.622934（loss_full=0.278653, loss_base=0.344281）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=134 step=14100 loss=0.614719 loss_full=0.270758 loss_base=0.343960 ms_per_step=3725.55
[profile] window=356.45s updates=100 data_wait=1.5ms h2d=1.2ms stage1=242.1ms dit=640.1ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 11:48] monitor: epoch=134 step=14100，最新 loss=0.614719（loss_full=0.270758, loss_base=0.343960）；profile window=356.45s 完成，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=134 step=14120 loss=0.660777 loss_full=0.298317 loss_base=0.362460 ms_per_step=3695.02
[2026-09-05 11:49] monitor: epoch=134 step=14120，最新 loss=0.660777（loss_full=0.298317, loss_base=0.362460）；trainer PID=17993 存活，CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=134 step=14140 loss=0.561063 loss_full=0.254768 loss_base=0.306295 ms_per_step=3821.34
[2026-09-05 11:50] monitor: epoch=134 step=14140，最新 loss=0.561063（loss_full=0.254768, loss_base=0.306295）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=134 step=14160 loss=0.610327 loss_full=0.261246 loss_base=0.349081 ms_per_step=3677.48
[2026-09-05 11:51] monitor: epoch=134 step=14160，最新 loss=0.610327（loss_full=0.261246, loss_base=0.349081）；trainer PID=17993 存活，CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
[epoch-end] epoch=134 elapsed=383.85s step=14175
[epoch-start] epoch=135 step=14175 steps_per_epoch=105
[2026-09-05 11:52] monitor: epoch=134 完成（step=14175，elapsed=383.85s），已进入 epoch=135；trainer PID=17993 存活，CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=135 step=14180 loss=0.659890 loss_full=0.298146 loss_base=0.361744 ms_per_step=3412.51
[2026-09-05 11:53] monitor: epoch=135 step=14180，最新 loss=0.659890（loss_full=0.298146, loss_base=0.361744）；trainer PID=17993 存活，CPU 约 301%，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=135 step=14200 loss=0.605554 loss_full=0.267750 loss_base=0.337803 ms_per_step=3430.40
[profile] window=361.00s updates=100 data_wait=1.8ms h2d=1.3ms stage1=251.7ms dit=641.2ms update=29.6ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 11:54] monitor: epoch=135 step=14200，最新 loss=0.605554（loss_full=0.267750, loss_base=0.337803）；profile window=361.00s 完成，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=135 step=14220 loss=0.564711 loss_full=0.239727 loss_base=0.324984 ms_per_step=3745.56
[2026-09-05 11:55] monitor: epoch=135 step=14220，最新 loss=0.564711（loss_full=0.239727, loss_base=0.324984）；GPU 利用率约 69%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=135 step=14240 loss=0.639454 loss_full=0.279197 loss_base=0.360257 ms_per_step=3729.33
[2026-09-05 11:56] monitor: epoch=135 step=14240，最新 loss=0.639454（loss_full=0.279197, loss_base=0.360257）；GPU 利用率约 89%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=135 step=14260 loss=0.519180 loss_full=0.223174 loss_base=0.296006 ms_per_step=3693.82
[2026-09-05 11:57] monitor: epoch=135 step=14260，最新 loss=0.519180（loss_full=0.223174, loss_base=0.296006）；GPU 利用率约 98%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
[2026-09-05 11:58] monitor: epoch=135 step=14260，最新 loss=0.519180（loss_full=0.223174, loss_base=0.296006）；GPU 利用率约 98%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=135 step=14280 loss=0.587168 loss_full=0.247008 loss_base=0.340160 ms_per_step=3656.38
[epoch-end] epoch=135 elapsed=382.90s step=14280
[epoch-start] epoch=136 step=14280 steps_per_epoch=105
[2026-09-05 11:58] monitor: epoch=135 完成（step=14280，elapsed=382.90s），已进入 epoch=136；trainer PID=17993 存活，GPU 利用率约 100%，显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=136 step=14300 loss=0.570463 loss_full=0.255198 loss_base=0.315265 ms_per_step=3804.99
[profile] window=372.86s updates=100 data_wait=1.7ms h2d=1.5ms stage1=279.1ms dit=643.5ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 12:00] monitor: epoch=136 step=14300，最新 loss=0.570463（loss_full=0.255198, loss_base=0.315265）；profile window=372.86s 完成，GPU 利用率约 100%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=136 step=14320 loss=0.580860 loss_full=0.249042 loss_base=0.331817 ms_per_step=3784.63
[2026-09-05 12:01] monitor: epoch=136 step=14320，最新 loss=0.580860（loss_full=0.249042, loss_base=0.331817）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=136 step=14340 loss=0.601127 loss_full=0.267750 loss_base=0.333377 ms_per_step=3811.57
[2026-09-05 12:03] monitor: epoch=136 step=14340，最新 loss=0.601127（loss_full=0.267750, loss_base=0.333377）；GPU 利用率约 91%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=136 step=14360 loss=0.519379 loss_full=0.219707 loss_base=0.299672 ms_per_step=3827.42
[2026-09-05 12:03] monitor: epoch=136 step=14360，最新 loss=0.519379（loss_full=0.219707, loss_base=0.299672）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=136 step=14380 loss=0.648322 loss_full=0.291872 loss_base=0.356450 ms_per_step=3804.72
[2026-09-05 12:05] monitor: epoch=136 step=14380，最新 loss=0.648322（loss_full=0.291872, loss_base=0.356450）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[epoch-end] epoch=136 elapsed=399.23s step=14385
[epoch-start] epoch=137 step=14385 steps_per_epoch=105
[2026-09-05 12:05] monitor: epoch=136 完成（step=14385，elapsed=399.23s），已进入 epoch=137；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=137 step=14400 loss=0.517760 loss_full=0.220942 loss_base=0.296818 ms_per_step=3606.31
[profile] window=376.95s updates=100 data_wait=1.9ms h2d=1.6ms stage1=286.9ms dit=645.4ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 12:06] monitor: epoch=137 step=14400，最新 loss=0.517760（loss_full=0.220942, loss_base=0.296818）；profile window=376.95s 完成，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=137 step=14420 loss=0.560994 loss_full=0.236092 loss_base=0.324902 ms_per_step=3783.68
[2026-09-05 12:07] monitor: epoch=137 step=14420，最新 loss=0.560994（loss_full=0.236092, loss_base=0.324902）；GPU 利用率约 100%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=137 step=14440 loss=0.589004 loss_full=0.259646 loss_base=0.329357 ms_per_step=3489.37
[2026-09-05 12:09] monitor: epoch=137 step=14440，最新 loss=0.589004（loss_full=0.259646, loss_base=0.329357）；GPU 利用率约 91%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=137 step=14460 loss=0.588323 loss_full=0.250038 loss_base=0.338285 ms_per_step=3768.97
[2026-09-05 12:10] monitor: epoch=137 step=14460，最新 loss=0.588323（loss_full=0.250038, loss_base=0.338285）；GPU 利用率约 98%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=137 step=14480 loss=0.575462 loss_full=0.259442 loss_base=0.316020 ms_per_step=3726.54
[epoch-end] epoch=137 elapsed=385.88s step=14490
[epoch-start] epoch=138 step=14490 steps_per_epoch=105
[2026-09-05 12:11] monitor: epoch=137 step=14480，最新 loss=0.575462（loss_full=0.259442, loss_base=0.316020）；GPU 利用率约 100%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
[2026-09-05 12:12] monitor: epoch=137 完成（step=14490，elapsed=385.88s），已进入 epoch=138；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=138 step=14500 loss=0.588198 loss_full=0.261116 loss_base=0.327081 ms_per_step=3674.23
[profile] window=369.12s updates=100 data_wait=2.0ms h2d=1.5ms stage1=271.4ms dit=641.5ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 12:12] monitor: epoch=138 step=14500，最新 loss=0.588198（loss_full=0.261116, loss_base=0.327081）；profile window=369.12s 完成，GPU 利用率约 100%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=138 step=14520 loss=0.590572 loss_full=0.261950 loss_base=0.328622 ms_per_step=3752.29
[2026-09-05 12:13] monitor: epoch=138 step=14520，最新 loss=0.590572（loss_full=0.261950, loss_base=0.328622）；GPU 利用率约 100%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=138 step=14540 loss=0.556527 loss_full=0.240366 loss_base=0.316160 ms_per_step=3732.17
[2026-09-05 12:15] monitor: epoch=138 step=14540，最新 loss=0.556527（loss_full=0.240366, loss_base=0.316160）；GPU 利用率约 98%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=138 step=14560 loss=0.637754 loss_full=0.276876 loss_base=0.360878 ms_per_step=3749.63
[2026-09-05 12:16] monitor: epoch=138 step=14560，最新 loss=0.637754（loss_full=0.276876, loss_base=0.360878）；GPU 利用率约 100%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=138 step=14580 loss=0.501204 loss_full=0.212818 loss_base=0.288386 ms_per_step=3640.36
[2026-09-05 12:17] monitor: epoch=138 step=14580，最新 loss=0.501204（loss_full=0.212818, loss_base=0.288386）；GPU 利用率约 100%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
[epoch-end] epoch=138 elapsed=387.79s step=14595
[epoch-start] epoch=139 step=14595 steps_per_epoch=105
task=ci2v epoch=139 step=14600 loss=0.568178 loss_full=0.246372 loss_base=0.321807 ms_per_step=3595.88
[profile] window=369.68s updates=100 data_wait=1.9ms h2d=1.5ms stage1=271.4ms dit=643.5ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 12:18] monitor: epoch=138 完成（step=14595，elapsed=387.79s），已进入 epoch=139；step=14600 loss=0.568178（loss_full=0.246372, loss_base=0.321807），profile window=369.68s 完成；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=139 step=14620 loss=0.659973 loss_full=0.295343 loss_base=0.364629 ms_per_step=3561.75
[2026-09-05 12:19] monitor: epoch=139 step=14620，最新 loss=0.659973（loss_full=0.295343, loss_base=0.364629）；GPU 利用率约 100%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=139 step=14640 loss=0.580732 loss_full=0.251137 loss_base=0.329595 ms_per_step=3756.00
[2026-09-05 12:21] monitor: epoch=139 step=14640，最新 loss=0.580732（loss_full=0.251137, loss_base=0.329595）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=139 step=14660 loss=0.652111 loss_full=0.292825 loss_base=0.359286 ms_per_step=3844.14
[2026-09-05 12:22] monitor: epoch=139 step=14660，最新 loss=0.652111（loss_full=0.292825, loss_base=0.359286）；GPU 利用率约 98%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=139 step=14680 loss=0.612709 loss_full=0.266832 loss_base=0.345877 ms_per_step=3791.95
[2026-09-05 12:23] monitor: epoch=139 step=14680，最新 loss=0.612709（loss_full=0.266832, loss_base=0.345877）；GPU 利用率约 88%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=139 step=14700 loss=0.513571 loss_full=0.210617 loss_base=0.302954 ms_per_step=3642.77
[profile] window=372.18s updates=100 data_wait=1.1ms h2d=1.5ms stage1=277.2ms dit=644.7ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[epoch-end] epoch=139 elapsed=391.72s step=14700
[epoch-start] epoch=140 step=14700 steps_per_epoch=105
[2026-09-05 12:24] monitor: epoch=139 完成（step=14700，elapsed=391.72s），已进入 epoch=140；step=14700 loss=0.513571（loss_full=0.210617, loss_base=0.302954），profile window=372.18s 完成；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=140 step=14720 loss=0.604137 loss_full=0.271358 loss_base=0.332779 ms_per_step=3785.76
[2026-09-05 12:26] monitor: epoch=140 step=14720，最新 loss=0.604137（loss_full=0.271358, loss_base=0.332779）；GPU 利用率约 100%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=140 step=14740 loss=0.556998 loss_full=0.237154 loss_base=0.319844 ms_per_step=3670.17
[2026-09-05 12:27] monitor: epoch=140 step=14740，最新 loss=0.556998（loss_full=0.237154, loss_base=0.319844）；GPU 利用率约 100%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=140 step=14760 loss=0.582127 loss_full=0.253520 loss_base=0.328607 ms_per_step=3500.26
[2026-09-05 12:28] monitor: epoch=140 step=14760，最新 loss=0.582127（loss_full=0.253520, loss_base=0.328607）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=140 step=14780 loss=0.593153 loss_full=0.269250 loss_base=0.323903 ms_per_step=3798.37
[2026-09-05 12:30] monitor: epoch=140 step=14780，最新 loss=0.593153（loss_full=0.269250, loss_base=0.323903）；GPU 利用率约 100%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=140 step=14800 loss=0.582223 loss_full=0.244879 loss_base=0.337344 ms_per_step=3783.62
[profile] window=371.04s updates=100 data_wait=2.0ms h2d=1.5ms stage1=273.9ms dit=643.7ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 12:31] monitor: epoch=140 step=14800，最新 loss=0.582223（loss_full=0.244879, loss_base=0.337344）；profile window=371.04s 完成，GPU 利用率约 83%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
[epoch-end] epoch=140 elapsed=389.15s step=14805
[epoch-start] epoch=141 step=14805 steps_per_epoch=105
[2026-09-05 12:31] monitor: epoch=140 完成（step=14805，elapsed=389.15s），已进入 epoch=141；profile window=371.04s 完成，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=141 step=14820 loss=0.593189 loss_full=0.267013 loss_base=0.326176 ms_per_step=3782.59
[2026-09-05 12:32] monitor: epoch=141 step=14820，最新 loss=0.593189（loss_full=0.267013, loss_base=0.326176）；GPU 利用率约 97%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=141 step=14840 loss=0.514306 loss_full=0.214800 loss_base=0.299506 ms_per_step=3820.09
[2026-09-05 12:34] monitor: epoch=141 step=14840，最新 loss=0.514306（loss_full=0.214800, loss_base=0.299506）；GPU 利用率约 100%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=141 step=14860 loss=0.557633 loss_full=0.234980 loss_base=0.322653 ms_per_step=3663.10
[2026-09-05 12:35] monitor: epoch=141 step=14860，最新 loss=0.557633（loss_full=0.234980, loss_base=0.322653）；GPU 利用率约 57%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=141 step=14880 loss=0.623555 loss_full=0.289343 loss_base=0.334212 ms_per_step=3793.92
[2026-09-05 12:36] monitor: epoch=141 step=14880，最新 loss=0.623555（loss_full=0.289343, loss_base=0.334212）；GPU 利用率约 91%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=141 step=14900 loss=0.562573 loss_full=0.249272 loss_base=0.313301 ms_per_step=3783.32
[profile] window=377.12s updates=100 data_wait=2.0ms h2d=1.6ms stage1=287.3ms dit=645.6ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[epoch-end] epoch=141 elapsed=396.01s step=14910
[epoch-start] epoch=142 step=14910 steps_per_epoch=105
[2026-09-05 12:37] monitor: epoch=141 step=14900，最新 loss=0.562573（loss_full=0.249272, loss_base=0.313301）；profile window=377.12s 完成，GPU 利用率约 100%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
[2026-09-05 12:38] monitor: epoch=141 完成（step=14910，elapsed=396.01s），已进入 epoch=142；profile window=377.12s 完成，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=142 step=14920 loss=0.608626 loss_full=0.270159 loss_base=0.338467 ms_per_step=3621.26
[2026-09-05 12:38] monitor: epoch=142 step=14920，最新 loss=0.608626（loss_full=0.270159, loss_base=0.338467）；GPU 利用率约 100%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=142 step=14940 loss=0.585308 loss_full=0.261726 loss_base=0.323581 ms_per_step=3678.67
[2026-09-05 12:40] monitor: epoch=142 step=14940，最新 loss=0.585308（loss_full=0.261726, loss_base=0.323581）；GPU 利用率约 100%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=142 step=14960 loss=0.528377 loss_full=0.218630 loss_base=0.309747 ms_per_step=3762.29
[2026-09-05 12:41] monitor: epoch=142 step=14960，最新 loss=0.528377（loss_full=0.218630, loss_base=0.309747）；GPU 利用率约 58%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=142 step=14980 loss=0.680185 loss_full=0.308978 loss_base=0.371207 ms_per_step=3793.47
[2026-09-05 12:42] monitor: epoch=142 step=14980，最新 loss=0.680185（loss_full=0.308978, loss_base=0.371207）；GPU 利用率约 100%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=142 step=15000 loss=0.520397 loss_full=0.222014 loss_base=0.298383 ms_per_step=3764.66
[profile] window=372.67s updates=100 data_wait=2.0ms h2d=1.6ms stage1=278.8ms dit=643.5ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=15000 elapsed=3.00s
[checkpoint] step=15000 elapsed=32.49s
[2026-09-05 12:44] monitor: step=15000 sample 与 checkpoint 均完成（checkpoint elapsed=32.49s，文件已确认写入）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
[epoch-end] epoch=142 elapsed=424.33s step=15015
[epoch-start] epoch=143 step=15015 steps_per_epoch=105
task=ci2v epoch=143 step=15020 loss=0.533678 loss_full=0.220884 loss_base=0.312794 ms_per_step=3631.32
[2026-09-05 12:45] monitor: checkpoint 后训练恢复，epoch=142 完成（step=15015，elapsed=424.33s），已进入 epoch=143；step=15020 loss=0.533678（loss_full=0.220884, loss_base=0.312794）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=143 step=15040 loss=0.616510 loss_full=0.272098 loss_base=0.344411 ms_per_step=3774.07
[2026-09-05 12:47] monitor: epoch=143 step=15040，最新 loss=0.616510（loss_full=0.272098, loss_base=0.344411）；GPU 利用率约 100%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=143 step=15060 loss=0.597953 loss_full=0.262538 loss_base=0.335415 ms_per_step=3719.15
[2026-09-05 12:48] monitor: epoch=143 step=15060，最新 loss=0.597953（loss_full=0.262538, loss_base=0.335415）；GPU 利用率约 35%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=143 step=15080 loss=0.669497 loss_full=0.309847 loss_base=0.359650 ms_per_step=3772.57
[2026-09-05 12:49] monitor: epoch=143 step=15080，最新 loss=0.669497（loss_full=0.309847, loss_base=0.359650）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=143 step=15100 loss=0.646880 loss_full=0.295932 loss_base=0.350948 ms_per_step=3772.56
[profile] window=373.66s updates=100 data_wait=2.0ms h2d=1.6ms stage1=281.3ms dit=642.4ms update=30.1ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 12:50] monitor: epoch=143 step=15100，最新 loss=0.646880（loss_full=0.295932, loss_base=0.350948）；profile window=373.66s 完成，GPU 利用率约 97%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=143 step=15120 loss=0.631419 loss_full=0.299649 loss_base=0.331770 ms_per_step=3668.27
[epoch-end] epoch=143 elapsed=394.36s step=15120
[epoch-start] epoch=144 step=15120 steps_per_epoch=105
[2026-09-05 12:51] monitor: epoch=143 完成（step=15120，elapsed=394.36s），已进入 epoch=144；profile window=373.66s 完成，trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=144 step=15140 loss=0.459470 loss_full=0.185663 loss_base=0.273807 ms_per_step=3821.70
[2026-09-05 12:53] monitor: epoch=144 step=15140，最新 loss=0.459470（loss_full=0.185663, loss_base=0.273807）；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=144 step=15160 loss=0.569801 loss_full=0.252902 loss_base=0.316900 ms_per_step=3613.37
[2026-09-05 12:54] monitor: epoch=144 step=15160，最新 loss=0.569801（loss_full=0.252902, loss_base=0.316900）；GPU 利用率约 100%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=144 step=15180 loss=0.653317 loss_full=0.278808 loss_base=0.374508 ms_per_step=3590.76
[2026-09-05 12:55] monitor: epoch=144 step=15180，最新 loss=0.653317（loss_full=0.278808, loss_base=0.374508）；GPU 利用率约 50%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=144 step=15200 loss=0.589018 loss_full=0.246792 loss_base=0.342226 ms_per_step=3529.13
[profile] window=364.73s updates=100 data_wait=2.0ms h2d=1.4ms stage1=260.5ms dit=641.4ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 12:56] monitor: epoch=144 step=15200，最新 loss=0.589018（loss_full=0.246792, loss_base=0.342226）；profile window=364.73s 完成，GPU 利用率约 100%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=144 step=15220 loss=0.621025 loss_full=0.277117 loss_base=0.343908 ms_per_step=3725.08
[2026-09-05 12:57] monitor: epoch=144 step=15220，最新 loss=0.621025（loss_full=0.277117, loss_base=0.343908）；GPU 利用率约 100%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
[epoch-end] epoch=144 elapsed=383.69s step=15225
[epoch-start] epoch=145 step=15225 steps_per_epoch=105
[2026-09-05 12:58] monitor: epoch=144 完成（step=15225，elapsed=383.69s），已进入 epoch=145；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=145 step=15240 loss=0.574835 loss_full=0.244923 loss_base=0.329911 ms_per_step=3753.12
[2026-09-05 12:58] monitor: epoch=145 step=15240，最新 loss=0.574835（loss_full=0.244923, loss_base=0.329911）；GPU 利用率约 100%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=145 step=15260 loss=0.602323 loss_full=0.260731 loss_base=0.341592 ms_per_step=3693.77
[2026-09-05 13:00] monitor: epoch=145 step=15260，最新 loss=0.602323（loss_full=0.260731, loss_base=0.341592）；GPU 利用率约 51%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=145 step=15280 loss=0.625169 loss_full=0.277109 loss_base=0.348060 ms_per_step=3594.39
[2026-09-05 13:01] monitor: epoch=145 step=15280，最新 loss=0.625169（loss_full=0.277109, loss_base=0.348060）；GPU 利用率约 91%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=145 step=15300 loss=0.635403 loss_full=0.298128 loss_base=0.337275 ms_per_step=3780.77
[profile] window=371.20s updates=100 data_wait=2.2ms h2d=1.5ms stage1=275.9ms dit=642.3ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
[2026-09-05 13:02] monitor: epoch=145 step=15300，最新 loss=0.635403（loss_full=0.298128, loss_base=0.337275）；profile window=371.20s 完成，GPU 利用率约 100%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=145 step=15320 loss=0.607522 loss_full=0.270470 loss_base=0.337053 ms_per_step=3775.15
[epoch-end] epoch=145 elapsed=389.00s step=15330
[epoch-start] epoch=146 step=15330 steps_per_epoch=105
task=ci2v epoch=146 step=15340 loss=0.525371 loss_full=0.223834 loss_base=0.301537 ms_per_step=3676.67
[2026-09-05 13:06] monitor: epoch=146 step=15360，最新 loss=0.573617（loss_full=0.245040, loss_base=0.328577）；GPU 利用率约 100%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
[2026-09-05 13:05] monitor: epoch=146 step=15340，最新 loss=0.525371（loss_full=0.223834, loss_base=0.301537）；GPU 利用率约 100%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
[2026-09-05 13:05] monitor: 复核时 trainer PID=17993、torchrun PID=17918、W&B PID=18255 均存活；日志暂无新 step 输出（仍处于正常输出间隔内），GPU 利用率约 100%，显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=146 step=15360 loss=0.573617 loss_full=0.245040 loss_base=0.328577 ms_per_step=3558.79
[2026-09-05 13:07] monitor: epoch=146 step=15380，最新 loss=0.619516（loss_full=0.266421, loss_base=0.353094）；GPU 利用率约 45%，显存进程仍由 trainer 持有，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=146 step=15380 loss=0.619516 loss_full=0.266421 loss_base=0.353094 ms_per_step=3771.94
[2026-09-05 13:08] monitor: epoch=146 step=15400，最新 loss=0.622566（loss_full=0.273080, loss_base=0.349486）；profile window=369.28s 完成，GPU 利用率约 100%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=146 step=15400 loss=0.622566 loss_full=0.273080 loss_base=0.349486 ms_per_step=3668.62
[2026-09-05 13:10] monitor: epoch=146 step=15420，最新 loss=0.588323（loss_full=0.260576, loss_base=0.327746）；GPU 利用率约 97%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
[profile] window=369.28s updates=100 data_wait=2.1ms h2d=1.5ms stage1=270.8ms dit=642.4ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=146 step=15420 loss=0.588323 loss_full=0.260576 loss_base=0.327746 ms_per_step=3692.84
[epoch-end] epoch=146 elapsed=388.85s step=15435
[epoch-start] epoch=147 step=15435 steps_per_epoch=105
[2026-09-05 13:11] monitor: epoch=146 完成（step=15435，elapsed=388.85s），已进入 epoch=147；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=147 step=15440 loss=0.637605 loss_full=0.283544 loss_base=0.354061 ms_per_step=3782.99
[2026-09-05 13:11] monitor: epoch=147 step=15440，最新 loss=0.637605（loss_full=0.283544, loss_base=0.354061）；GPU 利用率约 91%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=147 step=15460 loss=0.538435 loss_full=0.226670 loss_base=0.311766 ms_per_step=3777.39
[2026-09-05 13:12] monitor: epoch=147 step=15460，最新 loss=0.538435（loss_full=0.226670, loss_base=0.311766）；GPU 利用率约 100%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=147 step=15480 loss=0.534758 loss_full=0.227158 loss_base=0.307600 ms_per_step=3640.99
[2026-09-05 13:13] monitor: epoch=147 step=15480，最新 loss=0.534758（loss_full=0.227158, loss_base=0.307600）；GPU 利用率约 100%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=147 step=15500 loss=0.628867 loss_full=0.279806 loss_base=0.349061 ms_per_step=3770.83
[2026-09-05 13:15] monitor: epoch=147 step=15500，最新 loss=0.628867（loss_full=0.279806, loss_base=0.349061）；profile window=373.56s 完成，GPU 利用率约 100%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
[profile] window=373.56s updates=100 data_wait=2.0ms h2d=1.7ms stage1=281.3ms dit=642.5ms update=27.8ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=147 step=15520 loss=0.563101 loss_full=0.246346 loss_base=0.316755 ms_per_step=3608.32
[2026-09-05 13:16] monitor: epoch=147 step=15520，最新 loss=0.563101（loss_full=0.246346, loss_base=0.316755）；GPU 利用率约 100%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=147 step=15540 loss=0.557322 loss_full=0.236076 loss_base=0.321246 ms_per_step=3494.64
[epoch-end] epoch=147 elapsed=386.37s step=15540
[epoch-start] epoch=148 step=15540 steps_per_epoch=105
[2026-09-05 13:17] monitor: epoch=147 完成（step=15540，elapsed=386.37s），最后记录 loss=0.557322（loss_full=0.236076, loss_base=0.321246），已进入 epoch=148；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=148 step=15560 loss=0.507874 loss_full=0.207958 loss_base=0.299916 ms_per_step=3587.46
[2026-09-05 13:18] monitor: epoch=148 step=15560，最新 loss=0.507874（loss_full=0.207958, loss_base=0.299916）；GPU 利用率约 72%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=148 step=15580 loss=0.556828 loss_full=0.246911 loss_base=0.309917 ms_per_step=3686.28
[2026-09-05 13:19] monitor: epoch=148 step=15580，最新 loss=0.556828（loss_full=0.246911, loss_base=0.309917）；GPU 利用率约 98%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=148 step=15600 loss=0.540752 loss_full=0.236538 loss_base=0.304214 ms_per_step=3511.46
[2026-09-05 13:21] monitor: epoch=148 step=15600，最新 loss=0.540752（loss_full=0.236538, loss_base=0.304214）；profile window=358.03s 完成，GPU 利用率约 41%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
[profile] window=358.03s updates=100 data_wait=1.7ms h2d=1.3ms stage1=245.5ms dit=640.4ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=148 step=15620 loss=0.522084 loss_full=0.225349 loss_base=0.296735 ms_per_step=3773.87
[2026-09-05 13:22] monitor: epoch=148 step=15620，最新 loss=0.522084（loss_full=0.225349, loss_base=0.296735）；GPU 利用率约 88%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=148 step=15640 loss=0.591854 loss_full=0.258251 loss_base=0.333603 ms_per_step=3671.38
[2026-09-05 13:23] monitor: epoch=148 step=15640，最新 loss=0.591854（loss_full=0.258251, loss_base=0.333603）；GPU 利用率约 100%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
[epoch-end] epoch=148 elapsed=381.22s step=15645
[epoch-start] epoch=149 step=15645 steps_per_epoch=105
[2026-09-05 13:24] monitor: epoch=148 完成（step=15645，elapsed=381.22s），最后记录 loss=0.591854（loss_full=0.258251, loss_base=0.333603），已进入 epoch=149；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=149 step=15660 loss=0.523619 loss_full=0.224211 loss_base=0.299408 ms_per_step=3569.52
[2026-09-05 13:24] monitor: epoch=149 step=15660，最新 loss=0.523619（loss_full=0.224211, loss_base=0.299408）；GPU 利用率约 99%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=149 step=15680 loss=0.571031 loss_full=0.247776 loss_base=0.323256 ms_per_step=3588.01
[2026-09-05 13:26] monitor: epoch=149 step=15680，最新 loss=0.571031（loss_full=0.247776, loss_base=0.323256）；GPU 利用率约 95%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=149 step=15700 loss=0.578624 loss_full=0.246664 loss_base=0.331960 ms_per_step=3686.51
[2026-09-05 13:27] monitor: epoch=149 step=15700，最新 loss=0.578624（loss_full=0.246664, loss_base=0.331960）；profile window=366.04s 完成，GPU 利用率约 52%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
[profile] window=366.04s updates=100 data_wait=2.0ms h2d=1.5ms stage1=263.7ms dit=641.7ms update=29.1ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=149 step=15720 loss=0.615035 loss_full=0.270387 loss_base=0.344649 ms_per_step=3440.88
[2026-09-05 13:28] monitor: epoch=149 step=15720，最新 loss=0.615035（loss_full=0.270387, loss_base=0.344649）；GPU 利用率约 99%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=149 step=15740 loss=0.565671 loss_full=0.246677 loss_base=0.318994 ms_per_step=3761.19
[2026-09-05 13:29] monitor: epoch=149 step=15740，最新 loss=0.565671（loss_full=0.246677, loss_base=0.318994）；GPU 利用率约 8%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
[epoch-end] epoch=149 elapsed=382.51s step=15750
[epoch-start] epoch=150 step=15750 steps_per_epoch=105
[2026-09-05 13:30] monitor: epoch=149 完成（step=15750，elapsed=382.51s），最后记录 loss=0.565671（loss_full=0.246677, loss_base=0.318994），已进入 epoch=150；trainer PID=17993 存活，GPU 显存约 41410 MiB；错误扫描无命中。
task=ci2v epoch=150 step=15760 loss=0.715685 loss_full=0.334449 loss_base=0.381236 ms_per_step=3717.86
[2026-09-05 13:30] monitor: epoch=150 step=15760，最新 loss=0.715685（loss_full=0.334449, loss_base=0.381236）；GPU 利用率约 94%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=150 step=15780 loss=0.570359 loss_full=0.247361 loss_base=0.322998 ms_per_step=3655.35
[2026-09-05 13:31] monitor: epoch=150 step=15780，最新 loss=0.570359（loss_full=0.247361, loss_base=0.322998）；GPU 利用率约 83%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
task=ci2v epoch=150 step=15800 loss=0.611511 loss_full=0.289518 loss_base=0.321994 ms_per_step=3638.06
[2026-09-05 13:33] monitor: epoch=150 step=15800，最新 loss=0.611511（loss_full=0.289518, loss_base=0.321994）；profile window=364.53s 完成，GPU 利用率约 95%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
[2026-09-05 13:36] monitor: epoch=150 step=15840，最新 loss=0.619572（loss_full=0.271893, loss_base=0.347679）；GPU 利用率约 12%，显存约 41410 MiB，trainer PID=17993 存活；错误扫描无命中。
[profile] window=364.53s updates=100 data_wait=1.9ms h2d=1.6ms stage1=260.2ms dit=641.2ms update=27.7ms allocated=29.0GiB reserved=39.7GiB peak=38.0GiB
task=ci2v epoch=150 step=15820 loss=0.736888 loss_full=0.328119 loss_base=0.408769 ms_per_step=3459.64
task=ci2v epoch=150 step=15840 loss=0.619572 loss_full=0.271893 loss_base=0.347679 ms_per_step=3654.32
[epoch-end] epoch=150 elapsed=381.26s step=15855
[epoch-start] epoch=151 step=15855 steps_per_epoch=105
task=ci2v epoch=151 step=15860 loss=0.629063 loss_full=0.285387 loss_base=0.343676 ms_per_step=3744.52
task=ci2v epoch=151 step=15880 loss=0.604830 loss_full=0.260351 loss_base=0.344479 ms_per_step=3640.77
[2026-09-05 13:38] control: 按用户要求中断旧训练；旧 trainer PID=17993，停止前已到 epoch=151 step=15880；相关 torchrun/tee 进程已确认退出。
[2026-09-05 13:39] control: 启动全新训练配置 configs/training/generation/libero_ci2v_from_multiview_L_restart.yaml（继承原配置，仅使用新 run_name）；launcher PID=34876，torchrun PID=34878，trainer PID=34953；W&B=https://wandb.ai/Aircraft/V-RAE/runs/9717f9c457364e9095e70d7cdabd7504。
[2026-09-05 13:43] monitor: 新 run epoch=0 step=20，最新 loss=2.062750（loss_full=1.031667, loss_base=1.031083）；trainer PID=34953 存活；错误扫描无命中。
[latent-stats] using existing file: /zsh/code/V-RAE/ckpts/libero_videogen/latent_stats/ci2v_shared_20f_multiview.pt
/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/timm/models/layers/__init__.py:49: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
[wandb] API key detected (WANDB_API_KEY); online logging enabled.
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from WANDB_API_KEY.
wandb: Currently logged in as: shuhangzheng20 (Aircraft) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: setting up run 9717f9c457364e9095e70d7cdabd7504
wandb: Tracking run with wandb version 0.24.2
wandb: Run data is saved locally in /zsh/code/V-RAE/ckpts/libero_videogen/vjepa2_1_libero_ci2v_from_multiview_l_gc_restart_20260905/wandb/wandb/run-20260905_134125-9717f9c457364e9095e70d7cdabd7504
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run vjepa2_1_libero_ci2v_from_multiview_l_gc_restart_20260905
wandb: ⭐️ View project at https://wandb.ai/Aircraft/V-RAE
wandb: 🚀 View run at https://wandb.ai/Aircraft/V-RAE/runs/9717f9c457364e9095e70d7cdabd7504
[epoch-start] epoch=0 step=0 steps_per_epoch=105
/zsh/miniconda3/envs/vrae/lib/python3.10/contextlib.py:103: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the future, this context manager will be removed. Please see `torch.nn.attention.sdpa_kernel()` for the new context manager, with updated signature.
  self.gen = func(*args, **kwds)
/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/nvidia_cutlass_dsl/dsl_packages/cutlass/base_dsl/dsl.py:2309: UserWarning: Argument aux_data (position 17) cannot be converted to a JitArgument for function 'cutlass___call___flash_attncuteflash_fwd_sm90FlashAttentionForwardSm90_object_at__Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmem'. Its type `<class 'flash_attn.cute.utils.AuxData'>` is not supported. If the value is known at compile time, annotate it: `aux_data: Constexpr`. Otherwise, implement the `JitArgument` or `DynamicExpression` protocol for type `<class 'flash_attn.cute.utils.AuxData'>`, or register a custom argument adapter.
  exe_args, func_types, adapted_args = self.generate_mlir_function_types(
/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/nvidia_cutlass_dsl/dsl_packages/cutlass/base_dsl/dsl.py:2309: UserWarning: Argument aux_data (position 21) cannot be converted to a JitArgument for function 'cutlass___call___flash_attncuteflash_bwd_sm90FlashAttentionBackwardSm90_object_at__Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgme'. Its type `<class 'flash_attn.cute.utils.AuxData'>` is not supported. If the value is known at compile time, annotate it: `aux_data: Constexpr`. Otherwise, implement the `JitArgument` or `DynamicExpression` protocol for type `<class 'flash_attn.cute.utils.AuxData'>`, or register a custom argument adapter.
  exe_args, func_types, adapted_args = self.generate_mlir_function_types(
/zsh/miniconda3/envs/vrae/lib/python3.10/contextlib.py:103: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the future, this context manager will be removed. Please see `torch.nn.attention.sdpa_kernel()` for the new context manager, with updated signature.
  self.gen = func(*args, **kwds)
task=ci2v epoch=0 step=20 loss=2.062750 loss_full=1.031667 loss_base=1.031083 ms_per_step=4511.51
task=ci2v epoch=0 step=40 loss=2.211474 loss_full=1.106233 loss_base=1.105241 ms_per_step=3734.51
[2026-09-05 13:44] monitor: 新 run epoch=0 step=40，最新 loss=2.211474（loss_full=1.106233, loss_base=1.105241）；GPU 利用率约 100%，显存约 37092 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=0 step=60 loss=2.230375 loss_full=1.116418 loss_base=1.113956 ms_per_step=3703.68
[2026-09-05 13:45] monitor: 新 run epoch=0 step=60，最新 loss=2.230375（loss_full=1.116418, loss_base=1.113956）；GPU 利用率约 100%，显存约 37092 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=0 step=80 loss=2.216860 loss_full=1.107027 loss_base=1.109833 ms_per_step=3608.12
[2026-09-05 13:47] monitor: 新 run epoch=0 step=80，最新 loss=2.216860（loss_full=1.107027, loss_base=1.109833）；GPU 利用率约 100%，显存约 37092 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=0 step=100 loss=2.775942 loss_full=1.372347 loss_base=1.403596 ms_per_step=3439.46
[profile] window=289.91s updates=80 data_wait=1.0ms h2d=1.3ms stage1=256.2ms dit=641.4ms update=27.8ms allocated=28.7GiB reserved=35.5GiB peak=34.1GiB
[epoch-end] epoch=0 elapsed=396.92s step=105
[epoch-start] epoch=1 step=105 steps_per_epoch=105
task=ci2v epoch=1 step=120 loss=2.287875 loss_full=1.132116 loss_base=1.155758 ms_per_step=3407.67
[2026-09-05 13:49] monitor: 新 run epoch=1 step=120，最新 loss=2.287875（loss_full=1.132116, loss_base=1.155758）；GPU 利用率约 100%，显存约 37092 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=1 step=140 loss=2.120821 loss_full=1.042100 loss_base=1.078720 ms_per_step=3848.47
[2026-09-05 13:50] monitor: 新 run epoch=1 step=140，最新 loss=2.120821（loss_full=1.042100, loss_base=1.078720）；trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=1 step=160 loss=2.050770 loss_full=1.005317 loss_base=1.045453 ms_per_step=3757.29
[2026-09-05 13:52] monitor: 新 run epoch=1 step=160，最新 loss=2.050770（loss_full=1.005317, loss_base=1.045453）；trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=1 step=180 loss=2.543122 loss_full=1.220508 loss_base=1.322614 ms_per_step=3483.17
[2026-09-05 13:53] monitor: 新 run epoch=1 step=180，最新 loss=2.543122（loss_full=1.220508, loss_base=1.322614）；trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=1 step=200 loss=1.966539 loss_full=0.983038 loss_base=0.983502 ms_per_step=3533.60
[profile] window=360.87s updates=100 data_wait=1.7ms h2d=1.3ms stage1=249.3ms dit=643.4ms update=27.7ms allocated=28.7GiB reserved=35.5GiB peak=34.1GiB
[epoch-end] epoch=1 elapsed=380.81s step=210
[epoch-start] epoch=2 step=210 steps_per_epoch=105
task=ci2v epoch=2 step=220 loss=2.134678 loss_full=1.036130 loss_base=1.098549 ms_per_step=3785.84
[2026-09-05 13:55] monitor: 新 run epoch=2 step=220，最新 loss=2.134678（loss_full=1.036130, loss_base=1.098549）；trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=2 step=240 loss=1.998210 loss_full=0.984469 loss_base=1.013741 ms_per_step=3811.60
[2026-09-05 13:57] monitor: 新 run epoch=2 step=240，最新 loss=1.998210（loss_full=0.984469, loss_base=1.013741）；trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=2 step=260 loss=1.943248 loss_full=0.942929 loss_base=1.000319 ms_per_step=3668.05
[2026-09-05 13:58] monitor: 新 run epoch=2 step=260，最新 loss=1.943248（loss_full=0.942929, loss_base=1.000319）；trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=2 step=280 loss=1.976387 loss_full=0.947493 loss_base=1.028894 ms_per_step=3671.51
[2026-09-05 13:59] monitor: 新 run epoch=2 step=280，最新 loss=1.976387（loss_full=0.947493, loss_base=1.028894）；trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=2 step=300 loss=1.888921 loss_full=0.908219 loss_base=0.980702 ms_per_step=3657.51
[profile] window=372.15s updates=100 data_wait=2.1ms h2d=1.5ms stage1=275.7ms dit=644.5ms update=27.7ms allocated=28.7GiB reserved=35.5GiB peak=34.1GiB
[2026-09-05 14:00] monitor: 新 run epoch=2 step=300，最新 loss=1.888921（loss_full=0.908219, loss_base=0.980702）；profile window=372.15s 完成，trainer PID=34953 存活；错误扫描无命中。
[epoch-end] epoch=2 elapsed=391.99s step=315
[epoch-start] epoch=3 step=315 steps_per_epoch=105
task=ci2v epoch=3 step=320 loss=1.880382 loss_full=0.898152 loss_base=0.982231 ms_per_step=3808.99
[2026-09-05 14:01] monitor: 新 run epoch=3 step=320，最新 loss=1.880382（loss_full=0.898152, loss_base=0.982231）；GPU 利用率约 100%，显存约 37092 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=3 step=340 loss=1.884776 loss_full=0.893651 loss_base=0.991125 ms_per_step=3426.95
[2026-09-05 14:02] monitor: 新 run epoch=3 step=340，最新 loss=1.884776（loss_full=0.893651, loss_base=0.991125）；trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=3 step=360 loss=1.728987 loss_full=0.799755 loss_base=0.929232 ms_per_step=3637.27
[2026-09-05 14:04] monitor: 新 run epoch=3 step=360，最新 loss=1.728987（loss_full=0.799755, loss_base=0.929232）；trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=3 step=380 loss=1.842943 loss_full=0.814773 loss_base=1.028170 ms_per_step=3739.02
[2026-09-05 14:05] monitor: 新 run epoch=3 step=380，最新 loss=1.842943（loss_full=0.814773, loss_base=1.028170）；trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=3 step=400 loss=1.551610 loss_full=0.679375 loss_base=0.872235 ms_per_step=3743.08
[profile] window=367.38s updates=100 data_wait=1.8ms h2d=1.4ms stage1=264.9ms dit=643.7ms update=27.7ms allocated=28.7GiB reserved=35.5GiB peak=34.1GiB
[2026-09-05 14:06] monitor: 新 run epoch=3 step=400，最新 loss=1.551610（loss_full=0.679375, loss_base=0.872235）；profile window=367.38s 完成，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=3 step=420 loss=1.533401 loss_full=0.668495 loss_base=0.864906 ms_per_step=3738.51
[epoch-end] epoch=3 elapsed=386.35s step=420
[epoch-start] epoch=4 step=420 steps_per_epoch=105
task=ci2v epoch=4 step=440 loss=1.472192 loss_full=0.657131 loss_base=0.815061 ms_per_step=3613.40
[2026-09-05 14:09] monitor: 新 run epoch=4 step=440，最新 loss=1.472192（loss_full=0.657131, loss_base=0.815061）；trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=4 step=460 loss=1.285283 loss_full=0.568947 loss_base=0.716336 ms_per_step=3622.08
[2026-09-05 14:09] monitor: 新 run epoch=4 step=460，最新 loss=1.285283（loss_full=0.568947, loss_base=0.716336）；GPU 利用率约 100%，显存约 37092 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=4 step=480 loss=1.416007 loss_full=0.637214 loss_base=0.778793 ms_per_step=3619.82
[2026-09-05 14:11] monitor: 新 run epoch=4 step=480，最新 loss=1.416007（loss_full=0.637214, loss_base=0.778793）；trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=4 step=500 loss=1.226312 loss_full=0.542007 loss_base=0.684305 ms_per_step=3674.35
[profile] window=365.63s updates=100 data_wait=2.0ms h2d=1.4ms stage1=262.4ms dit=641.7ms update=27.7ms allocated=28.7GiB reserved=35.5GiB peak=34.1GiB
[2026-09-05 14:12] monitor: 新 run epoch=4 step=500，最新 loss=1.226312（loss_full=0.542007, loss_base=0.684305）；profile window=365.63s 完成，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=4 step=520 loss=1.356135 loss_full=0.614144 loss_base=0.741991 ms_per_step=3492.25
[2026-09-05 14:13] monitor: 新 run epoch=4 step=520，最新 loss=1.356135（loss_full=0.614144, loss_base=0.741991）；trainer PID=34953 存活；错误扫描无命中。
[epoch-end] epoch=4 elapsed=378.06s step=525
[epoch-start] epoch=5 step=525 steps_per_epoch=105
task=ci2v epoch=5 step=540 loss=1.389740 loss_full=0.609793 loss_base=0.779948 ms_per_step=3632.72
[2026-09-05 14:15] monitor: 新 run epoch=5 step=540，最新 loss=1.389740（loss_full=0.609793, loss_base=0.779948）；trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=5 step=560 loss=1.291177 loss_full=0.591787 loss_base=0.699390 ms_per_step=3319.18
[2026-09-05 14:15] monitor: 新 run epoch=5 step=560，最新 loss=1.291177（loss_full=0.591787, loss_base=0.699390）；GPU 利用率约 99%，显存约 37092 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=5 step=580 loss=1.269093 loss_full=0.591621 loss_base=0.677472 ms_per_step=3421.86
[2026-09-05 14:17] monitor: 新 run epoch=5 step=580，最新 loss=1.269093（loss_full=0.591621, loss_base=0.677472）；trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=5 step=600 loss=1.313483 loss_full=0.607653 loss_base=0.705830 ms_per_step=3737.78
[profile] window=352.35s updates=100 data_wait=2.0ms h2d=1.2ms stage1=233.7ms dit=637.5ms update=27.7ms allocated=28.7GiB reserved=35.5GiB peak=34.1GiB
[2026-09-05 14:18] monitor: 新 run epoch=5 step=600，最新 loss=1.313483（loss_full=0.607653, loss_base=0.705830）；profile window=352.35s 完成，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=5 step=620 loss=1.166715 loss_full=0.550461 loss_base=0.616254 ms_per_step=3733.51
[2026-09-05 14:19] monitor: 新 run epoch=5 step=620，最新 loss=1.166715（loss_full=0.550461, loss_base=0.616254）；trainer PID=34953 存活；错误扫描无命中。
[epoch-end] epoch=5 elapsed=376.64s step=630
[epoch-start] epoch=6 step=630 steps_per_epoch=105
task=ci2v epoch=6 step=640 loss=1.232879 loss_full=0.575921 loss_base=0.656958 ms_per_step=3770.31
[2026-09-05 14:20] monitor: 新 run epoch=6 step=640，最新 loss=1.232879（loss_full=0.575921, loss_base=0.656958）；GPU 利用率约 97%，显存约 37092 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=6 step=660 loss=1.201806 loss_full=0.564435 loss_base=0.637370 ms_per_step=3731.97
[2026-09-05 14:22] monitor: 新 run epoch=6 step=660，最新 loss=1.201806（loss_full=0.564435, loss_base=0.637370）；trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=6 step=680 loss=1.176481 loss_full=0.553926 loss_base=0.622555 ms_per_step=3586.33
[2026-09-05 14:23] monitor: 新 run epoch=6 step=680，最新 loss=1.176481（loss_full=0.553926, loss_base=0.622555）；trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=6 step=700 loss=1.101647 loss_full=0.518618 loss_base=0.583029 ms_per_step=3541.32
[profile] window=367.54s updates=100 data_wait=2.0ms h2d=1.5ms stage1=267.4ms dit=641.6ms update=27.7ms allocated=28.7GiB reserved=35.5GiB peak=34.1GiB
[2026-09-05 14:24] monitor: 新 run epoch=6 step=700，最新 loss=1.101647（loss_full=0.518618, loss_base=0.583029）；profile window=367.54s 完成，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=6 step=720 loss=1.241845 loss_full=0.588060 loss_base=0.653785 ms_per_step=3427.06
[2026-09-05 14:25] monitor: 新 run epoch=6 step=720，最新 loss=1.241845（loss_full=0.588060, loss_base=0.653785）；GPU 利用率约 98%，显存约 37092 MiB，trainer PID=34953 存活；错误扫描无命中。
[epoch-end] epoch=6 elapsed=375.73s step=735
[epoch-start] epoch=7 step=735 steps_per_epoch=105
task=ci2v epoch=7 step=740 loss=1.146924 loss_full=0.539966 loss_base=0.606959 ms_per_step=3513.78
task=ci2v epoch=7 step=760 loss=1.206658 loss_full=0.576402 loss_base=0.630255 ms_per_step=3702.29
[2026-09-05 14:28] monitor: 新 run epoch=7 step=760，最新 loss=1.206658（loss_full=0.576402, loss_base=0.630255）；trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=7 step=780 loss=1.066963 loss_full=0.513486 loss_base=0.553477 ms_per_step=3711.06
[2026-09-05 14:29] monitor: 新 run epoch=7 step=780，最新 loss=1.066963（loss_full=0.513486, loss_base=0.553477）；trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=7 step=800 loss=1.144744 loss_full=0.540074 loss_base=0.604671 ms_per_step=3587.55
[profile] window=359.10s updates=100 data_wait=1.7ms h2d=1.3ms stage1=249.4ms dit=639.2ms update=27.7ms allocated=28.7GiB reserved=35.5GiB peak=34.1GiB
[2026-09-05 14:30] monitor: 新 run epoch=7 step=800，最新 loss=1.144744（loss_full=0.540074, loss_base=0.604671）；profile window=359.10s 完成，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=7 step=820 loss=1.043258 loss_full=0.501490 loss_base=0.541768 ms_per_step=3442.12
[2026-09-05 14:31] monitor: 新 run epoch=7 step=820，最新 loss=1.043258（loss_full=0.501490, loss_base=0.541768）；trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=7 step=840 loss=1.199482 loss_full=0.566522 loss_base=0.632960 ms_per_step=3563.51
[epoch-end] epoch=7 elapsed=380.08s step=840
[epoch-start] epoch=8 step=840 steps_per_epoch=105
task=ci2v epoch=8 step=860 loss=1.079347 loss_full=0.517771 loss_base=0.561576 ms_per_step=3821.35
[2026-09-05 14:33] monitor: 新 run epoch=8 step=860，最新 loss=1.079347（loss_full=0.517771, loss_base=0.561576）；GPU 利用率约 91%，显存约 37092 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=8 step=880 loss=1.111311 loss_full=0.537240 loss_base=0.574071 ms_per_step=3567.19
[2026-09-05 14:35] monitor: 新 run epoch=8 step=880，最新 loss=1.111311（loss_full=0.537240, loss_base=0.574071）；GPU 利用率约 100%，显存约 37101 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=8 step=900 loss=1.030742 loss_full=0.494492 loss_base=0.536250 ms_per_step=3579.46
[profile] window=359.74s updates=100 data_wait=2.0ms h2d=1.3ms stage1=249.1ms dit=640.4ms update=27.7ms allocated=28.7GiB reserved=35.5GiB peak=34.1GiB
[2026-09-05 14:36] monitor: 新 run epoch=8 step=900，最新 loss=1.030742（loss_full=0.494492, loss_base=0.536250）；profile window=359.74s 完成，GPU 利用率约 100%，显存约 37101 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=8 step=920 loss=1.057784 loss_full=0.503453 loss_base=0.554331 ms_per_step=3718.91
[2026-09-05 14:38] monitor: 新 run epoch=8 step=920，最新 loss=1.057784（loss_full=0.503453, loss_base=0.554331）；GPU 利用率约 79%，显存约 37101 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=8 step=940 loss=1.054158 loss_full=0.504238 loss_base=0.549920 ms_per_step=3612.16
[epoch-end] epoch=8 elapsed=382.93s step=945
[epoch-start] epoch=9 step=945 steps_per_epoch=105
[2026-09-05 14:39] monitor: 新 run epoch=8 step=940，最新 loss=1.054158（loss_full=0.504238, loss_base=0.549920）；epoch=8 已完成（step=945，耗时 382.93s）并进入 epoch=9，GPU 利用率约 11%，显存约 37101 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=9 step=960 loss=0.998512 loss_full=0.478393 loss_base=0.520119 ms_per_step=3681.99
[2026-09-05 14:40] monitor: 新 run epoch=9 step=960，最新 loss=0.998512（loss_full=0.478393, loss_base=0.520119）；GPU 利用率约 13%，显存约 37101 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=9 step=980 loss=1.088969 loss_full=0.527159 loss_base=0.561810 ms_per_step=3648.85
[2026-09-05 14:42] monitor: 新 run epoch=9 step=980，最新 loss=1.088969（loss_full=0.527159, loss_base=0.561810）；GPU 利用率约 100%，显存约 37101 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=9 step=1000 loss=1.038422 loss_full=0.498081 loss_base=0.540341 ms_per_step=3691.86
[profile] window=367.34s updates=100 data_wait=1.9ms h2d=1.5ms stage1=266.7ms dit=641.9ms update=27.7ms allocated=28.7GiB reserved=35.5GiB peak=34.1GiB
/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/nvidia_cutlass_dsl/dsl_packages/cutlass/base_dsl/dsl.py:2309: UserWarning: Argument aux_data (position 17) cannot be converted to a JitArgument for function 'cutlass___call___flash_attncuteflash_fwd_sm90FlashAttentionForwardSm90_object_at__Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_None_0125_'. Its type `<class 'flash_attn.cute.utils.AuxData'>` is not supported. If the value is known at compile time, annotate it: `aux_data: Constexpr`. Otherwise, implement the `JitArgument` or `DynamicExpression` protocol for type `<class 'flash_attn.cute.utils.AuxData'>`, or register a custom argument adapter.
  exe_args, func_types, adapted_args = self.generate_mlir_function_types(
/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/nvidia_cutlass_dsl/dsl_packages/cutlass/base_dsl/dsl.py:2309: UserWarning: Argument aux_data (position 17) cannot be converted to a JitArgument for function 'cutlass___call___flash_attncuteflash_fwd_sm90FlashAttentionForwardSm90_object_at__Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_None_00883'. Its type `<class 'flash_attn.cute.utils.AuxData'>` is not supported. If the value is known at compile time, annotate it: `aux_data: Constexpr`. Otherwise, implement the `JitArgument` or `DynamicExpression` protocol for type `<class 'flash_attn.cute.utils.AuxData'>`, or register a custom argument adapter.
  exe_args, func_types, adapted_args = self.generate_mlir_function_types(
/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/nvidia_cutlass_dsl/dsl_packages/cutlass/base_dsl/dsl.py:2309: UserWarning: Argument aux_data (position 17) cannot be converted to a JitArgument for function 'cutlass___call___flash_attncuteflash_fwd_sm90FlashAttentionForwardSm90_object_at__Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_None_01178'. Its type `<class 'flash_attn.cute.utils.AuxData'>` is not supported. If the value is known at compile time, annotate it: `aux_data: Constexpr`. Otherwise, implement the `JitArgument` or `DynamicExpression` protocol for type `<class 'flash_attn.cute.utils.AuxData'>`, or register a custom argument adapter.
  exe_args, func_types, adapted_args = self.generate_mlir_function_types(
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=1000 elapsed=17.81s
[checkpoint] step=1000 elapsed=27.36s
[2026-09-05 14:43] monitor: 新 run epoch=9 step=1000，最新 loss=1.038422（loss_full=0.498081, loss_base=0.540341）；profile window=367.34s 完成，sample 用时 17.81s、checkpoint 用时 27.36s，step-00001000.pt 已写出；trainer PID=34953 存活，错误扫描无命中。
/zsh/miniconda3/envs/vrae/lib/python3.10/contextlib.py:103: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the future, this context manager will be removed. Please see `torch.nn.attention.sdpa_kernel()` for the new context manager, with updated signature.
  self.gen = func(*args, **kwds)
task=ci2v epoch=9 step=1020 loss=0.979406 loss_full=0.468137 loss_base=0.511268 ms_per_step=3329.15
[2026-09-05 14:44] monitor: 新 run epoch=9 step=1020，最新 loss=0.979406（loss_full=0.468137, loss_base=0.511268）；sample 后 GPU 利用率约 100%，显存约 40843 MiB，trainer PID=34953 存活；CUTLASS/FutureWarning 仅为警告，错误扫描无命中。
task=ci2v epoch=9 step=1040 loss=1.073038 loss_full=0.500219 loss_base=0.572819 ms_per_step=3777.72
[epoch-end] epoch=9 elapsed=427.42s step=1050
[epoch-start] epoch=10 step=1050 steps_per_epoch=105
[2026-09-05 14:46] monitor: 新 run epoch=9 step=1040，最新 loss=1.073038（loss_full=0.500219, loss_base=0.572819）；epoch=9 已完成（step=1050，耗时 427.42s）并进入 epoch=10，GPU 利用率约 100%，显存约 40843 MiB，trainer PID=34953 存活；CUTLASS/FutureWarning 仅为警告，错误扫描无命中。
task=ci2v epoch=10 step=1060 loss=0.999927 loss_full=0.475385 loss_base=0.524541 ms_per_step=3615.82
[2026-09-05 14:47] monitor: 新 run epoch=10 step=1060，最新 loss=0.999927（loss_full=0.475385, loss_base=0.524541）；GPU 利用率约 100%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=10 step=1080 loss=1.086478 loss_full=0.524083 loss_base=0.562395 ms_per_step=3517.91
[2026-09-05 14:48] monitor: 新 run epoch=10 step=1080，最新 loss=1.086478（loss_full=0.524083, loss_base=0.562395）；GPU 利用率约 98%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=10 step=1100 loss=1.100446 loss_full=0.522088 loss_base=0.578357 ms_per_step=3394.85
[profile] window=352.98s updates=100 data_wait=2.0ms h2d=1.2ms stage1=231.4ms dit=641.5ms update=27.7ms allocated=29.0GiB reserved=39.1GiB peak=37.9GiB
[2026-09-05 14:50] monitor: 新 run epoch=10 step=1100，最新 loss=1.100446（loss_full=0.522088, loss_base=0.578357）；profile window=352.98s 完成，allocated=29.0GiB、reserved=39.1GiB、peak=37.9GiB；GPU 利用率约 39%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=10 step=1120 loss=0.986661 loss_full=0.475350 loss_base=0.511312 ms_per_step=3425.56
task=ci2v epoch=10 step=1140 loss=0.987173 loss_full=0.475739 loss_base=0.511434 ms_per_step=3654.70
[2026-09-05 14:51] monitor: 新 run epoch=10 step=1140，最新 loss=0.987173（loss_full=0.475739, loss_base=0.511434）；GPU 利用率约 100%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
[epoch-end] epoch=10 elapsed=365.97s step=1155
[epoch-start] epoch=11 step=1155 steps_per_epoch=105
task=ci2v epoch=11 step=1160 loss=0.959453 loss_full=0.459377 loss_base=0.500076 ms_per_step=3327.38
[2026-09-05 14:52] monitor: 新 run epoch=10 step=1140，epoch=10 已完成（step=1155，耗时 365.97s）并进入 epoch=11；最新 epoch=11 step=1160 loss=0.959453（loss_full=0.459377, loss_base=0.500076），GPU 利用率约 94%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=11 step=1180 loss=1.001439 loss_full=0.475455 loss_base=0.525984 ms_per_step=3480.51
[2026-09-05 14:53] monitor: 新 run epoch=11 step=1180，最新 loss=1.001439（loss_full=0.475455, loss_base=0.525984）；GPU 利用率约 100%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=11 step=1200 loss=1.039936 loss_full=0.494232 loss_base=0.545704 ms_per_step=3402.45
[profile] window=346.08s updates=100 data_wait=1.6ms h2d=1.0ms stage1=219.2ms dit=637.3ms update=27.7ms allocated=29.0GiB reserved=39.1GiB peak=37.9GiB
[2026-09-05 14:55] monitor: 新 run epoch=11 step=1200，最新 loss=1.039936（loss_full=0.494232, loss_base=0.545704）；profile window=346.08s 完成，allocated=29.0GiB、reserved=39.1GiB、peak=37.9GiB；GPU 利用率约 91%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=11 step=1220 loss=0.951096 loss_full=0.458732 loss_base=0.492365 ms_per_step=3766.96
[2026-09-05 14:56] monitor: 新 run epoch=11 step=1220，最新 loss=0.951096（loss_full=0.458732, loss_base=0.492365）；GPU 利用率约 100%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=11 step=1240 loss=0.903578 loss_full=0.432235 loss_base=0.471342 ms_per_step=3716.60
[2026-09-05 14:58] monitor: 新 run epoch=11 step=1240，最新 loss=0.903578（loss_full=0.432235, loss_base=0.471342）；GPU 利用率约 22%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=11 step=1260 loss=0.947097 loss_full=0.449675 loss_base=0.497422 ms_per_step=3633.74
[epoch-end] epoch=11 elapsed=377.66s step=1260
[epoch-start] epoch=12 step=1260 steps_per_epoch=105
[2026-09-05 14:58] monitor: 新 run epoch=11 step=1260，最新 loss=0.947097（loss_full=0.449675, loss_base=0.497422）；epoch=11 已完成（step=1260，耗时 377.66s）并进入 epoch=12，GPU 利用率约 94%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=12 step=1280 loss=0.996604 loss_full=0.477931 loss_base=0.518672 ms_per_step=3546.52
[2026-09-05 14:59] monitor: 新 run epoch=12 step=1280，最新 loss=0.996604（loss_full=0.477931, loss_base=0.518672）；GPU 利用率约 100%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=12 step=1300 loss=0.961752 loss_full=0.457650 loss_base=0.504102 ms_per_step=3374.97
[profile] window=361.04s updates=100 data_wait=1.7ms h2d=1.4ms stage1=253.8ms dit=639.5ms update=27.7ms allocated=29.0GiB reserved=39.1GiB peak=37.9GiB
[2026-09-05 15:01] monitor: 新 run epoch=12 step=1300，最新 loss=0.961752（loss_full=0.457650, loss_base=0.504102）；profile window=361.04s 完成，allocated=29.0GiB、reserved=39.1GiB、peak=37.9GiB；GPU 利用率约 99%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=12 step=1320 loss=0.975226 loss_full=0.464998 loss_base=0.510228 ms_per_step=3358.78
[2026-09-05 15:02] monitor: 新 run epoch=12 step=1320，最新 loss=0.975226（loss_full=0.464998, loss_base=0.510228）；GPU 利用率约 98%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=12 step=1340 loss=0.955319 loss_full=0.453479 loss_base=0.501841 ms_per_step=3548.23
[2026-09-05 15:03] monitor: 新 run epoch=12 step=1340，最新 loss=0.955319（loss_full=0.453479, loss_base=0.501841）；GPU 利用率约 100%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=12 step=1360 loss=1.023026 loss_full=0.487888 loss_base=0.535137 ms_per_step=3624.61
[2026-09-05 15:04] monitor: 新 run epoch=12 step=1360，最新 loss=1.023026（loss_full=0.487888, loss_base=0.535137）；GPU 利用率约 100%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
[epoch-end] epoch=12 elapsed=365.69s step=1365
[epoch-start] epoch=13 step=1365 steps_per_epoch=105
[2026-09-05 15:04] monitor: 新 run epoch=12 已完成（step=1365，耗时 365.69s）并进入 epoch=13；GPU 利用率约 75%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=13 step=1380 loss=0.913432 loss_full=0.436668 loss_base=0.476764 ms_per_step=3465.83
[2026-09-05 15:05] monitor: 新 run epoch=13 step=1380，最新 loss=0.913432（loss_full=0.436668, loss_base=0.476764）；GPU 利用率约 41%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=13 step=1400 loss=0.913488 loss_full=0.437481 loss_base=0.476007 ms_per_step=3697.02
[profile] window=354.16s updates=100 data_wait=1.6ms h2d=1.2ms stage1=238.6ms dit=637.8ms update=27.7ms allocated=29.0GiB reserved=39.1GiB peak=37.9GiB
[2026-09-05 15:07] monitor: 新 run epoch=13 step=1400，最新 loss=0.913488（loss_full=0.437481, loss_base=0.476007）；profile window=354.16s 完成，allocated=29.0GiB、reserved=39.1GiB、peak=37.9GiB；GPU 利用率约 100%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=13 step=1420 loss=0.941287 loss_full=0.456413 loss_base=0.484875 ms_per_step=3699.88
[2026-09-05 15:08] monitor: 新 run epoch=13 step=1420，最新 loss=0.941287（loss_full=0.456413, loss_base=0.484875）；GPU 利用率约 100%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=13 step=1440 loss=0.993173 loss_full=0.471079 loss_base=0.522094 ms_per_step=3626.79
[2026-09-05 15:09] monitor: 新 run epoch=13 step=1440，最新 loss=0.993173（loss_full=0.471079, loss_base=0.522094）；GPU 利用率约 100%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=13 step=1460 loss=0.918482 loss_full=0.448099 loss_base=0.470383 ms_per_step=3431.01
[2026-09-05 15:10] monitor: 新 run epoch=13 step=1460，最新 loss=0.918482（loss_full=0.448099, loss_base=0.470383）；GPU 利用率约 100%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
[epoch-end] epoch=13 elapsed=376.87s step=1470
[epoch-start] epoch=14 step=1470 steps_per_epoch=105
task=ci2v epoch=14 step=1480 loss=0.939126 loss_full=0.432179 loss_base=0.506947 ms_per_step=3668.05
[2026-09-05 15:11] monitor: 新 run epoch=13 已完成（step=1470，耗时 376.87s）并进入 epoch=14；最新 step=1480 loss=0.939126（loss_full=0.432179, loss_base=0.506947），GPU 利用率约 98%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=14 step=1500 loss=0.973855 loss_full=0.462925 loss_base=0.510930 ms_per_step=3774.87
[profile] window=364.28s updates=100 data_wait=2.0ms h2d=1.4ms stage1=260.4ms dit=640.6ms update=27.7ms allocated=29.0GiB reserved=39.1GiB peak=37.9GiB
[2026-09-05 15:13] monitor: 新 run epoch=14 step=1500，最新 loss=0.973855（loss_full=0.462925, loss_base=0.510930）；profile window=364.28s 完成，allocated=29.0GiB、reserved=39.1GiB、peak=37.9GiB；GPU 利用率约 100%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=14 step=1520 loss=0.921109 loss_full=0.438334 loss_base=0.482774 ms_per_step=3623.38
[2026-09-05 15:14] monitor: 新 run epoch=14 step=1520，最新 loss=0.921109（loss_full=0.438334, loss_base=0.482774）；GPU 利用率约 100%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=14 step=1540 loss=0.922151 loss_full=0.439196 loss_base=0.482954 ms_per_step=3527.30
[2026-09-05 15:15] monitor: 新 run epoch=14 step=1540，最新 loss=0.922151（loss_full=0.439196, loss_base=0.482954）；GPU 利用率约 100%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=14 step=1560 loss=0.961248 loss_full=0.453783 loss_base=0.507465 ms_per_step=3537.81
[2026-09-05 15:16] monitor: 新 run epoch=14 step=1560，最新 loss=0.961248（loss_full=0.453783, loss_base=0.507465）；GPU 利用率约 88%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
[epoch-end] epoch=14 elapsed=380.66s step=1575
[epoch-start] epoch=15 step=1575 steps_per_epoch=105
task=ci2v epoch=15 step=1580 loss=0.927760 loss_full=0.423491 loss_base=0.504269 ms_per_step=3606.35
[2026-09-05 15:17] monitor: 新 run epoch=14 已完成（step=1575，耗时 380.66s）并进入 epoch=15；最新 step=1580 loss=0.927760（loss_full=0.423491, loss_base=0.504269），GPU 利用率约 100%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=15 step=1600 loss=0.913578 loss_full=0.435890 loss_base=0.477688 ms_per_step=3673.57
[profile] window=359.63s updates=100 data_wait=2.2ms h2d=1.3ms stage1=249.7ms dit=639.6ms update=27.7ms allocated=29.0GiB reserved=39.1GiB peak=37.9GiB
[2026-09-05 15:19] monitor: 新 run epoch=15 step=1600，最新 loss=0.913578（loss_full=0.435890, loss_base=0.477688）；profile window=359.63s 完成，allocated=29.0GiB、reserved=39.1GiB、peak=37.9GiB；GPU 利用率约 99%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=15 step=1620 loss=0.940976 loss_full=0.447738 loss_base=0.493237 ms_per_step=3495.84
[2026-09-05 15:20] monitor: 新 run epoch=15 step=1620，最新 loss=0.940976（loss_full=0.447738, loss_base=0.493237）；GPU 利用率约 100%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=15 step=1640 loss=0.903561 loss_full=0.427967 loss_base=0.475594 ms_per_step=3652.30
[2026-09-05 15:21] monitor: 新 run epoch=15 step=1640，最新 loss=0.903561（loss_full=0.427967, loss_base=0.475594）；GPU 利用率约 50%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=15 step=1660 loss=0.869155 loss_full=0.417361 loss_base=0.451794 ms_per_step=3648.36
[2026-09-05 15:23] monitor: 新 run epoch=15 step=1660，最新 loss=0.869155（loss_full=0.417361, loss_base=0.451794）；GPU 利用率约 62%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=15 step=1680 loss=0.911738 loss_full=0.432798 loss_base=0.478940 ms_per_step=3641.23
[epoch-end] epoch=15 elapsed=382.84s step=1680
[epoch-start] epoch=16 step=1680 steps_per_epoch=105
[2026-09-05 15:24] monitor: 新 run epoch=15 已完成（step=1680，耗时 382.84s）并进入 epoch=16；最新 loss=0.911738（loss_full=0.432798, loss_base=0.478940），GPU 利用率约 91%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=16 step=1700 loss=0.955848 loss_full=0.457258 loss_base=0.498590 ms_per_step=3665.75
[profile] window=362.34s updates=100 data_wait=1.9ms h2d=1.4ms stage1=253.9ms dit=642.2ms update=27.7ms allocated=29.0GiB reserved=39.1GiB peak=37.9GiB
[2026-09-05 15:24] monitor: 新 run epoch=16 step=1700，最新 loss=0.955848（loss_full=0.457258, loss_base=0.498590）；profile window=362.34s 完成，allocated=29.0GiB、reserved=39.1GiB、peak=37.9GiB；GPU 利用率约 13%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=16 step=1720 loss=0.839872 loss_full=0.399613 loss_base=0.440258 ms_per_step=3499.42
[2026-09-05 15:26] monitor: 新 run epoch=16 step=1720，最新 loss=0.839872（loss_full=0.399613, loss_base=0.440258）；GPU 利用率约 100%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=16 step=1740 loss=0.941127 loss_full=0.450764 loss_base=0.490363 ms_per_step=3674.69
[2026-09-05 15:27] monitor: 新 run epoch=16 step=1740，最新 loss=0.941127（loss_full=0.450764, loss_base=0.490363）；GPU 利用率约 100%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=16 step=1760 loss=0.953056 loss_full=0.455926 loss_base=0.497130 ms_per_step=3680.08
[2026-09-05 15:29] monitor: 新 run epoch=16 step=1760，最新 loss=0.953056（loss_full=0.455926, loss_base=0.497130）；GPU 利用率约 100%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=16 step=1780 loss=0.907121 loss_full=0.424038 loss_base=0.483083 ms_per_step=3516.45
[epoch-end] epoch=16 elapsed=377.28s step=1785
[epoch-start] epoch=17 step=1785 steps_per_epoch=105
[2026-09-05 15:30] monitor: 新 run epoch=16 已完成（step=1785，耗时 377.28s）并进入 epoch=17；最新 step=1780 loss=0.907121（loss_full=0.424038, loss_base=0.483083），GPU 利用率约 100%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=17 step=1800 loss=0.947357 loss_full=0.447757 loss_base=0.499600 ms_per_step=3480.44
[profile] window=357.29s updates=100 data_wait=1.7ms h2d=1.3ms stage1=244.5ms dit=639.7ms update=27.7ms allocated=29.0GiB reserved=39.1GiB peak=37.9GiB
[2026-09-05 15:31] monitor: 新 run epoch=17 step=1800，最新 loss=0.947357（loss_full=0.447757, loss_base=0.499600）；profile window=357.29s 完成，allocated=29.0GiB、reserved=39.1GiB、peak=37.9GiB；GPU 利用率约 100%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=17 step=1820 loss=0.867459 loss_full=0.407119 loss_base=0.460340 ms_per_step=3510.55
[2026-09-05 15:32] monitor: 新 run epoch=17 step=1820，最新 loss=0.867459（loss_full=0.407119, loss_base=0.460340）；GPU 利用率约 100%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=17 step=1840 loss=0.854543 loss_full=0.402238 loss_base=0.452305 ms_per_step=3478.85
[2026-09-05 15:33] monitor: 新 run epoch=17 step=1840，最新 loss=0.854543（loss_full=0.402238, loss_base=0.452305）；GPU 利用率约 100%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=17 step=1860 loss=0.841653 loss_full=0.401456 loss_base=0.440197 ms_per_step=3743.21
[2026-09-05 15:35] monitor: 新 run epoch=17 step=1860，最新 loss=0.841653（loss_full=0.401456, loss_base=0.440197）；GPU 利用率约 100%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=17 step=1880 loss=0.833924 loss_full=0.379218 loss_base=0.454707 ms_per_step=3681.82
[2026-09-05 15:35] monitor: 新 run epoch=17 step=1880，最新 loss=0.833924（loss_full=0.379218, loss_base=0.454707）；GPU 利用率约 100%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
[epoch-end] epoch=17 elapsed=378.87s step=1890
[epoch-start] epoch=18 step=1890 steps_per_epoch=105
task=ci2v epoch=18 step=1900 loss=0.966574 loss_full=0.449073 loss_base=0.517501 ms_per_step=3794.34
[profile] window=364.44s updates=100 data_wait=1.9ms h2d=1.5ms stage1=259.8ms dit=641.3ms update=27.7ms allocated=29.0GiB reserved=39.1GiB peak=37.9GiB
[2026-09-05 15:37] monitor: 新 run epoch=17 已完成（step=1890，耗时 378.87s）并进入 epoch=18；最新 step=1900 loss=0.966574（loss_full=0.449073, loss_base=0.517501），profile window=364.44s 完成；GPU 利用率约 100%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=18 step=1920 loss=0.896139 loss_full=0.425552 loss_base=0.470586 ms_per_step=3555.60
[2026-09-05 15:38] monitor: 新 run epoch=18 step=1920，最新 loss=0.896139（loss_full=0.425552, loss_base=0.470586）；GPU 利用率约 100%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=18 step=1940 loss=0.913574 loss_full=0.433451 loss_base=0.480122 ms_per_step=3531.09
[2026-09-05 15:39] monitor: 新 run epoch=18 step=1940，最新 loss=0.913574（loss_full=0.433451, loss_base=0.480122）；GPU 利用率约 28%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=18 step=1960 loss=0.872911 loss_full=0.421003 loss_base=0.451908 ms_per_step=3678.03
[2026-09-05 15:41] monitor: 新 run epoch=18 step=1960，最新 loss=0.872911（loss_full=0.421003, loss_base=0.451908）；GPU 利用率约 100%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=18 step=1980 loss=0.911192 loss_full=0.425296 loss_base=0.485895 ms_per_step=3579.16
[2026-09-05 15:41] monitor: 新 run epoch=18 step=1980，最新 loss=0.911192（loss_full=0.425296, loss_base=0.485895）；GPU 利用率约 14%，显存约 40843 MiB，trainer PID=34953 存活；错误扫描无命中。
[epoch-end] epoch=18 elapsed=380.23s step=1995
[epoch-start] epoch=19 step=1995 steps_per_epoch=105
task=ci2v epoch=19 step=2000 loss=0.848011 loss_full=0.394152 loss_base=0.453859 ms_per_step=3553.95
[profile] window=358.23s updates=100 data_wait=1.6ms h2d=1.3ms stage1=247.3ms dit=639.1ms update=27.7ms allocated=29.0GiB reserved=39.1GiB peak=37.9GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=2000 elapsed=2.21s
[checkpoint] step=2000 elapsed=27.68s
[2026-09-05 15:44] monitor: 新 run epoch=18 已完成（step=1995，耗时 380.23s）并进入 epoch=19；step=2000 loss=0.848011（loss_full=0.394152, loss_base=0.453859），sample 用时 2.21s、checkpoint 用时 27.68s，step-00002000.pt 已写出；GPU 利用率约 64%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=19 step=2020 loss=0.876568 loss_full=0.417852 loss_base=0.458716 ms_per_step=3688.06
[2026-09-05 15:44] monitor: step=2000 sample/checkpoint 后训练继续到 epoch=19 step=2020，最新 loss=0.876568（loss_full=0.417852, loss_base=0.458716）；GPU 利用率约 92%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=19 step=2040 loss=0.842226 loss_full=0.393418 loss_base=0.448808 ms_per_step=3720.25
[2026-09-05 15:46] monitor: 新 run epoch=19 step=2040，最新 loss=0.842226（loss_full=0.393418, loss_base=0.448808）；step=2000 sample 后 GPU 利用率约 100%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=19 step=2060 loss=0.803642 loss_full=0.381659 loss_base=0.421983 ms_per_step=3699.14
[2026-09-05 15:47] monitor: 新 run epoch=19 step=2060，最新 loss=0.803642（loss_full=0.381659, loss_base=0.421983）；step=2000 sample 后 GPU 利用率约 100%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=19 step=2080 loss=0.836021 loss_full=0.389296 loss_base=0.446725 ms_per_step=3427.23
[2026-09-05 15:48] monitor: 新 run epoch=19 step=2080，最新 loss=0.836021（loss_full=0.389296, loss_base=0.446725）；GPU 利用率约 100%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=19 step=2100 loss=0.871942 loss_full=0.415958 loss_base=0.455985 ms_per_step=3499.93
[profile] window=360.96s updates=100 data_wait=0.9ms h2d=1.4ms stage1=253.6ms dit=640.3ms update=27.7ms allocated=29.0GiB reserved=39.4GiB peak=38.0GiB
[epoch-end] epoch=19 elapsed=408.30s step=2100
[epoch-start] epoch=20 step=2100 steps_per_epoch=105
[2026-09-05 15:50] monitor: 新 run epoch=19 已完成（step=2100，耗时 408.30s）并进入 epoch=20；最新 loss=0.871942（loss_full=0.415958, loss_base=0.455985），profile window=360.96s 完成，allocated=29.0GiB、reserved=39.4GiB、peak=38.0GiB；GPU 利用率约 8%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=20 step=2120 loss=0.860284 loss_full=0.398512 loss_base=0.461772 ms_per_step=3815.17
[2026-09-05 15:50] monitor: 新 run epoch=20 step=2120，最新 loss=0.860284（loss_full=0.398512, loss_base=0.461772）；GPU 利用率约 100%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=20 step=2140 loss=0.872997 loss_full=0.414173 loss_base=0.458824 ms_per_step=3584.83
[2026-09-05 15:52] monitor: 新 run epoch=20 step=2140，最新 loss=0.872997（loss_full=0.414173, loss_base=0.458824）；GPU 利用率约 56%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=20 step=2160 loss=0.829834 loss_full=0.392871 loss_base=0.436962 ms_per_step=3699.70
[2026-09-05 15:54] monitor: 新 run epoch=20 step=2160，最新 loss=0.829834（loss_full=0.392871, loss_base=0.436962）；GPU 利用率约 48%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=20 step=2180 loss=0.834735 loss_full=0.384023 loss_base=0.450713 ms_per_step=3614.88
task=ci2v epoch=20 step=2200 loss=0.895094 loss_full=0.430865 loss_base=0.464229 ms_per_step=3776.75
[profile] window=370.09s updates=100 data_wait=2.1ms h2d=1.5ms stage1=271.7ms dit=643.6ms update=27.7ms allocated=29.0GiB reserved=39.4GiB peak=38.0GiB
[2026-09-05 15:55] monitor: 新 run epoch=20 step=2200，最新 loss=0.895094（loss_full=0.430865, loss_base=0.464229）；GPU 利用率约 100%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[epoch-end] epoch=20 elapsed=387.94s step=2205
[epoch-start] epoch=21 step=2205 steps_per_epoch=105
[2026-09-05 15:56] monitor: 新 run epoch=20 已正常收尾（elapsed=387.94s, step=2205），epoch=21 已启动；GPU 利用率约 28%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=21 step=2220 loss=0.863495 loss_full=0.411459 loss_base=0.452036 ms_per_step=3745.54
[2026-09-05 15:58] monitor: 新 run epoch=21 step=2220，最新 loss=0.863495（loss_full=0.411459, loss_base=0.452036）；trainer PID=34953 存活，训练持续推进；错误扫描无命中。
task=ci2v epoch=21 step=2240 loss=0.855393 loss_full=0.395162 loss_base=0.460231 ms_per_step=3682.25
[2026-09-05 15:58] monitor: 新 run epoch=21 step=2240，最新 loss=0.855393（loss_full=0.395162, loss_base=0.460231）；GPU 利用率约 94%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=21 step=2260 loss=0.970432 loss_full=0.465449 loss_base=0.504984 ms_per_step=3484.14
[2026-09-05 15:59] monitor: 新 run epoch=21 step=2260，最新 loss=0.970432（loss_full=0.465449, loss_base=0.504984）；GPU 利用率约 69%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=21 step=2280 loss=0.838869 loss_full=0.386865 loss_base=0.452003 ms_per_step=3771.53
[2026-09-05 16:00] monitor: 新 run epoch=21 step=2280，最新 loss=0.838869（loss_full=0.386865, loss_base=0.452003）；GPU 利用率约 99%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=21 step=2300 loss=0.803795 loss_full=0.370890 loss_base=0.432905 ms_per_step=3600.14
[profile] window=365.93s updates=100 data_wait=2.1ms h2d=1.5ms stage1=264.2ms dit=640.7ms update=27.7ms allocated=29.0GiB reserved=39.4GiB peak=38.0GiB
[2026-09-05 16:01] monitor: 新 run epoch=21 step=2300，最新 loss=0.803795（loss_full=0.370890, loss_base=0.432905）；profile 正常完成，GPU 利用率约 94%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[epoch-end] epoch=21 elapsed=383.23s step=2310
[epoch-start] epoch=22 step=2310 steps_per_epoch=105
task=ci2v epoch=22 step=2320 loss=0.824806 loss_full=0.386095 loss_base=0.438711 ms_per_step=3628.09
[2026-09-05 16:02] monitor: 新 run epoch=21 已正常收尾（elapsed=383.23s, step=2310），epoch=22 已启动；GPU 利用率约 100%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[2026-09-05 16:03] monitor: 新 run epoch=22 step=2320，最新 loss=0.824806（loss_full=0.386095, loss_base=0.438711）；GPU 利用率约 64%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[2026-09-05 16:03] monitor: 新 run epoch=22 step=2320 后持续运行；GPU 利用率约 99%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=22 step=2340 loss=0.851293 loss_full=0.396061 loss_base=0.455232 ms_per_step=3772.88
[2026-09-05 16:04] monitor: 新 run epoch=22 step=2340，最新 loss=0.851293（loss_full=0.396061, loss_base=0.455232）；GPU 利用率约 94%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[2026-09-05 16:05] monitor: 新 run epoch=22 step=2340 后持续运行；GPU 利用率约 98%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=22 step=2360 loss=0.843556 loss_full=0.383679 loss_base=0.459877 ms_per_step=3481.05
[2026-09-05 16:05] monitor: 新 run epoch=22 step=2360，最新 loss=0.843556（loss_full=0.383679, loss_base=0.459877）；GPU 利用率约 54%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[2026-09-05 16:06] monitor: 新 run epoch=22 step=2360 后持续运行；GPU 利用率约 100%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=22 step=2380 loss=0.830346 loss_full=0.389307 loss_base=0.441038 ms_per_step=3736.14
[2026-09-05 16:06] monitor: 新 run epoch=22 step=2380，最新 loss=0.830346（loss_full=0.389307, loss_base=0.441038）；GPU 利用率约 97%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[2026-09-05 16:07] monitor: 新 run epoch=22 step=2380，最新 loss=0.830346（loss_full=0.389307, loss_base=0.441038）；GPU 利用率约 100%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=22 step=2400 loss=0.888005 loss_full=0.419783 loss_base=0.468222 ms_per_step=3624.15
[profile] window=365.11s updates=100 data_wait=2.2ms h2d=1.4ms stage1=259.3ms dit=643.6ms update=27.7ms allocated=29.0GiB reserved=39.4GiB peak=38.0GiB
[2026-09-05 16:08] monitor: 新 run epoch=22 step=2400，最新 loss=0.888005（loss_full=0.419783, loss_base=0.468222）；profile 正常完成，GPU 利用率约 91%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[epoch-end] epoch=22 elapsed=385.28s step=2415
[epoch-start] epoch=23 step=2415 steps_per_epoch=105
[2026-09-05 16:08] monitor: 新 run epoch=22 已正常收尾（elapsed=385.28s, step=2415），epoch=23 已启动；GPU 利用率约 94%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=23 step=2420 loss=0.877077 loss_full=0.409063 loss_base=0.468013 ms_per_step=3636.23
[2026-09-05 16:09] monitor: 新 run epoch=23 step=2420，最新 loss=0.877077（loss_full=0.409063, loss_base=0.468013）；GPU 利用率约 95%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[2026-09-05 16:09] monitor: 新 run epoch=23 step=2420 后持续运行；GPU 利用率约 100%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=23 step=2440 loss=0.774895 loss_full=0.366366 loss_base=0.408529 ms_per_step=3621.44
[2026-09-05 16:10] monitor: 新 run epoch=23 step=2440，最新 loss=0.774895（loss_full=0.366366, loss_base=0.408529）；GPU 利用率约 93%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[2026-09-05 16:11] monitor: 新 run epoch=23 step=2440，最新 loss=0.774895（loss_full=0.366366, loss_base=0.408529）；GPU 利用率约 98%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=23 step=2460 loss=0.882288 loss_full=0.420472 loss_base=0.461816 ms_per_step=3768.95
[2026-09-05 16:11] monitor: 新 run epoch=23 step=2460，最新 loss=0.882288（loss_full=0.420472, loss_base=0.461816）；GPU 利用率约 68%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[2026-09-05 16:12] monitor: 新 run epoch=23 step=2460 后持续运行；GPU 利用率约 86%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=23 step=2480 loss=0.828197 loss_full=0.400024 loss_base=0.428173 ms_per_step=3571.16
[2026-09-05 16:12] monitor: 新 run epoch=23 step=2460 后持续运行；GPU 利用率约 100%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[2026-09-05 16:12] monitor: 新 run epoch=23 step=2480，最新 loss=0.828197（loss_full=0.400024, loss_base=0.428173）；GPU 利用率约 32%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[2026-09-05 16:13] monitor: 新 run epoch=23 step=2480，最新 loss=0.828197（loss_full=0.400024, loss_base=0.428173）；GPU 利用率约 54%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=23 step=2500 loss=0.850018 loss_full=0.406040 loss_base=0.443978 ms_per_step=3777.06
[profile] window=367.76s updates=100 data_wait=1.8ms h2d=1.5ms stage1=268.3ms dit=641.8ms update=27.7ms allocated=29.0GiB reserved=39.4GiB peak=38.0GiB
[2026-09-05 16:13] monitor: 新 run epoch=23 step=2480 后持续运行；GPU 利用率约 88%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[2026-09-05 16:14] monitor: 新 run epoch=23 step=2500，最新 loss=0.850018（loss_full=0.406040, loss_base=0.443978）；profile 正常完成，GPU 利用率约 60%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[2026-09-05 16:14] monitor: 新 run epoch=23 step=2500 后持续运行；GPU 利用率约 37%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=23 step=2520 loss=0.811683 loss_full=0.381108 loss_base=0.430575 ms_per_step=3597.50
[epoch-end] epoch=23 elapsed=384.89s step=2520
[epoch-start] epoch=24 step=2520 steps_per_epoch=105
[2026-09-05 16:15] monitor: 新 run epoch=23 已正常收尾（elapsed=384.89s, step=2520），epoch=24 已启动；GPU 利用率约 100%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[2026-09-05 16:15] monitor: 新 run epoch=24 已启动（step=2520, steps_per_epoch=105）；GPU 利用率约 100%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=24 step=2540 loss=0.890154 loss_full=0.419584 loss_base=0.470570 ms_per_step=3813.48
[2026-09-05 16:16] monitor: 新 run epoch=24 已启动后持续运行（step=2520）；GPU 利用率约 100%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[2026-09-05 16:16] monitor: 新 run epoch=24 step=2540，最新 loss=0.890154（loss_full=0.419584, loss_base=0.470570）；GPU 利用率约 96%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[2026-09-05 16:17] monitor: 新 run epoch=24 step=2540，最新 loss=0.890154（loss_full=0.419584, loss_base=0.470570）；GPU 利用率约 94%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=24 step=2560 loss=0.806370 loss_full=0.385829 loss_base=0.420541 ms_per_step=3601.91
[2026-09-05 16:17] monitor: 新 run epoch=24 step=2540 后持续运行；GPU 利用率约 98%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[2026-09-05 16:17] monitor: 新 run epoch=24 step=2560，最新 loss=0.806370（loss_full=0.385829, loss_base=0.420541）；GPU 利用率约 22%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[2026-09-05 16:18] monitor: 新 run epoch=24 step=2560 后持续运行；GPU 利用率约 100%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=24 step=2580 loss=0.821328 loss_full=0.383503 loss_base=0.437825 ms_per_step=3665.10
[2026-09-05 16:18] monitor: 新 run epoch=24 step=2580，最新 loss=0.821328（loss_full=0.383503, loss_base=0.437825）；GPU 利用率约 100%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[2026-09-05 16:19] monitor: 新 run epoch=24 step=2580 后持续运行；GPU 利用率约 100%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=24 step=2600 loss=0.782093 loss_full=0.365976 loss_base=0.416117 ms_per_step=3511.41
[profile] window=364.05s updates=100 data_wait=2.0ms h2d=1.4ms stage1=256.6ms dit=643.8ms update=27.7ms allocated=29.0GiB reserved=39.4GiB peak=38.0GiB
[2026-09-05 16:20] monitor: 新 run epoch=24 step=2600，最新 loss=0.782093（loss_full=0.365976, loss_base=0.416117）；profile 正常完成，GPU 利用率约 99%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[2026-09-05 16:20] monitor: 新 run epoch=24 step=2600 后持续运行；profile 正常完成，GPU 利用率约 100%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=24 step=2620 loss=0.806122 loss_full=0.374879 loss_base=0.431243 ms_per_step=3411.36
[2026-09-05 16:21] monitor: 新 run epoch=24 step=2600 后持续运行；GPU 利用率约 10%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[epoch-end] epoch=24 elapsed=377.99s step=2625
[epoch-start] epoch=25 step=2625 steps_per_epoch=105
[2026-09-05 16:21] monitor: 新 run epoch=24 已正常收尾（elapsed=377.99s, step=2625），epoch=25 已启动；GPU 利用率约 53%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[2026-09-05 16:22] monitor: 新 run epoch=25 已启动（step=2625, steps_per_epoch=105）；GPU 利用率约 100%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=25 step=2640 loss=0.806494 loss_full=0.381614 loss_base=0.424880 ms_per_step=3588.86
[2026-09-05 16:22] monitor: 新 run epoch=25 step=2640，最新 loss=0.806494（loss_full=0.381614, loss_base=0.424880）；GPU 利用率约 100%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[2026-09-05 16:22] monitor: 新 run epoch=25 step=2640 后持续运行；GPU 利用率约 0%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=25 step=2660 loss=0.817855 loss_full=0.389716 loss_base=0.428139 ms_per_step=3674.93
[2026-09-05 16:23] monitor: 新 run epoch=25 step=2640 后持续运行；GPU 利用率约 92%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[2026-09-05 16:23] monitor: 新 run epoch=25 step=2660，最新 loss=0.817855（loss_full=0.389716, loss_base=0.428139）；GPU 利用率约 100%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=25 step=2680 loss=0.897595 loss_full=0.422663 loss_base=0.474932 ms_per_step=3721.25
[2026-09-05 16:24] monitor: 新 run epoch=25 step=2660，最新 loss=0.817855（loss_full=0.389716, loss_base=0.428139）；GPU 利用率约 100%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[2026-09-05 16:25] monitor: 新 run epoch=25 step=2680，最新 loss=0.897595（loss_full=0.422663, loss_base=0.474932）；GPU 利用率约 15%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[2026-09-05 16:25] monitor: 新 run epoch=25 step=2680 后持续运行；GPU 利用率约 46%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=25 step=2700 loss=0.822350 loss_full=0.385933 loss_base=0.436417 ms_per_step=3773.12
[profile] window=363.65s updates=100 data_wait=1.7ms h2d=1.4ms stage1=257.0ms dit=642.9ms update=27.7ms allocated=29.0GiB reserved=39.4GiB peak=38.0GiB
[2026-09-05 16:26] monitor: 新 run epoch=25 step=2700，最新 loss=0.822350（loss_full=0.385933, loss_base=0.436417）；profile 正常完成，GPU 利用率约 100%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[2026-09-05 16:26] monitor: 新 run epoch=25 step=2700 后持续运行；profile 正常完成，GPU 利用率约 98%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=25 step=2720 loss=0.794807 loss_full=0.365196 loss_base=0.429611 ms_per_step=3603.49
[2026-09-05 16:27] monitor: 新 run epoch=25 step=2720，最新 loss=0.794807（loss_full=0.365196, loss_base=0.429611）；GPU 利用率约 100%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[epoch-end] epoch=25 elapsed=385.24s step=2730
[epoch-start] epoch=26 step=2730 steps_per_epoch=105
[2026-09-05 16:27] monitor: 新 run epoch=25 step=2720 后持续运行；GPU 利用率约 100%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[2026-09-05 16:28] monitor: 新 run epoch=25 已正常收尾（elapsed=385.24s, step=2730），epoch=26 已启动；GPU 利用率约 100%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=26 step=2740 loss=0.815248 loss_full=0.388370 loss_base=0.426879 ms_per_step=3730.63
[2026-09-05 16:28] monitor: 新 run epoch=26 step=2740，最新 loss=0.815248（loss_full=0.388370, loss_base=0.426879）；GPU 利用率约 47%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[2026-09-05 16:29] monitor: 新 run epoch=26 step=2740 后持续运行；GPU 利用率约 100%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=26 step=2760 loss=0.793980 loss_full=0.370750 loss_base=0.423230 ms_per_step=3561.14
[2026-09-05 16:29] monitor: 新 run epoch=26 step=2740 后持续运行；GPU 利用率约 92%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[2026-09-05 16:30] monitor: 新 run epoch=26 step=2760，最新 loss=0.793980（loss_full=0.370750, loss_base=0.423230）；GPU 利用率约 100%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[2026-09-05 16:30] monitor: 新 run epoch=26 step=2760 后持续运行；GPU 利用率约 100%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=26 step=2780 loss=0.828395 loss_full=0.392145 loss_base=0.436250 ms_per_step=3489.30
[2026-09-05 16:31] monitor: 新 run epoch=26 step=2780，最新 loss=0.828395（loss_full=0.392145, loss_base=0.436250）；GPU 利用率约 100%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
[2026-09-05 16:31] monitor: 新 run epoch=26 step=2780 后持续运行；GPU 利用率约 53%，显存约 41163 MiB，trainer PID=34953 存活；错误扫描无命中。
task=ci2v epoch=26 step=2800 loss=0.786380 loss_full=0.363360 loss_base=0.423020 ms_per_step=3426.45
[profile] window=356.48s updates=100 data_wait=2.2ms h2d=1.3ms stage1=240.4ms dit=641.2ms update=27.7ms allocated=29.0GiB reserved=39.4GiB peak=38.0GiB
task=ci2v epoch=26 step=2820 loss=0.778992 loss_full=0.355248 loss_base=0.423743 ms_per_step=3782.19
[epoch-end] epoch=26 elapsed=379.74s step=2835
[epoch-start] epoch=27 step=2835 steps_per_epoch=105
task=ci2v epoch=27 step=2840 loss=0.865453 loss_full=0.415339 loss_base=0.450114 ms_per_step=3747.96
task=ci2v epoch=27 step=2860 loss=0.858281 loss_full=0.394030 loss_base=0.464251 ms_per_step=3674.52
task=ci2v epoch=27 step=2880 loss=0.886038 loss_full=0.415788 loss_base=0.470250 ms_per_step=3601.13
task=ci2v epoch=27 step=2900 loss=0.794113 loss_full=0.374804 loss_base=0.419309 ms_per_step=3390.26
[profile] window=364.19s updates=100 data_wait=2.0ms h2d=1.4ms stage1=258.9ms dit=641.6ms update=27.7ms allocated=29.0GiB reserved=39.4GiB peak=38.0GiB
task=ci2v epoch=27 step=2920 loss=0.843539 loss_full=0.386542 loss_base=0.456997 ms_per_step=3495.70
task=ci2v epoch=27 step=2940 loss=0.765057 loss_full=0.346451 loss_base=0.418606 ms_per_step=3643.23
[epoch-end] epoch=27 elapsed=376.68s step=2940
[epoch-start] epoch=28 step=2940 steps_per_epoch=105
task=ci2v epoch=28 step=2960 loss=0.826675 loss_full=0.385621 loss_base=0.441055 ms_per_step=3519.23
task=ci2v epoch=28 step=2980 loss=0.757448 loss_full=0.338536 loss_base=0.418913 ms_per_step=3466.93
task=ci2v epoch=28 step=3000 loss=0.826636 loss_full=0.390313 loss_base=0.436323 ms_per_step=3562.01
[profile] window=354.01s updates=100 data_wait=1.8ms h2d=1.3ms stage1=237.1ms dit=638.3ms update=27.7ms allocated=29.0GiB reserved=39.4GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=3000 elapsed=2.25s
[checkpoint] step=3000 elapsed=18.75s
task=ci2v epoch=28 step=3020 loss=0.788173 loss_full=0.376560 loss_base=0.411613 ms_per_step=3564.82
task=ci2v epoch=28 step=3040 loss=0.792453 loss_full=0.379572 loss_base=0.412881 ms_per_step=3514.98
[epoch-end] epoch=28 elapsed=390.08s step=3045
[epoch-start] epoch=29 step=3045 steps_per_epoch=105
task=ci2v epoch=29 step=3060 loss=0.769684 loss_full=0.363218 loss_base=0.406466 ms_per_step=3617.85
task=ci2v epoch=29 step=3080 loss=0.827722 loss_full=0.392271 loss_base=0.435451 ms_per_step=3760.68
task=ci2v epoch=29 step=3100 loss=0.792192 loss_full=0.356107 loss_base=0.436085 ms_per_step=3578.04
[profile] window=361.00s updates=100 data_wait=1.8ms h2d=1.4ms stage1=253.7ms dit=639.6ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=29 step=3120 loss=0.803010 loss_full=0.381433 loss_base=0.421577 ms_per_step=3616.90
task=ci2v epoch=29 step=3140 loss=0.768154 loss_full=0.360612 loss_base=0.407542 ms_per_step=3597.08
[epoch-end] epoch=29 elapsed=380.48s step=3150
[epoch-start] epoch=30 step=3150 steps_per_epoch=105
task=ci2v epoch=30 step=3160 loss=0.805786 loss_full=0.382072 loss_base=0.423714 ms_per_step=3436.10
task=ci2v epoch=30 step=3180 loss=0.753195 loss_full=0.348575 loss_base=0.404620 ms_per_step=3432.34
task=ci2v epoch=30 step=3200 loss=0.716448 loss_full=0.333698 loss_base=0.382750 ms_per_step=3489.16
[profile] window=351.69s updates=100 data_wait=1.7ms h2d=1.3ms stage1=231.8ms dit=638.3ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=30 step=3220 loss=0.807317 loss_full=0.378481 loss_base=0.428836 ms_per_step=3427.85
task=ci2v epoch=30 step=3240 loss=0.730114 loss_full=0.325964 loss_base=0.404151 ms_per_step=3704.84
[epoch-end] epoch=30 elapsed=368.78s step=3255
[epoch-start] epoch=31 step=3255 steps_per_epoch=105
task=ci2v epoch=31 step=3260 loss=0.751000 loss_full=0.344831 loss_base=0.406169 ms_per_step=3546.26
task=ci2v epoch=31 step=3280 loss=0.746592 loss_full=0.342863 loss_base=0.403728 ms_per_step=3755.66
task=ci2v epoch=31 step=3300 loss=0.813545 loss_full=0.383354 loss_base=0.430191 ms_per_step=3531.18
[profile] window=359.58s updates=100 data_wait=1.6ms h2d=1.4ms stage1=249.3ms dit=640.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=31 step=3320 loss=0.806497 loss_full=0.370676 loss_base=0.435821 ms_per_step=3613.26
task=ci2v epoch=31 step=3340 loss=0.830627 loss_full=0.396353 loss_base=0.434274 ms_per_step=3770.92
task=ci2v epoch=31 step=3360 loss=0.728810 loss_full=0.332954 loss_base=0.395856 ms_per_step=3401.33
[epoch-end] epoch=31 elapsed=381.38s step=3360
[epoch-start] epoch=32 step=3360 steps_per_epoch=105
task=ci2v epoch=32 step=3380 loss=0.747208 loss_full=0.354826 loss_base=0.392382 ms_per_step=3664.36
task=ci2v epoch=32 step=3400 loss=0.769984 loss_full=0.363170 loss_base=0.406814 ms_per_step=3586.03
[profile] window=360.99s updates=100 data_wait=1.7ms h2d=1.4ms stage1=252.0ms dit=641.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=32 step=3420 loss=0.807607 loss_full=0.385938 loss_base=0.421669 ms_per_step=3451.74
task=ci2v epoch=32 step=3440 loss=0.813166 loss_full=0.388342 loss_base=0.424824 ms_per_step=3390.40
task=ci2v epoch=32 step=3460 loss=0.763807 loss_full=0.354919 loss_base=0.408888 ms_per_step=3524.93
[epoch-end] epoch=32 elapsed=369.19s step=3465
[epoch-start] epoch=33 step=3465 steps_per_epoch=105
task=ci2v epoch=33 step=3480 loss=0.702489 loss_full=0.327780 loss_base=0.374709 ms_per_step=3455.82
task=ci2v epoch=33 step=3500 loss=0.788206 loss_full=0.371270 loss_base=0.416936 ms_per_step=3637.05
[profile] window=349.47s updates=100 data_wait=1.6ms h2d=1.2ms stage1=226.6ms dit=638.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=33 step=3520 loss=0.714961 loss_full=0.336540 loss_base=0.378420 ms_per_step=3462.27
task=ci2v epoch=33 step=3540 loss=0.752644 loss_full=0.341881 loss_base=0.410763 ms_per_step=3320.71
task=ci2v epoch=33 step=3560 loss=0.769983 loss_full=0.358983 loss_base=0.411000 ms_per_step=3643.66
[epoch-end] epoch=33 elapsed=367.41s step=3570
[epoch-start] epoch=34 step=3570 steps_per_epoch=105
task=ci2v epoch=34 step=3580 loss=0.837510 loss_full=0.393145 loss_base=0.444364 ms_per_step=3525.05
task=ci2v epoch=34 step=3600 loss=0.761652 loss_full=0.352422 loss_base=0.409229 ms_per_step=3561.37
[profile] window=350.53s updates=100 data_wait=1.6ms h2d=1.1ms stage1=229.5ms dit=638.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=34 step=3620 loss=0.810755 loss_full=0.383564 loss_base=0.427191 ms_per_step=3567.55
task=ci2v epoch=34 step=3640 loss=0.775065 loss_full=0.352476 loss_base=0.422590 ms_per_step=3698.72
task=ci2v epoch=34 step=3660 loss=0.811059 loss_full=0.379891 loss_base=0.431168 ms_per_step=3721.18
[epoch-end] epoch=34 elapsed=384.29s step=3675
[epoch-start] epoch=35 step=3675 steps_per_epoch=105
task=ci2v epoch=35 step=3680 loss=0.711995 loss_full=0.333297 loss_base=0.378699 ms_per_step=3664.36
task=ci2v epoch=35 step=3700 loss=0.672390 loss_full=0.301162 loss_base=0.371228 ms_per_step=3545.75
[profile] window=364.22s updates=100 data_wait=2.2ms h2d=1.5ms stage1=257.5ms dit=643.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=35 step=3720 loss=0.807139 loss_full=0.373196 loss_base=0.433943 ms_per_step=3758.18
task=ci2v epoch=35 step=3740 loss=0.754497 loss_full=0.345087 loss_base=0.409410 ms_per_step=3550.64
task=ci2v epoch=35 step=3760 loss=0.719395 loss_full=0.320560 loss_base=0.398836 ms_per_step=3606.79
task=ci2v epoch=35 step=3780 loss=0.860094 loss_full=0.394221 loss_base=0.465873 ms_per_step=3616.10
[epoch-end] epoch=35 elapsed=379.76s step=3780
[epoch-start] epoch=36 step=3780 steps_per_epoch=105
task=ci2v epoch=36 step=3800 loss=0.778145 loss_full=0.361320 loss_base=0.416825 ms_per_step=3505.89
[profile] window=361.02s updates=100 data_wait=2.0ms h2d=1.3ms stage1=249.8ms dit=642.9ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=36 step=3820 loss=0.706325 loss_full=0.328496 loss_base=0.377829 ms_per_step=3540.94
task=ci2v epoch=36 step=3840 loss=0.758682 loss_full=0.357440 loss_base=0.401243 ms_per_step=3311.23
task=ci2v epoch=36 step=3860 loss=0.758839 loss_full=0.348705 loss_base=0.410134 ms_per_step=3443.60
task=ci2v epoch=36 step=3880 loss=0.674246 loss_full=0.297973 loss_base=0.376272 ms_per_step=3626.76
[epoch-end] epoch=36 elapsed=366.22s step=3885
[epoch-start] epoch=37 step=3885 steps_per_epoch=105
task=ci2v epoch=37 step=3900 loss=0.706358 loss_full=0.323832 loss_base=0.382526 ms_per_step=3430.16
[profile] window=347.33s updates=100 data_wait=1.9ms h2d=1.1ms stage1=222.8ms dit=636.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=37 step=3920 loss=0.732097 loss_full=0.341269 loss_base=0.390828 ms_per_step=3629.83
task=ci2v epoch=37 step=3940 loss=0.771650 loss_full=0.357888 loss_base=0.413762 ms_per_step=3380.10
task=ci2v epoch=37 step=3960 loss=0.806902 loss_full=0.380120 loss_base=0.426782 ms_per_step=3338.55
task=ci2v epoch=37 step=3980 loss=0.748539 loss_full=0.345543 loss_base=0.402996 ms_per_step=3593.74
[epoch-end] epoch=37 elapsed=365.32s step=3990
[epoch-start] epoch=38 step=3990 steps_per_epoch=105
task=ci2v epoch=38 step=4000 loss=0.822152 loss_full=0.382878 loss_base=0.439274 ms_per_step=3607.65
[profile] window=351.26s updates=100 data_wait=1.9ms h2d=1.2ms stage1=231.2ms dit=637.6ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=4000 elapsed=2.89s
[checkpoint] step=4000 elapsed=34.52s
task=ci2v epoch=38 step=4020 loss=0.781951 loss_full=0.370392 loss_base=0.411560 ms_per_step=3573.36
task=ci2v epoch=38 step=4040 loss=0.657190 loss_full=0.302707 loss_base=0.354483 ms_per_step=3548.59
task=ci2v epoch=38 step=4060 loss=0.766524 loss_full=0.354471 loss_base=0.412053 ms_per_step=3607.65
task=ci2v epoch=38 step=4080 loss=0.837649 loss_full=0.403702 loss_base=0.433947 ms_per_step=3709.51
[epoch-end] epoch=38 elapsed=419.61s step=4095
[epoch-start] epoch=39 step=4095 steps_per_epoch=105
task=ci2v epoch=39 step=4100 loss=0.774555 loss_full=0.358002 loss_base=0.416553 ms_per_step=3711.88
[profile] window=363.29s updates=100 data_wait=1.9ms h2d=1.4ms stage1=258.1ms dit=640.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=39 step=4120 loss=0.867621 loss_full=0.418599 loss_base=0.449022 ms_per_step=3451.28
task=ci2v epoch=39 step=4140 loss=0.736197 loss_full=0.341401 loss_base=0.394795 ms_per_step=3702.70
task=ci2v epoch=39 step=4160 loss=0.684158 loss_full=0.312784 loss_base=0.371374 ms_per_step=3612.52
task=ci2v epoch=39 step=4180 loss=0.800218 loss_full=0.380890 loss_base=0.419328 ms_per_step=3594.41
task=ci2v epoch=39 step=4200 loss=0.714933 loss_full=0.321149 loss_base=0.393783 ms_per_step=3664.65
[profile] window=360.78s updates=100 data_wait=1.0ms h2d=1.4ms stage1=251.6ms dit=641.9ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=39 elapsed=379.61s step=4200
[epoch-start] epoch=40 step=4200 steps_per_epoch=105
task=ci2v epoch=40 step=4220 loss=0.723873 loss_full=0.324738 loss_base=0.399135 ms_per_step=3481.80
task=ci2v epoch=40 step=4240 loss=0.720647 loss_full=0.332147 loss_base=0.388500 ms_per_step=3467.19
task=ci2v epoch=40 step=4260 loss=0.773440 loss_full=0.365220 loss_base=0.408220 ms_per_step=3441.85
task=ci2v epoch=40 step=4280 loss=0.771503 loss_full=0.365791 loss_base=0.405712 ms_per_step=3614.86
task=ci2v epoch=40 step=4300 loss=0.693620 loss_full=0.319154 loss_base=0.374466 ms_per_step=3642.97
[profile] window=353.24s updates=100 data_wait=1.9ms h2d=1.2ms stage1=235.5ms dit=638.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=40 elapsed=369.73s step=4305
[epoch-start] epoch=41 step=4305 steps_per_epoch=105
task=ci2v epoch=41 step=4320 loss=0.705659 loss_full=0.315060 loss_base=0.390599 ms_per_step=3432.35
task=ci2v epoch=41 step=4340 loss=0.718775 loss_full=0.328383 loss_base=0.390392 ms_per_step=3456.92
task=ci2v epoch=41 step=4360 loss=0.796349 loss_full=0.373558 loss_base=0.422791 ms_per_step=3766.14
task=ci2v epoch=41 step=4380 loss=0.732539 loss_full=0.342862 loss_base=0.389677 ms_per_step=3702.22
task=ci2v epoch=41 step=4400 loss=0.659492 loss_full=0.290005 loss_base=0.369487 ms_per_step=3621.65
[profile] window=359.85s updates=100 data_wait=1.5ms h2d=1.3ms stage1=250.1ms dit=640.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=41 elapsed=380.29s step=4410
[epoch-start] epoch=42 step=4410 steps_per_epoch=105
task=ci2v epoch=42 step=4420 loss=0.740935 loss_full=0.341295 loss_base=0.399640 ms_per_step=3637.05
task=ci2v epoch=42 step=4440 loss=0.720790 loss_full=0.322570 loss_base=0.398220 ms_per_step=3543.69
task=ci2v epoch=42 step=4460 loss=0.714279 loss_full=0.331141 loss_base=0.383138 ms_per_step=3472.39
task=ci2v epoch=42 step=4480 loss=0.802250 loss_full=0.373493 loss_base=0.428757 ms_per_step=3387.80
task=ci2v epoch=42 step=4500 loss=0.768314 loss_full=0.358276 loss_base=0.410038 ms_per_step=3717.27
[profile] window=355.43s updates=100 data_wait=1.5ms h2d=1.3ms stage1=240.1ms dit=638.8ms update=30.5ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=42 elapsed=372.75s step=4515
[epoch-start] epoch=43 step=4515 steps_per_epoch=105
task=ci2v epoch=43 step=4520 loss=0.735682 loss_full=0.341659 loss_base=0.394023 ms_per_step=3677.27
task=ci2v epoch=43 step=4540 loss=0.666163 loss_full=0.310164 loss_base=0.355999 ms_per_step=3725.87
task=ci2v epoch=43 step=4560 loss=0.798596 loss_full=0.368029 loss_base=0.430567 ms_per_step=3540.65
task=ci2v epoch=43 step=4580 loss=0.778295 loss_full=0.365577 loss_base=0.412718 ms_per_step=3544.55
task=ci2v epoch=43 step=4600 loss=0.723483 loss_full=0.325903 loss_base=0.397580 ms_per_step=3486.99
[profile] window=359.77s updates=100 data_wait=1.9ms h2d=1.3ms stage1=249.4ms dit=640.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=43 step=4620 loss=0.671956 loss_full=0.298463 loss_base=0.373493 ms_per_step=3442.79
[epoch-end] epoch=43 elapsed=374.92s step=4620
[epoch-start] epoch=44 step=4620 steps_per_epoch=105
task=ci2v epoch=44 step=4640 loss=0.748365 loss_full=0.345985 loss_base=0.402380 ms_per_step=3804.70
task=ci2v epoch=44 step=4660 loss=0.668526 loss_full=0.306175 loss_base=0.362351 ms_per_step=3681.43
task=ci2v epoch=44 step=4680 loss=0.746539 loss_full=0.341993 loss_base=0.404546 ms_per_step=3431.01
task=ci2v epoch=44 step=4700 loss=0.768373 loss_full=0.354261 loss_base=0.414112 ms_per_step=3478.42
[profile] window=357.03s updates=100 data_wait=1.9ms h2d=1.2ms stage1=238.8ms dit=644.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=44 step=4720 loss=0.723307 loss_full=0.331811 loss_base=0.391496 ms_per_step=3635.08
[epoch-end] epoch=44 elapsed=377.30s step=4725
[epoch-start] epoch=45 step=4725 steps_per_epoch=105
task=ci2v epoch=45 step=4740 loss=0.709079 loss_full=0.323532 loss_base=0.385547 ms_per_step=3569.65
task=ci2v epoch=45 step=4760 loss=0.660631 loss_full=0.293741 loss_base=0.366891 ms_per_step=3784.63
task=ci2v epoch=45 step=4780 loss=0.642535 loss_full=0.281241 loss_base=0.361294 ms_per_step=3620.04
task=ci2v epoch=45 step=4800 loss=0.740154 loss_full=0.345986 loss_base=0.394168 ms_per_step=3477.09
[profile] window=361.99s updates=100 data_wait=1.8ms h2d=1.4ms stage1=253.9ms dit=641.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=45 step=4820 loss=0.658442 loss_full=0.311073 loss_base=0.347369 ms_per_step=3725.04
[epoch-end] epoch=45 elapsed=381.58s step=4830
[epoch-start] epoch=46 step=4830 steps_per_epoch=105
task=ci2v epoch=46 step=4840 loss=0.770782 loss_full=0.363189 loss_base=0.407593 ms_per_step=3403.77
task=ci2v epoch=46 step=4860 loss=0.690251 loss_full=0.317666 loss_base=0.372584 ms_per_step=3647.66
task=ci2v epoch=46 step=4880 loss=0.786558 loss_full=0.362053 loss_base=0.424505 ms_per_step=3740.84
task=ci2v epoch=46 step=4900 loss=0.762643 loss_full=0.356348 loss_base=0.406295 ms_per_step=3662.73
[profile] window=363.86s updates=100 data_wait=1.8ms h2d=1.4ms stage1=256.7ms dit=643.7ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=46 step=4920 loss=0.638908 loss_full=0.293631 loss_base=0.345277 ms_per_step=3648.27
[epoch-end] epoch=46 elapsed=381.39s step=4935
[epoch-start] epoch=47 step=4935 steps_per_epoch=105
task=ci2v epoch=47 step=4940 loss=0.726279 loss_full=0.326136 loss_base=0.400142 ms_per_step=3631.06
task=ci2v epoch=47 step=4960 loss=0.709216 loss_full=0.323014 loss_base=0.386202 ms_per_step=3599.80
task=ci2v epoch=47 step=4980 loss=0.730820 loss_full=0.330796 loss_base=0.400023 ms_per_step=3441.21
task=ci2v epoch=47 step=5000 loss=0.712629 loss_full=0.324759 loss_base=0.387870 ms_per_step=3575.04
[profile] window=358.22s updates=100 data_wait=2.0ms h2d=1.3ms stage1=245.5ms dit=640.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=5000 elapsed=2.68s
[checkpoint] step=5000 elapsed=26.09s
task=ci2v epoch=47 step=5020 loss=0.777604 loss_full=0.365385 loss_base=0.412219 ms_per_step=3675.36
task=ci2v epoch=47 step=5040 loss=0.679879 loss_full=0.303678 loss_base=0.376202 ms_per_step=3475.15
[epoch-end] epoch=47 elapsed=404.32s step=5040
[epoch-start] epoch=48 step=5040 steps_per_epoch=105
task=ci2v epoch=48 step=5060 loss=0.759286 loss_full=0.345703 loss_base=0.413583 ms_per_step=3742.59
task=ci2v epoch=48 step=5080 loss=0.854284 loss_full=0.399455 loss_base=0.454829 ms_per_step=3728.00
task=ci2v epoch=48 step=5100 loss=0.723986 loss_full=0.330052 loss_base=0.393934 ms_per_step=3535.23
[profile] window=363.39s updates=100 data_wait=2.0ms h2d=1.6ms stage1=257.0ms dit=641.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=48 step=5120 loss=0.693016 loss_full=0.314211 loss_base=0.378805 ms_per_step=3684.32
task=ci2v epoch=48 step=5140 loss=0.673695 loss_full=0.302797 loss_base=0.370898 ms_per_step=3723.20
[epoch-end] epoch=48 elapsed=385.65s step=5145
[epoch-start] epoch=49 step=5145 steps_per_epoch=105
task=ci2v epoch=49 step=5160 loss=0.730446 loss_full=0.330666 loss_base=0.399780 ms_per_step=3547.00
task=ci2v epoch=49 step=5180 loss=0.767654 loss_full=0.354325 loss_base=0.413329 ms_per_step=3521.94
task=ci2v epoch=49 step=5200 loss=0.756523 loss_full=0.362444 loss_base=0.394078 ms_per_step=3632.41
[profile] window=362.44s updates=100 data_wait=1.7ms h2d=1.4ms stage1=256.4ms dit=640.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=49 step=5220 loss=0.626528 loss_full=0.282249 loss_base=0.344279 ms_per_step=3658.66
task=ci2v epoch=49 step=5240 loss=0.674780 loss_full=0.314157 loss_base=0.360623 ms_per_step=3650.96
[epoch-end] epoch=49 elapsed=376.65s step=5250
[epoch-start] epoch=50 step=5250 steps_per_epoch=105
task=ci2v epoch=50 step=5260 loss=0.682843 loss_full=0.312874 loss_base=0.369968 ms_per_step=3572.36
task=ci2v epoch=50 step=5280 loss=0.754858 loss_full=0.345099 loss_base=0.409759 ms_per_step=3713.83
task=ci2v epoch=50 step=5300 loss=0.694079 loss_full=0.319566 loss_base=0.374513 ms_per_step=3534.72
[profile] window=362.87s updates=100 data_wait=1.8ms h2d=1.4ms stage1=256.8ms dit=641.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=50 step=5320 loss=0.734210 loss_full=0.341699 loss_base=0.392510 ms_per_step=3646.10
task=ci2v epoch=50 step=5340 loss=0.733923 loss_full=0.333504 loss_base=0.400419 ms_per_step=3775.74
[epoch-end] epoch=50 elapsed=388.08s step=5355
[epoch-start] epoch=51 step=5355 steps_per_epoch=105
task=ci2v epoch=51 step=5360 loss=0.660891 loss_full=0.307291 loss_base=0.353601 ms_per_step=3746.83
task=ci2v epoch=51 step=5380 loss=0.766960 loss_full=0.361054 loss_base=0.405906 ms_per_step=3779.46
task=ci2v epoch=51 step=5400 loss=0.767856 loss_full=0.353634 loss_base=0.414222 ms_per_step=3792.42
[profile] window=375.07s updates=100 data_wait=2.3ms h2d=1.6ms stage1=283.7ms dit=643.8ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=51 step=5420 loss=0.727064 loss_full=0.322038 loss_base=0.405027 ms_per_step=3802.25
task=ci2v epoch=51 step=5440 loss=0.680963 loss_full=0.308935 loss_base=0.372028 ms_per_step=3823.81
task=ci2v epoch=51 step=5460 loss=0.750890 loss_full=0.347296 loss_base=0.403594 ms_per_step=3750.38
[epoch-end] epoch=51 elapsed=398.48s step=5460
[epoch-start] epoch=52 step=5460 steps_per_epoch=105
task=ci2v epoch=52 step=5480 loss=0.743826 loss_full=0.339051 loss_base=0.404775 ms_per_step=3666.12
task=ci2v epoch=52 step=5500 loss=0.782774 loss_full=0.361868 loss_base=0.420906 ms_per_step=3642.27
[profile] window=373.95s updates=100 data_wait=2.0ms h2d=1.6ms stage1=277.0ms dit=648.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=52 step=5520 loss=0.707907 loss_full=0.331392 loss_base=0.376515 ms_per_step=3552.50
task=ci2v epoch=52 step=5540 loss=0.713227 loss_full=0.323245 loss_base=0.389981 ms_per_step=3397.16
task=ci2v epoch=52 step=5560 loss=0.683493 loss_full=0.302313 loss_base=0.381180 ms_per_step=3773.22
[epoch-end] epoch=52 elapsed=378.47s step=5565
[epoch-start] epoch=53 step=5565 steps_per_epoch=105
task=ci2v epoch=53 step=5580 loss=0.741278 loss_full=0.336174 loss_base=0.405104 ms_per_step=3593.57
task=ci2v epoch=53 step=5600 loss=0.735232 loss_full=0.336870 loss_base=0.398361 ms_per_step=3599.10
[profile] window=358.57s updates=100 data_wait=1.6ms h2d=1.3ms stage1=245.7ms dit=641.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=53 step=5620 loss=0.761181 loss_full=0.346990 loss_base=0.414191 ms_per_step=3772.59
task=ci2v epoch=53 step=5640 loss=0.699785 loss_full=0.312124 loss_base=0.387661 ms_per_step=3701.51
task=ci2v epoch=53 step=5660 loss=0.748072 loss_full=0.342141 loss_base=0.405931 ms_per_step=3785.68
[epoch-end] epoch=53 elapsed=387.76s step=5670
[epoch-start] epoch=54 step=5670 steps_per_epoch=105
task=ci2v epoch=54 step=5680 loss=0.723726 loss_full=0.337993 loss_base=0.385733 ms_per_step=3665.05
task=ci2v epoch=54 step=5700 loss=0.723074 loss_full=0.337702 loss_base=0.385372 ms_per_step=3486.42
[profile] window=368.49s updates=100 data_wait=1.8ms h2d=1.5ms stage1=268.2ms dit=643.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=54 step=5720 loss=0.728493 loss_full=0.330411 loss_base=0.398082 ms_per_step=3616.21
task=ci2v epoch=54 step=5740 loss=0.768797 loss_full=0.356954 loss_base=0.411844 ms_per_step=3629.11
task=ci2v epoch=54 step=5760 loss=0.740209 loss_full=0.331020 loss_base=0.409188 ms_per_step=3708.11
[epoch-end] epoch=54 elapsed=382.09s step=5775
[epoch-start] epoch=55 step=5775 steps_per_epoch=105
task=ci2v epoch=55 step=5780 loss=0.609878 loss_full=0.269563 loss_base=0.340315 ms_per_step=3667.25
task=ci2v epoch=55 step=5800 loss=0.632361 loss_full=0.282605 loss_base=0.349757 ms_per_step=3458.11
[profile] window=361.84s updates=100 data_wait=1.6ms h2d=1.4ms stage1=253.5ms dit=641.9ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=55 step=5820 loss=0.646292 loss_full=0.284342 loss_base=0.361950 ms_per_step=3357.36
task=ci2v epoch=55 step=5840 loss=0.719979 loss_full=0.336153 loss_base=0.383826 ms_per_step=3764.65
task=ci2v epoch=55 step=5860 loss=0.684355 loss_full=0.311572 loss_base=0.372783 ms_per_step=3787.20
task=ci2v epoch=55 step=5880 loss=0.757259 loss_full=0.353122 loss_base=0.404137 ms_per_step=3702.07
[epoch-end] epoch=55 elapsed=379.76s step=5880
[epoch-start] epoch=56 step=5880 steps_per_epoch=105
task=ci2v epoch=56 step=5900 loss=0.765149 loss_full=0.359698 loss_base=0.405451 ms_per_step=3805.85
[profile] window=368.60s updates=100 data_wait=2.1ms h2d=1.5ms stage1=268.4ms dit=642.9ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=56 step=5920 loss=0.690013 loss_full=0.311445 loss_base=0.378568 ms_per_step=3752.40
task=ci2v epoch=56 step=5940 loss=0.598406 loss_full=0.264233 loss_base=0.334173 ms_per_step=3758.93
task=ci2v epoch=56 step=5960 loss=0.774421 loss_full=0.366482 loss_base=0.407938 ms_per_step=3771.13
task=ci2v epoch=56 step=5980 loss=0.687267 loss_full=0.312059 loss_base=0.375208 ms_per_step=3763.61
[epoch-end] epoch=56 elapsed=395.36s step=5985
[epoch-start] epoch=57 step=5985 steps_per_epoch=105
task=ci2v epoch=57 step=6000 loss=0.666740 loss_full=0.315016 loss_base=0.351724 ms_per_step=3770.28
[profile] window=376.59s updates=100 data_wait=1.8ms h2d=1.7ms stage1=287.1ms dit=644.3ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=6000 elapsed=2.70s
[checkpoint] step=6000 elapsed=37.83s
task=ci2v epoch=57 step=6020 loss=0.704229 loss_full=0.320597 loss_base=0.383633 ms_per_step=3766.91
task=ci2v epoch=57 step=6040 loss=0.660533 loss_full=0.299408 loss_base=0.361126 ms_per_step=3756.88
task=ci2v epoch=57 step=6060 loss=0.762691 loss_full=0.358816 loss_base=0.403875 ms_per_step=3807.47
task=ci2v epoch=57 step=6080 loss=0.599294 loss_full=0.266113 loss_base=0.333181 ms_per_step=3698.64
[epoch-end] epoch=57 elapsed=435.86s step=6090
[epoch-start] epoch=58 step=6090 steps_per_epoch=105
task=ci2v epoch=58 step=6100 loss=0.758587 loss_full=0.360465 loss_base=0.398122 ms_per_step=3767.49
[profile] window=376.21s updates=100 data_wait=1.9ms h2d=1.7ms stage1=284.8ms dit=645.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=58 step=6120 loss=0.682666 loss_full=0.305577 loss_base=0.377089 ms_per_step=3771.79
task=ci2v epoch=58 step=6140 loss=0.711650 loss_full=0.316614 loss_base=0.395037 ms_per_step=3754.97
task=ci2v epoch=58 step=6160 loss=0.676376 loss_full=0.312712 loss_base=0.363664 ms_per_step=3770.54
task=ci2v epoch=58 step=6180 loss=0.715476 loss_full=0.324347 loss_base=0.391129 ms_per_step=3755.93
[epoch-end] epoch=58 elapsed=395.26s step=6195
[epoch-start] epoch=59 step=6195 steps_per_epoch=105
task=ci2v epoch=59 step=6200 loss=0.790500 loss_full=0.361294 loss_base=0.429206 ms_per_step=3769.18
[profile] window=376.71s updates=100 data_wait=2.2ms h2d=1.7ms stage1=286.2ms dit=645.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=59 step=6220 loss=0.696349 loss_full=0.311043 loss_base=0.385306 ms_per_step=3741.55
task=ci2v epoch=59 step=6240 loss=0.685810 loss_full=0.318148 loss_base=0.367663 ms_per_step=3750.32
task=ci2v epoch=59 step=6260 loss=0.816247 loss_full=0.370813 loss_base=0.445435 ms_per_step=3754.98
task=ci2v epoch=59 step=6280 loss=0.740797 loss_full=0.338991 loss_base=0.401806 ms_per_step=3754.54
task=ci2v epoch=59 step=6300 loss=0.763895 loss_full=0.354425 loss_base=0.409471 ms_per_step=3719.02
[profile] window=374.67s updates=100 data_wait=1.0ms h2d=1.7ms stage1=284.3ms dit=643.6ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=59 elapsed=395.03s step=6300
[epoch-start] epoch=60 step=6300 steps_per_epoch=105
task=ci2v epoch=60 step=6320 loss=0.672266 loss_full=0.307509 loss_base=0.364757 ms_per_step=3809.61
task=ci2v epoch=60 step=6340 loss=0.672890 loss_full=0.297635 loss_base=0.375256 ms_per_step=3753.32
task=ci2v epoch=60 step=6360 loss=0.687412 loss_full=0.314409 loss_base=0.373003 ms_per_step=3726.90
task=ci2v epoch=60 step=6380 loss=0.655141 loss_full=0.282683 loss_base=0.372458 ms_per_step=3743.92
task=ci2v epoch=60 step=6400 loss=0.696896 loss_full=0.312749 loss_base=0.384147 ms_per_step=3768.92
[profile] window=376.32s updates=100 data_wait=2.0ms h2d=1.9ms stage1=287.8ms dit=642.6ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=60 elapsed=394.66s step=6405
[epoch-start] epoch=61 step=6405 steps_per_epoch=105
task=ci2v epoch=61 step=6420 loss=0.634396 loss_full=0.285016 loss_base=0.349380 ms_per_step=3785.06
task=ci2v epoch=61 step=6440 loss=0.653557 loss_full=0.287980 loss_base=0.365578 ms_per_step=3758.20
task=ci2v epoch=61 step=6460 loss=0.635910 loss_full=0.274306 loss_base=0.361605 ms_per_step=3768.98
task=ci2v epoch=61 step=6480 loss=0.675688 loss_full=0.308944 loss_base=0.366744 ms_per_step=3761.81
task=ci2v epoch=61 step=6500 loss=0.682323 loss_full=0.318000 loss_base=0.364323 ms_per_step=3766.41
[profile] window=377.07s updates=100 data_wait=1.8ms h2d=1.8ms stage1=286.5ms dit=646.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=61 elapsed=395.88s step=6510
[epoch-start] epoch=62 step=6510 steps_per_epoch=105
task=ci2v epoch=62 step=6520 loss=0.637134 loss_full=0.283177 loss_base=0.353957 ms_per_step=3772.32
task=ci2v epoch=62 step=6540 loss=0.637188 loss_full=0.281983 loss_base=0.355205 ms_per_step=3765.40
task=ci2v epoch=62 step=6560 loss=0.724204 loss_full=0.341658 loss_base=0.382546 ms_per_step=3760.19
task=ci2v epoch=62 step=6580 loss=0.669665 loss_full=0.305741 loss_base=0.363924 ms_per_step=3768.48
task=ci2v epoch=62 step=6600 loss=0.661511 loss_full=0.303273 loss_base=0.358238 ms_per_step=3784.08
[profile] window=377.27s updates=100 data_wait=1.9ms h2d=1.7ms stage1=287.1ms dit=646.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=62 elapsed=396.24s step=6615
[epoch-start] epoch=63 step=6615 steps_per_epoch=105
task=ci2v epoch=63 step=6620 loss=0.665897 loss_full=0.303335 loss_base=0.362562 ms_per_step=3782.41
task=ci2v epoch=63 step=6640 loss=0.699902 loss_full=0.326843 loss_base=0.373058 ms_per_step=3754.32
task=ci2v epoch=63 step=6660 loss=0.740445 loss_full=0.335639 loss_base=0.404806 ms_per_step=3767.58
task=ci2v epoch=63 step=6680 loss=0.695277 loss_full=0.318645 loss_base=0.376632 ms_per_step=3776.28
task=ci2v epoch=63 step=6700 loss=0.671544 loss_full=0.303096 loss_base=0.368448 ms_per_step=3787.91
[profile] window=377.63s updates=100 data_wait=1.8ms h2d=1.7ms stage1=286.2ms dit=647.9ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=63 step=6720 loss=0.682775 loss_full=0.307155 loss_base=0.375619 ms_per_step=3712.87
[epoch-end] epoch=63 elapsed=396.32s step=6720
[epoch-start] epoch=64 step=6720 steps_per_epoch=105
task=ci2v epoch=64 step=6740 loss=0.594943 loss_full=0.257388 loss_base=0.337555 ms_per_step=3818.76
task=ci2v epoch=64 step=6760 loss=0.676347 loss_full=0.315478 loss_base=0.360869 ms_per_step=3771.30
task=ci2v epoch=64 step=6780 loss=0.611414 loss_full=0.276910 loss_base=0.334503 ms_per_step=3761.06
task=ci2v epoch=64 step=6800 loss=0.646022 loss_full=0.295675 loss_base=0.350347 ms_per_step=3763.91
[profile] window=376.82s updates=100 data_wait=2.1ms h2d=1.7ms stage1=286.6ms dit=645.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=64 step=6820 loss=0.776777 loss_full=0.337102 loss_base=0.439675 ms_per_step=3745.84
[epoch-end] epoch=64 elapsed=395.30s step=6825
[epoch-start] epoch=65 step=6825 steps_per_epoch=105
task=ci2v epoch=65 step=6840 loss=0.683684 loss_full=0.313197 loss_base=0.370488 ms_per_step=3791.12
task=ci2v epoch=65 step=6860 loss=0.668606 loss_full=0.306795 loss_base=0.361812 ms_per_step=3760.30
task=ci2v epoch=65 step=6880 loss=0.654392 loss_full=0.296480 loss_base=0.357911 ms_per_step=3755.74
task=ci2v epoch=65 step=6900 loss=0.675795 loss_full=0.309646 loss_base=0.366149 ms_per_step=3757.61
[profile] window=376.48s updates=100 data_wait=2.0ms h2d=1.7ms stage1=286.1ms dit=643.8ms update=32.0ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=65 step=6920 loss=0.565469 loss_full=0.245902 loss_base=0.319566 ms_per_step=3779.73
[epoch-end] epoch=65 elapsed=396.77s step=6930
[epoch-start] epoch=66 step=6930 steps_per_epoch=105
task=ci2v epoch=66 step=6940 loss=0.673111 loss_full=0.304438 loss_base=0.368673 ms_per_step=3820.02
task=ci2v epoch=66 step=6960 loss=0.699281 loss_full=0.315392 loss_base=0.383889 ms_per_step=3744.00
task=ci2v epoch=66 step=6980 loss=0.681800 loss_full=0.308343 loss_base=0.373457 ms_per_step=3757.18
task=ci2v epoch=66 step=7000 loss=0.589904 loss_full=0.257246 loss_base=0.332659 ms_per_step=3820.82
[profile] window=378.69s updates=100 data_wait=2.0ms h2d=1.7ms stage1=287.7ms dit=648.8ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=7000 elapsed=2.76s
[checkpoint] step=7000 elapsed=27.42s
task=ci2v epoch=66 step=7020 loss=0.714360 loss_full=0.334527 loss_base=0.379833 ms_per_step=3785.58
[epoch-end] epoch=66 elapsed=427.42s step=7035
[epoch-start] epoch=67 step=7035 steps_per_epoch=105
task=ci2v epoch=67 step=7040 loss=0.632601 loss_full=0.277458 loss_base=0.355143 ms_per_step=3779.70
task=ci2v epoch=67 step=7060 loss=0.678862 loss_full=0.294714 loss_base=0.384147 ms_per_step=3768.01
task=ci2v epoch=67 step=7080 loss=0.713643 loss_full=0.324668 loss_base=0.388975 ms_per_step=3771.01
task=ci2v epoch=67 step=7100 loss=0.640929 loss_full=0.293666 loss_base=0.347262 ms_per_step=3766.61
[profile] window=377.69s updates=100 data_wait=2.1ms h2d=1.8ms stage1=288.9ms dit=645.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=67 step=7120 loss=0.685714 loss_full=0.308748 loss_base=0.376966 ms_per_step=3755.06
task=ci2v epoch=67 step=7140 loss=0.663680 loss_full=0.288100 loss_base=0.375581 ms_per_step=3716.33
[epoch-end] epoch=67 elapsed=396.05s step=7140
[epoch-start] epoch=68 step=7140 steps_per_epoch=105
task=ci2v epoch=68 step=7160 loss=0.725991 loss_full=0.333135 loss_base=0.392856 ms_per_step=3823.38
task=ci2v epoch=68 step=7180 loss=0.686295 loss_full=0.311133 loss_base=0.375162 ms_per_step=3744.03
task=ci2v epoch=68 step=7200 loss=0.632951 loss_full=0.296760 loss_base=0.336191 ms_per_step=3736.23
[profile] window=375.76s updates=100 data_wait=2.1ms h2d=1.7ms stage1=285.7ms dit=643.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=68 step=7220 loss=0.716818 loss_full=0.332523 loss_base=0.384295 ms_per_step=3768.14
task=ci2v epoch=68 step=7240 loss=0.798815 loss_full=0.360105 loss_base=0.438710 ms_per_step=3768.91
[epoch-end] epoch=68 elapsed=395.18s step=7245
[epoch-start] epoch=69 step=7245 steps_per_epoch=105
task=ci2v epoch=69 step=7260 loss=0.703520 loss_full=0.319531 loss_base=0.383989 ms_per_step=3843.50
task=ci2v epoch=69 step=7280 loss=0.631108 loss_full=0.279196 loss_base=0.351912 ms_per_step=3835.64
task=ci2v epoch=69 step=7300 loss=0.646250 loss_full=0.297466 loss_base=0.348783 ms_per_step=3825.40
[profile] window=381.09s updates=100 data_wait=2.1ms h2d=2.0ms stage1=292.0ms dit=650.3ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=69 step=7320 loss=0.696517 loss_full=0.321705 loss_base=0.374811 ms_per_step=3796.16
task=ci2v epoch=69 step=7340 loss=0.637184 loss_full=0.286295 loss_base=0.350889 ms_per_step=3790.87
[epoch-end] epoch=69 elapsed=401.99s step=7350
[epoch-start] epoch=70 step=7350 steps_per_epoch=105
task=ci2v epoch=70 step=7360 loss=0.715949 loss_full=0.337244 loss_base=0.378705 ms_per_step=3857.13
task=ci2v epoch=70 step=7380 loss=0.658886 loss_full=0.305447 loss_base=0.353439 ms_per_step=3828.06
task=ci2v epoch=70 step=7400 loss=0.611678 loss_full=0.277801 loss_base=0.333876 ms_per_step=3811.75
[profile] window=381.93s updates=100 data_wait=2.2ms h2d=1.7ms stage1=292.9ms dit=651.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=70 step=7420 loss=0.710902 loss_full=0.328713 loss_base=0.382189 ms_per_step=3816.87
task=ci2v epoch=70 step=7440 loss=0.608311 loss_full=0.265619 loss_base=0.342692 ms_per_step=3818.14
[epoch-end] epoch=70 elapsed=401.19s step=7455
[epoch-start] epoch=71 step=7455 steps_per_epoch=105
task=ci2v epoch=71 step=7460 loss=0.662545 loss_full=0.297694 loss_base=0.364851 ms_per_step=3837.94
task=ci2v epoch=71 step=7480 loss=0.647895 loss_full=0.284874 loss_base=0.363020 ms_per_step=3783.71
task=ci2v epoch=71 step=7500 loss=0.658298 loss_full=0.290236 loss_base=0.368062 ms_per_step=3796.48
[profile] window=381.30s updates=100 data_wait=2.3ms h2d=1.7ms stage1=295.8ms dit=647.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=71 step=7520 loss=0.704402 loss_full=0.321966 loss_base=0.382436 ms_per_step=3812.24
task=ci2v epoch=71 step=7540 loss=0.713187 loss_full=0.312845 loss_base=0.400342 ms_per_step=3792.14
task=ci2v epoch=71 step=7560 loss=0.548872 loss_full=0.229936 loss_base=0.318935 ms_per_step=3755.32
[epoch-end] epoch=71 elapsed=400.00s step=7560
[epoch-start] epoch=72 step=7560 steps_per_epoch=105
task=ci2v epoch=72 step=7580 loss=0.677986 loss_full=0.307251 loss_base=0.370735 ms_per_step=3876.90
task=ci2v epoch=72 step=7600 loss=0.622652 loss_full=0.277235 loss_base=0.345418 ms_per_step=3774.33
[profile] window=380.47s updates=100 data_wait=2.0ms h2d=1.7ms stage1=291.9ms dit=649.0ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=72 step=7620 loss=0.706038 loss_full=0.318187 loss_base=0.387851 ms_per_step=3792.98
task=ci2v epoch=72 step=7640 loss=0.686961 loss_full=0.310004 loss_base=0.376957 ms_per_step=3806.18
task=ci2v epoch=72 step=7660 loss=0.659432 loss_full=0.295567 loss_base=0.363866 ms_per_step=3780.00
[epoch-end] epoch=72 elapsed=398.96s step=7665
[epoch-start] epoch=73 step=7665 steps_per_epoch=105
task=ci2v epoch=73 step=7680 loss=0.704348 loss_full=0.319436 loss_base=0.384912 ms_per_step=3869.18
task=ci2v epoch=73 step=7700 loss=0.644591 loss_full=0.278327 loss_base=0.366265 ms_per_step=3780.72
[profile] window=380.84s updates=100 data_wait=2.2ms h2d=1.7ms stage1=293.6ms dit=648.3ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=73 step=7720 loss=0.741220 loss_full=0.324662 loss_base=0.416558 ms_per_step=3806.51
task=ci2v epoch=73 step=7740 loss=0.789404 loss_full=0.370892 loss_base=0.418512 ms_per_step=3805.72
task=ci2v epoch=73 step=7760 loss=0.678869 loss_full=0.317942 loss_base=0.360927 ms_per_step=3613.48
[epoch-end] epoch=73 elapsed=393.36s step=7770
[epoch-start] epoch=74 step=7770 steps_per_epoch=105
task=ci2v epoch=74 step=7780 loss=0.586873 loss_full=0.258530 loss_base=0.328343 ms_per_step=3563.54
task=ci2v epoch=74 step=7800 loss=0.667615 loss_full=0.306615 loss_base=0.361001 ms_per_step=3837.48
[profile] window=372.79s updates=100 data_wait=1.9ms h2d=1.5ms stage1=273.5ms dit=649.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=74 step=7820 loss=0.618516 loss_full=0.274470 loss_base=0.344046 ms_per_step=3843.45
task=ci2v epoch=74 step=7840 loss=0.629383 loss_full=0.284741 loss_base=0.344643 ms_per_step=3851.38
task=ci2v epoch=74 step=7860 loss=0.616503 loss_full=0.270737 loss_base=0.345766 ms_per_step=3813.74
[epoch-end] epoch=74 elapsed=401.43s step=7875
[epoch-start] epoch=75 step=7875 steps_per_epoch=105
task=ci2v epoch=75 step=7880 loss=0.645011 loss_full=0.288585 loss_base=0.356427 ms_per_step=3825.21
task=ci2v epoch=75 step=7900 loss=0.625360 loss_full=0.285117 loss_base=0.340243 ms_per_step=3524.41
[profile] window=377.43s updates=100 data_wait=2.4ms h2d=1.6ms stage1=279.7ms dit=653.9ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=75 step=7920 loss=0.584257 loss_full=0.254646 loss_base=0.329611 ms_per_step=3853.56
task=ci2v epoch=75 step=7940 loss=0.617301 loss_full=0.278682 loss_base=0.338619 ms_per_step=3848.55
task=ci2v epoch=75 step=7960 loss=0.612783 loss_full=0.272767 loss_base=0.340017 ms_per_step=3821.85
task=ci2v epoch=75 step=7980 loss=0.697486 loss_full=0.319696 loss_base=0.377790 ms_per_step=3589.74
[epoch-end] epoch=75 elapsed=393.34s step=7980
[epoch-start] epoch=76 step=7980 steps_per_epoch=105
task=ci2v epoch=76 step=8000 loss=0.640521 loss_full=0.288667 loss_base=0.351854 ms_per_step=3858.85
[profile] window=379.69s updates=100 data_wait=2.1ms h2d=1.6ms stage1=286.1ms dit=653.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=8000 elapsed=2.82s
[checkpoint] step=8000 elapsed=40.78s
task=ci2v epoch=76 step=8020 loss=0.588207 loss_full=0.257129 loss_base=0.331078 ms_per_step=3824.81
task=ci2v epoch=76 step=8040 loss=0.687958 loss_full=0.316043 loss_base=0.371915 ms_per_step=3797.11
task=ci2v epoch=76 step=8060 loss=0.680912 loss_full=0.310869 loss_base=0.370043 ms_per_step=3809.60
task=ci2v epoch=76 step=8080 loss=0.642888 loss_full=0.289428 loss_base=0.353461 ms_per_step=3811.64
[epoch-end] epoch=76 elapsed=443.88s step=8085
[epoch-start] epoch=77 step=8085 steps_per_epoch=105
task=ci2v epoch=77 step=8100 loss=0.642667 loss_full=0.287414 loss_base=0.355253 ms_per_step=3786.14
[profile] window=380.85s updates=100 data_wait=2.1ms h2d=1.7ms stage1=291.8ms dit=650.4ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=77 step=8120 loss=0.592927 loss_full=0.261080 loss_base=0.331847 ms_per_step=3757.31
task=ci2v epoch=77 step=8140 loss=0.675156 loss_full=0.303783 loss_base=0.371372 ms_per_step=3774.82
task=ci2v epoch=77 step=8160 loss=0.672048 loss_full=0.306125 loss_base=0.365922 ms_per_step=3763.58
task=ci2v epoch=77 step=8180 loss=0.558024 loss_full=0.239496 loss_base=0.318528 ms_per_step=3511.25
[epoch-end] epoch=77 elapsed=388.56s step=8190
[epoch-start] epoch=78 step=8190 steps_per_epoch=105
task=ci2v epoch=78 step=8200 loss=0.580551 loss_full=0.246673 loss_base=0.333877 ms_per_step=3559.02
[profile] window=367.58s updates=100 data_wait=2.0ms h2d=1.5ms stage1=265.3ms dit=643.8ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=78 step=8220 loss=0.649653 loss_full=0.293805 loss_base=0.355847 ms_per_step=3327.87
task=ci2v epoch=78 step=8240 loss=0.625015 loss_full=0.276372 loss_base=0.348643 ms_per_step=3531.91
task=ci2v epoch=78 step=8260 loss=0.613799 loss_full=0.264813 loss_base=0.348985 ms_per_step=3776.26
task=ci2v epoch=78 step=8280 loss=0.631093 loss_full=0.275604 loss_base=0.355490 ms_per_step=3623.28
[epoch-end] epoch=78 elapsed=371.79s step=8295
[epoch-start] epoch=79 step=8295 steps_per_epoch=105
task=ci2v epoch=79 step=8300 loss=0.634903 loss_full=0.293762 loss_base=0.341141 ms_per_step=3406.85
[profile] window=353.59s updates=100 data_wait=1.7ms h2d=1.2ms stage1=234.3ms dit=640.8ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=79 step=8320 loss=0.664218 loss_full=0.301187 loss_base=0.363031 ms_per_step=3733.22
task=ci2v epoch=79 step=8340 loss=0.712181 loss_full=0.323916 loss_base=0.388265 ms_per_step=3651.67
task=ci2v epoch=79 step=8360 loss=0.607958 loss_full=0.262106 loss_base=0.345852 ms_per_step=3562.12
task=ci2v epoch=79 step=8380 loss=0.679752 loss_full=0.310963 loss_base=0.368789 ms_per_step=3318.85
task=ci2v epoch=79 step=8400 loss=0.627524 loss_full=0.282114 loss_base=0.345410 ms_per_step=3672.85
[profile] window=359.05s updates=100 data_wait=1.1ms h2d=1.3ms stage1=247.5ms dit=641.8ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=79 elapsed=378.16s step=8400
[epoch-start] epoch=80 step=8400 steps_per_epoch=105
task=ci2v epoch=80 step=8420 loss=0.649635 loss_full=0.290900 loss_base=0.358735 ms_per_step=3823.28
task=ci2v epoch=80 step=8440 loss=0.584095 loss_full=0.252578 loss_base=0.331517 ms_per_step=3770.32
task=ci2v epoch=80 step=8460 loss=0.708781 loss_full=0.328682 loss_base=0.380100 ms_per_step=3755.46
task=ci2v epoch=80 step=8480 loss=0.664588 loss_full=0.289622 loss_base=0.374966 ms_per_step=3776.87
task=ci2v epoch=80 step=8500 loss=0.738295 loss_full=0.344330 loss_base=0.393965 ms_per_step=3477.53
[profile] window=372.33s updates=100 data_wait=1.9ms h2d=1.6ms stage1=276.5ms dit=644.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=80 elapsed=390.56s step=8505
[epoch-start] epoch=81 step=8505 steps_per_epoch=105
task=ci2v epoch=81 step=8520 loss=0.686845 loss_full=0.318419 loss_base=0.368426 ms_per_step=3812.73
task=ci2v epoch=81 step=8540 loss=0.603970 loss_full=0.270328 loss_base=0.333642 ms_per_step=3758.19
task=ci2v epoch=81 step=8560 loss=0.628164 loss_full=0.271690 loss_base=0.356473 ms_per_step=3555.63
task=ci2v epoch=81 step=8580 loss=0.692362 loss_full=0.328347 loss_base=0.364015 ms_per_step=3717.05
task=ci2v epoch=81 step=8600 loss=0.616302 loss_full=0.278482 loss_base=0.337820 ms_per_step=3749.17
[profile] window=372.12s updates=100 data_wait=2.2ms h2d=1.6ms stage1=274.5ms dit=645.8ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=81 elapsed=391.13s step=8610
[epoch-start] epoch=82 step=8610 steps_per_epoch=105
task=ci2v epoch=82 step=8620 loss=0.652891 loss_full=0.294147 loss_base=0.358743 ms_per_step=3782.38
task=ci2v epoch=82 step=8640 loss=0.599333 loss_full=0.260171 loss_base=0.339162 ms_per_step=3700.57
task=ci2v epoch=82 step=8660 loss=0.650896 loss_full=0.288009 loss_base=0.362886 ms_per_step=3735.83
task=ci2v epoch=82 step=8680 loss=0.669562 loss_full=0.302309 loss_base=0.367253 ms_per_step=3769.51
task=ci2v epoch=82 step=8700 loss=0.725967 loss_full=0.332062 loss_base=0.393906 ms_per_step=3753.75
[profile] window=375.10s updates=100 data_wait=2.2ms h2d=1.6ms stage1=280.6ms dit=647.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=82 elapsed=393.79s step=8715
[epoch-start] epoch=83 step=8715 steps_per_epoch=105
task=ci2v epoch=83 step=8720 loss=0.608149 loss_full=0.269013 loss_base=0.339136 ms_per_step=3776.49
task=ci2v epoch=83 step=8740 loss=0.692505 loss_full=0.320834 loss_base=0.371672 ms_per_step=3777.42
task=ci2v epoch=83 step=8760 loss=0.692406 loss_full=0.306807 loss_base=0.385600 ms_per_step=3481.79
task=ci2v epoch=83 step=8780 loss=0.583224 loss_full=0.246580 loss_base=0.336645 ms_per_step=3584.14
task=ci2v epoch=83 step=8800 loss=0.741335 loss_full=0.348420 loss_base=0.392915 ms_per_step=3764.30
[profile] window=367.95s updates=100 data_wait=2.0ms h2d=1.5ms stage1=265.7ms dit=644.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=83 step=8820 loss=0.632044 loss_full=0.280229 loss_base=0.351815 ms_per_step=3547.40
[epoch-end] epoch=83 elapsed=383.41s step=8820
[epoch-start] epoch=84 step=8820 steps_per_epoch=105
task=ci2v epoch=84 step=8840 loss=0.685310 loss_full=0.312861 loss_base=0.372449 ms_per_step=3707.16
task=ci2v epoch=84 step=8860 loss=0.688336 loss_full=0.308527 loss_base=0.379809 ms_per_step=3760.87
task=ci2v epoch=84 step=8880 loss=0.738997 loss_full=0.338291 loss_base=0.400705 ms_per_step=3485.89
task=ci2v epoch=84 step=8900 loss=0.628066 loss_full=0.284487 loss_base=0.343580 ms_per_step=3372.55
[profile] window=357.74s updates=100 data_wait=1.8ms h2d=1.3ms stage1=243.8ms dit=641.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=84 step=8920 loss=0.714507 loss_full=0.329816 loss_base=0.384691 ms_per_step=3760.93
[epoch-end] epoch=84 elapsed=380.37s step=8925
[epoch-start] epoch=85 step=8925 steps_per_epoch=105
task=ci2v epoch=85 step=8940 loss=0.670809 loss_full=0.298103 loss_base=0.372706 ms_per_step=3800.61
task=ci2v epoch=85 step=8960 loss=0.661035 loss_full=0.291043 loss_base=0.369992 ms_per_step=3755.39
task=ci2v epoch=85 step=8980 loss=0.712863 loss_full=0.321013 loss_base=0.391850 ms_per_step=3336.49
task=ci2v epoch=85 step=9000 loss=0.666109 loss_full=0.294365 loss_base=0.371744 ms_per_step=3332.52
[profile] window=359.99s updates=100 data_wait=1.8ms h2d=1.3ms stage1=249.1ms dit=640.5ms update=31.5ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=9000 elapsed=3.05s
[checkpoint] step=9000 elapsed=35.62s
task=ci2v epoch=85 step=9020 loss=0.699503 loss_full=0.319346 loss_base=0.380157 ms_per_step=3566.78
[epoch-end] epoch=85 elapsed=412.94s step=9030
[epoch-start] epoch=86 step=9030 steps_per_epoch=105
task=ci2v epoch=86 step=9040 loss=0.631903 loss_full=0.282515 loss_base=0.349388 ms_per_step=3626.19
task=ci2v epoch=86 step=9060 loss=0.625979 loss_full=0.287927 loss_base=0.338051 ms_per_step=3759.60
task=ci2v epoch=86 step=9080 loss=0.688057 loss_full=0.320176 loss_base=0.367881 ms_per_step=3771.39
task=ci2v epoch=86 step=9100 loss=0.707070 loss_full=0.335975 loss_base=0.371095 ms_per_step=3760.91
[profile] window=369.95s updates=100 data_wait=1.8ms h2d=1.5ms stage1=272.0ms dit=643.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=86 step=9120 loss=0.613922 loss_full=0.266932 loss_base=0.346990 ms_per_step=3550.21
[epoch-end] epoch=86 elapsed=389.14s step=9135
[epoch-start] epoch=87 step=9135 steps_per_epoch=105
task=ci2v epoch=87 step=9140 loss=0.740701 loss_full=0.343322 loss_base=0.397379 ms_per_step=3782.24
task=ci2v epoch=87 step=9160 loss=0.640432 loss_full=0.292416 loss_base=0.348017 ms_per_step=3769.86
task=ci2v epoch=87 step=9180 loss=0.638475 loss_full=0.293078 loss_base=0.345397 ms_per_step=3771.54
task=ci2v epoch=87 step=9200 loss=0.656640 loss_full=0.299876 loss_base=0.356764 ms_per_step=3606.42
[profile] window=369.86s updates=100 data_wait=2.1ms h2d=1.6ms stage1=269.6ms dit=645.2ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=87 step=9220 loss=0.597530 loss_full=0.263339 loss_base=0.334190 ms_per_step=3341.30
task=ci2v epoch=87 step=9240 loss=0.613283 loss_full=0.268026 loss_base=0.345257 ms_per_step=3652.45
[epoch-end] epoch=87 elapsed=383.25s step=9240
[epoch-start] epoch=88 step=9240 steps_per_epoch=105
task=ci2v epoch=88 step=9260 loss=0.761868 loss_full=0.348310 loss_base=0.413558 ms_per_step=3857.14
task=ci2v epoch=88 step=9280 loss=0.651662 loss_full=0.295023 loss_base=0.356639 ms_per_step=3788.22
task=ci2v epoch=88 step=9300 loss=0.701768 loss_full=0.306303 loss_base=0.395466 ms_per_step=3508.39
[profile] window=363.20s updates=100 data_wait=2.1ms h2d=1.3ms stage1=255.1ms dit=643.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=88 step=9320 loss=0.651308 loss_full=0.288941 loss_base=0.362367 ms_per_step=3355.45
task=ci2v epoch=88 step=9340 loss=0.647907 loss_full=0.285790 loss_base=0.362117 ms_per_step=3359.23
[epoch-end] epoch=88 elapsed=374.44s step=9345
[epoch-start] epoch=89 step=9345 steps_per_epoch=105
task=ci2v epoch=89 step=9360 loss=0.656487 loss_full=0.300201 loss_base=0.356286 ms_per_step=3370.13
task=ci2v epoch=89 step=9380 loss=0.676486 loss_full=0.310805 loss_base=0.365681 ms_per_step=3350.13
task=ci2v epoch=89 step=9400 loss=0.687114 loss_full=0.317328 loss_base=0.369787 ms_per_step=3335.06
[profile] window=335.67s updates=100 data_wait=1.6ms h2d=0.8ms stage1=195.3ms dit=635.3ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=89 step=9420 loss=0.605085 loss_full=0.272186 loss_base=0.332900 ms_per_step=3370.25
task=ci2v epoch=89 step=9440 loss=0.654086 loss_full=0.298673 loss_base=0.355413 ms_per_step=3757.10
[epoch-end] epoch=89 elapsed=364.36s step=9450
[epoch-start] epoch=90 step=9450 steps_per_epoch=105
task=ci2v epoch=90 step=9460 loss=0.665310 loss_full=0.299386 loss_base=0.365924 ms_per_step=3793.21
task=ci2v epoch=90 step=9480 loss=0.647097 loss_full=0.287702 loss_base=0.359395 ms_per_step=3768.66
task=ci2v epoch=90 step=9500 loss=0.554001 loss_full=0.233365 loss_base=0.320637 ms_per_step=3792.95
[profile] window=369.89s updates=100 data_wait=2.1ms h2d=1.5ms stage1=269.3ms dit=645.5ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=90 step=9520 loss=0.687496 loss_full=0.309017 loss_base=0.378479 ms_per_step=3740.23
task=ci2v epoch=90 step=9540 loss=0.630541 loss_full=0.265857 loss_base=0.364685 ms_per_step=3768.51
[epoch-end] epoch=90 elapsed=396.84s step=9555
[epoch-start] epoch=91 step=9555 steps_per_epoch=105
task=ci2v epoch=91 step=9560 loss=0.667982 loss_full=0.312416 loss_base=0.355567 ms_per_step=3795.03
task=ci2v epoch=91 step=9580 loss=0.604575 loss_full=0.271886 loss_base=0.332688 ms_per_step=3763.20
task=ci2v epoch=91 step=9600 loss=0.689590 loss_full=0.322469 loss_base=0.367121 ms_per_step=3768.80
[profile] window=376.98s updates=100 data_wait=2.0ms h2d=1.7ms stage1=286.6ms dit=645.9ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=91 step=9620 loss=0.697257 loss_full=0.325236 loss_base=0.372021 ms_per_step=3759.10
task=ci2v epoch=91 step=9640 loss=0.542758 loss_full=0.232963 loss_base=0.309795 ms_per_step=3756.99
task=ci2v epoch=91 step=9660 loss=0.640973 loss_full=0.289101 loss_base=0.351872 ms_per_step=3748.20
[epoch-end] epoch=91 elapsed=396.14s step=9660
[epoch-start] epoch=92 step=9660 steps_per_epoch=105
task=ci2v epoch=92 step=9680 loss=0.657612 loss_full=0.300203 loss_base=0.357409 ms_per_step=3848.84
task=ci2v epoch=92 step=9700 loss=0.592489 loss_full=0.265709 loss_base=0.326780 ms_per_step=3783.60
[profile] window=378.18s updates=100 data_wait=2.2ms h2d=1.7ms stage1=288.2ms dit=647.0ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=92 step=9720 loss=0.620687 loss_full=0.278134 loss_base=0.342553 ms_per_step=3722.10
task=ci2v epoch=92 step=9740 loss=0.592586 loss_full=0.268316 loss_base=0.324270 ms_per_step=3548.99
task=ci2v epoch=92 step=9760 loss=0.616231 loss_full=0.260744 loss_base=0.355487 ms_per_step=3788.70
[epoch-end] epoch=92 elapsed=392.05s step=9765
[epoch-start] epoch=93 step=9765 steps_per_epoch=105
task=ci2v epoch=93 step=9780 loss=0.619057 loss_full=0.260251 loss_base=0.358806 ms_per_step=3783.04
task=ci2v epoch=93 step=9800 loss=0.572428 loss_full=0.254331 loss_base=0.318097 ms_per_step=3783.83
[profile] window=372.79s updates=100 data_wait=2.1ms h2d=1.6ms stage1=276.3ms dit=645.1ms update=30.5ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=93 step=9820 loss=0.747181 loss_full=0.347942 loss_base=0.399238 ms_per_step=3761.61
task=ci2v epoch=93 step=9840 loss=0.764019 loss_full=0.346193 loss_base=0.417826 ms_per_step=3770.72
task=ci2v epoch=93 step=9860 loss=0.587650 loss_full=0.262013 loss_base=0.325637 ms_per_step=3632.46
[epoch-end] epoch=93 elapsed=390.04s step=9870
[epoch-start] epoch=94 step=9870 steps_per_epoch=105
task=ci2v epoch=94 step=9880 loss=0.899647 loss_full=0.368122 loss_base=0.531525 ms_per_step=3614.26
task=ci2v epoch=94 step=9900 loss=0.687081 loss_full=0.315731 loss_base=0.371350 ms_per_step=3770.47
[profile] window=371.26s updates=100 data_wait=1.9ms h2d=1.5ms stage1=273.4ms dit=645.0ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=94 step=9920 loss=0.541856 loss_full=0.241491 loss_base=0.300365 ms_per_step=3686.83
task=ci2v epoch=94 step=9940 loss=0.640176 loss_full=0.289969 loss_base=0.350207 ms_per_step=3558.80
task=ci2v epoch=94 step=9960 loss=0.619959 loss_full=0.282542 loss_base=0.337417 ms_per_step=3755.06
[epoch-end] epoch=94 elapsed=390.72s step=9975
[epoch-start] epoch=95 step=9975 steps_per_epoch=105
task=ci2v epoch=95 step=9980 loss=0.649504 loss_full=0.295529 loss_base=0.353975 ms_per_step=3776.43
task=ci2v epoch=95 step=10000 loss=0.645480 loss_full=0.286786 loss_base=0.358694 ms_per_step=3780.62
[profile] window=371.41s updates=100 data_wait=2.2ms h2d=1.6ms stage1=273.2ms dit=645.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=10000 elapsed=2.95s
[checkpoint] step=10000 elapsed=41.88s
task=ci2v epoch=95 step=10020 loss=0.677829 loss_full=0.304957 loss_base=0.372872 ms_per_step=3585.26
task=ci2v epoch=95 step=10040 loss=0.641426 loss_full=0.277789 loss_base=0.363637 ms_per_step=3770.33
task=ci2v epoch=95 step=10060 loss=0.634767 loss_full=0.291776 loss_base=0.342991 ms_per_step=3757.21
task=ci2v epoch=95 step=10080 loss=0.596758 loss_full=0.259242 loss_base=0.337516 ms_per_step=3744.58
[epoch-end] epoch=95 elapsed=438.04s step=10080
[epoch-start] epoch=96 step=10080 steps_per_epoch=105
task=ci2v epoch=96 step=10100 loss=0.679020 loss_full=0.300414 loss_base=0.378605 ms_per_step=3819.23
[profile] window=373.79s updates=100 data_wait=2.2ms h2d=1.6ms stage1=278.9ms dit=645.6ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=96 step=10120 loss=0.670426 loss_full=0.302666 loss_base=0.367760 ms_per_step=3754.79
task=ci2v epoch=96 step=10140 loss=0.624858 loss_full=0.276395 loss_base=0.348463 ms_per_step=3766.03
task=ci2v epoch=96 step=10160 loss=0.767922 loss_full=0.354951 loss_base=0.412971 ms_per_step=3757.68
task=ci2v epoch=96 step=10180 loss=0.648531 loss_full=0.297284 loss_base=0.351248 ms_per_step=3773.49
[epoch-end] epoch=96 elapsed=395.79s step=10185
[epoch-start] epoch=97 step=10185 steps_per_epoch=105
task=ci2v epoch=97 step=10200 loss=0.668717 loss_full=0.304398 loss_base=0.364319 ms_per_step=3774.75
[profile] window=376.78s updates=100 data_wait=2.2ms h2d=1.7ms stage1=286.1ms dit=645.8ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=97 step=10220 loss=0.663110 loss_full=0.295395 loss_base=0.367716 ms_per_step=3786.23
task=ci2v epoch=97 step=10240 loss=0.609311 loss_full=0.271388 loss_base=0.337922 ms_per_step=3758.44
task=ci2v epoch=97 step=10260 loss=0.618256 loss_full=0.277634 loss_base=0.340622 ms_per_step=3756.38
task=ci2v epoch=97 step=10280 loss=0.652489 loss_full=0.296145 loss_base=0.356345 ms_per_step=3751.19
[epoch-end] epoch=97 elapsed=396.09s step=10290
[epoch-start] epoch=98 step=10290 steps_per_epoch=105
task=ci2v epoch=98 step=10300 loss=0.584035 loss_full=0.260672 loss_base=0.323364 ms_per_step=3789.89
[profile] window=377.09s updates=100 data_wait=2.1ms h2d=1.7ms stage1=285.0ms dit=647.6ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=98 step=10320 loss=0.582117 loss_full=0.258853 loss_base=0.323263 ms_per_step=3363.19
task=ci2v epoch=98 step=10340 loss=0.660974 loss_full=0.297694 loss_base=0.363280 ms_per_step=3320.45
task=ci2v epoch=98 step=10360 loss=0.662968 loss_full=0.295006 loss_base=0.367962 ms_per_step=3618.68
task=ci2v epoch=98 step=10380 loss=0.643832 loss_full=0.297153 loss_base=0.346679 ms_per_step=3615.37
[epoch-end] epoch=98 elapsed=371.09s step=10395
[epoch-start] epoch=99 step=10395 steps_per_epoch=105
task=ci2v epoch=99 step=10400 loss=0.626879 loss_full=0.280382 loss_base=0.346497 ms_per_step=3561.05
[profile] window=349.84s updates=100 data_wait=1.7ms h2d=1.2ms stage1=227.1ms dit=638.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=99 step=10420 loss=0.655101 loss_full=0.295057 loss_base=0.360045 ms_per_step=3465.53
task=ci2v epoch=99 step=10440 loss=0.536780 loss_full=0.231929 loss_base=0.304851 ms_per_step=3437.57
task=ci2v epoch=99 step=10460 loss=0.773396 loss_full=0.354572 loss_base=0.418824 ms_per_step=3318.94
task=ci2v epoch=99 step=10480 loss=0.684079 loss_full=0.305726 loss_base=0.378353 ms_per_step=3729.04
task=ci2v epoch=99 step=10500 loss=0.695422 loss_full=0.321211 loss_base=0.374211 ms_per_step=3707.20
[profile] window=353.43s updates=100 data_wait=1.0ms h2d=1.2ms stage1=235.0ms dit=640.3ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=99 elapsed=371.12s step=10500
[epoch-start] epoch=100 step=10500 steps_per_epoch=105
task=ci2v epoch=100 step=10520 loss=0.645076 loss_full=0.295232 loss_base=0.349844 ms_per_step=3620.29
task=ci2v epoch=100 step=10540 loss=0.653697 loss_full=0.290391 loss_base=0.363305 ms_per_step=3799.13
task=ci2v epoch=100 step=10560 loss=0.605522 loss_full=0.269920 loss_base=0.335602 ms_per_step=3758.00
task=ci2v epoch=100 step=10580 loss=0.565145 loss_full=0.251040 loss_base=0.314105 ms_per_step=3765.03
task=ci2v epoch=100 step=10600 loss=0.582320 loss_full=0.252701 loss_base=0.329618 ms_per_step=3767.99
[profile] window=374.45s updates=100 data_wait=1.8ms h2d=1.6ms stage1=280.8ms dit=645.7ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=100 elapsed=392.62s step=10605
[epoch-start] epoch=101 step=10605 steps_per_epoch=105
task=ci2v epoch=101 step=10620 loss=0.637467 loss_full=0.281048 loss_base=0.356420 ms_per_step=3599.99
task=ci2v epoch=101 step=10640 loss=0.701347 loss_full=0.323274 loss_base=0.378074 ms_per_step=3757.08
task=ci2v epoch=101 step=10660 loss=0.696501 loss_full=0.318670 loss_base=0.377831 ms_per_step=3637.98
task=ci2v epoch=101 step=10680 loss=0.500661 loss_full=0.206918 loss_base=0.293743 ms_per_step=3484.27
task=ci2v epoch=101 step=10700 loss=0.628757 loss_full=0.280965 loss_base=0.347793 ms_per_step=3746.79
[profile] window=364.78s updates=100 data_wait=2.1ms h2d=1.5ms stage1=259.2ms dit=642.9ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=101 elapsed=381.63s step=10710
[epoch-start] epoch=102 step=10710 steps_per_epoch=105
task=ci2v epoch=102 step=10720 loss=0.571131 loss_full=0.251659 loss_base=0.319472 ms_per_step=3523.44
task=ci2v epoch=102 step=10740 loss=0.592802 loss_full=0.252841 loss_base=0.339961 ms_per_step=3768.00
task=ci2v epoch=102 step=10760 loss=0.598633 loss_full=0.264591 loss_base=0.334042 ms_per_step=3764.33
task=ci2v epoch=102 step=10780 loss=0.623790 loss_full=0.278935 loss_base=0.344855 ms_per_step=3757.42
task=ci2v epoch=102 step=10800 loss=0.610301 loss_full=0.267453 loss_base=0.342848 ms_per_step=3759.97
[profile] window=371.73s updates=100 data_wait=2.0ms h2d=1.6ms stage1=274.4ms dit=645.4ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=102 elapsed=392.65s step=10815
[epoch-start] epoch=103 step=10815 steps_per_epoch=105
task=ci2v epoch=103 step=10820 loss=0.608884 loss_full=0.271825 loss_base=0.337060 ms_per_step=3785.36
task=ci2v epoch=103 step=10840 loss=0.513251 loss_full=0.218773 loss_base=0.294477 ms_per_step=3679.60
task=ci2v epoch=103 step=10860 loss=0.570166 loss_full=0.244982 loss_base=0.325183 ms_per_step=3749.77
task=ci2v epoch=103 step=10880 loss=0.583218 loss_full=0.237910 loss_base=0.345308 ms_per_step=3769.32
task=ci2v epoch=103 step=10900 loss=0.657561 loss_full=0.292968 loss_base=0.364593 ms_per_step=3758.25
[profile] window=375.11s updates=100 data_wait=2.1ms h2d=1.7ms stage1=281.7ms dit=646.0ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=103 step=10920 loss=0.664265 loss_full=0.293500 loss_base=0.370765 ms_per_step=3728.42
[epoch-end] epoch=103 elapsed=394.34s step=10920
[epoch-start] epoch=104 step=10920 steps_per_epoch=105
task=ci2v epoch=104 step=10940 loss=0.540036 loss_full=0.231422 loss_base=0.308614 ms_per_step=3848.96
task=ci2v epoch=104 step=10960 loss=0.595749 loss_full=0.249033 loss_base=0.346716 ms_per_step=3760.50
task=ci2v epoch=104 step=10980 loss=0.737950 loss_full=0.344336 loss_base=0.393614 ms_per_step=3752.95
task=ci2v epoch=104 step=11000 loss=0.719600 loss_full=0.327649 loss_base=0.391951 ms_per_step=3786.13
[profile] window=377.80s updates=100 data_wait=2.2ms h2d=1.7ms stage1=287.0ms dit=647.3ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=11000 elapsed=2.92s
[checkpoint] step=11000 elapsed=31.02s
task=ci2v epoch=104 step=11020 loss=0.617488 loss_full=0.272543 loss_base=0.344945 ms_per_step=3487.99
[epoch-end] epoch=104 elapsed=423.35s step=11025
[epoch-start] epoch=105 step=11025 steps_per_epoch=105
task=ci2v epoch=105 step=11040 loss=0.562198 loss_full=0.248312 loss_base=0.313886 ms_per_step=3592.80
task=ci2v epoch=105 step=11060 loss=0.640987 loss_full=0.294755 loss_base=0.346233 ms_per_step=3777.51
task=ci2v epoch=105 step=11080 loss=0.524083 loss_full=0.218306 loss_base=0.305777 ms_per_step=3783.21
task=ci2v epoch=105 step=11100 loss=0.625539 loss_full=0.293182 loss_base=0.332358 ms_per_step=3757.90
[profile] window=368.24s updates=100 data_wait=1.7ms h2d=1.5ms stage1=267.8ms dit=643.4ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=105 step=11120 loss=0.677980 loss_full=0.304227 loss_base=0.373753 ms_per_step=3755.03
[epoch-end] epoch=105 elapsed=393.85s step=11130
[epoch-start] epoch=106 step=11130 steps_per_epoch=105
task=ci2v epoch=106 step=11140 loss=0.612898 loss_full=0.273510 loss_base=0.339388 ms_per_step=3761.61
task=ci2v epoch=106 step=11160 loss=0.623359 loss_full=0.271485 loss_base=0.351874 ms_per_step=3797.27
task=ci2v epoch=106 step=11180 loss=0.595548 loss_full=0.248473 loss_base=0.347074 ms_per_step=3766.89
task=ci2v epoch=106 step=11200 loss=0.587074 loss_full=0.251878 loss_base=0.335196 ms_per_step=3717.76
[profile] window=376.23s updates=100 data_wait=1.9ms h2d=1.7ms stage1=282.7ms dit=648.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=106 step=11220 loss=0.606589 loss_full=0.270492 loss_base=0.336097 ms_per_step=3586.00
[epoch-end] epoch=106 elapsed=392.16s step=11235
[epoch-start] epoch=107 step=11235 steps_per_epoch=105
task=ci2v epoch=107 step=11240 loss=0.676355 loss_full=0.300895 loss_base=0.375460 ms_per_step=3760.93
task=ci2v epoch=107 step=11260 loss=0.623080 loss_full=0.267673 loss_base=0.355407 ms_per_step=3788.41
task=ci2v epoch=107 step=11280 loss=0.583333 loss_full=0.256249 loss_base=0.327084 ms_per_step=3782.00
task=ci2v epoch=107 step=11300 loss=0.613515 loss_full=0.272270 loss_base=0.341244 ms_per_step=3767.86
[profile] window=373.96s updates=100 data_wait=1.9ms h2d=1.6ms stage1=279.0ms dit=646.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=107 step=11320 loss=0.621046 loss_full=0.284723 loss_base=0.336323 ms_per_step=3751.90
task=ci2v epoch=107 step=11340 loss=0.699279 loss_full=0.316974 loss_base=0.382305 ms_per_step=3733.25
[epoch-end] epoch=107 elapsed=396.52s step=11340
[epoch-start] epoch=108 step=11340 steps_per_epoch=105
task=ci2v epoch=108 step=11360 loss=0.587237 loss_full=0.256801 loss_base=0.330437 ms_per_step=3843.87
task=ci2v epoch=108 step=11380 loss=0.572491 loss_full=0.245748 loss_base=0.326744 ms_per_step=3752.25
task=ci2v epoch=108 step=11400 loss=0.577727 loss_full=0.248727 loss_base=0.329000 ms_per_step=3748.96
[profile] window=376.86s updates=100 data_wait=2.0ms h2d=1.7ms stage1=284.8ms dit=647.2ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=108 step=11420 loss=0.564733 loss_full=0.236566 loss_base=0.328167 ms_per_step=3745.52
task=ci2v epoch=108 step=11440 loss=0.492044 loss_full=0.204381 loss_base=0.287664 ms_per_step=3791.10
[epoch-end] epoch=108 elapsed=396.14s step=11445
[epoch-start] epoch=109 step=11445 steps_per_epoch=105
task=ci2v epoch=109 step=11460 loss=0.633472 loss_full=0.279257 loss_base=0.354215 ms_per_step=3786.60
task=ci2v epoch=109 step=11480 loss=0.544969 loss_full=0.234129 loss_base=0.310840 ms_per_step=3641.29
task=ci2v epoch=109 step=11500 loss=0.606344 loss_full=0.275564 loss_base=0.330781 ms_per_step=3691.52
[profile] window=373.38s updates=100 data_wait=1.9ms h2d=1.6ms stage1=277.0ms dit=646.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=109 step=11520 loss=0.635619 loss_full=0.288194 loss_base=0.347425 ms_per_step=3778.26
task=ci2v epoch=109 step=11540 loss=0.631516 loss_full=0.281739 loss_base=0.349776 ms_per_step=3790.22
[epoch-end] epoch=109 elapsed=390.59s step=11550
[epoch-start] epoch=110 step=11550 steps_per_epoch=105
task=ci2v epoch=110 step=11560 loss=0.591168 loss_full=0.248664 loss_base=0.342504 ms_per_step=3563.51
task=ci2v epoch=110 step=11580 loss=0.600119 loss_full=0.258827 loss_base=0.341292 ms_per_step=3608.10
task=ci2v epoch=110 step=11600 loss=0.717730 loss_full=0.327204 loss_base=0.390526 ms_per_step=3705.44
[profile] window=369.17s updates=100 data_wait=1.7ms h2d=1.5ms stage1=268.7ms dit=645.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=110 step=11620 loss=0.550040 loss_full=0.231946 loss_base=0.318094 ms_per_step=3779.63
task=ci2v epoch=110 step=11640 loss=0.656630 loss_full=0.298829 loss_base=0.357801 ms_per_step=3764.07
[epoch-end] epoch=110 elapsed=384.57s step=11655
[epoch-start] epoch=111 step=11655 steps_per_epoch=105
task=ci2v epoch=111 step=11660 loss=0.627636 loss_full=0.267965 loss_base=0.359671 ms_per_step=3386.34
task=ci2v epoch=111 step=11680 loss=0.598152 loss_full=0.257297 loss_base=0.340855 ms_per_step=3706.23
task=ci2v epoch=111 step=11700 loss=0.667516 loss_full=0.301274 loss_base=0.366242 ms_per_step=3507.69
[profile] window=363.14s updates=100 data_wait=1.8ms h2d=1.4ms stage1=255.3ms dit=643.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=111 step=11720 loss=0.710147 loss_full=0.339391 loss_base=0.370756 ms_per_step=3659.71
task=ci2v epoch=111 step=11740 loss=0.681564 loss_full=0.305113 loss_base=0.376451 ms_per_step=3472.85
task=ci2v epoch=111 step=11760 loss=0.575600 loss_full=0.258828 loss_base=0.316772 ms_per_step=3311.88
[epoch-end] epoch=111 elapsed=370.92s step=11760
[epoch-start] epoch=112 step=11760 steps_per_epoch=105
task=ci2v epoch=112 step=11780 loss=0.694764 loss_full=0.315504 loss_base=0.379260 ms_per_step=3344.87
task=ci2v epoch=112 step=11800 loss=0.614049 loss_full=0.267984 loss_base=0.346065 ms_per_step=3575.94
[profile] window=347.57s updates=100 data_wait=1.6ms h2d=1.1ms stage1=222.3ms dit=637.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=112 step=11820 loss=0.582235 loss_full=0.260537 loss_base=0.321698 ms_per_step=3524.81
task=ci2v epoch=112 step=11840 loss=0.654538 loss_full=0.293632 loss_base=0.360906 ms_per_step=3335.25
task=ci2v epoch=112 step=11860 loss=0.594444 loss_full=0.258139 loss_base=0.336305 ms_per_step=3319.57
[epoch-end] epoch=112 elapsed=358.72s step=11865
[epoch-start] epoch=113 step=11865 steps_per_epoch=105
task=ci2v epoch=113 step=11880 loss=0.672746 loss_full=0.314926 loss_base=0.357820 ms_per_step=3375.17
task=ci2v epoch=113 step=11900 loss=0.544684 loss_full=0.233481 loss_base=0.311203 ms_per_step=3761.31
[profile] window=346.59s updates=100 data_wait=1.7ms h2d=1.1ms stage1=219.0ms dit=638.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=113 step=11920 loss=0.647834 loss_full=0.295192 loss_base=0.352642 ms_per_step=3754.81
task=ci2v epoch=113 step=11940 loss=0.713272 loss_full=0.331434 loss_base=0.381838 ms_per_step=3781.47
task=ci2v epoch=113 step=11960 loss=0.716503 loss_full=0.321495 loss_base=0.395008 ms_per_step=3754.31
[epoch-end] epoch=113 elapsed=390.27s step=11970
[epoch-start] epoch=114 step=11970 steps_per_epoch=105
task=ci2v epoch=114 step=11980 loss=0.605872 loss_full=0.267036 loss_base=0.338835 ms_per_step=3834.18
task=ci2v epoch=114 step=12000 loss=0.573753 loss_full=0.250757 loss_base=0.322996 ms_per_step=3786.19
[profile] window=378.47s updates=100 data_wait=1.9ms h2d=1.7ms stage1=286.2ms dit=650.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=12000 elapsed=2.80s
[checkpoint] step=12000 elapsed=33.71s
task=ci2v epoch=114 step=12020 loss=0.699912 loss_full=0.324113 loss_base=0.375799 ms_per_step=3426.41
task=ci2v epoch=114 step=12040 loss=0.502556 loss_full=0.219354 loss_base=0.283202 ms_per_step=3790.71
task=ci2v epoch=114 step=12060 loss=0.694414 loss_full=0.318573 loss_base=0.375841 ms_per_step=3612.58
[epoch-end] epoch=114 elapsed=417.73s step=12075
[epoch-start] epoch=115 step=12075 steps_per_epoch=105
task=ci2v epoch=115 step=12080 loss=0.643818 loss_full=0.283447 loss_base=0.360371 ms_per_step=3337.65
task=ci2v epoch=115 step=12100 loss=0.577870 loss_full=0.251247 loss_base=0.326623 ms_per_step=3391.76
[profile] window=351.46s updates=100 data_wait=1.6ms h2d=1.2ms stage1=228.3ms dit=641.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=115 step=12120 loss=0.652104 loss_full=0.295735 loss_base=0.356369 ms_per_step=3748.71
task=ci2v epoch=115 step=12140 loss=0.667325 loss_full=0.305913 loss_base=0.361411 ms_per_step=3754.52
task=ci2v epoch=115 step=12160 loss=0.563666 loss_full=0.244621 loss_base=0.319045 ms_per_step=3776.00
task=ci2v epoch=115 step=12180 loss=0.680443 loss_full=0.312113 loss_base=0.368329 ms_per_step=3520.14
[epoch-end] epoch=115 elapsed=381.56s step=12180
[epoch-start] epoch=116 step=12180 steps_per_epoch=105
task=ci2v epoch=116 step=12200 loss=0.599004 loss_full=0.259506 loss_base=0.339498 ms_per_step=3808.91
[profile] window=372.43s updates=100 data_wait=2.1ms h2d=1.6ms stage1=275.9ms dit=645.1ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=116 step=12220 loss=0.594170 loss_full=0.261894 loss_base=0.332276 ms_per_step=3545.97
task=ci2v epoch=116 step=12240 loss=0.560883 loss_full=0.242095 loss_base=0.318788 ms_per_step=3322.31
task=ci2v epoch=116 step=12260 loss=0.656178 loss_full=0.296875 loss_base=0.359303 ms_per_step=3342.58
task=ci2v epoch=116 step=12280 loss=0.603188 loss_full=0.272520 loss_base=0.330668 ms_per_step=3306.07
[epoch-end] epoch=116 elapsed=363.22s step=12285
[epoch-start] epoch=117 step=12285 steps_per_epoch=105
task=ci2v epoch=117 step=12300 loss=0.625693 loss_full=0.280961 loss_base=0.344732 ms_per_step=3338.04
[profile] window=337.37s updates=100 data_wait=1.6ms h2d=0.9ms stage1=198.9ms dit=635.9ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=117 step=12320 loss=0.630693 loss_full=0.285486 loss_base=0.345207 ms_per_step=3533.35
task=ci2v epoch=117 step=12340 loss=0.571821 loss_full=0.248572 loss_base=0.323249 ms_per_step=3772.15
task=ci2v epoch=117 step=12360 loss=0.710707 loss_full=0.324265 loss_base=0.386442 ms_per_step=3769.06
task=ci2v epoch=117 step=12380 loss=0.609656 loss_full=0.266232 loss_base=0.343424 ms_per_step=3763.95
[epoch-end] epoch=117 elapsed=381.64s step=12390
[epoch-start] epoch=118 step=12390 steps_per_epoch=105
task=ci2v epoch=118 step=12400 loss=0.673304 loss_full=0.310132 loss_base=0.363172 ms_per_step=3478.63
[profile] window=366.60s updates=100 data_wait=1.8ms h2d=1.5ms stage1=263.5ms dit=643.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=118 step=12420 loss=0.634310 loss_full=0.282522 loss_base=0.351788 ms_per_step=3526.94
task=ci2v epoch=118 step=12440 loss=0.609511 loss_full=0.266689 loss_base=0.342823 ms_per_step=3771.63
task=ci2v epoch=118 step=12460 loss=0.627577 loss_full=0.278256 loss_base=0.349321 ms_per_step=3764.34
task=ci2v epoch=118 step=12480 loss=0.532266 loss_full=0.224898 loss_base=0.307368 ms_per_step=3763.61
[epoch-end] epoch=118 elapsed=387.86s step=12495
[epoch-start] epoch=119 step=12495 steps_per_epoch=105
task=ci2v epoch=119 step=12500 loss=0.609942 loss_full=0.273508 loss_base=0.336434 ms_per_step=3730.49
[profile] window=371.41s updates=100 data_wait=2.0ms h2d=1.6ms stage1=274.2ms dit=644.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=119 step=12520 loss=0.571258 loss_full=0.237306 loss_base=0.333952 ms_per_step=3313.05
task=ci2v epoch=119 step=12540 loss=0.587437 loss_full=0.263195 loss_base=0.324243 ms_per_step=3750.36
task=ci2v epoch=119 step=12560 loss=0.636119 loss_full=0.286552 loss_base=0.349567 ms_per_step=3760.22
task=ci2v epoch=119 step=12580 loss=0.589762 loss_full=0.266100 loss_base=0.323662 ms_per_step=3607.97
task=ci2v epoch=119 step=12600 loss=0.586550 loss_full=0.262726 loss_base=0.323824 ms_per_step=3703.55
[profile] window=362.96s updates=100 data_wait=1.0ms h2d=1.4ms stage1=256.0ms dit=642.9ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=119 elapsed=382.41s step=12600
[epoch-start] epoch=120 step=12600 steps_per_epoch=105
task=ci2v epoch=120 step=12620 loss=0.618141 loss_full=0.277180 loss_base=0.340961 ms_per_step=3705.12
task=ci2v epoch=120 step=12640 loss=0.557404 loss_full=0.234124 loss_base=0.323280 ms_per_step=3300.08
task=ci2v epoch=120 step=12660 loss=0.654210 loss_full=0.297463 loss_base=0.356747 ms_per_step=3475.89
task=ci2v epoch=120 step=12680 loss=0.678420 loss_full=0.313333 loss_base=0.365087 ms_per_step=3750.86
task=ci2v epoch=120 step=12700 loss=0.603149 loss_full=0.266070 loss_base=0.337079 ms_per_step=3710.19
[profile] window=359.11s updates=100 data_wait=2.1ms h2d=1.3ms stage1=246.8ms dit=641.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=120 elapsed=377.70s step=12705
[epoch-start] epoch=121 step=12705 steps_per_epoch=105
task=ci2v epoch=121 step=12720 loss=0.556751 loss_full=0.235696 loss_base=0.321055 ms_per_step=3776.57
task=ci2v epoch=121 step=12740 loss=0.644811 loss_full=0.288741 loss_base=0.356070 ms_per_step=3748.00
task=ci2v epoch=121 step=12760 loss=0.580537 loss_full=0.247732 loss_base=0.332805 ms_per_step=3709.05
task=ci2v epoch=121 step=12780 loss=0.709079 loss_full=0.315501 loss_base=0.393578 ms_per_step=3785.10
task=ci2v epoch=121 step=12800 loss=0.574360 loss_full=0.252725 loss_base=0.321635 ms_per_step=3762.30
[profile] window=375.88s updates=100 data_wait=1.9ms h2d=1.7ms stage1=282.4ms dit=647.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=121 elapsed=393.94s step=12810
[epoch-start] epoch=122 step=12810 steps_per_epoch=105
task=ci2v epoch=122 step=12820 loss=0.512914 loss_full=0.216268 loss_base=0.296647 ms_per_step=3720.28
task=ci2v epoch=122 step=12840 loss=0.558896 loss_full=0.241100 loss_base=0.317795 ms_per_step=3764.44
task=ci2v epoch=122 step=12860 loss=0.567366 loss_full=0.239912 loss_base=0.327454 ms_per_step=3319.76
task=ci2v epoch=122 step=12880 loss=0.588462 loss_full=0.265418 loss_base=0.323044 ms_per_step=3314.23
task=ci2v epoch=122 step=12900 loss=0.684576 loss_full=0.299021 loss_base=0.385555 ms_per_step=3309.14
[profile] window=348.83s updates=100 data_wait=1.7ms h2d=1.1ms stage1=224.5ms dit=638.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=122 elapsed=367.50s step=12915
[epoch-start] epoch=123 step=12915 steps_per_epoch=105
task=ci2v epoch=123 step=12920 loss=0.553834 loss_full=0.239136 loss_base=0.314698 ms_per_step=3735.41
task=ci2v epoch=123 step=12940 loss=0.645224 loss_full=0.289172 loss_base=0.356052 ms_per_step=3385.34
task=ci2v epoch=123 step=12960 loss=0.640624 loss_full=0.291631 loss_base=0.348993 ms_per_step=3492.12
task=ci2v epoch=123 step=12980 loss=0.572734 loss_full=0.252737 loss_base=0.319997 ms_per_step=3318.21
task=ci2v epoch=123 step=13000 loss=0.567327 loss_full=0.241909 loss_base=0.325418 ms_per_step=3454.32
[profile] window=347.97s updates=100 data_wait=2.1ms h2d=1.1ms stage1=221.2ms dit=639.3ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=13000 elapsed=2.51s
[checkpoint] step=13000 elapsed=26.01s
task=ci2v epoch=123 step=13020 loss=0.615457 loss_full=0.267614 loss_base=0.347843 ms_per_step=3381.01
[epoch-end] epoch=123 elapsed=389.61s step=13020
[epoch-start] epoch=124 step=13020 steps_per_epoch=105
task=ci2v epoch=124 step=13040 loss=0.564615 loss_full=0.253187 loss_base=0.311428 ms_per_step=3698.46
task=ci2v epoch=124 step=13060 loss=0.578145 loss_full=0.242896 loss_base=0.335249 ms_per_step=3702.92
task=ci2v epoch=124 step=13080 loss=0.553641 loss_full=0.240333 loss_base=0.313307 ms_per_step=3760.50
task=ci2v epoch=124 step=13100 loss=0.652623 loss_full=0.293925 loss_base=0.358697 ms_per_step=3493.31
[profile] window=360.98s updates=100 data_wait=2.0ms h2d=1.4ms stage1=250.9ms dit=641.9ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=124 step=13120 loss=0.614833 loss_full=0.268668 loss_base=0.346165 ms_per_step=3314.65
[epoch-end] epoch=124 elapsed=376.17s step=13125
[epoch-start] epoch=125 step=13125 steps_per_epoch=105
task=ci2v epoch=125 step=13140 loss=0.572576 loss_full=0.246193 loss_base=0.326383 ms_per_step=3655.44
task=ci2v epoch=125 step=13160 loss=0.608014 loss_full=0.265066 loss_base=0.342949 ms_per_step=3792.08
task=ci2v epoch=125 step=13180 loss=0.640082 loss_full=0.284985 loss_base=0.355098 ms_per_step=3598.20
task=ci2v epoch=125 step=13200 loss=0.569134 loss_full=0.249793 loss_base=0.319341 ms_per_step=3313.35
[profile] window=353.74s updates=100 data_wait=1.7ms h2d=1.2ms stage1=235.4ms dit=639.9ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=125 step=13220 loss=0.639452 loss_full=0.285341 loss_base=0.354111 ms_per_step=3354.97
[epoch-end] epoch=125 elapsed=372.84s step=13230
[epoch-start] epoch=126 step=13230 steps_per_epoch=105
task=ci2v epoch=126 step=13240 loss=0.586771 loss_full=0.265544 loss_base=0.321227 ms_per_step=3514.32
task=ci2v epoch=126 step=13260 loss=0.645774 loss_full=0.286958 loss_base=0.358816 ms_per_step=3762.79
task=ci2v epoch=126 step=13280 loss=0.496724 loss_full=0.201548 loss_base=0.295175 ms_per_step=3753.70
task=ci2v epoch=126 step=13300 loss=0.622711 loss_full=0.284669 loss_base=0.338042 ms_per_step=3453.93
[profile] window=357.05s updates=100 data_wait=1.6ms h2d=1.3ms stage1=242.5ms dit=641.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=126 step=13320 loss=0.668176 loss_full=0.312239 loss_base=0.355937 ms_per_step=3313.84
[epoch-end] epoch=126 elapsed=373.26s step=13335
[epoch-start] epoch=127 step=13335 steps_per_epoch=105
task=ci2v epoch=127 step=13340 loss=0.606420 loss_full=0.260500 loss_base=0.345920 ms_per_step=3567.14
task=ci2v epoch=127 step=13360 loss=0.656875 loss_full=0.301981 loss_base=0.354894 ms_per_step=3774.87
task=ci2v epoch=127 step=13380 loss=0.651129 loss_full=0.290644 loss_base=0.360485 ms_per_step=3459.43
task=ci2v epoch=127 step=13400 loss=0.595107 loss_full=0.265292 loss_base=0.329815 ms_per_step=3735.22
[profile] window=357.27s updates=100 data_wait=1.7ms h2d=1.3ms stage1=242.3ms dit=640.8ms update=30.5ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=127 step=13420 loss=0.568225 loss_full=0.248015 loss_base=0.320210 ms_per_step=3567.68
task=ci2v epoch=127 step=13440 loss=0.665194 loss_full=0.294580 loss_base=0.370614 ms_per_step=3300.14
[epoch-end] epoch=127 elapsed=377.00s step=13440
[epoch-start] epoch=128 step=13440 steps_per_epoch=105
task=ci2v epoch=128 step=13460 loss=0.567601 loss_full=0.245044 loss_base=0.322558 ms_per_step=3610.79
task=ci2v epoch=128 step=13480 loss=0.623995 loss_full=0.270989 loss_base=0.353006 ms_per_step=3563.00
task=ci2v epoch=128 step=13500 loss=0.624436 loss_full=0.279871 loss_base=0.344566 ms_per_step=3493.29
[profile] window=350.96s updates=100 data_wait=1.8ms h2d=1.2ms stage1=228.6ms dit=639.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=128 step=13520 loss=0.567500 loss_full=0.249172 loss_base=0.318328 ms_per_step=3634.22
task=ci2v epoch=128 step=13540 loss=0.491058 loss_full=0.209090 loss_base=0.281968 ms_per_step=3754.51
[epoch-end] epoch=128 elapsed=379.29s step=13545
[epoch-start] epoch=129 step=13545 steps_per_epoch=105
task=ci2v epoch=129 step=13560 loss=0.622149 loss_full=0.278126 loss_base=0.344023 ms_per_step=3717.58
task=ci2v epoch=129 step=13580 loss=0.613777 loss_full=0.276595 loss_base=0.337181 ms_per_step=3472.27
task=ci2v epoch=129 step=13600 loss=0.639867 loss_full=0.289088 loss_base=0.350779 ms_per_step=3768.47
[profile] window=367.20s updates=100 data_wait=1.9ms h2d=1.5ms stage1=264.2ms dit=644.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=129 step=13620 loss=0.590099 loss_full=0.256939 loss_base=0.333160 ms_per_step=3474.07
task=ci2v epoch=129 step=13640 loss=0.584126 loss_full=0.251267 loss_base=0.332859 ms_per_step=3377.18
[epoch-end] epoch=129 elapsed=374.54s step=13650
[epoch-start] epoch=130 step=13650 steps_per_epoch=105
task=ci2v epoch=130 step=13660 loss=0.594028 loss_full=0.252891 loss_base=0.341137 ms_per_step=3469.53
task=ci2v epoch=130 step=13680 loss=0.550333 loss_full=0.242212 loss_base=0.308121 ms_per_step=3318.03
task=ci2v epoch=130 step=13700 loss=0.598872 loss_full=0.261222 loss_base=0.337650 ms_per_step=3356.33
[profile] window=340.17s updates=100 data_wait=1.6ms h2d=1.0ms stage1=205.6ms dit=636.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=130 step=13720 loss=0.550771 loss_full=0.228793 loss_base=0.321978 ms_per_step=3559.18
task=ci2v epoch=130 step=13740 loss=0.531191 loss_full=0.223882 loss_base=0.307309 ms_per_step=3521.58
[epoch-end] epoch=130 elapsed=358.65s step=13755
[epoch-start] epoch=131 step=13755 steps_per_epoch=105
task=ci2v epoch=131 step=13760 loss=0.568449 loss_full=0.239169 loss_base=0.329280 ms_per_step=3344.80
task=ci2v epoch=131 step=13780 loss=0.639061 loss_full=0.295658 loss_base=0.343403 ms_per_step=3481.05
task=ci2v epoch=131 step=13800 loss=0.565938 loss_full=0.238640 loss_base=0.327298 ms_per_step=3493.66
[profile] window=348.27s updates=100 data_wait=1.6ms h2d=1.1ms stage1=222.9ms dit=639.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=131 step=13820 loss=0.554424 loss_full=0.246019 loss_base=0.308405 ms_per_step=3721.09
task=ci2v epoch=131 step=13840 loss=0.528032 loss_full=0.228172 loss_base=0.299860 ms_per_step=3448.61
task=ci2v epoch=131 step=13860 loss=0.518396 loss_full=0.218883 loss_base=0.299513 ms_per_step=3376.31
[epoch-end] epoch=131 elapsed=368.21s step=13860
[epoch-start] epoch=132 step=13860 steps_per_epoch=105
task=ci2v epoch=132 step=13880 loss=0.628305 loss_full=0.278670 loss_base=0.349635 ms_per_step=3358.84
task=ci2v epoch=132 step=13900 loss=0.618068 loss_full=0.272365 loss_base=0.345703 ms_per_step=3423.08
[profile] window=346.83s updates=100 data_wait=1.7ms h2d=1.1ms stage1=219.5ms dit=638.6ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=132 step=13920 loss=0.633428 loss_full=0.280200 loss_base=0.353229 ms_per_step=3768.76
task=ci2v epoch=132 step=13940 loss=0.653744 loss_full=0.294290 loss_base=0.359454 ms_per_step=3634.01
task=ci2v epoch=132 step=13960 loss=0.658596 loss_full=0.300830 loss_base=0.357766 ms_per_step=3349.23
[epoch-end] epoch=132 elapsed=367.40s step=13965
[epoch-start] epoch=133 step=13965 steps_per_epoch=105
task=ci2v epoch=133 step=13980 loss=0.538194 loss_full=0.230114 loss_base=0.308080 ms_per_step=3340.36
task=ci2v epoch=133 step=14000 loss=0.550161 loss_full=0.243480 loss_base=0.306681 ms_per_step=3717.37
[profile] window=356.46s updates=100 data_wait=1.8ms h2d=1.3ms stage1=241.1ms dit=641.0ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=14000 elapsed=2.90s
[checkpoint] step=14000 elapsed=29.10s
task=ci2v epoch=133 step=14020 loss=0.567029 loss_full=0.244679 loss_base=0.322350 ms_per_step=3718.29
task=ci2v epoch=133 step=14040 loss=0.611758 loss_full=0.272718 loss_base=0.339041 ms_per_step=3572.01
task=ci2v epoch=133 step=14060 loss=0.560919 loss_full=0.239946 loss_base=0.320973 ms_per_step=3680.12
[epoch-end] epoch=133 elapsed=411.08s step=14070
[epoch-start] epoch=134 step=14070 steps_per_epoch=105
task=ci2v epoch=134 step=14080 loss=0.625988 loss_full=0.279365 loss_base=0.346624 ms_per_step=3662.41
task=ci2v epoch=134 step=14100 loss=0.623289 loss_full=0.276340 loss_base=0.346949 ms_per_step=3789.70
[profile] window=368.72s updates=100 data_wait=2.1ms h2d=1.5ms stage1=266.7ms dit=645.3ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=134 step=14120 loss=0.666333 loss_full=0.301192 loss_base=0.365141 ms_per_step=3560.91
task=ci2v epoch=134 step=14140 loss=0.564894 loss_full=0.258423 loss_base=0.306471 ms_per_step=3581.91
task=ci2v epoch=134 step=14160 loss=0.616183 loss_full=0.267356 loss_base=0.348827 ms_per_step=3550.03
[epoch-end] epoch=134 elapsed=378.74s step=14175
[epoch-start] epoch=135 step=14175 steps_per_epoch=105
task=ci2v epoch=135 step=14180 loss=0.641673 loss_full=0.282535 loss_base=0.359138 ms_per_step=3360.84
task=ci2v epoch=135 step=14200 loss=0.606450 loss_full=0.269790 loss_base=0.336660 ms_per_step=3439.16
[profile] window=350.12s updates=100 data_wait=1.7ms h2d=1.1ms stage1=226.3ms dit=640.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=135 step=14220 loss=0.561693 loss_full=0.240818 loss_base=0.320875 ms_per_step=3381.96
task=ci2v epoch=135 step=14240 loss=0.643501 loss_full=0.283467 loss_base=0.360035 ms_per_step=3767.51
task=ci2v epoch=135 step=14260 loss=0.531913 loss_full=0.232217 loss_base=0.299696 ms_per_step=3382.80
task=ci2v epoch=135 step=14280 loss=0.593029 loss_full=0.249227 loss_base=0.343801 ms_per_step=3731.98
[epoch-end] epoch=135 elapsed=371.84s step=14280
[epoch-start] epoch=136 step=14280 steps_per_epoch=105
task=ci2v epoch=136 step=14300 loss=0.568316 loss_full=0.254066 loss_base=0.314250 ms_per_step=3846.59
[profile] window=362.48s updates=100 data_wait=1.9ms h2d=1.4ms stage1=253.0ms dit=643.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=136 step=14320 loss=0.579402 loss_full=0.248352 loss_base=0.331050 ms_per_step=3756.36
task=ci2v epoch=136 step=14340 loss=0.598370 loss_full=0.262674 loss_base=0.335696 ms_per_step=3774.95
task=ci2v epoch=136 step=14360 loss=0.514665 loss_full=0.217246 loss_base=0.297419 ms_per_step=3774.33
task=ci2v epoch=136 step=14380 loss=0.655262 loss_full=0.296703 loss_base=0.358559 ms_per_step=3764.17
[epoch-end] epoch=136 elapsed=395.11s step=14385
[epoch-start] epoch=137 step=14385 steps_per_epoch=105
task=ci2v epoch=137 step=14400 loss=0.521498 loss_full=0.223462 loss_base=0.298036 ms_per_step=3346.12
[profile] window=368.58s updates=100 data_wait=1.8ms h2d=1.5ms stage1=266.7ms dit=645.3ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=137 step=14420 loss=0.572409 loss_full=0.245221 loss_base=0.327188 ms_per_step=3669.71
task=ci2v epoch=137 step=14440 loss=0.588413 loss_full=0.258040 loss_base=0.330373 ms_per_step=3762.37
task=ci2v epoch=137 step=14460 loss=0.608722 loss_full=0.262773 loss_base=0.345950 ms_per_step=3539.29
task=ci2v epoch=137 step=14480 loss=0.572829 loss_full=0.256542 loss_base=0.316287 ms_per_step=3492.48
[epoch-end] epoch=137 elapsed=373.39s step=14490
[epoch-start] epoch=138 step=14490 steps_per_epoch=105
task=ci2v epoch=138 step=14500 loss=0.600272 loss_full=0.265618 loss_base=0.334654 ms_per_step=3367.70
[profile] window=356.90s updates=100 data_wait=2.0ms h2d=1.3ms stage1=241.0ms dit=641.8ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=138 step=14520 loss=0.589239 loss_full=0.261480 loss_base=0.327759 ms_per_step=3424.61
task=ci2v epoch=138 step=14540 loss=0.564331 loss_full=0.242932 loss_base=0.321399 ms_per_step=3649.65
task=ci2v epoch=138 step=14560 loss=0.637915 loss_full=0.275292 loss_base=0.362623 ms_per_step=3629.30
task=ci2v epoch=138 step=14580 loss=0.495045 loss_full=0.208648 loss_base=0.286396 ms_per_step=3553.88
[epoch-end] epoch=138 elapsed=370.17s step=14595
[epoch-start] epoch=139 step=14595 steps_per_epoch=105
task=ci2v epoch=139 step=14600 loss=0.568076 loss_full=0.245426 loss_base=0.322649 ms_per_step=3394.75
[profile] window=353.31s updates=100 data_wait=1.7ms h2d=1.2ms stage1=234.6ms dit=639.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=139 step=14620 loss=0.655819 loss_full=0.291829 loss_base=0.363991 ms_per_step=3512.29
task=ci2v epoch=139 step=14640 loss=0.581718 loss_full=0.252710 loss_base=0.329008 ms_per_step=3620.06
task=ci2v epoch=139 step=14660 loss=0.648079 loss_full=0.287439 loss_base=0.360640 ms_per_step=3587.93
task=ci2v epoch=139 step=14680 loss=0.607041 loss_full=0.261077 loss_base=0.345965 ms_per_step=3789.95
task=ci2v epoch=139 step=14700 loss=0.511382 loss_full=0.210683 loss_base=0.300699 ms_per_step=3510.47
[profile] window=360.68s updates=100 data_wait=1.0ms h2d=1.4ms stage1=250.8ms dit=642.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=139 elapsed=378.28s step=14700
[epoch-start] epoch=140 step=14700 steps_per_epoch=105
task=ci2v epoch=140 step=14720 loss=0.619371 loss_full=0.282569 loss_base=0.336802 ms_per_step=3702.46
task=ci2v epoch=140 step=14740 loss=0.552907 loss_full=0.233209 loss_base=0.319697 ms_per_step=3762.95
task=ci2v epoch=140 step=14760 loss=0.575375 loss_full=0.249979 loss_base=0.325396 ms_per_step=3768.87
task=ci2v epoch=140 step=14780 loss=0.593761 loss_full=0.272086 loss_base=0.321676 ms_per_step=3747.71
task=ci2v epoch=140 step=14800 loss=0.581374 loss_full=0.246189 loss_base=0.335185 ms_per_step=3684.47
[profile] window=373.58s updates=100 data_wait=1.7ms h2d=1.6ms stage1=278.4ms dit=646.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=140 elapsed=390.02s step=14805
[epoch-start] epoch=141 step=14805 steps_per_epoch=105
task=ci2v epoch=141 step=14820 loss=0.609567 loss_full=0.275649 loss_base=0.333918 ms_per_step=3343.00
task=ci2v epoch=141 step=14840 loss=0.510827 loss_full=0.212434 loss_base=0.298393 ms_per_step=3546.65
task=ci2v epoch=141 step=14860 loss=0.548324 loss_full=0.229346 loss_base=0.318978 ms_per_step=3771.78
task=ci2v epoch=141 step=14880 loss=0.632115 loss_full=0.293896 loss_base=0.338220 ms_per_step=3772.57
task=ci2v epoch=141 step=14900 loss=0.563963 loss_full=0.251497 loss_base=0.312466 ms_per_step=3775.32
[profile] window=364.45s updates=100 data_wait=1.8ms h2d=1.4ms stage1=257.2ms dit=644.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=141 elapsed=385.26s step=14910
[epoch-start] epoch=142 step=14910 steps_per_epoch=105
task=ci2v epoch=142 step=14920 loss=0.615332 loss_full=0.271719 loss_base=0.343613 ms_per_step=3801.41
task=ci2v epoch=142 step=14940 loss=0.592054 loss_full=0.270775 loss_base=0.321279 ms_per_step=3764.03
task=ci2v epoch=142 step=14960 loss=0.523688 loss_full=0.215876 loss_base=0.307812 ms_per_step=3363.77
task=ci2v epoch=142 step=14980 loss=0.666118 loss_full=0.298431 loss_base=0.367687 ms_per_step=3334.55
task=ci2v epoch=142 step=15000 loss=0.520973 loss_full=0.224065 loss_base=0.296907 ms_per_step=3682.56
[profile] window=359.19s updates=100 data_wait=2.1ms h2d=1.3ms stage1=246.9ms dit=641.3ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=15000 elapsed=2.59s
[checkpoint] step=15000 elapsed=27.38s
[epoch-end] epoch=142 elapsed=404.43s step=15015
[epoch-start] epoch=143 step=15015 steps_per_epoch=105
task=ci2v epoch=143 step=15020 loss=0.533514 loss_full=0.221107 loss_base=0.312407 ms_per_step=3526.19
task=ci2v epoch=143 step=15040 loss=0.623082 loss_full=0.273576 loss_base=0.349505 ms_per_step=3415.69
task=ci2v epoch=143 step=15060 loss=0.605244 loss_full=0.266074 loss_base=0.339171 ms_per_step=3570.07
task=ci2v epoch=143 step=15080 loss=0.664997 loss_full=0.308731 loss_base=0.356266 ms_per_step=3631.44
task=ci2v epoch=143 step=15100 loss=0.635399 loss_full=0.288157 loss_base=0.347241 ms_per_step=3329.25
[profile] window=349.71s updates=100 data_wait=1.7ms h2d=1.1ms stage1=225.6ms dit=639.8ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=143 step=15120 loss=0.621620 loss_full=0.291561 loss_base=0.330060 ms_per_step=3582.72
[epoch-end] epoch=143 elapsed=369.46s step=15120
[epoch-start] epoch=144 step=15120 steps_per_epoch=105
task=ci2v epoch=144 step=15140 loss=0.457655 loss_full=0.185132 loss_base=0.272524 ms_per_step=3630.96
task=ci2v epoch=144 step=15160 loss=0.563435 loss_full=0.249511 loss_base=0.313924 ms_per_step=3760.90
task=ci2v epoch=144 step=15180 loss=0.649666 loss_full=0.276484 loss_base=0.373182 ms_per_step=3788.93
task=ci2v epoch=144 step=15200 loss=0.584662 loss_full=0.246123 loss_base=0.338538 ms_per_step=3781.92
[profile] window=371.17s updates=100 data_wait=1.9ms h2d=1.6ms stage1=272.5ms dit=645.4ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=144 step=15220 loss=0.612179 loss_full=0.269189 loss_base=0.342990 ms_per_step=3636.44
[epoch-end] epoch=144 elapsed=388.71s step=15225
[epoch-start] epoch=145 step=15225 steps_per_epoch=105
task=ci2v epoch=145 step=15240 loss=0.573466 loss_full=0.243658 loss_base=0.329808 ms_per_step=3358.72
task=ci2v epoch=145 step=15260 loss=0.600551 loss_full=0.258794 loss_base=0.341757 ms_per_step=3553.68
task=ci2v epoch=145 step=15280 loss=0.628124 loss_full=0.278443 loss_base=0.349681 ms_per_step=3750.40
task=ci2v epoch=145 step=15300 loss=0.623980 loss_full=0.291456 loss_base=0.332524 ms_per_step=3560.46
[profile] window=357.47s updates=100 data_wait=1.6ms h2d=1.3ms stage1=243.4ms dit=641.3ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=145 step=15320 loss=0.602332 loss_full=0.263359 loss_base=0.338972 ms_per_step=3723.79
[epoch-end] epoch=145 elapsed=376.17s step=15330
[epoch-start] epoch=146 step=15330 steps_per_epoch=105
task=ci2v epoch=146 step=15340 loss=0.531629 loss_full=0.225282 loss_base=0.306348 ms_per_step=3352.06
task=ci2v epoch=146 step=15360 loss=0.572711 loss_full=0.245496 loss_base=0.327215 ms_per_step=3454.58
task=ci2v epoch=146 step=15380 loss=0.625465 loss_full=0.273438 loss_base=0.352027 ms_per_step=3766.13
task=ci2v epoch=146 step=15400 loss=0.609450 loss_full=0.266162 loss_base=0.343288 ms_per_step=3776.44
[profile] window=361.72s updates=100 data_wait=1.7ms h2d=1.4ms stage1=252.7ms dit=642.5ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=146 step=15420 loss=0.589886 loss_full=0.262971 loss_base=0.326916 ms_per_step=3765.48
[epoch-end] epoch=146 elapsed=385.25s step=15435
[epoch-start] epoch=147 step=15435 steps_per_epoch=105
task=ci2v epoch=147 step=15440 loss=0.634768 loss_full=0.285465 loss_base=0.349303 ms_per_step=3790.16
task=ci2v epoch=147 step=15460 loss=0.540270 loss_full=0.226363 loss_base=0.313908 ms_per_step=3748.61
task=ci2v epoch=147 step=15480 loss=0.532752 loss_full=0.225908 loss_base=0.306844 ms_per_step=3766.70
task=ci2v epoch=147 step=15500 loss=0.615540 loss_full=0.273910 loss_base=0.341630 ms_per_step=3550.20
[profile] window=372.69s updates=100 data_wait=1.8ms h2d=1.6ms stage1=276.3ms dit=645.7ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=147 step=15520 loss=0.570480 loss_full=0.250198 loss_base=0.320283 ms_per_step=3672.92
task=ci2v epoch=147 step=15540 loss=0.559015 loss_full=0.236671 loss_base=0.322344 ms_per_step=3727.64
[epoch-end] epoch=147 elapsed=389.78s step=15540
[epoch-start] epoch=148 step=15540 steps_per_epoch=105
task=ci2v epoch=148 step=15560 loss=0.502622 loss_full=0.207851 loss_base=0.294770 ms_per_step=3834.47
task=ci2v epoch=148 step=15580 loss=0.570111 loss_full=0.259092 loss_base=0.311019 ms_per_step=3777.30
task=ci2v epoch=148 step=15600 loss=0.526189 loss_full=0.226714 loss_base=0.299475 ms_per_step=3778.15
[profile] window=376.05s updates=100 data_wait=1.9ms h2d=1.6ms stage1=282.7ms dit=647.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=148 step=15620 loss=0.521254 loss_full=0.223352 loss_base=0.297902 ms_per_step=3349.11
task=ci2v epoch=148 step=15640 loss=0.594011 loss_full=0.259496 loss_base=0.334515 ms_per_step=3506.75
[epoch-end] epoch=148 elapsed=383.33s step=15645
[epoch-start] epoch=149 step=15645 steps_per_epoch=105
task=ci2v epoch=149 step=15660 loss=0.513140 loss_full=0.219197 loss_base=0.293943 ms_per_step=3780.84
task=ci2v epoch=149 step=15680 loss=0.576967 loss_full=0.254541 loss_base=0.322426 ms_per_step=3705.13
task=ci2v epoch=149 step=15700 loss=0.581413 loss_full=0.248120 loss_base=0.333293 ms_per_step=3324.64
[profile] window=353.59s updates=100 data_wait=2.0ms h2d=1.2ms stage1=234.0ms dit=640.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=149 step=15720 loss=0.603769 loss_full=0.264387 loss_base=0.339382 ms_per_step=3723.89
task=ci2v epoch=149 step=15740 loss=0.573279 loss_full=0.253396 loss_base=0.319883 ms_per_step=3537.43
[epoch-end] epoch=149 elapsed=378.99s step=15750
[epoch-start] epoch=150 step=15750 steps_per_epoch=105
task=ci2v epoch=150 step=15760 loss=0.738233 loss_full=0.347455 loss_base=0.390777 ms_per_step=3687.36
task=ci2v epoch=150 step=15780 loss=0.573457 loss_full=0.246507 loss_base=0.326949 ms_per_step=3765.87
task=ci2v epoch=150 step=15800 loss=0.571936 loss_full=0.256735 loss_base=0.315201 ms_per_step=3759.52
[profile] window=369.74s updates=100 data_wait=1.7ms h2d=1.5ms stage1=269.2ms dit=645.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=150 step=15820 loss=0.718822 loss_full=0.317821 loss_base=0.401001 ms_per_step=3772.07
task=ci2v epoch=150 step=15840 loss=0.613661 loss_full=0.264864 loss_base=0.348798 ms_per_step=3554.04
[epoch-end] epoch=150 elapsed=385.55s step=15855
[epoch-start] epoch=151 step=15855 steps_per_epoch=105
task=ci2v epoch=151 step=15860 loss=0.627687 loss_full=0.284771 loss_base=0.342916 ms_per_step=3451.25
task=ci2v epoch=151 step=15880 loss=0.602290 loss_full=0.257450 loss_base=0.344840 ms_per_step=3508.99
task=ci2v epoch=151 step=15900 loss=0.511566 loss_full=0.212730 loss_base=0.298835 ms_per_step=3533.66
[profile] window=356.66s updates=100 data_wait=1.6ms h2d=1.3ms stage1=241.8ms dit=640.8ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=151 step=15920 loss=0.605133 loss_full=0.270545 loss_base=0.334589 ms_per_step=3554.03
task=ci2v epoch=151 step=15940 loss=0.582505 loss_full=0.250619 loss_base=0.331885 ms_per_step=3481.00
task=ci2v epoch=151 step=15960 loss=0.551617 loss_full=0.233347 loss_base=0.318270 ms_per_step=3313.17
[epoch-end] epoch=151 elapsed=367.59s step=15960
[epoch-start] epoch=152 step=15960 steps_per_epoch=105
task=ci2v epoch=152 step=15980 loss=0.501605 loss_full=0.210604 loss_base=0.291001 ms_per_step=3386.72
task=ci2v epoch=152 step=16000 loss=0.601602 loss_full=0.262112 loss_base=0.339491 ms_per_step=3399.22
[profile] window=342.95s updates=100 data_wait=1.6ms h2d=1.0ms stage1=211.5ms dit=637.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=16000 elapsed=2.56s
[checkpoint] step=16000 elapsed=26.29s
task=ci2v epoch=152 step=16020 loss=0.603561 loss_full=0.267449 loss_base=0.336113 ms_per_step=3728.79
task=ci2v epoch=152 step=16040 loss=0.491114 loss_full=0.207903 loss_base=0.283210 ms_per_step=3349.07
task=ci2v epoch=152 step=16060 loss=0.710905 loss_full=0.326278 loss_base=0.384628 ms_per_step=3568.55
[epoch-end] epoch=152 elapsed=395.11s step=16065
[epoch-start] epoch=153 step=16065 steps_per_epoch=105
task=ci2v epoch=153 step=16080 loss=0.603883 loss_full=0.269835 loss_base=0.334048 ms_per_step=3740.14
task=ci2v epoch=153 step=16100 loss=0.618128 loss_full=0.269072 loss_base=0.349056 ms_per_step=3488.61
[profile] window=357.77s updates=100 data_wait=1.9ms h2d=1.3ms stage1=242.8ms dit=642.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=153 step=16120 loss=0.583691 loss_full=0.253267 loss_base=0.330424 ms_per_step=3518.36
task=ci2v epoch=153 step=16140 loss=0.491603 loss_full=0.203089 loss_base=0.288514 ms_per_step=3788.80
task=ci2v epoch=153 step=16160 loss=0.584713 loss_full=0.264093 loss_base=0.320619 ms_per_step=3751.35
[epoch-end] epoch=153 elapsed=383.31s step=16170
[epoch-start] epoch=154 step=16170 steps_per_epoch=105
task=ci2v epoch=154 step=16180 loss=0.579992 loss_full=0.246902 loss_base=0.333091 ms_per_step=3409.88
task=ci2v epoch=154 step=16200 loss=0.562758 loss_full=0.256038 loss_base=0.306720 ms_per_step=3307.64
[profile] window=355.78s updates=100 data_wait=1.6ms h2d=1.3ms stage1=239.6ms dit=640.8ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=154 step=16220 loss=0.742286 loss_full=0.349829 loss_base=0.392457 ms_per_step=3586.21
task=ci2v epoch=154 step=16240 loss=0.673502 loss_full=0.303920 loss_base=0.369582 ms_per_step=3789.13
task=ci2v epoch=154 step=16260 loss=0.562885 loss_full=0.241497 loss_base=0.321388 ms_per_step=3767.50
[epoch-end] epoch=154 elapsed=378.50s step=16275
[epoch-start] epoch=155 step=16275 steps_per_epoch=105
task=ci2v epoch=155 step=16280 loss=0.589077 loss_full=0.264404 loss_base=0.324673 ms_per_step=3650.64
task=ci2v epoch=155 step=16300 loss=0.581807 loss_full=0.261016 loss_base=0.320791 ms_per_step=3777.92
[profile] window=371.70s updates=100 data_wait=1.9ms h2d=1.6ms stage1=274.4ms dit=645.2ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=155 step=16320 loss=0.534069 loss_full=0.218967 loss_base=0.315103 ms_per_step=3783.00
task=ci2v epoch=155 step=16340 loss=0.606972 loss_full=0.272009 loss_base=0.334963 ms_per_step=3585.48
task=ci2v epoch=155 step=16360 loss=0.657462 loss_full=0.292034 loss_base=0.365428 ms_per_step=3663.75
task=ci2v epoch=155 step=16380 loss=0.483228 loss_full=0.205566 loss_base=0.277663 ms_per_step=3713.56
[epoch-end] epoch=155 elapsed=388.65s step=16380
[epoch-start] epoch=156 step=16380 steps_per_epoch=105
task=ci2v epoch=156 step=16400 loss=0.557747 loss_full=0.239386 loss_base=0.318362 ms_per_step=3819.66
[profile] window=371.58s updates=100 data_wait=2.0ms h2d=1.6ms stage1=273.9ms dit=645.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=156 step=16420 loss=0.574623 loss_full=0.245291 loss_base=0.329332 ms_per_step=3737.63
task=ci2v epoch=156 step=16440 loss=0.500119 loss_full=0.201477 loss_base=0.298642 ms_per_step=3681.74
task=ci2v epoch=156 step=16460 loss=0.571406 loss_full=0.246815 loss_base=0.324590 ms_per_step=3661.19
task=ci2v epoch=156 step=16480 loss=0.591636 loss_full=0.255800 loss_base=0.335836 ms_per_step=3357.34
[epoch-end] epoch=156 elapsed=382.04s step=16485
[epoch-start] epoch=157 step=16485 steps_per_epoch=105
task=ci2v epoch=157 step=16500 loss=0.501895 loss_full=0.208481 loss_base=0.293414 ms_per_step=3498.47
[profile] window=358.99s updates=100 data_wait=1.7ms h2d=1.3ms stage1=246.7ms dit=641.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=157 step=16520 loss=0.576584 loss_full=0.250899 loss_base=0.325684 ms_per_step=3480.03
task=ci2v epoch=157 step=16540 loss=0.607938 loss_full=0.264225 loss_base=0.343713 ms_per_step=3590.56
task=ci2v epoch=157 step=16560 loss=0.647259 loss_full=0.294444 loss_base=0.352815 ms_per_step=3784.16
task=ci2v epoch=157 step=16580 loss=0.604839 loss_full=0.266326 loss_base=0.338514 ms_per_step=3580.26
[epoch-end] epoch=157 elapsed=375.42s step=16590
[epoch-start] epoch=158 step=16590 steps_per_epoch=105
task=ci2v epoch=158 step=16600 loss=0.627763 loss_full=0.277870 loss_base=0.349892 ms_per_step=3381.57
[profile] window=356.60s updates=100 data_wait=1.8ms h2d=1.2ms stage1=241.8ms dit=640.6ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=158 step=16620 loss=0.663624 loss_full=0.306788 loss_base=0.356836 ms_per_step=3398.48
task=ci2v epoch=158 step=16640 loss=0.544368 loss_full=0.229022 loss_base=0.315346 ms_per_step=3508.89
task=ci2v epoch=158 step=16660 loss=0.515078 loss_full=0.211131 loss_base=0.303947 ms_per_step=3542.81
task=ci2v epoch=158 step=16680 loss=0.571020 loss_full=0.256390 loss_base=0.314630 ms_per_step=3406.27
[epoch-end] epoch=158 elapsed=368.33s step=16695
[epoch-start] epoch=159 step=16695 steps_per_epoch=105
task=ci2v epoch=159 step=16700 loss=0.549972 loss_full=0.240244 loss_base=0.309728 ms_per_step=3790.20
[profile] window=353.20s updates=100 data_wait=1.8ms h2d=1.2ms stage1=234.5ms dit=639.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=159 step=16720 loss=0.689813 loss_full=0.310053 loss_base=0.379760 ms_per_step=3683.13
task=ci2v epoch=159 step=16740 loss=0.652591 loss_full=0.292385 loss_base=0.360206 ms_per_step=3453.85
task=ci2v epoch=159 step=16760 loss=0.527272 loss_full=0.227574 loss_base=0.299698 ms_per_step=3410.78
task=ci2v epoch=159 step=16780 loss=0.554016 loss_full=0.242890 loss_base=0.311126 ms_per_step=3337.34
task=ci2v epoch=159 step=16800 loss=0.602186 loss_full=0.270317 loss_base=0.331869 ms_per_step=3741.68
[profile] window=352.80s updates=100 data_wait=1.1ms h2d=1.1ms stage1=228.2ms dit=645.6ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=159 elapsed=372.75s step=16800
[epoch-start] epoch=160 step=16800 steps_per_epoch=105
task=ci2v epoch=160 step=16820 loss=0.602102 loss_full=0.273755 loss_base=0.328347 ms_per_step=3863.27
task=ci2v epoch=160 step=16840 loss=0.557810 loss_full=0.244944 loss_base=0.312866 ms_per_step=3595.63
task=ci2v epoch=160 step=16860 loss=0.724172 loss_full=0.334073 loss_base=0.390099 ms_per_step=3488.47
task=ci2v epoch=160 step=16880 loss=0.587964 loss_full=0.265608 loss_base=0.322356 ms_per_step=3660.05
task=ci2v epoch=160 step=16900 loss=0.673215 loss_full=0.320599 loss_base=0.352617 ms_per_step=3627.15
[profile] window=364.96s updates=100 data_wait=1.9ms h2d=1.4ms stage1=257.4ms dit=645.2ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=160 elapsed=381.46s step=16905
[epoch-start] epoch=161 step=16905 steps_per_epoch=105
task=ci2v epoch=161 step=16920 loss=0.592657 loss_full=0.262395 loss_base=0.330262 ms_per_step=3698.39
task=ci2v epoch=161 step=16940 loss=0.603385 loss_full=0.258076 loss_base=0.345310 ms_per_step=3850.03
task=ci2v epoch=161 step=16960 loss=0.623975 loss_full=0.282150 loss_base=0.341825 ms_per_step=3809.31
task=ci2v epoch=161 step=16980 loss=0.503073 loss_full=0.213096 loss_base=0.289977 ms_per_step=3830.71
task=ci2v epoch=161 step=17000 loss=0.563036 loss_full=0.251181 loss_base=0.311855 ms_per_step=3564.61
[profile] window=375.31s updates=100 data_wait=1.8ms h2d=1.5ms stage1=277.9ms dit=650.9ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=17000 elapsed=2.77s
[checkpoint] step=17000 elapsed=36.38s
[epoch-end] epoch=161 elapsed=433.10s step=17010
[epoch-start] epoch=162 step=17010 steps_per_epoch=105
task=ci2v epoch=162 step=17020 loss=0.543596 loss_full=0.230827 loss_base=0.312770 ms_per_step=3753.75
task=ci2v epoch=162 step=17040 loss=0.656187 loss_full=0.291773 loss_base=0.364414 ms_per_step=3404.70
task=ci2v epoch=162 step=17060 loss=0.643786 loss_full=0.286511 loss_base=0.357275 ms_per_step=3803.07
task=ci2v epoch=162 step=17080 loss=0.607620 loss_full=0.267687 loss_base=0.339933 ms_per_step=3635.72
task=ci2v epoch=162 step=17100 loss=0.534506 loss_full=0.233672 loss_base=0.300834 ms_per_step=3339.73
[profile] window=359.01s updates=100 data_wait=2.1ms h2d=1.2ms stage1=242.9ms dit=644.9ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=162 elapsed=376.54s step=17115
[epoch-start] epoch=163 step=17115 steps_per_epoch=105
task=ci2v epoch=163 step=17120 loss=0.651231 loss_full=0.287385 loss_base=0.363846 ms_per_step=3630.93
task=ci2v epoch=163 step=17140 loss=0.714588 loss_full=0.323305 loss_base=0.391283 ms_per_step=3728.97
task=ci2v epoch=163 step=17160 loss=0.608265 loss_full=0.277724 loss_base=0.330542 ms_per_step=3479.97
task=ci2v epoch=163 step=17180 loss=0.543372 loss_full=0.230467 loss_base=0.312905 ms_per_step=3612.12
task=ci2v epoch=163 step=17200 loss=0.478533 loss_full=0.201972 loss_base=0.276561 ms_per_step=3441.17
[profile] window=358.12s updates=100 data_wait=2.0ms h2d=1.2ms stage1=239.1ms dit=646.8ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=163 step=17220 loss=0.552932 loss_full=0.242864 loss_base=0.310068 ms_per_step=3444.95
[epoch-end] epoch=163 elapsed=374.82s step=17220
[epoch-start] epoch=164 step=17220 steps_per_epoch=105
task=ci2v epoch=164 step=17240 loss=0.575274 loss_full=0.248911 loss_base=0.326363 ms_per_step=3897.05
task=ci2v epoch=164 step=17260 loss=0.619613 loss_full=0.278332 loss_base=0.341280 ms_per_step=3386.18
task=ci2v epoch=164 step=17280 loss=0.551119 loss_full=0.246736 loss_base=0.304383 ms_per_step=3574.76
task=ci2v epoch=164 step=17300 loss=0.660219 loss_full=0.309966 loss_base=0.350253 ms_per_step=3431.19
[profile] window=354.94s updates=100 data_wait=1.9ms h2d=1.2ms stage1=233.4ms dit=644.3ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=164 step=17320 loss=0.553224 loss_full=0.233455 loss_base=0.319769 ms_per_step=3644.24
[epoch-end] epoch=164 elapsed=377.44s step=17325
[epoch-start] epoch=165 step=17325 steps_per_epoch=105
task=ci2v epoch=165 step=17340 loss=0.609136 loss_full=0.267956 loss_base=0.341180 ms_per_step=3834.88
task=ci2v epoch=165 step=17360 loss=0.575362 loss_full=0.248756 loss_base=0.326606 ms_per_step=3678.12
task=ci2v epoch=165 step=17380 loss=0.581393 loss_full=0.245281 loss_base=0.336112 ms_per_step=3485.98
task=ci2v epoch=165 step=17400 loss=0.621301 loss_full=0.265625 loss_base=0.355676 ms_per_step=3540.50
[profile] window=364.02s updates=100 data_wait=2.1ms h2d=1.3ms stage1=253.3ms dit=647.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=165 step=17420 loss=0.624876 loss_full=0.274266 loss_base=0.350610 ms_per_step=3552.04
[epoch-end] epoch=165 elapsed=377.00s step=17430
[epoch-start] epoch=166 step=17430 steps_per_epoch=105
task=ci2v epoch=166 step=17440 loss=0.601800 loss_full=0.265397 loss_base=0.336403 ms_per_step=3499.29
task=ci2v epoch=166 step=17460 loss=0.498681 loss_full=0.219601 loss_base=0.279080 ms_per_step=3339.15
task=ci2v epoch=166 step=17480 loss=0.612597 loss_full=0.271205 loss_base=0.341392 ms_per_step=3352.05
task=ci2v epoch=166 step=17500 loss=0.646817 loss_full=0.290869 loss_base=0.355948 ms_per_step=3348.36
[profile] window=342.09s updates=100 data_wait=1.7ms h2d=0.9ms stage1=206.9ms dit=639.6ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=166 step=17520 loss=0.521127 loss_full=0.216053 loss_base=0.305074 ms_per_step=3338.87
[epoch-end] epoch=166 elapsed=354.84s step=17535
[epoch-start] epoch=167 step=17535 steps_per_epoch=105
task=ci2v epoch=167 step=17540 loss=0.628723 loss_full=0.276791 loss_base=0.351932 ms_per_step=3411.13
task=ci2v epoch=167 step=17560 loss=0.645976 loss_full=0.294530 loss_base=0.351446 ms_per_step=3377.58
task=ci2v epoch=167 step=17580 loss=0.623735 loss_full=0.272146 loss_base=0.351589 ms_per_step=3846.05
task=ci2v epoch=167 step=17600 loss=0.585244 loss_full=0.258935 loss_base=0.326308 ms_per_step=3723.62
[profile] window=354.19s updates=100 data_wait=1.7ms h2d=1.1ms stage1=231.5ms dit=645.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=167 step=17620 loss=0.493110 loss_full=0.202119 loss_base=0.290991 ms_per_step=3483.04
task=ci2v epoch=167 step=17640 loss=0.612809 loss_full=0.269827 loss_base=0.342982 ms_per_step=3423.53
[epoch-end] epoch=167 elapsed=375.67s step=17640
[epoch-start] epoch=168 step=17640 steps_per_epoch=105
task=ci2v epoch=168 step=17660 loss=0.683876 loss_full=0.309086 loss_base=0.374790 ms_per_step=3494.09
task=ci2v epoch=168 step=17680 loss=0.518357 loss_full=0.209824 loss_base=0.308534 ms_per_step=3621.12
task=ci2v epoch=168 step=17700 loss=0.575390 loss_full=0.248519 loss_base=0.326871 ms_per_step=3339.81
[profile] window=347.49s updates=100 data_wait=1.7ms h2d=1.1ms stage1=218.3ms dit=641.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=168 step=17720 loss=0.543842 loss_full=0.228782 loss_base=0.315060 ms_per_step=3347.14
task=ci2v epoch=168 step=17740 loss=0.590521 loss_full=0.258178 loss_base=0.332343 ms_per_step=3632.95
[epoch-end] epoch=168 elapsed=365.35s step=17745
[epoch-start] epoch=169 step=17745 steps_per_epoch=105
task=ci2v epoch=169 step=17760 loss=0.598537 loss_full=0.267482 loss_base=0.331055 ms_per_step=3438.16
task=ci2v epoch=169 step=17780 loss=0.561710 loss_full=0.239760 loss_base=0.321950 ms_per_step=3334.79
task=ci2v epoch=169 step=17800 loss=0.496634 loss_full=0.207277 loss_base=0.289357 ms_per_step=3372.24
[profile] window=342.77s updates=100 data_wait=1.7ms h2d=0.9ms stage1=209.0ms dit=639.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=169 step=17820 loss=0.584214 loss_full=0.262894 loss_base=0.321320 ms_per_step=3366.59
task=ci2v epoch=169 step=17840 loss=0.486403 loss_full=0.198164 loss_base=0.288238 ms_per_step=3457.38
[epoch-end] epoch=169 elapsed=356.75s step=17850
[epoch-start] epoch=170 step=17850 steps_per_epoch=105
task=ci2v epoch=170 step=17860 loss=0.582842 loss_full=0.262004 loss_base=0.320838 ms_per_step=3609.93
task=ci2v epoch=170 step=17880 loss=0.530327 loss_full=0.229518 loss_base=0.300809 ms_per_step=3807.74
task=ci2v epoch=170 step=17900 loss=0.542474 loss_full=0.236436 loss_base=0.306038 ms_per_step=3825.93
[profile] window=361.60s updates=100 data_wait=1.7ms h2d=1.3ms stage1=248.6ms dit=646.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=170 step=17920 loss=0.563719 loss_full=0.247412 loss_base=0.316306 ms_per_step=3583.10
task=ci2v epoch=170 step=17940 loss=0.632994 loss_full=0.289989 loss_base=0.343005 ms_per_step=3592.63
[epoch-end] epoch=170 elapsed=389.30s step=17955
[epoch-start] epoch=171 step=17955 steps_per_epoch=105
task=ci2v epoch=171 step=17960 loss=0.533838 loss_full=0.229572 loss_base=0.304266 ms_per_step=3568.95
task=ci2v epoch=171 step=17980 loss=0.601367 loss_full=0.265810 loss_base=0.335557 ms_per_step=3478.06
task=ci2v epoch=171 step=18000 loss=0.506974 loss_full=0.209529 loss_base=0.297446 ms_per_step=3737.94
[profile] window=359.47s updates=100 data_wait=1.9ms h2d=1.2ms stage1=245.1ms dit=644.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=18000 elapsed=2.65s
[checkpoint] step=18000 elapsed=34.81s
task=ci2v epoch=171 step=18020 loss=0.608603 loss_full=0.278882 loss_base=0.329720 ms_per_step=3346.73
task=ci2v epoch=171 step=18040 loss=0.514733 loss_full=0.212421 loss_base=0.302312 ms_per_step=3370.25
task=ci2v epoch=171 step=18060 loss=0.577926 loss_full=0.243871 loss_base=0.334055 ms_per_step=3555.12
[epoch-end] epoch=171 elapsed=405.23s step=18060
[epoch-start] epoch=172 step=18060 steps_per_epoch=105
task=ci2v epoch=172 step=18080 loss=0.567710 loss_full=0.249476 loss_base=0.318234 ms_per_step=3673.34
task=ci2v epoch=172 step=18100 loss=0.608134 loss_full=0.272136 loss_base=0.335998 ms_per_step=3741.29
[profile] window=353.99s updates=100 data_wait=2.0ms h2d=1.2ms stage1=233.2ms dit=642.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=172 step=18120 loss=0.544335 loss_full=0.229882 loss_base=0.314452 ms_per_step=3460.18
task=ci2v epoch=172 step=18140 loss=0.525046 loss_full=0.215839 loss_base=0.309206 ms_per_step=3391.45
task=ci2v epoch=172 step=18160 loss=0.645014 loss_full=0.289607 loss_base=0.355407 ms_per_step=3470.24
[epoch-end] epoch=172 elapsed=371.27s step=18165
[epoch-start] epoch=173 step=18165 steps_per_epoch=105
task=ci2v epoch=173 step=18180 loss=0.596327 loss_full=0.259877 loss_base=0.336450 ms_per_step=3640.46
task=ci2v epoch=173 step=18200 loss=0.517742 loss_full=0.225870 loss_base=0.291872 ms_per_step=3555.66
[profile] window=350.62s updates=100 data_wait=1.8ms h2d=1.1ms stage1=223.3ms dit=644.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=173 step=18220 loss=0.652526 loss_full=0.296147 loss_base=0.356379 ms_per_step=3494.62
task=ci2v epoch=173 step=18240 loss=0.545297 loss_full=0.230870 loss_base=0.314427 ms_per_step=3444.17
task=ci2v epoch=173 step=18260 loss=0.556074 loss_full=0.238852 loss_base=0.317222 ms_per_step=3611.14
[epoch-end] epoch=173 elapsed=372.20s step=18270
[epoch-start] epoch=174 step=18270 steps_per_epoch=105
task=ci2v epoch=174 step=18280 loss=0.512395 loss_full=0.217080 loss_base=0.295315 ms_per_step=3371.24
task=ci2v epoch=174 step=18300 loss=0.572038 loss_full=0.254481 loss_base=0.317557 ms_per_step=3357.22
[profile] window=345.84s updates=100 data_wait=1.9ms h2d=1.0ms stage1=215.5ms dit=640.3ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=174 step=18320 loss=0.648523 loss_full=0.291419 loss_base=0.357104 ms_per_step=3482.55
task=ci2v epoch=174 step=18340 loss=0.540882 loss_full=0.224754 loss_base=0.316128 ms_per_step=3599.31
task=ci2v epoch=174 step=18360 loss=0.532313 loss_full=0.224022 loss_base=0.308291 ms_per_step=3357.62
[epoch-end] epoch=174 elapsed=360.74s step=18375
[epoch-start] epoch=175 step=18375 steps_per_epoch=105
task=ci2v epoch=175 step=18380 loss=0.569650 loss_full=0.235591 loss_base=0.334059 ms_per_step=3386.46
task=ci2v epoch=175 step=18400 loss=0.498993 loss_full=0.214607 loss_base=0.284385 ms_per_step=3418.74
[profile] window=345.15s updates=100 data_wait=1.7ms h2d=1.0ms stage1=213.5ms dit=640.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=175 step=18420 loss=0.614947 loss_full=0.275481 loss_base=0.339466 ms_per_step=3383.58
task=ci2v epoch=175 step=18440 loss=0.576076 loss_full=0.254334 loss_base=0.321742 ms_per_step=3765.75
task=ci2v epoch=175 step=18460 loss=0.587946 loss_full=0.259747 loss_base=0.328199 ms_per_step=3360.97
task=ci2v epoch=175 step=18480 loss=0.558494 loss_full=0.236307 loss_base=0.322187 ms_per_step=3597.46
[epoch-end] epoch=175 elapsed=368.53s step=18480
[epoch-start] epoch=176 step=18480 steps_per_epoch=105
task=ci2v epoch=176 step=18500 loss=0.519694 loss_full=0.217109 loss_base=0.302585 ms_per_step=3474.29
[profile] window=351.90s updates=100 data_wait=1.7ms h2d=1.1ms stage1=228.0ms dit=642.6ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=176 step=18520 loss=0.661483 loss_full=0.293911 loss_base=0.367572 ms_per_step=3765.62
task=ci2v epoch=176 step=18540 loss=0.617203 loss_full=0.276529 loss_base=0.340674 ms_per_step=3810.54
task=ci2v epoch=176 step=18560 loss=0.560601 loss_full=0.238170 loss_base=0.322431 ms_per_step=3486.01
task=ci2v epoch=176 step=18580 loss=0.590174 loss_full=0.259069 loss_base=0.331106 ms_per_step=3538.01
[epoch-end] epoch=176 elapsed=378.24s step=18585
[epoch-start] epoch=177 step=18585 steps_per_epoch=105
task=ci2v epoch=177 step=18600 loss=0.524865 loss_full=0.228744 loss_base=0.296120 ms_per_step=3365.63
[profile] window=359.57s updates=100 data_wait=1.7ms h2d=1.3ms stage1=244.0ms dit=645.9ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=177 step=18620 loss=0.473341 loss_full=0.190435 loss_base=0.282907 ms_per_step=3622.94
task=ci2v epoch=177 step=18640 loss=0.538585 loss_full=0.229486 loss_base=0.309099 ms_per_step=3312.38
task=ci2v epoch=177 step=18660 loss=0.546811 loss_full=0.232010 loss_base=0.314801 ms_per_step=3424.66
task=ci2v epoch=177 step=18680 loss=0.577234 loss_full=0.245808 loss_base=0.331426 ms_per_step=3506.78
[epoch-end] epoch=177 elapsed=361.85s step=18690
[epoch-start] epoch=178 step=18690 steps_per_epoch=105
task=ci2v epoch=178 step=18700 loss=0.496631 loss_full=0.203776 loss_base=0.292855 ms_per_step=3375.78
[profile] window=345.11s updates=100 data_wait=1.8ms h2d=1.0ms stage1=214.0ms dit=639.8ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=178 step=18720 loss=0.624973 loss_full=0.276282 loss_base=0.348691 ms_per_step=3455.13
task=ci2v epoch=178 step=18740 loss=0.648695 loss_full=0.291171 loss_base=0.357524 ms_per_step=3331.18
task=ci2v epoch=178 step=18760 loss=0.582004 loss_full=0.249708 loss_base=0.332296 ms_per_step=3660.10
task=ci2v epoch=178 step=18780 loss=0.575553 loss_full=0.244211 loss_base=0.331342 ms_per_step=3322.63
[epoch-end] epoch=178 elapsed=359.77s step=18795
[epoch-start] epoch=179 step=18795 steps_per_epoch=105
task=ci2v epoch=179 step=18800 loss=0.531671 loss_full=0.225376 loss_base=0.306295 ms_per_step=3361.30
[profile] window=342.87s updates=100 data_wait=1.6ms h2d=1.0ms stage1=209.9ms dit=638.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=179 step=18820 loss=0.626905 loss_full=0.285455 loss_base=0.341450 ms_per_step=3633.49
task=ci2v epoch=179 step=18840 loss=0.513475 loss_full=0.216811 loss_base=0.296664 ms_per_step=3542.53
task=ci2v epoch=179 step=18860 loss=0.639952 loss_full=0.285597 loss_base=0.354355 ms_per_step=3358.69
task=ci2v epoch=179 step=18880 loss=0.546005 loss_full=0.243077 loss_base=0.302928 ms_per_step=3435.85
task=ci2v epoch=179 step=18900 loss=0.516178 loss_full=0.218126 loss_base=0.298052 ms_per_step=3372.95
[profile] window=347.13s updates=100 data_wait=1.0ms h2d=1.0ms stage1=217.8ms dit=641.9ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=179 elapsed=364.83s step=18900
[epoch-start] epoch=180 step=18900 steps_per_epoch=105
task=ci2v epoch=180 step=18920 loss=0.534006 loss_full=0.230231 loss_base=0.303775 ms_per_step=3534.63
task=ci2v epoch=180 step=18940 loss=0.535264 loss_full=0.224075 loss_base=0.311189 ms_per_step=3562.49
task=ci2v epoch=180 step=18960 loss=0.632525 loss_full=0.269434 loss_base=0.363091 ms_per_step=3485.83
task=ci2v epoch=180 step=18980 loss=0.578445 loss_full=0.254377 loss_base=0.324069 ms_per_step=3498.34
task=ci2v epoch=180 step=19000 loss=0.572165 loss_full=0.248472 loss_base=0.323693 ms_per_step=3342.49
[profile] window=348.74s updates=100 data_wait=1.8ms h2d=1.1ms stage1=220.9ms dit=641.9ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=19000 elapsed=2.55s
[checkpoint] step=19000 elapsed=26.17s
[epoch-end] epoch=180 elapsed=393.99s step=19005
[epoch-start] epoch=181 step=19005 steps_per_epoch=105
task=ci2v epoch=181 step=19020 loss=0.570295 loss_full=0.255794 loss_base=0.314500 ms_per_step=3461.02
task=ci2v epoch=181 step=19040 loss=0.651211 loss_full=0.278899 loss_base=0.372312 ms_per_step=3426.68
task=ci2v epoch=181 step=19060 loss=0.577894 loss_full=0.256554 loss_base=0.321340 ms_per_step=3682.97
task=ci2v epoch=181 step=19080 loss=0.633630 loss_full=0.274080 loss_base=0.359549 ms_per_step=3392.88
task=ci2v epoch=181 step=19100 loss=0.563032 loss_full=0.246777 loss_base=0.316255 ms_per_step=3354.44
[profile] window=346.61s updates=100 data_wait=2.1ms h2d=1.0ms stage1=215.8ms dit=641.3ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=181 elapsed=363.28s step=19110
[epoch-start] epoch=182 step=19110 steps_per_epoch=105
task=ci2v epoch=182 step=19120 loss=0.562684 loss_full=0.243075 loss_base=0.319609 ms_per_step=3435.65
task=ci2v epoch=182 step=19140 loss=0.580306 loss_full=0.244363 loss_base=0.335943 ms_per_step=3572.34
task=ci2v epoch=182 step=19160 loss=0.637413 loss_full=0.275123 loss_base=0.362290 ms_per_step=3366.87
task=ci2v epoch=182 step=19180 loss=0.499078 loss_full=0.208314 loss_base=0.290764 ms_per_step=3388.32
task=ci2v epoch=182 step=19200 loss=0.548799 loss_full=0.238811 loss_base=0.309988 ms_per_step=3449.16
[profile] window=344.51s updates=100 data_wait=1.7ms h2d=1.0ms stage1=213.0ms dit=639.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=182 elapsed=368.34s step=19215
[epoch-start] epoch=183 step=19215 steps_per_epoch=105
task=ci2v epoch=183 step=19220 loss=0.544414 loss_full=0.230380 loss_base=0.314033 ms_per_step=3724.85
task=ci2v epoch=183 step=19240 loss=0.559072 loss_full=0.243877 loss_base=0.315194 ms_per_step=3428.59
task=ci2v epoch=183 step=19260 loss=0.552483 loss_full=0.235578 loss_base=0.316905 ms_per_step=3516.89
task=ci2v epoch=183 step=19280 loss=0.578803 loss_full=0.241452 loss_base=0.337351 ms_per_step=3767.60
task=ci2v epoch=183 step=19300 loss=0.464755 loss_full=0.190775 loss_base=0.273981 ms_per_step=3897.41
[profile] window=366.96s updates=100 data_wait=2.2ms h2d=1.3ms stage1=257.5ms dit=650.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=183 step=19320 loss=0.516107 loss_full=0.210223 loss_base=0.305884 ms_per_step=3839.24
[epoch-end] epoch=183 elapsed=387.28s step=19320
[epoch-start] epoch=184 step=19320 steps_per_epoch=105
task=ci2v epoch=184 step=19340 loss=0.590534 loss_full=0.249322 loss_base=0.341212 ms_per_step=3926.87
task=ci2v epoch=184 step=19360 loss=0.539007 loss_full=0.226369 loss_base=0.312638 ms_per_step=3477.12
task=ci2v epoch=184 step=19380 loss=0.541641 loss_full=0.221747 loss_base=0.319894 ms_per_step=3341.82
task=ci2v epoch=184 step=19400 loss=0.571439 loss_full=0.253335 loss_base=0.318104 ms_per_step=3344.09
[profile] window=358.84s updates=100 data_wait=2.1ms h2d=1.2ms stage1=244.1ms dit=643.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=184 step=19420 loss=0.605053 loss_full=0.258892 loss_base=0.346161 ms_per_step=3336.10
[epoch-end] epoch=184 elapsed=365.16s step=19425
[epoch-start] epoch=185 step=19425 steps_per_epoch=105
task=ci2v epoch=185 step=19440 loss=0.510628 loss_full=0.219624 loss_base=0.291004 ms_per_step=3353.95
task=ci2v epoch=185 step=19460 loss=0.624131 loss_full=0.284482 loss_base=0.339648 ms_per_step=3327.01
task=ci2v epoch=185 step=19480 loss=0.573390 loss_full=0.250417 loss_base=0.322973 ms_per_step=3340.43
task=ci2v epoch=185 step=19500 loss=0.562338 loss_full=0.237748 loss_base=0.324590 ms_per_step=3343.90
[profile] window=334.29s updates=100 data_wait=1.7ms h2d=0.8ms stage1=192.0ms dit=635.3ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=185 step=19520 loss=0.601371 loss_full=0.270895 loss_base=0.330476 ms_per_step=3357.07
[epoch-end] epoch=185 elapsed=351.70s step=19530
[epoch-start] epoch=186 step=19530 steps_per_epoch=105
task=ci2v epoch=186 step=19540 loss=0.652601 loss_full=0.290317 loss_base=0.362284 ms_per_step=3365.31
task=ci2v epoch=186 step=19560 loss=0.482035 loss_full=0.204247 loss_base=0.277788 ms_per_step=3408.39
task=ci2v epoch=186 step=19580 loss=0.480098 loss_full=0.196410 loss_base=0.283687 ms_per_step=3673.18
task=ci2v epoch=186 step=19600 loss=0.517353 loss_full=0.219023 loss_base=0.298330 ms_per_step=3413.31
[profile] window=344.61s updates=100 data_wait=1.7ms h2d=1.0ms stage1=213.3ms dit=639.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=186 step=19620 loss=0.533142 loss_full=0.236552 loss_base=0.296590 ms_per_step=3341.13
[epoch-end] epoch=186 elapsed=361.34s step=19635
[epoch-start] epoch=187 step=19635 steps_per_epoch=105
task=ci2v epoch=187 step=19640 loss=0.539607 loss_full=0.222568 loss_base=0.317039 ms_per_step=3386.95
task=ci2v epoch=187 step=19660 loss=0.491661 loss_full=0.203807 loss_base=0.287854 ms_per_step=3730.59
task=ci2v epoch=187 step=19680 loss=0.623216 loss_full=0.262881 loss_base=0.360335 ms_per_step=3524.40
task=ci2v epoch=187 step=19700 loss=0.511599 loss_full=0.213685 loss_base=0.297915 ms_per_step=3555.65
[profile] window=351.03s updates=100 data_wait=1.8ms h2d=1.1ms stage1=226.8ms dit=641.8ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=187 step=19720 loss=0.466784 loss_full=0.187725 loss_base=0.279059 ms_per_step=3406.95
task=ci2v epoch=187 step=19740 loss=0.658137 loss_full=0.300677 loss_base=0.357460 ms_per_step=3348.14
[epoch-end] epoch=187 elapsed=369.33s step=19740
[epoch-start] epoch=188 step=19740 steps_per_epoch=105
task=ci2v epoch=188 step=19760 loss=0.554062 loss_full=0.242576 loss_base=0.311486 ms_per_step=3381.29
task=ci2v epoch=188 step=19780 loss=0.467918 loss_full=0.194064 loss_base=0.273854 ms_per_step=3564.25
task=ci2v epoch=188 step=19800 loss=0.571908 loss_full=0.252426 loss_base=0.319482 ms_per_step=3347.88
[profile] window=341.25s updates=100 data_wait=1.7ms h2d=0.9ms stage1=207.8ms dit=636.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=188 step=19820 loss=0.519991 loss_full=0.224360 loss_base=0.295631 ms_per_step=3347.10
task=ci2v epoch=188 step=19840 loss=0.545336 loss_full=0.243028 loss_base=0.302308 ms_per_step=3735.07
[epoch-end] epoch=188 elapsed=366.01s step=19845
[epoch-start] epoch=189 step=19845 steps_per_epoch=105
task=ci2v epoch=189 step=19860 loss=0.586835 loss_full=0.255982 loss_base=0.330852 ms_per_step=3724.08
task=ci2v epoch=189 step=19880 loss=0.533022 loss_full=0.225266 loss_base=0.307756 ms_per_step=3451.67
task=ci2v epoch=189 step=19900 loss=0.563885 loss_full=0.243263 loss_base=0.320622 ms_per_step=3858.93
[profile] window=362.59s updates=100 data_wait=2.1ms h2d=1.3ms stage1=251.4ms dit=645.3ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=189 step=19920 loss=0.678368 loss_full=0.305741 loss_base=0.372628 ms_per_step=3636.44
task=ci2v epoch=189 step=19940 loss=0.582844 loss_full=0.255274 loss_base=0.327570 ms_per_step=3516.14
[epoch-end] epoch=189 elapsed=379.77s step=19950
[epoch-start] epoch=190 step=19950 steps_per_epoch=105
task=ci2v epoch=190 step=19960 loss=0.523741 loss_full=0.220920 loss_base=0.302821 ms_per_step=3451.06
task=ci2v epoch=190 step=19980 loss=0.578318 loss_full=0.244376 loss_base=0.333943 ms_per_step=3321.73
task=ci2v epoch=190 step=20000 loss=0.598743 loss_full=0.267179 loss_base=0.331564 ms_per_step=3351.97
[profile] window=345.82s updates=100 data_wait=1.5ms h2d=1.0ms stage1=215.8ms dit=640.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=20000 elapsed=2.52s
[checkpoint] step=20000 elapsed=26.17s
task=ci2v epoch=190 step=20020 loss=0.663743 loss_full=0.305678 loss_base=0.358065 ms_per_step=3327.68
task=ci2v epoch=190 step=20040 loss=0.606186 loss_full=0.265712 loss_base=0.340475 ms_per_step=3308.99
[epoch-end] epoch=190 elapsed=380.22s step=20055
[epoch-start] epoch=191 step=20055 steps_per_epoch=105
task=ci2v epoch=191 step=20060 loss=0.597914 loss_full=0.259544 loss_base=0.338369 ms_per_step=3339.10
task=ci2v epoch=191 step=20080 loss=0.550481 loss_full=0.229439 loss_base=0.321042 ms_per_step=3500.01
task=ci2v epoch=191 step=20100 loss=0.591824 loss_full=0.260647 loss_base=0.331177 ms_per_step=3321.68
[profile] window=336.22s updates=100 data_wait=1.6ms h2d=0.9ms stage1=197.7ms dit=633.8ms update=29.2ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=191 step=20120 loss=0.517648 loss_full=0.214008 loss_base=0.303640 ms_per_step=3385.45
task=ci2v epoch=191 step=20140 loss=0.539949 loss_full=0.235952 loss_base=0.303997 ms_per_step=3326.83
task=ci2v epoch=191 step=20160 loss=0.575001 loss_full=0.246137 loss_base=0.328865 ms_per_step=3299.35
[epoch-end] epoch=191 elapsed=354.19s step=20160
[epoch-start] epoch=192 step=20160 steps_per_epoch=105
task=ci2v epoch=192 step=20180 loss=0.575813 loss_full=0.246614 loss_base=0.329199 ms_per_step=3403.07
task=ci2v epoch=192 step=20200 loss=0.591137 loss_full=0.254952 loss_base=0.336185 ms_per_step=3402.05
[profile] window=336.60s updates=100 data_wait=1.7ms h2d=0.9ms stage1=198.3ms dit=634.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=192 step=20220 loss=0.558563 loss_full=0.240990 loss_base=0.317574 ms_per_step=3378.07
task=ci2v epoch=192 step=20240 loss=0.643484 loss_full=0.290082 loss_base=0.353401 ms_per_step=3319.73
task=ci2v epoch=192 step=20260 loss=0.581049 loss_full=0.254431 loss_base=0.326618 ms_per_step=3386.69
[epoch-end] epoch=192 elapsed=354.45s step=20265
[epoch-start] epoch=193 step=20265 steps_per_epoch=105
task=ci2v epoch=193 step=20280 loss=0.650632 loss_full=0.295772 loss_base=0.354859 ms_per_step=3315.83
task=ci2v epoch=193 step=20300 loss=0.590425 loss_full=0.258732 loss_base=0.331693 ms_per_step=3308.47
[profile] window=334.44s updates=100 data_wait=1.5ms h2d=0.9ms stage1=193.8ms dit=633.9ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=193 step=20320 loss=0.510635 loss_full=0.225517 loss_base=0.285117 ms_per_step=3320.20
task=ci2v epoch=193 step=20340 loss=0.465319 loss_full=0.188342 loss_base=0.276977 ms_per_step=3305.83
task=ci2v epoch=193 step=20360 loss=0.590593 loss_full=0.258497 loss_base=0.332097 ms_per_step=3420.08
[epoch-end] epoch=193 elapsed=352.07s step=20370
[epoch-start] epoch=194 step=20370 steps_per_epoch=105
task=ci2v epoch=194 step=20380 loss=0.646221 loss_full=0.290733 loss_base=0.355488 ms_per_step=3452.90
task=ci2v epoch=194 step=20400 loss=0.581589 loss_full=0.251576 loss_base=0.330013 ms_per_step=3374.98
[profile] window=337.74s updates=100 data_wait=1.6ms h2d=0.9ms stage1=201.0ms dit=634.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=194 step=20420 loss=0.537798 loss_full=0.225813 loss_base=0.311985 ms_per_step=3331.29
task=ci2v epoch=194 step=20440 loss=0.573400 loss_full=0.245904 loss_base=0.327496 ms_per_step=3369.55
task=ci2v epoch=194 step=20460 loss=0.541638 loss_full=0.238551 loss_base=0.303088 ms_per_step=3314.66
[epoch-end] epoch=194 elapsed=353.70s step=20475
[epoch-start] epoch=195 step=20475 steps_per_epoch=105
task=ci2v epoch=195 step=20480 loss=0.669800 loss_full=0.308156 loss_base=0.361645 ms_per_step=3416.36
task=ci2v epoch=195 step=20500 loss=0.542242 loss_full=0.238859 loss_base=0.303383 ms_per_step=3367.48
[profile] window=336.26s updates=100 data_wait=1.7ms h2d=0.9ms stage1=196.3ms dit=635.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=195 step=20520 loss=0.522260 loss_full=0.219069 loss_base=0.303190 ms_per_step=3312.97
task=ci2v epoch=195 step=20540 loss=0.457296 loss_full=0.185693 loss_base=0.271604 ms_per_step=3316.75
task=ci2v epoch=195 step=20560 loss=0.582203 loss_full=0.252059 loss_base=0.330145 ms_per_step=3312.96
task=ci2v epoch=195 step=20580 loss=0.616827 loss_full=0.271164 loss_base=0.345663 ms_per_step=3309.10
[epoch-end] epoch=195 elapsed=350.11s step=20580
[epoch-start] epoch=196 step=20580 steps_per_epoch=105
task=ci2v epoch=196 step=20600 loss=0.510493 loss_full=0.220570 loss_base=0.289923 ms_per_step=3542.32
[profile] window=336.15s updates=100 data_wait=1.6ms h2d=0.9ms stage1=196.2ms dit=635.2ms update=29.2ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=196 step=20620 loss=0.613351 loss_full=0.268018 loss_base=0.345333 ms_per_step=3652.45
task=ci2v epoch=196 step=20640 loss=0.590329 loss_full=0.257223 loss_base=0.333106 ms_per_step=3426.37
task=ci2v epoch=196 step=20660 loss=0.575918 loss_full=0.248707 loss_base=0.327211 ms_per_step=3341.05
task=ci2v epoch=196 step=20680 loss=0.546436 loss_full=0.240866 loss_base=0.305570 ms_per_step=3331.40
[epoch-end] epoch=196 elapsed=362.57s step=20685
[epoch-start] epoch=197 step=20685 steps_per_epoch=105
task=ci2v epoch=197 step=20700 loss=0.688608 loss_full=0.314533 loss_base=0.374074 ms_per_step=3355.04
[profile] window=342.37s updates=100 data_wait=1.5ms h2d=1.0ms stage1=209.4ms dit=637.2ms update=30.6ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=197 step=20720 loss=0.577255 loss_full=0.252708 loss_base=0.324546 ms_per_step=3311.80
task=ci2v epoch=197 step=20740 loss=0.499575 loss_full=0.207678 loss_base=0.291896 ms_per_step=3451.38
task=ci2v epoch=197 step=20760 loss=0.540893 loss_full=0.229820 loss_base=0.311073 ms_per_step=3650.05
task=ci2v epoch=197 step=20780 loss=0.519873 loss_full=0.215887 loss_base=0.303986 ms_per_step=3320.32
[epoch-end] epoch=197 elapsed=360.14s step=20790
[epoch-start] epoch=198 step=20790 steps_per_epoch=105
task=ci2v epoch=198 step=20800 loss=0.551546 loss_full=0.233132 loss_base=0.318414 ms_per_step=3408.73
[profile] window=343.12s updates=100 data_wait=1.5ms h2d=1.0ms stage1=212.7ms dit=636.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=198 step=20820 loss=0.551066 loss_full=0.231979 loss_base=0.319087 ms_per_step=3425.19
task=ci2v epoch=198 step=20840 loss=0.604856 loss_full=0.265636 loss_base=0.339220 ms_per_step=3549.21
task=ci2v epoch=198 step=20860 loss=0.594091 loss_full=0.262283 loss_base=0.331808 ms_per_step=3452.77
task=ci2v epoch=198 step=20880 loss=0.554090 loss_full=0.241723 loss_base=0.312368 ms_per_step=3419.66
[epoch-end] epoch=198 elapsed=361.45s step=20895
[epoch-start] epoch=199 step=20895 steps_per_epoch=105
task=ci2v epoch=199 step=20900 loss=0.594525 loss_full=0.263243 loss_base=0.331282 ms_per_step=3377.20
[profile] window=344.75s updates=100 data_wait=1.6ms h2d=1.0ms stage1=215.9ms dit=637.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=199 step=20920 loss=0.554892 loss_full=0.242008 loss_base=0.312884 ms_per_step=3294.39
task=ci2v epoch=199 step=20940 loss=0.508646 loss_full=0.218124 loss_base=0.290521 ms_per_step=3466.80
task=ci2v epoch=199 step=20960 loss=0.576485 loss_full=0.253770 loss_base=0.322715 ms_per_step=3615.24
task=ci2v epoch=199 step=20980 loss=0.521792 loss_full=0.225939 loss_base=0.295853 ms_per_step=3337.40
task=ci2v epoch=199 step=21000 loss=0.486988 loss_full=0.200519 loss_base=0.286469 ms_per_step=3322.73
[profile] window=341.00s updates=100 data_wait=0.9ms h2d=1.0ms stage1=207.2ms dit=637.3ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=21000 elapsed=2.28s
[checkpoint] step=21000 elapsed=16.69s
[epoch-end] epoch=199 elapsed=377.51s step=21000
[epoch-start] epoch=200 step=21000 steps_per_epoch=105
task=ci2v epoch=200 step=21020 loss=0.552700 loss_full=0.240044 loss_base=0.312656 ms_per_step=3351.83
task=ci2v epoch=200 step=21040 loss=0.547853 loss_full=0.232767 loss_base=0.315085 ms_per_step=3377.07
task=ci2v epoch=200 step=21060 loss=0.618138 loss_full=0.274924 loss_base=0.343215 ms_per_step=3367.35
task=ci2v epoch=200 step=21080 loss=0.563904 loss_full=0.233832 loss_base=0.330073 ms_per_step=3342.63
task=ci2v epoch=200 step=21100 loss=0.592946 loss_full=0.262314 loss_base=0.330633 ms_per_step=3339.15
[profile] window=335.83s updates=100 data_wait=1.6ms h2d=0.9ms stage1=196.2ms dit=634.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=200 elapsed=353.79s step=21105
[epoch-start] epoch=201 step=21105 steps_per_epoch=105
task=ci2v epoch=201 step=21120 loss=0.509613 loss_full=0.216771 loss_base=0.292842 ms_per_step=3465.02
task=ci2v epoch=201 step=21140 loss=0.664809 loss_full=0.291148 loss_base=0.373661 ms_per_step=3540.20
task=ci2v epoch=201 step=21160 loss=0.536955 loss_full=0.225356 loss_base=0.311599 ms_per_step=3379.77
task=ci2v epoch=201 step=21180 loss=0.509419 loss_full=0.214468 loss_base=0.294951 ms_per_step=3307.23
task=ci2v epoch=201 step=21200 loss=0.483416 loss_full=0.199709 loss_base=0.283706 ms_per_step=3362.77
[profile] window=341.37s updates=100 data_wait=1.6ms h2d=1.0ms stage1=206.5ms dit=638.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=201 elapsed=356.53s step=21210
[epoch-start] epoch=202 step=21210 steps_per_epoch=105
task=ci2v epoch=202 step=21220 loss=0.670579 loss_full=0.302516 loss_base=0.368063 ms_per_step=3362.36
task=ci2v epoch=202 step=21240 loss=0.556153 loss_full=0.234959 loss_base=0.321194 ms_per_step=3358.97
task=ci2v epoch=202 step=21260 loss=0.463008 loss_full=0.179547 loss_base=0.283460 ms_per_step=3354.12
task=ci2v epoch=202 step=21280 loss=0.611476 loss_full=0.268842 loss_base=0.342634 ms_per_step=3374.61
task=ci2v epoch=202 step=21300 loss=0.602460 loss_full=0.269002 loss_base=0.333459 ms_per_step=3353.94
[profile] window=336.35s updates=100 data_wait=1.6ms h2d=0.9ms stage1=197.2ms dit=635.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=202 elapsed=352.97s step=21315
[epoch-start] epoch=203 step=21315 steps_per_epoch=105
task=ci2v epoch=203 step=21320 loss=0.614755 loss_full=0.271814 loss_base=0.342941 ms_per_step=3343.66
task=ci2v epoch=203 step=21340 loss=0.682845 loss_full=0.314775 loss_base=0.368070 ms_per_step=3320.08
task=ci2v epoch=203 step=21360 loss=0.549452 loss_full=0.233860 loss_base=0.315592 ms_per_step=3331.98
task=ci2v epoch=203 step=21380 loss=0.563072 loss_full=0.245424 loss_base=0.317647 ms_per_step=3320.84
task=ci2v epoch=203 step=21400 loss=0.575607 loss_full=0.251396 loss_base=0.324211 ms_per_step=3351.69
[profile] window=333.64s updates=100 data_wait=1.6ms h2d=0.8ms stage1=191.7ms dit=633.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=203 step=21420 loss=0.593121 loss_full=0.263947 loss_base=0.329174 ms_per_step=3296.89
[epoch-end] epoch=203 elapsed=350.31s step=21420
[epoch-start] epoch=204 step=21420 steps_per_epoch=105
task=ci2v epoch=204 step=21440 loss=0.453675 loss_full=0.187167 loss_base=0.266509 ms_per_step=3384.06
task=ci2v epoch=204 step=21460 loss=0.580347 loss_full=0.255464 loss_base=0.324883 ms_per_step=3303.79
task=ci2v epoch=204 step=21480 loss=0.529241 loss_full=0.225341 loss_base=0.303900 ms_per_step=3319.82
task=ci2v epoch=204 step=21500 loss=0.519901 loss_full=0.217609 loss_base=0.302292 ms_per_step=3410.88
[profile] window=334.58s updates=100 data_wait=1.6ms h2d=0.9ms stage1=194.2ms dit=633.6ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=204 step=21520 loss=0.565099 loss_full=0.240713 loss_base=0.324387 ms_per_step=3331.59
[epoch-end] epoch=204 elapsed=351.68s step=21525
[epoch-start] epoch=205 step=21525 steps_per_epoch=105
task=ci2v epoch=205 step=21540 loss=0.592242 loss_full=0.264169 loss_base=0.328072 ms_per_step=3355.65
task=ci2v epoch=205 step=21560 loss=0.575168 loss_full=0.257109 loss_base=0.318059 ms_per_step=3314.02
task=ci2v epoch=205 step=21580 loss=0.570854 loss_full=0.244468 loss_base=0.326386 ms_per_step=3313.17
task=ci2v epoch=205 step=21600 loss=0.542170 loss_full=0.229148 loss_base=0.313021 ms_per_step=3335.04
[profile] window=333.26s updates=100 data_wait=1.6ms h2d=0.8ms stage1=190.2ms dit=634.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=205 step=21620 loss=0.465553 loss_full=0.189686 loss_base=0.275867 ms_per_step=3408.37
[epoch-end] epoch=205 elapsed=351.53s step=21630
[epoch-start] epoch=206 step=21630 steps_per_epoch=105
task=ci2v epoch=206 step=21640 loss=0.616656 loss_full=0.269297 loss_base=0.347359 ms_per_step=3398.87
task=ci2v epoch=206 step=21660 loss=0.556327 loss_full=0.243721 loss_base=0.312605 ms_per_step=3314.92
task=ci2v epoch=206 step=21680 loss=0.524458 loss_full=0.228892 loss_base=0.295566 ms_per_step=3384.99
task=ci2v epoch=206 step=21700 loss=0.589775 loss_full=0.258581 loss_base=0.331194 ms_per_step=3326.18
[profile] window=336.94s updates=100 data_wait=1.6ms h2d=0.9ms stage1=197.6ms dit=636.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=206 step=21720 loss=0.577787 loss_full=0.251486 loss_base=0.326301 ms_per_step=3354.96
[epoch-end] epoch=206 elapsed=352.41s step=21735
[epoch-start] epoch=207 step=21735 steps_per_epoch=105
task=ci2v epoch=207 step=21740 loss=0.499302 loss_full=0.212411 loss_base=0.286891 ms_per_step=3339.85
task=ci2v epoch=207 step=21760 loss=0.450121 loss_full=0.182363 loss_base=0.267758 ms_per_step=3311.42
task=ci2v epoch=207 step=21780 loss=0.442535 loss_full=0.176011 loss_base=0.266524 ms_per_step=3345.63
task=ci2v epoch=207 step=21800 loss=0.542046 loss_full=0.232556 loss_base=0.309490 ms_per_step=3327.53
[profile] window=333.86s updates=100 data_wait=1.6ms h2d=0.8ms stage1=191.9ms dit=634.3ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=207 step=21820 loss=0.526823 loss_full=0.222676 loss_base=0.304147 ms_per_step=3315.31
task=ci2v epoch=207 step=21840 loss=0.588037 loss_full=0.254702 loss_base=0.333335 ms_per_step=3329.85
[epoch-end] epoch=207 elapsed=350.49s step=21840
[epoch-start] epoch=208 step=21840 steps_per_epoch=105
task=ci2v epoch=208 step=21860 loss=0.551318 loss_full=0.236498 loss_base=0.314820 ms_per_step=3348.05
task=ci2v epoch=208 step=21880 loss=0.556081 loss_full=0.239559 loss_base=0.316523 ms_per_step=3328.65
task=ci2v epoch=208 step=21900 loss=0.507076 loss_full=0.211752 loss_base=0.295324 ms_per_step=3376.83
[profile] window=334.24s updates=100 data_wait=1.6ms h2d=0.8ms stage1=191.8ms dit=635.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=208 step=21920 loss=0.571352 loss_full=0.248977 loss_base=0.322375 ms_per_step=3527.77
task=ci2v epoch=208 step=21940 loss=0.595975 loss_full=0.253207 loss_base=0.342767 ms_per_step=3310.89
[epoch-end] epoch=208 elapsed=354.57s step=21945
[epoch-start] epoch=209 step=21945 steps_per_epoch=105
task=ci2v epoch=209 step=21960 loss=0.609393 loss_full=0.270117 loss_base=0.339276 ms_per_step=3538.43
task=ci2v epoch=209 step=21980 loss=0.482702 loss_full=0.207721 loss_base=0.274981 ms_per_step=3314.19
task=ci2v epoch=209 step=22000 loss=0.585635 loss_full=0.254361 loss_base=0.331274 ms_per_step=3392.52
[profile] window=341.95s updates=100 data_wait=1.8ms h2d=1.0ms stage1=209.7ms dit=635.7ms update=30.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=22000 elapsed=2.97s
[checkpoint] step=22000 elapsed=26.75s
task=ci2v epoch=209 step=22020 loss=0.561723 loss_full=0.244395 loss_base=0.317327 ms_per_step=3417.92
task=ci2v epoch=209 step=22040 loss=0.592192 loss_full=0.263467 loss_base=0.328726 ms_per_step=3346.90
[epoch-end] epoch=209 elapsed=386.87s step=22050
[epoch-start] epoch=210 step=22050 steps_per_epoch=105
task=ci2v epoch=210 step=22060 loss=0.532629 loss_full=0.232214 loss_base=0.300415 ms_per_step=3400.80
task=ci2v epoch=210 step=22080 loss=0.472524 loss_full=0.196031 loss_base=0.276492 ms_per_step=3400.68
task=ci2v epoch=210 step=22100 loss=0.586038 loss_full=0.250747 loss_base=0.335291 ms_per_step=3421.74
[profile] window=340.03s updates=100 data_wait=1.9ms h2d=0.9ms stage1=204.7ms dit=636.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=210 step=22120 loss=0.543767 loss_full=0.232563 loss_base=0.311204 ms_per_step=3338.84
task=ci2v epoch=210 step=22140 loss=0.493247 loss_full=0.212815 loss_base=0.280433 ms_per_step=3336.20
[epoch-end] epoch=210 elapsed=355.01s step=22155
[epoch-start] epoch=211 step=22155 steps_per_epoch=105
task=ci2v epoch=211 step=22160 loss=0.520988 loss_full=0.221700 loss_base=0.299288 ms_per_step=3352.17
task=ci2v epoch=211 step=22180 loss=0.581164 loss_full=0.256521 loss_base=0.324642 ms_per_step=3470.44
task=ci2v epoch=211 step=22200 loss=0.493099 loss_full=0.200737 loss_base=0.292362 ms_per_step=3712.71
[profile] window=344.48s updates=100 data_wait=1.5ms h2d=1.0ms stage1=213.0ms dit=637.9ms update=34.3ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=211 step=22220 loss=0.494760 loss_full=0.192619 loss_base=0.302142 ms_per_step=3336.12
task=ci2v epoch=211 step=22240 loss=0.556450 loss_full=0.250447 loss_base=0.306003 ms_per_step=3346.07
task=ci2v epoch=211 step=22260 loss=0.556591 loss_full=0.235468 loss_base=0.321123 ms_per_step=3342.23
[epoch-end] epoch=211 elapsed=362.08s step=22260
[epoch-start] epoch=212 step=22260 steps_per_epoch=105
task=ci2v epoch=212 step=22280 loss=0.575825 loss_full=0.254424 loss_base=0.321401 ms_per_step=3360.68
task=ci2v epoch=212 step=22300 loss=0.531419 loss_full=0.219488 loss_base=0.311931 ms_per_step=3325.84
[profile] window=334.49s updates=100 data_wait=1.6ms h2d=0.8ms stage1=190.7ms dit=636.8ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=212 step=22320 loss=0.543770 loss_full=0.232125 loss_base=0.311645 ms_per_step=3339.10
task=ci2v epoch=212 step=22340 loss=0.619121 loss_full=0.273785 loss_base=0.345337 ms_per_step=3369.76
task=ci2v epoch=212 step=22360 loss=0.524196 loss_full=0.227335 loss_base=0.296861 ms_per_step=3327.22
[epoch-end] epoch=212 elapsed=351.24s step=22365
[epoch-start] epoch=213 step=22365 steps_per_epoch=105
task=ci2v epoch=213 step=22380 loss=0.654721 loss_full=0.294583 loss_base=0.360139 ms_per_step=3345.34
task=ci2v epoch=213 step=22400 loss=0.499664 loss_full=0.212468 loss_base=0.287197 ms_per_step=3317.59
[profile] window=334.24s updates=100 data_wait=1.6ms h2d=0.8ms stage1=190.0ms dit=637.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=213 step=22420 loss=0.543416 loss_full=0.236266 loss_base=0.307150 ms_per_step=3523.42
task=ci2v epoch=213 step=22440 loss=0.595842 loss_full=0.248496 loss_base=0.347347 ms_per_step=3306.96
task=ci2v epoch=213 step=22460 loss=0.482800 loss_full=0.198981 loss_base=0.283819 ms_per_step=3317.37
[epoch-end] epoch=213 elapsed=353.01s step=22470
[epoch-start] epoch=214 step=22470 steps_per_epoch=105
task=ci2v epoch=214 step=22480 loss=0.605049 loss_full=0.274602 loss_base=0.330447 ms_per_step=3560.06
task=ci2v epoch=214 step=22500 loss=0.612563 loss_full=0.276696 loss_base=0.335867 ms_per_step=3377.55
[profile] window=341.97s updates=100 data_wait=1.7ms h2d=1.0ms stage1=207.7ms dit=638.1ms update=29.2ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=214 step=22520 loss=0.634572 loss_full=0.274412 loss_base=0.360160 ms_per_step=3336.89
task=ci2v epoch=214 step=22540 loss=0.469360 loss_full=0.194851 loss_base=0.274508 ms_per_step=3407.70
task=ci2v epoch=214 step=22560 loss=0.576541 loss_full=0.250625 loss_base=0.325917 ms_per_step=3317.51
[epoch-end] epoch=214 elapsed=356.71s step=22575
[epoch-start] epoch=215 step=22575 steps_per_epoch=105
task=ci2v epoch=215 step=22580 loss=0.686530 loss_full=0.306480 loss_base=0.380050 ms_per_step=3329.25
task=ci2v epoch=215 step=22600 loss=0.502625 loss_full=0.209989 loss_base=0.292636 ms_per_step=3306.94
[profile] window=334.24s updates=100 data_wait=1.7ms h2d=0.8ms stage1=192.2ms dit=634.8ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=215 step=22620 loss=0.532429 loss_full=0.225536 loss_base=0.306893 ms_per_step=3427.81
task=ci2v epoch=215 step=22640 loss=0.576471 loss_full=0.253769 loss_base=0.322702 ms_per_step=3338.53
task=ci2v epoch=215 step=22660 loss=0.523558 loss_full=0.222422 loss_base=0.301136 ms_per_step=3304.12
task=ci2v epoch=215 step=22680 loss=0.640172 loss_full=0.285700 loss_base=0.354473 ms_per_step=3401.02
[epoch-end] epoch=215 elapsed=353.42s step=22680
[epoch-start] epoch=216 step=22680 steps_per_epoch=105
task=ci2v epoch=216 step=22700 loss=0.662941 loss_full=0.300215 loss_base=0.362725 ms_per_step=3364.14
[profile] window=336.98s updates=100 data_wait=1.4ms h2d=0.9ms stage1=198.1ms dit=635.2ms update=29.3ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=216 step=22720 loss=0.573660 loss_full=0.255607 loss_base=0.318054 ms_per_step=3307.41
task=ci2v epoch=216 step=22740 loss=0.507801 loss_full=0.210895 loss_base=0.296906 ms_per_step=3326.18
task=ci2v epoch=216 step=22760 loss=0.591531 loss_full=0.259467 loss_base=0.332064 ms_per_step=3307.88
task=ci2v epoch=216 step=22780 loss=0.546057 loss_full=0.229627 loss_base=0.316429 ms_per_step=3337.69
[epoch-end] epoch=216 elapsed=349.69s step=22785
[epoch-start] epoch=217 step=22785 steps_per_epoch=105
task=ci2v epoch=217 step=22800 loss=0.597106 loss_full=0.262121 loss_base=0.334985 ms_per_step=3358.92
[profile] window=333.03s updates=100 data_wait=1.8ms h2d=0.8ms stage1=190.5ms dit=633.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=217 step=22820 loss=0.530049 loss_full=0.220134 loss_base=0.309915 ms_per_step=3390.83
task=ci2v epoch=217 step=22840 loss=0.504487 loss_full=0.211396 loss_base=0.293091 ms_per_step=3334.32
task=ci2v epoch=217 step=22860 loss=0.524846 loss_full=0.227903 loss_base=0.296943 ms_per_step=3352.92
task=ci2v epoch=217 step=22880 loss=0.616939 loss_full=0.276905 loss_base=0.340034 ms_per_step=3323.96
[epoch-end] epoch=217 elapsed=352.09s step=22890
[epoch-start] epoch=218 step=22890 steps_per_epoch=105
task=ci2v epoch=218 step=22900 loss=0.452255 loss_full=0.181474 loss_base=0.270781 ms_per_step=3392.37
[profile] window=336.16s updates=100 data_wait=1.6ms h2d=0.9ms stage1=197.6ms dit=634.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=218 step=22920 loss=0.541832 loss_full=0.242656 loss_base=0.299177 ms_per_step=3344.38
task=ci2v epoch=218 step=22940 loss=0.562579 loss_full=0.247955 loss_base=0.314625 ms_per_step=3384.44
task=ci2v epoch=218 step=22960 loss=0.523835 loss_full=0.219423 loss_base=0.304413 ms_per_step=3317.08
task=ci2v epoch=218 step=22980 loss=0.526187 loss_full=0.217091 loss_base=0.309095 ms_per_step=3306.95
[epoch-end] epoch=218 elapsed=353.06s step=22995
[epoch-start] epoch=219 step=22995 steps_per_epoch=105
task=ci2v epoch=219 step=23000 loss=0.620255 loss_full=0.282296 loss_base=0.337959 ms_per_step=3411.46
[profile] window=335.56s updates=100 data_wait=1.6ms h2d=0.9ms stage1=196.0ms dit=634.3ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=23000 elapsed=2.75s
[checkpoint] step=23000 elapsed=26.78s
task=ci2v epoch=219 step=23020 loss=0.569859 loss_full=0.255777 loss_base=0.314082 ms_per_step=3381.16
task=ci2v epoch=219 step=23040 loss=0.474092 loss_full=0.188116 loss_base=0.285976 ms_per_step=3396.87
task=ci2v epoch=219 step=23060 loss=0.549310 loss_full=0.245325 loss_base=0.303985 ms_per_step=3317.81
task=ci2v epoch=219 step=23080 loss=0.575075 loss_full=0.252173 loss_base=0.322902 ms_per_step=3314.94
task=ci2v epoch=219 step=23100 loss=0.547267 loss_full=0.225578 loss_base=0.321688 ms_per_step=3386.90
[profile] window=336.23s updates=100 data_wait=0.9ms h2d=0.9ms stage1=198.1ms dit=634.6ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=219 elapsed=383.49s step=23100
[epoch-start] epoch=220 step=23100 steps_per_epoch=105
task=ci2v epoch=220 step=23120 loss=0.463359 loss_full=0.187547 loss_base=0.275812 ms_per_step=3553.11
task=ci2v epoch=220 step=23140 loss=0.520165 loss_full=0.225874 loss_base=0.294291 ms_per_step=3303.70
task=ci2v epoch=220 step=23160 loss=0.538370 loss_full=0.222006 loss_base=0.316365 ms_per_step=3338.12
task=ci2v epoch=220 step=23180 loss=0.460502 loss_full=0.193202 loss_base=0.267300 ms_per_step=3330.94
task=ci2v epoch=220 step=23200 loss=0.625817 loss_full=0.277838 loss_base=0.347980 ms_per_step=3317.35
[profile] window=337.13s updates=100 data_wait=1.5ms h2d=0.9ms stage1=198.4ms dit=635.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=220 elapsed=353.74s step=23205
[epoch-start] epoch=221 step=23205 steps_per_epoch=105
task=ci2v epoch=221 step=23220 loss=0.585524 loss_full=0.256853 loss_base=0.328671 ms_per_step=3336.78
task=ci2v epoch=221 step=23240 loss=0.617086 loss_full=0.263686 loss_base=0.353400 ms_per_step=3310.09
task=ci2v epoch=221 step=23260 loss=0.531128 loss_full=0.222891 loss_base=0.308237 ms_per_step=3316.03
task=ci2v epoch=221 step=23280 loss=0.537739 loss_full=0.231840 loss_base=0.305899 ms_per_step=3313.12
task=ci2v epoch=221 step=23300 loss=0.531007 loss_full=0.219070 loss_base=0.311937 ms_per_step=3456.99
[profile] window=334.93s updates=100 data_wait=1.7ms h2d=0.9ms stage1=194.9ms dit=633.8ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=221 elapsed=351.47s step=23310
[epoch-start] epoch=222 step=23310 steps_per_epoch=105
task=ci2v epoch=222 step=23320 loss=0.477327 loss_full=0.193608 loss_base=0.283719 ms_per_step=3341.59
task=ci2v epoch=222 step=23340 loss=0.505899 loss_full=0.213037 loss_base=0.292862 ms_per_step=3325.06
task=ci2v epoch=222 step=23360 loss=0.502469 loss_full=0.210456 loss_base=0.292012 ms_per_step=3431.14
task=ci2v epoch=222 step=23380 loss=0.562551 loss_full=0.244852 loss_base=0.317699 ms_per_step=3431.92
task=ci2v epoch=222 step=23400 loss=0.528255 loss_full=0.229415 loss_base=0.298840 ms_per_step=3419.44
[profile] window=339.26s updates=100 data_wait=1.4ms h2d=0.9ms stage1=204.2ms dit=635.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=222 elapsed=356.22s step=23415
[epoch-start] epoch=223 step=23415 steps_per_epoch=105
task=ci2v epoch=223 step=23420 loss=0.526198 loss_full=0.225635 loss_base=0.300562 ms_per_step=3346.77
task=ci2v epoch=223 step=23440 loss=0.528698 loss_full=0.217134 loss_base=0.311564 ms_per_step=3375.85
task=ci2v epoch=223 step=23460 loss=0.560098 loss_full=0.236826 loss_base=0.323272 ms_per_step=3454.90
task=ci2v epoch=223 step=23480 loss=0.579980 loss_full=0.248151 loss_base=0.331829 ms_per_step=3307.31
task=ci2v epoch=223 step=23500 loss=0.536323 loss_full=0.225547 loss_base=0.310775 ms_per_step=3331.93
[profile] window=336.61s updates=100 data_wait=1.6ms h2d=0.9ms stage1=196.7ms dit=636.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=223 step=23520 loss=0.449362 loss_full=0.184734 loss_base=0.264628 ms_per_step=3316.20
[epoch-end] epoch=223 elapsed=353.33s step=23520
[epoch-start] epoch=224 step=23520 steps_per_epoch=105
task=ci2v epoch=224 step=23540 loss=0.541948 loss_full=0.225178 loss_base=0.316769 ms_per_step=3415.92
task=ci2v epoch=224 step=23560 loss=0.552406 loss_full=0.232368 loss_base=0.320038 ms_per_step=3408.39
task=ci2v epoch=224 step=23580 loss=0.577082 loss_full=0.233492 loss_base=0.343590 ms_per_step=3314.97
task=ci2v epoch=224 step=23600 loss=0.591101 loss_full=0.258085 loss_base=0.333017 ms_per_step=3334.88
[profile] window=336.08s updates=100 data_wait=1.7ms h2d=0.9ms stage1=196.6ms dit=634.9ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=224 step=23620 loss=0.550761 loss_full=0.237290 loss_base=0.313471 ms_per_step=3327.86
[epoch-end] epoch=224 elapsed=352.75s step=23625
[epoch-start] epoch=225 step=23625 steps_per_epoch=105
task=ci2v epoch=225 step=23640 loss=0.512525 loss_full=0.216456 loss_base=0.296069 ms_per_step=3390.66
task=ci2v epoch=225 step=23660 loss=0.591616 loss_full=0.252932 loss_base=0.338684 ms_per_step=3333.38
task=ci2v epoch=225 step=23680 loss=0.537035 loss_full=0.225734 loss_base=0.311301 ms_per_step=3325.17
task=ci2v epoch=225 step=23700 loss=0.489799 loss_full=0.209728 loss_base=0.280071 ms_per_step=3496.96
[profile] window=337.75s updates=100 data_wait=1.5ms h2d=0.9ms stage1=201.3ms dit=634.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=225 step=23720 loss=0.499258 loss_full=0.204427 loss_base=0.294831 ms_per_step=3304.65
[epoch-end] epoch=225 elapsed=353.96s step=23730
[epoch-start] epoch=226 step=23730 steps_per_epoch=105
task=ci2v epoch=226 step=23740 loss=0.503964 loss_full=0.207906 loss_base=0.296058 ms_per_step=3363.67
task=ci2v epoch=226 step=23760 loss=0.474598 loss_full=0.193604 loss_base=0.280994 ms_per_step=3376.82
task=ci2v epoch=226 step=23780 loss=0.494770 loss_full=0.207450 loss_base=0.287319 ms_per_step=3493.98
task=ci2v epoch=226 step=23800 loss=0.534017 loss_full=0.226835 loss_base=0.307182 ms_per_step=3464.92
[profile] window=340.35s updates=100 data_wait=1.6ms h2d=0.9ms stage1=206.4ms dit=635.8ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=226 step=23820 loss=0.614857 loss_full=0.280465 loss_base=0.334393 ms_per_step=3429.67
[epoch-end] epoch=226 elapsed=359.56s step=23835
[epoch-start] epoch=227 step=23835 steps_per_epoch=105
task=ci2v epoch=227 step=23840 loss=0.478054 loss_full=0.202361 loss_base=0.275694 ms_per_step=3345.37
task=ci2v epoch=227 step=23860 loss=0.559750 loss_full=0.247578 loss_base=0.312171 ms_per_step=3327.12
task=ci2v epoch=227 step=23880 loss=0.518044 loss_full=0.224974 loss_base=0.293071 ms_per_step=3566.54
task=ci2v epoch=227 step=23900 loss=0.519505 loss_full=0.221715 loss_base=0.297790 ms_per_step=3343.03
[profile] window=340.50s updates=100 data_wait=1.5ms h2d=0.9ms stage1=206.7ms dit=635.4ms update=29.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=227 step=23920 loss=0.556333 loss_full=0.242503 loss_base=0.313829 ms_per_step=3347.17
task=ci2v epoch=227 step=23940 loss=0.557407 loss_full=0.236985 loss_base=0.320422 ms_per_step=3358.90
[epoch-end] epoch=227 elapsed=356.70s step=23940
[epoch-start] epoch=228 step=23940 steps_per_epoch=105
task=ci2v epoch=228 step=23960 loss=0.487096 loss_full=0.198312 loss_base=0.288784 ms_per_step=3440.13
task=ci2v epoch=228 step=23980 loss=0.624311 loss_full=0.282778 loss_base=0.341532 ms_per_step=3307.85
task=ci2v epoch=228 step=24000 loss=0.596254 loss_full=0.262431 loss_base=0.333823 ms_per_step=3336.10
[profile] window=336.08s updates=100 data_wait=1.4ms h2d=0.9ms stage1=196.9ms dit=634.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=24000 elapsed=2.43s
[checkpoint] step=24000 elapsed=19.45s
task=ci2v epoch=228 step=24020 loss=0.484561 loss_full=0.203555 loss_base=0.281006 ms_per_step=3331.12
task=ci2v epoch=228 step=24040 loss=0.542686 loss_full=0.235513 loss_base=0.307174 ms_per_step=3389.15
[epoch-end] epoch=228 elapsed=374.76s step=24045
[epoch-start] epoch=229 step=24045 steps_per_epoch=105
task=ci2v epoch=229 step=24060 loss=0.565849 loss_full=0.249577 loss_base=0.316273 ms_per_step=3340.94
task=ci2v epoch=229 step=24080 loss=0.457602 loss_full=0.187769 loss_base=0.269833 ms_per_step=3365.75
task=ci2v epoch=229 step=24100 loss=0.503336 loss_full=0.217660 loss_base=0.285676 ms_per_step=3411.79
[profile] window=337.05s updates=100 data_wait=1.6ms h2d=0.9ms stage1=198.7ms dit=635.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=229 step=24120 loss=0.492855 loss_full=0.208199 loss_base=0.284656 ms_per_step=3379.24
task=ci2v epoch=229 step=24140 loss=0.538990 loss_full=0.230658 loss_base=0.308333 ms_per_step=3332.19
[epoch-end] epoch=229 elapsed=354.08s step=24150
[epoch-start] epoch=230 step=24150 steps_per_epoch=105
task=ci2v epoch=230 step=24160 loss=0.467159 loss_full=0.190024 loss_base=0.277135 ms_per_step=3388.10
task=ci2v epoch=230 step=24180 loss=0.622869 loss_full=0.289304 loss_base=0.333565 ms_per_step=3303.65
task=ci2v epoch=230 step=24200 loss=0.559201 loss_full=0.237478 loss_base=0.321723 ms_per_step=3317.36
[profile] window=334.68s updates=100 data_wait=1.6ms h2d=0.8ms stage1=194.0ms dit=634.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=230 step=24220 loss=0.710911 loss_full=0.333239 loss_base=0.377672 ms_per_step=3327.46
task=ci2v epoch=230 step=24240 loss=0.526541 loss_full=0.224922 loss_base=0.301619 ms_per_step=3347.28
[epoch-end] epoch=230 elapsed=349.95s step=24255
[epoch-start] epoch=231 step=24255 steps_per_epoch=105
task=ci2v epoch=231 step=24260 loss=0.627910 loss_full=0.283902 loss_base=0.344009 ms_per_step=3337.33
task=ci2v epoch=231 step=24280 loss=0.594362 loss_full=0.271836 loss_base=0.322526 ms_per_step=3473.22
task=ci2v epoch=231 step=24300 loss=0.546796 loss_full=0.224162 loss_base=0.322634 ms_per_step=3346.99
[profile] window=336.92s updates=100 data_wait=1.5ms h2d=0.9ms stage1=199.5ms dit=634.3ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=231 step=24320 loss=0.552662 loss_full=0.242244 loss_base=0.310418 ms_per_step=3331.91
task=ci2v epoch=231 step=24340 loss=0.497326 loss_full=0.210127 loss_base=0.287198 ms_per_step=3367.82
task=ci2v epoch=231 step=24360 loss=0.597544 loss_full=0.267743 loss_base=0.329801 ms_per_step=3388.37
[epoch-end] epoch=231 elapsed=356.00s step=24360
[epoch-start] epoch=232 step=24360 steps_per_epoch=105
task=ci2v epoch=232 step=24380 loss=0.541265 loss_full=0.240417 loss_base=0.300848 ms_per_step=3509.19
task=ci2v epoch=232 step=24400 loss=0.458838 loss_full=0.180389 loss_base=0.278449 ms_per_step=3316.18
[profile] window=338.54s updates=100 data_wait=2.0ms h2d=0.9ms stage1=201.8ms dit=635.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=232 step=24420 loss=0.566022 loss_full=0.237743 loss_base=0.328279 ms_per_step=3401.55
task=ci2v epoch=232 step=24440 loss=0.552118 loss_full=0.246757 loss_base=0.305361 ms_per_step=3338.73
task=ci2v epoch=232 step=24460 loss=0.620415 loss_full=0.267228 loss_base=0.353187 ms_per_step=3379.34
[epoch-end] epoch=232 elapsed=355.90s step=24465
[epoch-start] epoch=233 step=24465 steps_per_epoch=105
task=ci2v epoch=233 step=24480 loss=0.523062 loss_full=0.224037 loss_base=0.299026 ms_per_step=3376.20
task=ci2v epoch=233 step=24500 loss=0.500078 loss_full=0.214354 loss_base=0.285725 ms_per_step=3551.58
[profile] window=341.21s updates=100 data_wait=1.9ms h2d=1.0ms stage1=207.6ms dit=636.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=233 step=24520 loss=0.544927 loss_full=0.223116 loss_base=0.321811 ms_per_step=3426.68
task=ci2v epoch=233 step=24540 loss=0.516225 loss_full=0.215934 loss_base=0.300290 ms_per_step=3321.58
task=ci2v epoch=233 step=24560 loss=0.638014 loss_full=0.295673 loss_base=0.342340 ms_per_step=3317.97
[epoch-end] epoch=233 elapsed=356.79s step=24570
[epoch-start] epoch=234 step=24570 steps_per_epoch=105
task=ci2v epoch=234 step=24580 loss=0.500134 loss_full=0.209082 loss_base=0.291053 ms_per_step=3354.49
task=ci2v epoch=234 step=24600 loss=0.582007 loss_full=0.253997 loss_base=0.328010 ms_per_step=3380.86
[profile] window=336.30s updates=100 data_wait=1.5ms h2d=0.9ms stage1=196.2ms dit=636.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=234 step=24620 loss=0.475707 loss_full=0.193316 loss_base=0.282391 ms_per_step=3312.06
task=ci2v epoch=234 step=24640 loss=0.583455 loss_full=0.258496 loss_base=0.324959 ms_per_step=3317.37
task=ci2v epoch=234 step=24660 loss=0.593628 loss_full=0.260650 loss_base=0.332977 ms_per_step=3326.10
[epoch-end] epoch=234 elapsed=350.36s step=24675
[epoch-start] epoch=235 step=24675 steps_per_epoch=105
task=ci2v epoch=235 step=24680 loss=0.496056 loss_full=0.208806 loss_base=0.287250 ms_per_step=3340.38
task=ci2v epoch=235 step=24700 loss=0.623538 loss_full=0.263836 loss_base=0.359702 ms_per_step=3331.87
[profile] window=332.83s updates=100 data_wait=1.8ms h2d=0.8ms stage1=189.5ms dit=633.8ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=235 step=24720 loss=0.510519 loss_full=0.220938 loss_base=0.289581 ms_per_step=3410.32
task=ci2v epoch=235 step=24740 loss=0.634459 loss_full=0.271387 loss_base=0.363072 ms_per_step=3601.67
task=ci2v epoch=235 step=24760 loss=0.631907 loss_full=0.288451 loss_base=0.343456 ms_per_step=3572.66
task=ci2v epoch=235 step=24780 loss=0.534639 loss_full=0.226411 loss_base=0.308228 ms_per_step=3388.81
[epoch-end] epoch=235 elapsed=363.99s step=24780
[epoch-start] epoch=236 step=24780 steps_per_epoch=105
task=ci2v epoch=236 step=24800 loss=0.518328 loss_full=0.225331 loss_base=0.292997 ms_per_step=3352.86
[profile] window=346.79s updates=100 data_wait=1.5ms h2d=1.0ms stage1=217.7ms dit=640.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=236 step=24820 loss=0.606076 loss_full=0.270003 loss_base=0.336073 ms_per_step=3456.14
task=ci2v epoch=236 step=24840 loss=0.415145 loss_full=0.160409 loss_base=0.254736 ms_per_step=3376.81
task=ci2v epoch=236 step=24860 loss=0.558043 loss_full=0.251852 loss_base=0.306190 ms_per_step=3532.72
task=ci2v epoch=236 step=24880 loss=0.509066 loss_full=0.207500 loss_base=0.301566 ms_per_step=3452.39
[epoch-end] epoch=236 elapsed=360.26s step=24885
[epoch-start] epoch=237 step=24885 steps_per_epoch=105
task=ci2v epoch=237 step=24900 loss=0.412815 loss_full=0.161780 loss_base=0.251035 ms_per_step=3553.34
[profile] window=347.69s updates=100 data_wait=1.5ms h2d=1.1ms stage1=222.1ms dit=638.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=237 step=24920 loss=0.542636 loss_full=0.235569 loss_base=0.307066 ms_per_step=3333.98
task=ci2v epoch=237 step=24940 loss=0.520060 loss_full=0.220051 loss_base=0.300009 ms_per_step=3315.96
task=ci2v epoch=237 step=24960 loss=0.574782 loss_full=0.251820 loss_base=0.322963 ms_per_step=3307.96
task=ci2v epoch=237 step=24980 loss=0.620974 loss_full=0.290384 loss_base=0.330591 ms_per_step=3342.72
[epoch-end] epoch=237 elapsed=353.66s step=24990
[epoch-start] epoch=238 step=24990 steps_per_epoch=105
task=ci2v epoch=238 step=25000 loss=0.560984 loss_full=0.249826 loss_base=0.311158 ms_per_step=3330.47
[profile] window=332.90s updates=100 data_wait=1.6ms h2d=0.8ms stage1=190.5ms dit=633.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=25000 elapsed=2.81s
[checkpoint] step=25000 elapsed=19.86s
task=ci2v epoch=238 step=25020 loss=0.537478 loss_full=0.233523 loss_base=0.303956 ms_per_step=3317.03
task=ci2v epoch=238 step=25040 loss=0.488059 loss_full=0.200916 loss_base=0.287143 ms_per_step=3314.24
task=ci2v epoch=238 step=25060 loss=0.461754 loss_full=0.191574 loss_base=0.270180 ms_per_step=3313.95
task=ci2v epoch=238 step=25080 loss=0.629632 loss_full=0.277460 loss_base=0.352172 ms_per_step=3323.00
[epoch-end] epoch=238 elapsed=372.71s step=25095
[epoch-start] epoch=239 step=25095 steps_per_epoch=105
task=ci2v epoch=239 step=25100 loss=0.541808 loss_full=0.223638 loss_base=0.318170 ms_per_step=3377.61
[profile] window=333.19s updates=100 data_wait=1.6ms h2d=0.8ms stage1=190.0ms dit=634.3ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=239 step=25120 loss=0.614986 loss_full=0.283846 loss_base=0.331140 ms_per_step=3339.72
task=ci2v epoch=239 step=25140 loss=0.487776 loss_full=0.199696 loss_base=0.288080 ms_per_step=3397.25
task=ci2v epoch=239 step=25160 loss=0.511550 loss_full=0.222989 loss_base=0.288560 ms_per_step=3579.01
task=ci2v epoch=239 step=25180 loss=0.581889 loss_full=0.256433 loss_base=0.325457 ms_per_step=3316.23
task=ci2v epoch=239 step=25200 loss=0.558750 loss_full=0.243671 loss_base=0.315079 ms_per_step=3301.75
[profile] window=339.00s updates=100 data_wait=0.9ms h2d=0.9ms stage1=203.3ms dit=636.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=239 elapsed=356.47s step=25200
[epoch-start] epoch=240 step=25200 steps_per_epoch=105
task=ci2v epoch=240 step=25220 loss=0.568450 loss_full=0.256595 loss_base=0.311855 ms_per_step=3346.68
task=ci2v epoch=240 step=25240 loss=0.658692 loss_full=0.287988 loss_base=0.370704 ms_per_step=3313.26
task=ci2v epoch=240 step=25260 loss=0.556450 loss_full=0.241444 loss_base=0.315005 ms_per_step=3361.33
task=ci2v epoch=240 step=25280 loss=0.538868 loss_full=0.230966 loss_base=0.307902 ms_per_step=3398.36
task=ci2v epoch=240 step=25300 loss=0.496954 loss_full=0.200857 loss_base=0.296097 ms_per_step=3468.43
[profile] window=338.03s updates=100 data_wait=1.6ms h2d=0.9ms stage1=200.0ms dit=636.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=240 elapsed=354.31s step=25305
[epoch-start] epoch=241 step=25305 steps_per_epoch=105
task=ci2v epoch=241 step=25320 loss=0.546191 loss_full=0.226978 loss_base=0.319213 ms_per_step=3329.76
task=ci2v epoch=241 step=25340 loss=0.460130 loss_full=0.188421 loss_base=0.271709 ms_per_step=3395.52
task=ci2v epoch=241 step=25360 loss=0.484088 loss_full=0.199921 loss_base=0.284167 ms_per_step=3506.15
task=ci2v epoch=241 step=25380 loss=0.553014 loss_full=0.239465 loss_base=0.313550 ms_per_step=3324.70
task=ci2v epoch=241 step=25400 loss=0.549539 loss_full=0.229361 loss_base=0.320178 ms_per_step=3421.40
[profile] window=339.82s updates=100 data_wait=1.7ms h2d=0.9ms stage1=205.5ms dit=635.3ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=241 elapsed=356.68s step=25410
[epoch-start] epoch=242 step=25410 steps_per_epoch=105
task=ci2v epoch=242 step=25420 loss=0.609378 loss_full=0.267101 loss_base=0.342277 ms_per_step=3344.49
task=ci2v epoch=242 step=25440 loss=0.528791 loss_full=0.226491 loss_base=0.302300 ms_per_step=3342.80
task=ci2v epoch=242 step=25460 loss=0.631134 loss_full=0.278209 loss_base=0.352925 ms_per_step=3392.73
task=ci2v epoch=242 step=25480 loss=0.525024 loss_full=0.226088 loss_base=0.298935 ms_per_step=3347.72
task=ci2v epoch=242 step=25500 loss=0.533529 loss_full=0.213942 loss_base=0.319588 ms_per_step=3342.70
[profile] window=335.68s updates=100 data_wait=1.5ms h2d=0.9ms stage1=196.2ms dit=634.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=242 elapsed=352.19s step=25515
[epoch-start] epoch=243 step=25515 steps_per_epoch=105
task=ci2v epoch=243 step=25520 loss=0.567850 loss_full=0.234220 loss_base=0.333630 ms_per_step=3353.28
task=ci2v epoch=243 step=25540 loss=0.497883 loss_full=0.205340 loss_base=0.292543 ms_per_step=3503.67
task=ci2v epoch=243 step=25560 loss=0.564627 loss_full=0.246740 loss_base=0.317887 ms_per_step=3452.99
task=ci2v epoch=243 step=25580 loss=0.562361 loss_full=0.256295 loss_base=0.306066 ms_per_step=3444.36
task=ci2v epoch=243 step=25600 loss=0.511248 loss_full=0.216868 loss_base=0.294380 ms_per_step=3312.20
[profile] window=341.60s updates=100 data_wait=1.5ms h2d=1.0ms stage1=208.7ms dit=636.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=243 step=25620 loss=0.479994 loss_full=0.200803 loss_base=0.279191 ms_per_step=3291.14
[epoch-end] epoch=243 elapsed=358.27s step=25620
[epoch-start] epoch=244 step=25620 steps_per_epoch=105
task=ci2v epoch=244 step=25640 loss=0.505329 loss_full=0.199148 loss_base=0.306181 ms_per_step=3362.69
task=ci2v epoch=244 step=25660 loss=0.514309 loss_full=0.212723 loss_base=0.301585 ms_per_step=3330.44
task=ci2v epoch=244 step=25680 loss=0.588766 loss_full=0.266033 loss_base=0.322732 ms_per_step=3331.98
task=ci2v epoch=244 step=25700 loss=0.569739 loss_full=0.238960 loss_base=0.330779 ms_per_step=3320.00
[profile] window=333.00s updates=100 data_wait=1.7ms h2d=0.8ms stage1=189.9ms dit=633.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=244 step=25720 loss=0.561035 loss_full=0.250227 loss_base=0.310808 ms_per_step=3314.34
[epoch-end] epoch=244 elapsed=349.72s step=25725
[epoch-start] epoch=245 step=25725 steps_per_epoch=105
task=ci2v epoch=245 step=25740 loss=0.531838 loss_full=0.224996 loss_base=0.306842 ms_per_step=3366.32
task=ci2v epoch=245 step=25760 loss=0.623764 loss_full=0.282303 loss_base=0.341462 ms_per_step=3323.90
task=ci2v epoch=245 step=25780 loss=0.523981 loss_full=0.215320 loss_base=0.308661 ms_per_step=3410.64
task=ci2v epoch=245 step=25800 loss=0.661187 loss_full=0.295837 loss_base=0.365350 ms_per_step=3316.99
[profile] window=334.92s updates=100 data_wait=1.4ms h2d=0.9ms stage1=194.3ms dit=634.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=245 step=25820 loss=0.545260 loss_full=0.247491 loss_base=0.297769 ms_per_step=3335.15
[epoch-end] epoch=245 elapsed=352.45s step=25830
[epoch-start] epoch=246 step=25830 steps_per_epoch=105
task=ci2v epoch=246 step=25840 loss=0.488955 loss_full=0.203620 loss_base=0.285334 ms_per_step=3428.74
task=ci2v epoch=246 step=25860 loss=0.547983 loss_full=0.224683 loss_base=0.323301 ms_per_step=3409.38
task=ci2v epoch=246 step=25880 loss=0.559180 loss_full=0.239441 loss_base=0.319739 ms_per_step=3303.32
task=ci2v epoch=246 step=25900 loss=0.503862 loss_full=0.210385 loss_base=0.293476 ms_per_step=3322.43
[profile] window=336.25s updates=100 data_wait=1.6ms h2d=0.9ms stage1=197.4ms dit=634.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=246 step=25920 loss=0.554888 loss_full=0.241486 loss_base=0.313402 ms_per_step=3409.78
[epoch-end] epoch=246 elapsed=354.33s step=25935
[epoch-start] epoch=247 step=25935 steps_per_epoch=105
task=ci2v epoch=247 step=25940 loss=0.526882 loss_full=0.220194 loss_base=0.306688 ms_per_step=3341.29
task=ci2v epoch=247 step=25960 loss=0.514510 loss_full=0.209917 loss_base=0.304593 ms_per_step=3298.03
task=ci2v epoch=247 step=25980 loss=0.545587 loss_full=0.236302 loss_base=0.309284 ms_per_step=3330.94
task=ci2v epoch=247 step=26000 loss=0.463144 loss_full=0.185373 loss_base=0.277771 ms_per_step=3309.21
[profile] window=334.05s updates=100 data_wait=1.5ms h2d=0.9ms stage1=191.8ms dit=634.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=26000 elapsed=2.60s
[checkpoint] step=26000 elapsed=19.62s
task=ci2v epoch=247 step=26020 loss=0.544853 loss_full=0.240257 loss_base=0.304596 ms_per_step=3463.02
task=ci2v epoch=247 step=26040 loss=0.551713 loss_full=0.228971 loss_base=0.322742 ms_per_step=3344.92
[epoch-end] epoch=247 elapsed=374.75s step=26040
[epoch-start] epoch=248 step=26040 steps_per_epoch=105
task=ci2v epoch=248 step=26060 loss=0.565391 loss_full=0.247891 loss_base=0.317500 ms_per_step=3647.40
task=ci2v epoch=248 step=26080 loss=0.547518 loss_full=0.235473 loss_base=0.312045 ms_per_step=3342.44
task=ci2v epoch=248 step=26100 loss=0.565579 loss_full=0.241961 loss_base=0.323618 ms_per_step=3394.36
[profile] window=344.11s updates=100 data_wait=1.7ms h2d=1.0ms stage1=214.9ms dit=636.3ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=248 step=26120 loss=0.494525 loss_full=0.209785 loss_base=0.284740 ms_per_step=3322.43
task=ci2v epoch=248 step=26140 loss=0.526647 loss_full=0.218936 loss_base=0.307712 ms_per_step=3321.43
[epoch-end] epoch=248 elapsed=357.49s step=26145
[epoch-start] epoch=249 step=26145 steps_per_epoch=105
task=ci2v epoch=249 step=26160 loss=0.598413 loss_full=0.258212 loss_base=0.340201 ms_per_step=3358.52
task=ci2v epoch=249 step=26180 loss=0.480771 loss_full=0.198426 loss_base=0.282345 ms_per_step=3346.15
task=ci2v epoch=249 step=26200 loss=0.619094 loss_full=0.269305 loss_base=0.349789 ms_per_step=3444.74
[profile] window=336.13s updates=100 data_wait=1.7ms h2d=0.9ms stage1=195.6ms dit=635.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=249 step=26220 loss=0.510666 loss_full=0.211891 loss_base=0.298776 ms_per_step=3314.85
task=ci2v epoch=249 step=26240 loss=0.577893 loss_full=0.261325 loss_base=0.316568 ms_per_step=3329.72
[epoch-end] epoch=249 elapsed=352.89s step=26250
[epoch-start] epoch=250 step=26250 steps_per_epoch=105
task=ci2v epoch=250 step=26260 loss=0.561078 loss_full=0.256224 loss_base=0.304854 ms_per_step=3469.76
task=ci2v epoch=250 step=26280 loss=0.504826 loss_full=0.211921 loss_base=0.292906 ms_per_step=3431.00
task=ci2v epoch=250 step=26300 loss=0.487417 loss_full=0.198874 loss_base=0.288542 ms_per_step=3317.41
[profile] window=337.53s updates=100 data_wait=1.8ms h2d=0.9ms stage1=200.0ms dit=634.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=250 step=26320 loss=0.554617 loss_full=0.250225 loss_base=0.304392 ms_per_step=3324.83
task=ci2v epoch=250 step=26340 loss=0.530364 loss_full=0.227935 loss_base=0.302429 ms_per_step=3312.57
[epoch-end] epoch=250 elapsed=353.56s step=26355
[epoch-start] epoch=251 step=26355 steps_per_epoch=105
task=ci2v epoch=251 step=26360 loss=0.512324 loss_full=0.211027 loss_base=0.301297 ms_per_step=3372.27
task=ci2v epoch=251 step=26380 loss=0.538661 loss_full=0.245703 loss_base=0.292959 ms_per_step=3333.63
task=ci2v epoch=251 step=26400 loss=0.459888 loss_full=0.189983 loss_base=0.269904 ms_per_step=3375.41
[profile] window=334.64s updates=100 data_wait=1.5ms h2d=0.8ms stage1=194.0ms dit=634.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=251 step=26420 loss=0.557788 loss_full=0.234475 loss_base=0.323313 ms_per_step=3401.14
task=ci2v epoch=251 step=26440 loss=0.502670 loss_full=0.205953 loss_base=0.296717 ms_per_step=3322.44
task=ci2v epoch=251 step=26460 loss=0.491042 loss_full=0.209759 loss_base=0.281283 ms_per_step=3307.81
[epoch-end] epoch=251 elapsed=353.50s step=26460
[epoch-start] epoch=252 step=26460 steps_per_epoch=105
task=ci2v epoch=252 step=26480 loss=0.515883 loss_full=0.204793 loss_base=0.311089 ms_per_step=3360.70
task=ci2v epoch=252 step=26500 loss=0.583479 loss_full=0.254612 loss_base=0.328866 ms_per_step=3385.10
[profile] window=335.82s updates=100 data_wait=1.5ms h2d=0.9ms stage1=195.5ms dit=635.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=252 step=26520 loss=0.527731 loss_full=0.219847 loss_base=0.307884 ms_per_step=3311.81
task=ci2v epoch=252 step=26540 loss=0.528014 loss_full=0.219910 loss_base=0.308104 ms_per_step=3313.38
task=ci2v epoch=252 step=26560 loss=0.525852 loss_full=0.211459 loss_base=0.314394 ms_per_step=3306.47
[epoch-end] epoch=252 elapsed=350.26s step=26565
[epoch-start] epoch=253 step=26565 steps_per_epoch=105
task=ci2v epoch=253 step=26580 loss=0.512776 loss_full=0.207492 loss_base=0.305284 ms_per_step=3336.36
task=ci2v epoch=253 step=26600 loss=0.566193 loss_full=0.248747 loss_base=0.317446 ms_per_step=3326.87
[profile] window=332.17s updates=100 data_wait=1.6ms h2d=0.8ms stage1=187.9ms dit=633.9ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=253 step=26620 loss=0.595191 loss_full=0.264365 loss_base=0.330826 ms_per_step=3411.77
task=ci2v epoch=253 step=26640 loss=0.608664 loss_full=0.261224 loss_base=0.347439 ms_per_step=3316.01
task=ci2v epoch=253 step=26660 loss=0.494570 loss_full=0.205670 loss_base=0.288900 ms_per_step=3308.52
[epoch-end] epoch=253 elapsed=350.93s step=26670
[epoch-start] epoch=254 step=26670 steps_per_epoch=105
task=ci2v epoch=254 step=26680 loss=0.572205 loss_full=0.257880 loss_base=0.314325 ms_per_step=3326.40
task=ci2v epoch=254 step=26700 loss=0.463103 loss_full=0.193803 loss_base=0.269300 ms_per_step=3320.20
[profile] window=333.93s updates=100 data_wait=1.6ms h2d=0.9ms stage1=192.2ms dit=634.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=254 step=26720 loss=0.481073 loss_full=0.194630 loss_base=0.286444 ms_per_step=3412.74
task=ci2v epoch=254 step=26740 loss=0.549664 loss_full=0.229773 loss_base=0.319891 ms_per_step=3329.61
task=ci2v epoch=254 step=26760 loss=0.522485 loss_full=0.223291 loss_base=0.299195 ms_per_step=3329.99
[epoch-end] epoch=254 elapsed=351.09s step=26775
[epoch-start] epoch=255 step=26775 steps_per_epoch=105
task=ci2v epoch=255 step=26780 loss=0.433571 loss_full=0.176710 loss_base=0.256861 ms_per_step=3326.85
task=ci2v epoch=255 step=26800 loss=0.565558 loss_full=0.242773 loss_base=0.322785 ms_per_step=3294.22
[profile] window=334.14s updates=100 data_wait=1.6ms h2d=0.9ms stage1=193.1ms dit=633.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=255 step=26820 loss=0.552526 loss_full=0.236199 loss_base=0.316327 ms_per_step=3326.89
task=ci2v epoch=255 step=26840 loss=0.478932 loss_full=0.193531 loss_base=0.285400 ms_per_step=3319.10
task=ci2v epoch=255 step=26860 loss=0.552508 loss_full=0.240421 loss_base=0.312087 ms_per_step=3344.72
task=ci2v epoch=255 step=26880 loss=0.611153 loss_full=0.278935 loss_base=0.332218 ms_per_step=3327.50
[epoch-end] epoch=255 elapsed=350.04s step=26880
[epoch-start] epoch=256 step=26880 steps_per_epoch=105
task=ci2v epoch=256 step=26900 loss=0.586396 loss_full=0.245447 loss_base=0.340949 ms_per_step=3436.12
[profile] window=335.36s updates=100 data_wait=1.8ms h2d=0.9ms stage1=194.6ms dit=634.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=256 step=26920 loss=0.511211 loss_full=0.217087 loss_base=0.294124 ms_per_step=3311.06
task=ci2v epoch=256 step=26940 loss=0.536543 loss_full=0.231875 loss_base=0.304668 ms_per_step=3320.25
task=ci2v epoch=256 step=26960 loss=0.463505 loss_full=0.188417 loss_base=0.275088 ms_per_step=3331.45
task=ci2v epoch=256 step=26980 loss=0.534984 loss_full=0.230327 loss_base=0.304657 ms_per_step=3390.91
[epoch-end] epoch=256 elapsed=352.34s step=26985
[epoch-start] epoch=257 step=26985 steps_per_epoch=105
task=ci2v epoch=257 step=27000 loss=0.495724 loss_full=0.205796 loss_base=0.289928 ms_per_step=3342.41
[profile] window=334.19s updates=100 data_wait=1.4ms h2d=0.8ms stage1=192.5ms dit=634.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=27000 elapsed=2.47s
[checkpoint] step=27000 elapsed=19.57s
task=ci2v epoch=257 step=27020 loss=0.596502 loss_full=0.262535 loss_base=0.333967 ms_per_step=3316.21
task=ci2v epoch=257 step=27040 loss=0.617248 loss_full=0.275097 loss_base=0.342151 ms_per_step=3309.52
task=ci2v epoch=257 step=27060 loss=0.564692 loss_full=0.255935 loss_base=0.308757 ms_per_step=3329.75
task=ci2v epoch=257 step=27080 loss=0.545164 loss_full=0.242388 loss_base=0.302776 ms_per_step=3332.76
[epoch-end] epoch=257 elapsed=372.68s step=27090
[epoch-start] epoch=258 step=27090 steps_per_epoch=105
task=ci2v epoch=258 step=27100 loss=0.623943 loss_full=0.275331 loss_base=0.348612 ms_per_step=3390.47
[profile] window=333.85s updates=100 data_wait=1.6ms h2d=0.8ms stage1=191.6ms dit=634.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=258 step=27120 loss=0.607610 loss_full=0.271032 loss_base=0.336577 ms_per_step=3339.55
task=ci2v epoch=258 step=27140 loss=0.566470 loss_full=0.244304 loss_base=0.322166 ms_per_step=3350.89
task=ci2v epoch=258 step=27160 loss=0.617611 loss_full=0.267189 loss_base=0.350423 ms_per_step=3386.91
task=ci2v epoch=258 step=27180 loss=0.521227 loss_full=0.219569 loss_base=0.301658 ms_per_step=3359.47
[epoch-end] epoch=258 elapsed=353.81s step=27195
[epoch-start] epoch=259 step=27195 steps_per_epoch=105
task=ci2v epoch=259 step=27200 loss=0.603995 loss_full=0.274007 loss_base=0.329988 ms_per_step=3402.75
[profile] window=337.06s updates=100 data_wait=1.4ms h2d=0.9ms stage1=199.0ms dit=635.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=259 step=27220 loss=0.607557 loss_full=0.267238 loss_base=0.340319 ms_per_step=3315.57
task=ci2v epoch=259 step=27240 loss=0.577381 loss_full=0.251593 loss_base=0.325788 ms_per_step=3921.88
task=ci2v epoch=259 step=27260 loss=0.478170 loss_full=0.189364 loss_base=0.288807 ms_per_step=3915.20
task=ci2v epoch=259 step=27280 loss=0.628235 loss_full=0.274763 loss_base=0.353472 ms_per_step=3921.41
task=ci2v epoch=259 step=27300 loss=0.618112 loss_full=0.280499 loss_base=0.337614 ms_per_step=3824.04
[profile] window=378.20s updates=100 data_wait=1.3ms h2d=1.5ms stage1=285.5ms dit=651.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=259 elapsed=395.77s step=27300
[epoch-start] epoch=260 step=27300 steps_per_epoch=105
task=ci2v epoch=260 step=27320 loss=0.576893 loss_full=0.248340 loss_base=0.328553 ms_per_step=3418.84
task=ci2v epoch=260 step=27340 loss=0.450872 loss_full=0.184598 loss_base=0.266274 ms_per_step=3321.25
task=ci2v epoch=260 step=27360 loss=0.520553 loss_full=0.218929 loss_base=0.301624 ms_per_step=3320.89
task=ci2v epoch=260 step=27380 loss=0.485452 loss_full=0.196284 loss_base=0.289169 ms_per_step=3333.50
task=ci2v epoch=260 step=27400 loss=0.534230 loss_full=0.224693 loss_base=0.309536 ms_per_step=3330.56
[profile] window=334.77s updates=100 data_wait=2.0ms h2d=0.8ms stage1=193.2ms dit=634.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=260 elapsed=351.06s step=27405
[epoch-start] epoch=261 step=27405 steps_per_epoch=105
task=ci2v epoch=261 step=27420 loss=0.516867 loss_full=0.214318 loss_base=0.302550 ms_per_step=3360.60
task=ci2v epoch=261 step=27440 loss=0.563658 loss_full=0.239928 loss_base=0.323730 ms_per_step=3410.12
task=ci2v epoch=261 step=27460 loss=0.560374 loss_full=0.235913 loss_base=0.324461 ms_per_step=3348.64
task=ci2v epoch=261 step=27480 loss=0.508696 loss_full=0.207934 loss_base=0.300762 ms_per_step=3653.10
task=ci2v epoch=261 step=27500 loss=0.509197 loss_full=0.210891 loss_base=0.298305 ms_per_step=3412.98
[profile] window=343.97s updates=100 data_wait=1.7ms h2d=1.0ms stage1=214.2ms dit=637.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=261 elapsed=365.67s step=27510
[epoch-start] epoch=262 step=27510 steps_per_epoch=105
task=ci2v epoch=262 step=27520 loss=0.582159 loss_full=0.247524 loss_base=0.334635 ms_per_step=3699.13
task=ci2v epoch=262 step=27540 loss=0.512799 loss_full=0.218404 loss_base=0.294396 ms_per_step=3389.66
task=ci2v epoch=262 step=27560 loss=0.559744 loss_full=0.243658 loss_base=0.316087 ms_per_step=3500.72
task=ci2v epoch=262 step=27580 loss=0.451556 loss_full=0.183201 loss_base=0.268355 ms_per_step=3712.86
task=ci2v epoch=262 step=27600 loss=0.556595 loss_full=0.241395 loss_base=0.315200 ms_per_step=3581.77
[profile] window=357.95s updates=100 data_wait=2.1ms h2d=1.2ms stage1=245.1ms dit=640.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=262 elapsed=374.80s step=27615
[epoch-start] epoch=263 step=27615 steps_per_epoch=105
task=ci2v epoch=263 step=27620 loss=0.632524 loss_full=0.268732 loss_base=0.363792 ms_per_step=3598.51
task=ci2v epoch=263 step=27640 loss=0.493770 loss_full=0.207766 loss_base=0.286004 ms_per_step=3772.20
task=ci2v epoch=263 step=27660 loss=0.470163 loss_full=0.194465 loss_base=0.275698 ms_per_step=3656.83
task=ci2v epoch=263 step=27680 loss=0.509572 loss_full=0.218475 loss_base=0.291096 ms_per_step=3538.22
task=ci2v epoch=263 step=27700 loss=0.487658 loss_full=0.200757 loss_base=0.286902 ms_per_step=3369.59
[profile] window=358.95s updates=100 data_wait=1.7ms h2d=1.5ms stage1=244.4ms dit=643.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=263 step=27720 loss=0.622046 loss_full=0.282547 loss_base=0.339500 ms_per_step=3433.29
[epoch-end] epoch=263 elapsed=373.28s step=27720
[epoch-start] epoch=264 step=27720 steps_per_epoch=105
task=ci2v epoch=264 step=27740 loss=0.496877 loss_full=0.210396 loss_base=0.286481 ms_per_step=3555.35
task=ci2v epoch=264 step=27760 loss=0.454737 loss_full=0.182912 loss_base=0.271825 ms_per_step=3676.90
task=ci2v epoch=264 step=27780 loss=0.510354 loss_full=0.217010 loss_base=0.293344 ms_per_step=3396.10
task=ci2v epoch=264 step=27800 loss=0.477019 loss_full=0.202979 loss_base=0.274041 ms_per_step=3369.21
[profile] window=348.88s updates=100 data_wait=1.7ms h2d=1.0ms stage1=220.4ms dit=642.6ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=264 step=27820 loss=0.550064 loss_full=0.236112 loss_base=0.313952 ms_per_step=3362.79
[epoch-end] epoch=264 elapsed=365.22s step=27825
[epoch-start] epoch=265 step=27825 steps_per_epoch=105
task=ci2v epoch=265 step=27840 loss=0.546184 loss_full=0.240085 loss_base=0.306099 ms_per_step=3646.04
task=ci2v epoch=265 step=27860 loss=0.522571 loss_full=0.226329 loss_base=0.296242 ms_per_step=3533.41
task=ci2v epoch=265 step=27880 loss=0.630015 loss_full=0.280808 loss_base=0.349207 ms_per_step=3585.32
task=ci2v epoch=265 step=27900 loss=0.577531 loss_full=0.250588 loss_base=0.326943 ms_per_step=3331.04
[profile] window=349.43s updates=100 data_wait=2.1ms h2d=1.1ms stage1=221.8ms dit=642.3ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=265 step=27920 loss=0.497930 loss_full=0.209126 loss_base=0.288804 ms_per_step=3541.25
[epoch-end] epoch=265 elapsed=372.02s step=27930
[epoch-start] epoch=266 step=27930 steps_per_epoch=105
task=ci2v epoch=266 step=27940 loss=0.517476 loss_full=0.219066 loss_base=0.298410 ms_per_step=3644.44
task=ci2v epoch=266 step=27960 loss=0.568627 loss_full=0.247220 loss_base=0.321408 ms_per_step=3446.56
task=ci2v epoch=266 step=27980 loss=0.511177 loss_full=0.206112 loss_base=0.305065 ms_per_step=3410.95
task=ci2v epoch=266 step=28000 loss=0.586279 loss_full=0.235860 loss_base=0.350419 ms_per_step=3544.59
[profile] window=352.02s updates=100 data_wait=1.8ms h2d=1.2ms stage1=232.2ms dit=638.9ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=28000 elapsed=2.65s
[checkpoint] step=28000 elapsed=25.83s
task=ci2v epoch=266 step=28020 loss=0.486229 loss_full=0.198664 loss_base=0.287565 ms_per_step=3363.78
[epoch-end] epoch=266 elapsed=394.64s step=28035
[epoch-start] epoch=267 step=28035 steps_per_epoch=105
task=ci2v epoch=267 step=28040 loss=0.566207 loss_full=0.252814 loss_base=0.313393 ms_per_step=3581.45
task=ci2v epoch=267 step=28060 loss=0.511190 loss_full=0.214625 loss_base=0.296566 ms_per_step=3333.87
task=ci2v epoch=267 step=28080 loss=0.508390 loss_full=0.207037 loss_base=0.301354 ms_per_step=3383.33
task=ci2v epoch=267 step=28100 loss=0.495524 loss_full=0.201490 loss_base=0.294035 ms_per_step=3314.22
[profile] window=339.81s updates=100 data_wait=1.5ms h2d=0.9ms stage1=205.7ms dit=635.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=267 step=28120 loss=0.440817 loss_full=0.173313 loss_base=0.267504 ms_per_step=3318.64
task=ci2v epoch=267 step=28140 loss=0.593343 loss_full=0.262172 loss_base=0.331171 ms_per_step=3307.57
[epoch-end] epoch=267 elapsed=351.06s step=28140
[epoch-start] epoch=268 step=28140 steps_per_epoch=105
task=ci2v epoch=268 step=28160 loss=0.476448 loss_full=0.195290 loss_base=0.281158 ms_per_step=3367.51
task=ci2v epoch=268 step=28180 loss=0.517171 loss_full=0.217443 loss_base=0.299727 ms_per_step=3319.20
task=ci2v epoch=268 step=28200 loss=0.599244 loss_full=0.258561 loss_base=0.340683 ms_per_step=3331.50
[profile] window=333.16s updates=100 data_wait=1.5ms h2d=0.8ms stage1=190.2ms dit=634.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=268 step=28220 loss=0.522711 loss_full=0.219711 loss_base=0.303001 ms_per_step=3310.13
task=ci2v epoch=268 step=28240 loss=0.524239 loss_full=0.226585 loss_base=0.297654 ms_per_step=3348.47
[epoch-end] epoch=268 elapsed=350.45s step=28245
[epoch-start] epoch=269 step=28245 steps_per_epoch=105
task=ci2v epoch=269 step=28260 loss=0.526565 loss_full=0.222597 loss_base=0.303969 ms_per_step=3343.38
task=ci2v epoch=269 step=28280 loss=0.500129 loss_full=0.197921 loss_base=0.302208 ms_per_step=3305.83
task=ci2v epoch=269 step=28300 loss=0.472081 loss_full=0.194311 loss_base=0.277771 ms_per_step=3312.15
[profile] window=332.67s updates=100 data_wait=1.5ms h2d=0.8ms stage1=189.2ms dit=633.6ms update=29.4ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=269 step=28320 loss=0.586586 loss_full=0.252662 loss_base=0.333923 ms_per_step=3437.38
task=ci2v epoch=269 step=28340 loss=0.650557 loss_full=0.292644 loss_base=0.357913 ms_per_step=3336.17
[epoch-end] epoch=269 elapsed=351.66s step=28350
[epoch-start] epoch=270 step=28350 steps_per_epoch=105
task=ci2v epoch=270 step=28360 loss=0.550832 loss_full=0.236161 loss_base=0.314671 ms_per_step=3359.62
task=ci2v epoch=270 step=28380 loss=0.480621 loss_full=0.193993 loss_base=0.286628 ms_per_step=3316.46
task=ci2v epoch=270 step=28400 loss=0.636763 loss_full=0.291347 loss_base=0.345416 ms_per_step=3315.64
[profile] window=335.57s updates=100 data_wait=1.6ms h2d=0.9ms stage1=194.6ms dit=635.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=270 step=28420 loss=0.589714 loss_full=0.259534 loss_base=0.330180 ms_per_step=3309.84
task=ci2v epoch=270 step=28440 loss=0.442447 loss_full=0.177929 loss_base=0.264518 ms_per_step=3459.67
[epoch-end] epoch=270 elapsed=352.40s step=28455
[epoch-start] epoch=271 step=28455 steps_per_epoch=105
task=ci2v epoch=271 step=28460 loss=0.537069 loss_full=0.236314 loss_base=0.300755 ms_per_step=3356.58
task=ci2v epoch=271 step=28480 loss=0.718883 loss_full=0.337016 loss_base=0.381868 ms_per_step=3321.51
task=ci2v epoch=271 step=28500 loss=0.510699 loss_full=0.218001 loss_base=0.292698 ms_per_step=3334.93
[profile] window=335.90s updates=100 data_wait=1.6ms h2d=0.9ms stage1=196.4ms dit=634.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=271 step=28520 loss=0.490137 loss_full=0.210067 loss_base=0.280070 ms_per_step=3312.52
task=ci2v epoch=271 step=28540 loss=0.551849 loss_full=0.247280 loss_base=0.304568 ms_per_step=3350.49
task=ci2v epoch=271 step=28560 loss=0.500054 loss_full=0.211143 loss_base=0.288912 ms_per_step=3297.44
[epoch-end] epoch=271 elapsed=350.01s step=28560
[epoch-start] epoch=272 step=28560 steps_per_epoch=105
task=ci2v epoch=272 step=28580 loss=0.542408 loss_full=0.231742 loss_base=0.310667 ms_per_step=3335.62
task=ci2v epoch=272 step=28600 loss=0.633055 loss_full=0.287845 loss_base=0.345210 ms_per_step=3513.96
[profile] window=336.47s updates=100 data_wait=1.6ms h2d=0.9ms stage1=197.4ms dit=635.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=272 step=28620 loss=0.495767 loss_full=0.203638 loss_base=0.292129 ms_per_step=3298.42
task=ci2v epoch=272 step=28640 loss=0.477422 loss_full=0.192441 loss_base=0.284981 ms_per_step=3318.73
task=ci2v epoch=272 step=28660 loss=0.596565 loss_full=0.266791 loss_base=0.329774 ms_per_step=3325.22
[epoch-end] epoch=272 elapsed=352.58s step=28665
[epoch-start] epoch=273 step=28665 steps_per_epoch=105
task=ci2v epoch=273 step=28680 loss=0.504601 loss_full=0.215329 loss_base=0.289272 ms_per_step=3357.29
task=ci2v epoch=273 step=28700 loss=0.508818 loss_full=0.215028 loss_base=0.293790 ms_per_step=3330.58
[profile] window=332.88s updates=100 data_wait=1.9ms h2d=0.8ms stage1=189.5ms dit=633.9ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=273 step=28720 loss=0.501899 loss_full=0.213918 loss_base=0.287982 ms_per_step=3327.65
task=ci2v epoch=273 step=28740 loss=0.551900 loss_full=0.231368 loss_base=0.320532 ms_per_step=3328.60
task=ci2v epoch=273 step=28760 loss=0.539247 loss_full=0.235496 loss_base=0.303751 ms_per_step=3327.14
[epoch-end] epoch=273 elapsed=350.81s step=28770
[epoch-start] epoch=274 step=28770 steps_per_epoch=105
task=ci2v epoch=274 step=28780 loss=0.552895 loss_full=0.238222 loss_base=0.314673 ms_per_step=3362.28
task=ci2v epoch=274 step=28800 loss=0.534613 loss_full=0.224536 loss_base=0.310078 ms_per_step=3323.70
[profile] window=333.66s updates=100 data_wait=1.6ms h2d=0.8ms stage1=191.3ms dit=634.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=274 step=28820 loss=0.534972 loss_full=0.229661 loss_base=0.305311 ms_per_step=3376.68
task=ci2v epoch=274 step=28840 loss=0.468498 loss_full=0.195018 loss_base=0.273480 ms_per_step=3313.78
task=ci2v epoch=274 step=28860 loss=0.472964 loss_full=0.193603 loss_base=0.279361 ms_per_step=3317.13
[epoch-end] epoch=274 elapsed=350.36s step=28875
[epoch-start] epoch=275 step=28875 steps_per_epoch=105
task=ci2v epoch=275 step=28880 loss=0.511553 loss_full=0.210777 loss_base=0.300776 ms_per_step=3339.24
task=ci2v epoch=275 step=28900 loss=0.443783 loss_full=0.178453 loss_base=0.265330 ms_per_step=3302.20
[profile] window=333.25s updates=100 data_wait=1.6ms h2d=0.8ms stage1=189.9ms dit=634.6ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=275 step=28920 loss=0.539283 loss_full=0.238024 loss_base=0.301258 ms_per_step=3316.76
task=ci2v epoch=275 step=28940 loss=0.552299 loss_full=0.236668 loss_base=0.315631 ms_per_step=3321.44
task=ci2v epoch=275 step=28960 loss=0.576341 loss_full=0.242776 loss_base=0.333565 ms_per_step=3311.79
task=ci2v epoch=275 step=28980 loss=0.430323 loss_full=0.174483 loss_base=0.255840 ms_per_step=3305.62
[epoch-end] epoch=275 elapsed=348.93s step=28980
[epoch-start] epoch=276 step=28980 steps_per_epoch=105
task=ci2v epoch=276 step=29000 loss=0.456494 loss_full=0.181728 loss_base=0.274766 ms_per_step=3353.27
[profile] window=332.45s updates=100 data_wait=1.7ms h2d=0.8ms stage1=188.5ms dit=634.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=29000 elapsed=2.52s
[checkpoint] step=29000 elapsed=19.44s
task=ci2v epoch=276 step=29020 loss=0.557772 loss_full=0.243468 loss_base=0.314304 ms_per_step=3306.33
task=ci2v epoch=276 step=29040 loss=0.494239 loss_full=0.201348 loss_base=0.292891 ms_per_step=3381.09
task=ci2v epoch=276 step=29060 loss=0.530967 loss_full=0.226727 loss_base=0.304240 ms_per_step=3437.68
task=ci2v epoch=276 step=29080 loss=0.552372 loss_full=0.237708 loss_base=0.314664 ms_per_step=3421.32
[epoch-end] epoch=276 elapsed=376.59s step=29085
[epoch-start] epoch=277 step=29085 steps_per_epoch=105
task=ci2v epoch=277 step=29100 loss=0.515661 loss_full=0.213701 loss_base=0.301961 ms_per_step=3349.50
[profile] window=338.19s updates=100 data_wait=1.6ms h2d=0.9ms stage1=201.3ms dit=635.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=277 step=29120 loss=0.521523 loss_full=0.213019 loss_base=0.308504 ms_per_step=3312.57
task=ci2v epoch=277 step=29140 loss=0.505636 loss_full=0.211321 loss_base=0.294315 ms_per_step=3335.57
task=ci2v epoch=277 step=29160 loss=0.521375 loss_full=0.214864 loss_base=0.306511 ms_per_step=3347.94
task=ci2v epoch=277 step=29180 loss=0.573221 loss_full=0.247010 loss_base=0.326211 ms_per_step=3330.43
[epoch-end] epoch=277 elapsed=350.59s step=29190
[epoch-start] epoch=278 step=29190 steps_per_epoch=105
task=ci2v epoch=278 step=29200 loss=0.536830 loss_full=0.221222 loss_base=0.315608 ms_per_step=3338.25
[profile] window=333.57s updates=100 data_wait=1.6ms h2d=0.8ms stage1=191.0ms dit=634.3ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=278 step=29220 loss=0.638335 loss_full=0.284410 loss_base=0.353924 ms_per_step=3334.03
task=ci2v epoch=278 step=29240 loss=0.447710 loss_full=0.188169 loss_base=0.259540 ms_per_step=3324.30
task=ci2v epoch=278 step=29260 loss=0.524903 loss_full=0.220072 loss_base=0.304831 ms_per_step=3331.65
task=ci2v epoch=278 step=29280 loss=0.525296 loss_full=0.215430 loss_base=0.309866 ms_per_step=3310.47
[epoch-end] epoch=278 elapsed=349.77s step=29295
[epoch-start] epoch=279 step=29295 steps_per_epoch=105
task=ci2v epoch=279 step=29300 loss=0.530744 loss_full=0.224005 loss_base=0.306739 ms_per_step=3348.26
[profile] window=333.25s updates=100 data_wait=1.7ms h2d=0.8ms stage1=190.6ms dit=633.9ms update=28.0ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=279 step=29320 loss=0.585969 loss_full=0.255197 loss_base=0.330772 ms_per_step=3337.62
task=ci2v epoch=279 step=29340 loss=0.541200 loss_full=0.232138 loss_base=0.309062 ms_per_step=3337.92
task=ci2v epoch=279 step=29360 loss=0.601531 loss_full=0.268694 loss_base=0.332837 ms_per_step=3390.74
task=ci2v epoch=279 step=29380 loss=0.471241 loss_full=0.193516 loss_base=0.277725 ms_per_step=3317.82
task=ci2v epoch=279 step=29400 loss=0.464045 loss_full=0.188946 loss_base=0.275099 ms_per_step=3304.27
[profile] window=334.04s updates=100 data_wait=0.9ms h2d=0.9ms stage1=192.8ms dit=634.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=279 elapsed=351.65s step=29400
[epoch-start] epoch=280 step=29400 steps_per_epoch=105
task=ci2v epoch=280 step=29420 loss=0.459954 loss_full=0.189628 loss_base=0.270326 ms_per_step=3386.95
task=ci2v epoch=280 step=29440 loss=0.609770 loss_full=0.280062 loss_base=0.329708 ms_per_step=3328.22
task=ci2v epoch=280 step=29460 loss=0.517955 loss_full=0.218815 loss_base=0.299140 ms_per_step=3433.68
task=ci2v epoch=280 step=29480 loss=0.510002 loss_full=0.217146 loss_base=0.292857 ms_per_step=3439.18
task=ci2v epoch=280 step=29500 loss=0.546008 loss_full=0.227746 loss_base=0.318262 ms_per_step=3371.46
[profile] window=339.46s updates=100 data_wait=1.5ms h2d=0.9ms stage1=203.2ms dit=636.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=280 elapsed=355.94s step=29505
[epoch-start] epoch=281 step=29505 steps_per_epoch=105
task=ci2v epoch=281 step=29520 loss=0.592984 loss_full=0.257554 loss_base=0.335430 ms_per_step=3386.90
task=ci2v epoch=281 step=29540 loss=0.681485 loss_full=0.309506 loss_base=0.371978 ms_per_step=3313.32
task=ci2v epoch=281 step=29560 loss=0.486910 loss_full=0.196164 loss_base=0.290746 ms_per_step=3321.98
task=ci2v epoch=281 step=29580 loss=0.588685 loss_full=0.257271 loss_base=0.331414 ms_per_step=3315.45
task=ci2v epoch=281 step=29600 loss=0.495448 loss_full=0.199931 loss_base=0.295517 ms_per_step=3373.58
[profile] window=334.50s updates=100 data_wait=1.5ms h2d=0.8ms stage1=193.5ms dit=634.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=281 elapsed=351.51s step=29610
[epoch-start] epoch=282 step=29610 steps_per_epoch=105
task=ci2v epoch=282 step=29620 loss=0.591456 loss_full=0.259122 loss_base=0.332334 ms_per_step=3361.44
task=ci2v epoch=282 step=29640 loss=0.541209 loss_full=0.234915 loss_base=0.306294 ms_per_step=3326.83
task=ci2v epoch=282 step=29660 loss=0.476609 loss_full=0.190993 loss_base=0.285616 ms_per_step=3380.91
task=ci2v epoch=282 step=29680 loss=0.511949 loss_full=0.215173 loss_base=0.296777 ms_per_step=3336.89
task=ci2v epoch=282 step=29700 loss=0.502848 loss_full=0.211126 loss_base=0.291722 ms_per_step=3333.15
[profile] window=335.06s updates=100 data_wait=1.6ms h2d=0.8ms stage1=192.3ms dit=636.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=282 elapsed=351.35s step=29715
[epoch-start] epoch=283 step=29715 steps_per_epoch=105
task=ci2v epoch=283 step=29720 loss=0.429020 loss_full=0.167115 loss_base=0.261905 ms_per_step=3338.82
task=ci2v epoch=283 step=29740 loss=0.519812 loss_full=0.220641 loss_base=0.299171 ms_per_step=3333.57
task=ci2v epoch=283 step=29760 loss=0.558673 loss_full=0.238663 loss_base=0.320009 ms_per_step=3329.45
task=ci2v epoch=283 step=29780 loss=0.512422 loss_full=0.214623 loss_base=0.297798 ms_per_step=3377.27
task=ci2v epoch=283 step=29800 loss=0.578672 loss_full=0.241694 loss_base=0.336978 ms_per_step=3344.42
[profile] window=334.75s updates=100 data_wait=1.6ms h2d=0.8ms stage1=192.1ms dit=636.3ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=283 step=29820 loss=0.526115 loss_full=0.231906 loss_base=0.294209 ms_per_step=3311.87
[epoch-end] epoch=283 elapsed=351.68s step=29820
[epoch-start] epoch=284 step=29820 steps_per_epoch=105
task=ci2v epoch=284 step=29840 loss=0.565986 loss_full=0.245619 loss_base=0.320367 ms_per_step=3545.50
task=ci2v epoch=284 step=29860 loss=0.490887 loss_full=0.181507 loss_base=0.309380 ms_per_step=3438.54
task=ci2v epoch=284 step=29880 loss=0.558118 loss_full=0.243797 loss_base=0.314321 ms_per_step=3418.57
task=ci2v epoch=284 step=29900 loss=0.574088 loss_full=0.247827 loss_base=0.326262 ms_per_step=3361.50
[profile] window=341.79s updates=100 data_wait=1.6ms h2d=1.0ms stage1=209.2ms dit=636.3ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=284 step=29920 loss=0.520573 loss_full=0.220156 loss_base=0.300417 ms_per_step=3400.51
[epoch-end] epoch=284 elapsed=360.07s step=29925
[epoch-start] epoch=285 step=29925 steps_per_epoch=105
task=ci2v epoch=285 step=29940 loss=0.480763 loss_full=0.200615 loss_base=0.280148 ms_per_step=3337.95
task=ci2v epoch=285 step=29960 loss=0.638717 loss_full=0.283023 loss_base=0.355695 ms_per_step=3339.32
task=ci2v epoch=285 step=29980 loss=0.614292 loss_full=0.266661 loss_base=0.347631 ms_per_step=3441.56
task=ci2v epoch=285 step=30000 loss=0.498295 loss_full=0.204124 loss_base=0.294170 ms_per_step=3390.09
[profile] window=338.46s updates=100 data_wait=1.6ms h2d=0.9ms stage1=202.4ms dit=635.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=30000 elapsed=2.28s
[checkpoint] step=30000 elapsed=24.79s
task=ci2v epoch=285 step=30020 loss=0.493053 loss_full=0.200935 loss_base=0.292119 ms_per_step=3317.97
[epoch-end] epoch=285 elapsed=380.66s step=30030
[epoch-start] epoch=286 step=30030 steps_per_epoch=105
task=ci2v epoch=286 step=30040 loss=0.531793 loss_full=0.220495 loss_base=0.311298 ms_per_step=3356.55
task=ci2v epoch=286 step=30060 loss=0.505541 loss_full=0.210604 loss_base=0.294937 ms_per_step=3311.38
task=ci2v epoch=286 step=30080 loss=0.541308 loss_full=0.235672 loss_base=0.305635 ms_per_step=3418.50
task=ci2v epoch=286 step=30100 loss=0.517459 loss_full=0.218373 loss_base=0.299086 ms_per_step=3310.26
[profile] window=334.57s updates=100 data_wait=1.5ms h2d=0.8ms stage1=193.1ms dit=634.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=286 step=30120 loss=0.496447 loss_full=0.202409 loss_base=0.294038 ms_per_step=3324.01
[epoch-end] epoch=286 elapsed=351.20s step=30135
[epoch-start] epoch=287 step=30135 steps_per_epoch=105
task=ci2v epoch=287 step=30140 loss=0.558827 loss_full=0.244910 loss_base=0.313917 ms_per_step=3342.97
task=ci2v epoch=287 step=30160 loss=0.464854 loss_full=0.191090 loss_base=0.273764 ms_per_step=3326.76
task=ci2v epoch=287 step=30180 loss=0.530924 loss_full=0.226089 loss_base=0.304835 ms_per_step=3320.61
task=ci2v epoch=287 step=30200 loss=0.614419 loss_full=0.272011 loss_base=0.342408 ms_per_step=3492.00
[profile] window=336.39s updates=100 data_wait=1.5ms h2d=0.9ms stage1=197.8ms dit=634.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=287 step=30220 loss=0.498300 loss_full=0.206926 loss_base=0.291374 ms_per_step=3376.38
task=ci2v epoch=287 step=30240 loss=0.512461 loss_full=0.211199 loss_base=0.301262 ms_per_step=3339.73
[epoch-end] epoch=287 elapsed=354.96s step=30240
[epoch-start] epoch=288 step=30240 steps_per_epoch=105
task=ci2v epoch=288 step=30260 loss=0.637838 loss_full=0.280352 loss_base=0.357486 ms_per_step=3359.37
task=ci2v epoch=288 step=30280 loss=0.554266 loss_full=0.237794 loss_base=0.316473 ms_per_step=3341.05
task=ci2v epoch=288 step=30300 loss=0.488707 loss_full=0.212132 loss_base=0.276575 ms_per_step=3318.36
[profile] window=334.96s updates=100 data_wait=1.6ms h2d=0.8ms stage1=193.8ms dit=634.9ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=288 step=30320 loss=0.525781 loss_full=0.227939 loss_base=0.297842 ms_per_step=3402.12
task=ci2v epoch=288 step=30340 loss=0.502963 loss_full=0.205504 loss_base=0.297459 ms_per_step=3322.14
[epoch-end] epoch=288 elapsed=351.81s step=30345
[epoch-start] epoch=289 step=30345 steps_per_epoch=105
task=ci2v epoch=289 step=30360 loss=0.463397 loss_full=0.181299 loss_base=0.282098 ms_per_step=3420.46
task=ci2v epoch=289 step=30380 loss=0.467268 loss_full=0.193639 loss_base=0.273629 ms_per_step=3315.87
task=ci2v epoch=289 step=30400 loss=0.474388 loss_full=0.189607 loss_base=0.284781 ms_per_step=3405.27
[profile] window=337.59s updates=100 data_wait=1.7ms h2d=0.9ms stage1=200.5ms dit=634.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=289 step=30420 loss=0.467275 loss_full=0.199437 loss_base=0.267838 ms_per_step=3367.01
task=ci2v epoch=289 step=30440 loss=0.465529 loss_full=0.182954 loss_base=0.282575 ms_per_step=3339.21
[epoch-end] epoch=289 elapsed=354.07s step=30450
[epoch-start] epoch=290 step=30450 steps_per_epoch=105
task=ci2v epoch=290 step=30460 loss=0.602056 loss_full=0.262182 loss_base=0.339874 ms_per_step=3411.16
task=ci2v epoch=290 step=30480 loss=0.560256 loss_full=0.245881 loss_base=0.314375 ms_per_step=3366.35
task=ci2v epoch=290 step=30500 loss=0.538420 loss_full=0.232127 loss_base=0.306294 ms_per_step=3301.18
[profile] window=335.97s updates=100 data_wait=1.4ms h2d=0.9ms stage1=195.9ms dit=635.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=290 step=30520 loss=0.500051 loss_full=0.202546 loss_base=0.297505 ms_per_step=3426.56
task=ci2v epoch=290 step=30540 loss=0.549271 loss_full=0.240068 loss_base=0.309203 ms_per_step=3343.86
[epoch-end] epoch=290 elapsed=354.08s step=30555
[epoch-start] epoch=291 step=30555 steps_per_epoch=105
task=ci2v epoch=291 step=30560 loss=0.523418 loss_full=0.219850 loss_base=0.303567 ms_per_step=3349.32
task=ci2v epoch=291 step=30580 loss=0.544930 loss_full=0.239820 loss_base=0.305110 ms_per_step=3307.25
task=ci2v epoch=291 step=30600 loss=0.544604 loss_full=0.238589 loss_base=0.306015 ms_per_step=3321.91
[profile] window=335.25s updates=100 data_wait=1.4ms h2d=0.9ms stage1=194.8ms dit=634.9ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=291 step=30620 loss=0.615285 loss_full=0.265648 loss_base=0.349637 ms_per_step=3384.92
task=ci2v epoch=291 step=30640 loss=0.591714 loss_full=0.253060 loss_base=0.338654 ms_per_step=3363.16
task=ci2v epoch=291 step=30660 loss=0.495090 loss_full=0.199454 loss_base=0.295636 ms_per_step=3318.01
[epoch-end] epoch=291 elapsed=351.34s step=30660
[epoch-start] epoch=292 step=30660 steps_per_epoch=105
task=ci2v epoch=292 step=30680 loss=0.458742 loss_full=0.183407 loss_base=0.275336 ms_per_step=3395.91
task=ci2v epoch=292 step=30700 loss=0.697829 loss_full=0.331127 loss_base=0.366703 ms_per_step=3301.13
[profile] window=335.53s updates=100 data_wait=1.7ms h2d=0.8ms stage1=194.5ms dit=635.6ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=292 step=30720 loss=0.564237 loss_full=0.221519 loss_base=0.342718 ms_per_step=3329.92
task=ci2v epoch=292 step=30740 loss=0.560666 loss_full=0.242326 loss_base=0.318340 ms_per_step=3314.81
task=ci2v epoch=292 step=30760 loss=0.576419 loss_full=0.253298 loss_base=0.323121 ms_per_step=3310.15
[epoch-end] epoch=292 elapsed=349.86s step=30765
[epoch-start] epoch=293 step=30765 steps_per_epoch=105
task=ci2v epoch=293 step=30780 loss=0.512066 loss_full=0.211583 loss_base=0.300482 ms_per_step=3350.62
task=ci2v epoch=293 step=30800 loss=0.508006 loss_full=0.217390 loss_base=0.290617 ms_per_step=3455.20
[profile] window=335.48s updates=100 data_wait=1.5ms h2d=0.9ms stage1=195.5ms dit=634.6ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=293 step=30820 loss=0.482471 loss_full=0.203789 loss_base=0.278683 ms_per_step=3364.57
task=ci2v epoch=293 step=30840 loss=0.588410 loss_full=0.269336 loss_base=0.319074 ms_per_step=3348.25
task=ci2v epoch=293 step=30860 loss=0.485607 loss_full=0.195993 loss_base=0.289613 ms_per_step=3333.52
[epoch-end] epoch=293 elapsed=354.17s step=30870
[epoch-start] epoch=294 step=30870 steps_per_epoch=105
task=ci2v epoch=294 step=30880 loss=0.555034 loss_full=0.235030 loss_base=0.320005 ms_per_step=3388.65
task=ci2v epoch=294 step=30900 loss=0.557367 loss_full=0.235727 loss_base=0.321641 ms_per_step=3305.99
[profile] window=335.09s updates=100 data_wait=1.7ms h2d=0.8ms stage1=194.6ms dit=634.6ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=294 step=30920 loss=0.641259 loss_full=0.281962 loss_base=0.359297 ms_per_step=3315.47
task=ci2v epoch=294 step=30940 loss=0.457002 loss_full=0.190963 loss_base=0.266040 ms_per_step=3319.54
task=ci2v epoch=294 step=30960 loss=0.431237 loss_full=0.175214 loss_base=0.256023 ms_per_step=3352.64
[epoch-end] epoch=294 elapsed=353.97s step=30975
[epoch-start] epoch=295 step=30975 steps_per_epoch=105
task=ci2v epoch=295 step=30980 loss=0.506339 loss_full=0.209376 loss_base=0.296962 ms_per_step=3585.48
task=ci2v epoch=295 step=31000 loss=0.488782 loss_full=0.205657 loss_base=0.283125 ms_per_step=3345.24
[profile] window=338.64s updates=100 data_wait=1.8ms h2d=0.9ms stage1=202.3ms dit=635.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=31000 elapsed=2.54s
[checkpoint] step=31000 elapsed=29.79s
task=ci2v epoch=295 step=31020 loss=0.475794 loss_full=0.198601 loss_base=0.277193 ms_per_step=3598.10
task=ci2v epoch=295 step=31040 loss=0.497956 loss_full=0.220213 loss_base=0.277743 ms_per_step=3334.03
task=ci2v epoch=295 step=31060 loss=0.475941 loss_full=0.198854 loss_base=0.277087 ms_per_step=3319.89
task=ci2v epoch=295 step=31080 loss=0.500089 loss_full=0.207317 loss_base=0.292772 ms_per_step=3493.88
[epoch-end] epoch=295 elapsed=393.19s step=31080
[epoch-start] epoch=296 step=31080 steps_per_epoch=105
task=ci2v epoch=296 step=31100 loss=0.518508 loss_full=0.211769 loss_base=0.306739 ms_per_step=3527.63
[profile] window=345.74s updates=100 data_wait=1.7ms h2d=1.0ms stage1=218.0ms dit=637.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=296 step=31120 loss=0.506622 loss_full=0.212149 loss_base=0.294473 ms_per_step=3325.45
task=ci2v epoch=296 step=31140 loss=0.538454 loss_full=0.224426 loss_base=0.314028 ms_per_step=3481.35
task=ci2v epoch=296 step=31160 loss=0.487184 loss_full=0.198583 loss_base=0.288602 ms_per_step=3800.42
task=ci2v epoch=296 step=31180 loss=0.493941 loss_full=0.196768 loss_base=0.297173 ms_per_step=3363.93
[epoch-end] epoch=296 elapsed=366.90s step=31185
[epoch-start] epoch=297 step=31185 steps_per_epoch=105
task=ci2v epoch=297 step=31200 loss=0.585195 loss_full=0.260898 loss_base=0.324298 ms_per_step=3368.81
[profile] window=347.05s updates=100 data_wait=1.9ms h2d=1.0ms stage1=217.0ms dit=641.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=297 step=31220 loss=0.548327 loss_full=0.227021 loss_base=0.321306 ms_per_step=3355.91
task=ci2v epoch=297 step=31240 loss=0.471401 loss_full=0.194110 loss_base=0.277291 ms_per_step=3532.58
task=ci2v epoch=297 step=31260 loss=0.551141 loss_full=0.242428 loss_base=0.308712 ms_per_step=3420.26
task=ci2v epoch=297 step=31280 loss=0.483524 loss_full=0.204217 loss_base=0.279307 ms_per_step=3425.30
[epoch-end] epoch=297 elapsed=359.17s step=31290
[epoch-start] epoch=298 step=31290 steps_per_epoch=105
task=ci2v epoch=298 step=31300 loss=0.549151 loss_full=0.228802 loss_base=0.320349 ms_per_step=3577.70
[profile] window=346.49s updates=100 data_wait=1.6ms h2d=1.0ms stage1=217.5ms dit=639.9ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=298 step=31320 loss=0.477097 loss_full=0.196765 loss_base=0.280331 ms_per_step=3413.67
task=ci2v epoch=298 step=31340 loss=0.553929 loss_full=0.246519 loss_base=0.307410 ms_per_step=3653.62
task=ci2v epoch=298 step=31360 loss=0.473829 loss_full=0.195062 loss_base=0.278767 ms_per_step=3852.49
task=ci2v epoch=298 step=31380 loss=0.482213 loss_full=0.198253 loss_base=0.283960 ms_per_step=3778.40
[epoch-end] epoch=298 elapsed=388.18s step=31395
[epoch-start] epoch=299 step=31395 steps_per_epoch=105
task=ci2v epoch=299 step=31400 loss=0.462312 loss_full=0.193538 loss_base=0.268773 ms_per_step=3715.59
[profile] window=368.53s updates=100 data_wait=2.2ms h2d=1.4ms stage1=261.8ms dit=649.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=299 step=31420 loss=0.538209 loss_full=0.234453 loss_base=0.303757 ms_per_step=3538.64
task=ci2v epoch=299 step=31440 loss=0.544142 loss_full=0.231703 loss_base=0.312440 ms_per_step=3510.52
task=ci2v epoch=299 step=31460 loss=0.453716 loss_full=0.181517 loss_base=0.272200 ms_per_step=3763.92
task=ci2v epoch=299 step=31480 loss=0.633564 loss_full=0.283514 loss_base=0.350050 ms_per_step=3491.70
task=ci2v epoch=299 step=31500 loss=0.565690 loss_full=0.240262 loss_base=0.325428 ms_per_step=3391.84
[profile] window=354.19s updates=100 data_wait=1.1ms h2d=1.2ms stage1=232.1ms dit=645.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=299 elapsed=373.20s step=31500
[epoch-start] epoch=300 step=31500 steps_per_epoch=105
task=ci2v epoch=300 step=31520 loss=0.575090 loss_full=0.244612 loss_base=0.330478 ms_per_step=3390.95
task=ci2v epoch=300 step=31540 loss=0.550443 loss_full=0.233382 loss_base=0.317061 ms_per_step=3468.57
task=ci2v epoch=300 step=31560 loss=0.677850 loss_full=0.309307 loss_base=0.368543 ms_per_step=3755.03
task=ci2v epoch=300 step=31580 loss=0.561919 loss_full=0.242991 loss_base=0.318929 ms_per_step=3414.02
task=ci2v epoch=300 step=31600 loss=0.521421 loss_full=0.222068 loss_base=0.299353 ms_per_step=3330.53
[profile] window=347.45s updates=100 data_wait=1.7ms h2d=1.0ms stage1=218.0ms dit=641.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=300 elapsed=363.89s step=31605
[epoch-start] epoch=301 step=31605 steps_per_epoch=105
task=ci2v epoch=301 step=31620 loss=0.516703 loss_full=0.220119 loss_base=0.296584 ms_per_step=3344.28
task=ci2v epoch=301 step=31640 loss=0.525101 loss_full=0.227369 loss_base=0.297732 ms_per_step=3431.08
task=ci2v epoch=301 step=31660 loss=0.505268 loss_full=0.208071 loss_base=0.297197 ms_per_step=3339.72
task=ci2v epoch=301 step=31680 loss=0.518926 loss_full=0.222396 loss_base=0.296530 ms_per_step=3601.74
task=ci2v epoch=301 step=31700 loss=0.624269 loss_full=0.280808 loss_base=0.343461 ms_per_step=3949.63
[profile] window=353.59s updates=100 data_wait=1.8ms h2d=1.1ms stage1=231.8ms dit=643.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=301 elapsed=375.62s step=31710
[epoch-start] epoch=302 step=31710 steps_per_epoch=105
task=ci2v epoch=302 step=31720 loss=0.554516 loss_full=0.241965 loss_base=0.312551 ms_per_step=3935.01
task=ci2v epoch=302 step=31740 loss=0.586666 loss_full=0.254376 loss_base=0.332290 ms_per_step=3759.29
task=ci2v epoch=302 step=31760 loss=0.513202 loss_full=0.223806 loss_base=0.289396 ms_per_step=3347.60
task=ci2v epoch=302 step=31780 loss=0.590059 loss_full=0.260975 loss_base=0.329084 ms_per_step=3342.64
task=ci2v epoch=302 step=31800 loss=0.505499 loss_full=0.211287 loss_base=0.294212 ms_per_step=3343.22
[profile] window=354.90s updates=100 data_wait=2.1ms h2d=1.1ms stage1=233.5ms dit=644.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=302 elapsed=366.53s step=31815
[epoch-start] epoch=303 step=31815 steps_per_epoch=105
task=ci2v epoch=303 step=31820 loss=0.549412 loss_full=0.244292 loss_base=0.305120 ms_per_step=3356.59
task=ci2v epoch=303 step=31840 loss=0.640898 loss_full=0.273778 loss_base=0.367120 ms_per_step=3342.79
task=ci2v epoch=303 step=31860 loss=0.600315 loss_full=0.259966 loss_base=0.340349 ms_per_step=3313.80
task=ci2v epoch=303 step=31880 loss=0.485142 loss_full=0.200085 loss_base=0.285057 ms_per_step=3357.25
task=ci2v epoch=303 step=31900 loss=0.502059 loss_full=0.212094 loss_base=0.289966 ms_per_step=3553.71
[profile] window=338.74s updates=100 data_wait=1.6ms h2d=0.9ms stage1=201.1ms dit=637.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=303 step=31920 loss=0.550232 loss_full=0.237983 loss_base=0.312249 ms_per_step=3424.23
[epoch-end] epoch=303 elapsed=357.63s step=31920
[epoch-start] epoch=304 step=31920 steps_per_epoch=105
task=ci2v epoch=304 step=31940 loss=0.477172 loss_full=0.191596 loss_base=0.285576 ms_per_step=3462.79
task=ci2v epoch=304 step=31960 loss=0.478117 loss_full=0.193689 loss_base=0.284428 ms_per_step=3806.16
task=ci2v epoch=304 step=31980 loss=0.570908 loss_full=0.259408 loss_base=0.311500 ms_per_step=3799.82
task=ci2v epoch=304 step=32000 loss=0.464388 loss_full=0.196518 loss_base=0.267870 ms_per_step=3399.47
[profile] window=358.11s updates=100 data_wait=1.8ms h2d=1.2ms stage1=243.4ms dit=642.6ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=32000 elapsed=2.74s
[checkpoint] step=32000 elapsed=25.87s
task=ci2v epoch=304 step=32020 loss=0.517024 loss_full=0.228287 loss_base=0.288737 ms_per_step=3432.41
[epoch-end] epoch=304 elapsed=403.31s step=32025
[epoch-start] epoch=305 step=32025 steps_per_epoch=105
task=ci2v epoch=305 step=32040 loss=0.463848 loss_full=0.198480 loss_base=0.265368 ms_per_step=3460.83
task=ci2v epoch=305 step=32060 loss=0.547042 loss_full=0.238341 loss_base=0.308701 ms_per_step=3333.97
task=ci2v epoch=305 step=32080 loss=0.512274 loss_full=0.205965 loss_base=0.306309 ms_per_step=3324.10
task=ci2v epoch=305 step=32100 loss=0.485707 loss_full=0.199949 loss_base=0.285758 ms_per_step=3386.04
[profile] window=339.02s updates=100 data_wait=1.7ms h2d=0.9ms stage1=204.1ms dit=634.8ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=305 step=32120 loss=0.455691 loss_full=0.179064 loss_base=0.276627 ms_per_step=3436.48
[epoch-end] epoch=305 elapsed=357.68s step=32130
[epoch-start] epoch=306 step=32130 steps_per_epoch=105
task=ci2v epoch=306 step=32140 loss=0.490577 loss_full=0.199645 loss_base=0.290932 ms_per_step=3704.85
task=ci2v epoch=306 step=32160 loss=0.621212 loss_full=0.268601 loss_base=0.352612 ms_per_step=3640.86
task=ci2v epoch=306 step=32180 loss=0.452628 loss_full=0.180564 loss_base=0.272064 ms_per_step=3789.40
task=ci2v epoch=306 step=32200 loss=0.473869 loss_full=0.199138 loss_base=0.274732 ms_per_step=3764.35
[profile] window=367.05s updates=100 data_wait=2.2ms h2d=1.4ms stage1=262.0ms dit=645.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=306 step=32220 loss=0.478919 loss_full=0.193767 loss_base=0.285152 ms_per_step=3511.97
[epoch-end] epoch=306 elapsed=383.76s step=32235
[epoch-start] epoch=307 step=32235 steps_per_epoch=105
task=ci2v epoch=307 step=32240 loss=0.564524 loss_full=0.246199 loss_base=0.318325 ms_per_step=3344.18
task=ci2v epoch=307 step=32260 loss=0.466080 loss_full=0.185193 loss_base=0.280887 ms_per_step=3368.94
task=ci2v epoch=307 step=32280 loss=0.477814 loss_full=0.190383 loss_base=0.287431 ms_per_step=3324.33
task=ci2v epoch=307 step=32300 loss=0.487576 loss_full=0.201474 loss_base=0.286102 ms_per_step=3437.31
[profile] window=340.01s updates=100 data_wait=1.5ms h2d=0.9ms stage1=205.2ms dit=636.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=307 step=32320 loss=0.549548 loss_full=0.237348 loss_base=0.312200 ms_per_step=3686.15
task=ci2v epoch=307 step=32340 loss=0.485457 loss_full=0.197399 loss_base=0.288058 ms_per_step=3485.71
[epoch-end] epoch=307 elapsed=363.59s step=32340
[epoch-start] epoch=308 step=32340 steps_per_epoch=105
task=ci2v epoch=308 step=32360 loss=0.503572 loss_full=0.214176 loss_base=0.289396 ms_per_step=3786.86
task=ci2v epoch=308 step=32380 loss=0.510267 loss_full=0.205613 loss_base=0.304654 ms_per_step=3740.24
task=ci2v epoch=308 step=32400 loss=0.579219 loss_full=0.258764 loss_base=0.320455 ms_per_step=3771.91
[profile] window=369.69s updates=100 data_wait=1.9ms h2d=1.5ms stage1=269.1ms dit=645.3ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=308 step=32420 loss=0.544820 loss_full=0.222298 loss_base=0.322523 ms_per_step=3539.05
task=ci2v epoch=308 step=32440 loss=0.581209 loss_full=0.254039 loss_base=0.327170 ms_per_step=3650.82
[epoch-end] epoch=308 elapsed=387.86s step=32445
[epoch-start] epoch=309 step=32445 steps_per_epoch=105
task=ci2v epoch=309 step=32460 loss=0.484947 loss_full=0.201916 loss_base=0.283031 ms_per_step=3576.08
task=ci2v epoch=309 step=32480 loss=0.594889 loss_full=0.269970 loss_base=0.324919 ms_per_step=3520.42
task=ci2v epoch=309 step=32500 loss=0.446179 loss_full=0.179520 loss_base=0.266659 ms_per_step=3665.80
[profile] window=359.30s updates=100 data_wait=2.2ms h2d=1.3ms stage1=246.7ms dit=641.8ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=309 step=32520 loss=0.472301 loss_full=0.190312 loss_base=0.281989 ms_per_step=3654.24
task=ci2v epoch=309 step=32540 loss=0.571595 loss_full=0.245645 loss_base=0.325950 ms_per_step=3506.37
[epoch-end] epoch=309 elapsed=374.32s step=32550
[epoch-start] epoch=310 step=32550 steps_per_epoch=105
task=ci2v epoch=310 step=32560 loss=0.533959 loss_full=0.222388 loss_base=0.311571 ms_per_step=3478.46
task=ci2v epoch=310 step=32580 loss=0.583478 loss_full=0.254559 loss_base=0.328920 ms_per_step=3802.22
task=ci2v epoch=310 step=32600 loss=0.425760 loss_full=0.172588 loss_base=0.253173 ms_per_step=3462.16
[profile] window=358.32s updates=100 data_wait=1.7ms h2d=1.2ms stage1=244.0ms dit=642.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=310 step=32620 loss=0.480303 loss_full=0.198714 loss_base=0.281589 ms_per_step=3334.42
task=ci2v epoch=310 step=32640 loss=0.512972 loss_full=0.221273 loss_base=0.291699 ms_per_step=3391.47
[epoch-end] epoch=310 elapsed=369.86s step=32655
[epoch-start] epoch=311 step=32655 steps_per_epoch=105
task=ci2v epoch=311 step=32660 loss=0.530747 loss_full=0.217907 loss_base=0.312840 ms_per_step=3620.03
task=ci2v epoch=311 step=32680 loss=0.552192 loss_full=0.243153 loss_base=0.309039 ms_per_step=3794.95
task=ci2v epoch=311 step=32700 loss=0.507647 loss_full=0.216720 loss_base=0.290928 ms_per_step=3736.29
[profile] window=357.81s updates=100 data_wait=1.9ms h2d=1.2ms stage1=243.4ms dit=641.9ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=311 step=32720 loss=0.509494 loss_full=0.218724 loss_base=0.290770 ms_per_step=3856.26
task=ci2v epoch=311 step=32740 loss=0.490949 loss_full=0.210023 loss_base=0.280926 ms_per_step=3761.87
task=ci2v epoch=311 step=32760 loss=0.434274 loss_full=0.171432 loss_base=0.262842 ms_per_step=3668.90
[epoch-end] epoch=311 elapsed=395.84s step=32760
[epoch-start] epoch=312 step=32760 steps_per_epoch=105
task=ci2v epoch=312 step=32780 loss=0.524365 loss_full=0.214972 loss_base=0.309393 ms_per_step=3436.73
task=ci2v epoch=312 step=32800 loss=0.502374 loss_full=0.203689 loss_base=0.298684 ms_per_step=3413.19
[profile] window=363.00s updates=100 data_wait=2.2ms h2d=1.3ms stage1=254.6ms dit=643.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=312 step=32820 loss=0.534119 loss_full=0.226978 loss_base=0.307141 ms_per_step=3459.71
task=ci2v epoch=312 step=32840 loss=0.586490 loss_full=0.259240 loss_base=0.327250 ms_per_step=3512.43
task=ci2v epoch=312 step=32860 loss=0.670626 loss_full=0.304229 loss_base=0.366397 ms_per_step=3357.61
[epoch-end] epoch=312 elapsed=360.47s step=32865
[epoch-start] epoch=313 step=32865 steps_per_epoch=105
task=ci2v epoch=313 step=32880 loss=0.529161 loss_full=0.219187 loss_base=0.309974 ms_per_step=3379.69
task=ci2v epoch=313 step=32900 loss=0.474645 loss_full=0.202129 loss_base=0.272516 ms_per_step=3390.05
[profile] window=342.25s updates=100 data_wait=1.7ms h2d=1.0ms stage1=211.7ms dit=635.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=313 step=32920 loss=0.543234 loss_full=0.232893 loss_base=0.310341 ms_per_step=3426.38
task=ci2v epoch=313 step=32940 loss=0.550577 loss_full=0.232883 loss_base=0.317694 ms_per_step=3315.53
task=ci2v epoch=313 step=32960 loss=0.495297 loss_full=0.207694 loss_base=0.287604 ms_per_step=3594.46
[epoch-end] epoch=313 elapsed=362.74s step=32970
[epoch-start] epoch=314 step=32970 steps_per_epoch=105
task=ci2v epoch=314 step=32980 loss=0.482140 loss_full=0.194378 loss_base=0.287762 ms_per_step=3556.90
task=ci2v epoch=314 step=33000 loss=0.519227 loss_full=0.225373 loss_base=0.293854 ms_per_step=3417.95
[profile] window=346.48s updates=100 data_wait=1.6ms h2d=1.0ms stage1=219.9ms dit=637.6ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=33000 elapsed=2.36s
[checkpoint] step=33000 elapsed=24.35s
task=ci2v epoch=314 step=33020 loss=0.544458 loss_full=0.228745 loss_base=0.315713 ms_per_step=3468.76
task=ci2v epoch=314 step=33040 loss=0.547319 loss_full=0.224590 loss_base=0.322729 ms_per_step=3334.21
task=ci2v epoch=314 step=33060 loss=0.408834 loss_full=0.157267 loss_base=0.251567 ms_per_step=3327.19
[epoch-end] epoch=314 elapsed=381.93s step=33075
[epoch-start] epoch=315 step=33075 steps_per_epoch=105
task=ci2v epoch=315 step=33080 loss=0.548427 loss_full=0.239860 loss_base=0.308566 ms_per_step=3346.44
task=ci2v epoch=315 step=33100 loss=0.473445 loss_full=0.194161 loss_base=0.279284 ms_per_step=3390.89
[profile] window=337.62s updates=100 data_wait=1.9ms h2d=0.9ms stage1=200.4ms dit=634.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=315 step=33120 loss=0.549867 loss_full=0.238370 loss_base=0.311497 ms_per_step=3310.17
task=ci2v epoch=315 step=33140 loss=0.521404 loss_full=0.212579 loss_base=0.308825 ms_per_step=3292.35
task=ci2v epoch=315 step=33160 loss=0.548052 loss_full=0.237792 loss_base=0.310260 ms_per_step=3296.05
task=ci2v epoch=315 step=33180 loss=0.559821 loss_full=0.241507 loss_base=0.318313 ms_per_step=3289.54
[epoch-end] epoch=315 elapsed=349.43s step=33180
[epoch-start] epoch=316 step=33180 steps_per_epoch=105
task=ci2v epoch=316 step=33200 loss=0.493905 loss_full=0.204383 loss_base=0.289522 ms_per_step=3365.98
[profile] window=331.35s updates=100 data_wait=1.6ms h2d=0.8ms stage1=185.9ms dit=633.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=316 step=33220 loss=0.541767 loss_full=0.228432 loss_base=0.313335 ms_per_step=3353.70
task=ci2v epoch=316 step=33240 loss=0.518271 loss_full=0.221633 loss_base=0.296638 ms_per_step=3327.15
task=ci2v epoch=316 step=33260 loss=0.553551 loss_full=0.236662 loss_base=0.316889 ms_per_step=3313.68
task=ci2v epoch=316 step=33280 loss=0.478697 loss_full=0.196810 loss_base=0.281887 ms_per_step=3311.34
[epoch-end] epoch=316 elapsed=350.20s step=33285
[epoch-start] epoch=317 step=33285 steps_per_epoch=105
task=ci2v epoch=317 step=33300 loss=0.465080 loss_full=0.188871 loss_base=0.276209 ms_per_step=3329.30
[profile] window=332.98s updates=100 data_wait=1.7ms h2d=0.8ms stage1=187.9ms dit=635.9ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=317 step=33320 loss=0.629660 loss_full=0.283606 loss_base=0.346054 ms_per_step=3312.51
task=ci2v epoch=317 step=33340 loss=0.486777 loss_full=0.206580 loss_base=0.280197 ms_per_step=3320.38
task=ci2v epoch=317 step=33360 loss=0.525994 loss_full=0.229233 loss_base=0.296761 ms_per_step=3312.29
task=ci2v epoch=317 step=33380 loss=0.561288 loss_full=0.258993 loss_base=0.302295 ms_per_step=3305.08
[epoch-end] epoch=317 elapsed=348.74s step=33390
[epoch-start] epoch=318 step=33390 steps_per_epoch=105
task=ci2v epoch=318 step=33400 loss=0.433952 loss_full=0.176092 loss_base=0.257860 ms_per_step=3362.80
[profile] window=332.53s updates=100 data_wait=1.5ms h2d=0.8ms stage1=187.7ms dit=635.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=318 step=33420 loss=0.482836 loss_full=0.199856 loss_base=0.282981 ms_per_step=3453.41
task=ci2v epoch=318 step=33440 loss=0.464726 loss_full=0.189308 loss_base=0.275418 ms_per_step=3475.67
task=ci2v epoch=318 step=33460 loss=0.519978 loss_full=0.220029 loss_base=0.299949 ms_per_step=3344.48
task=ci2v epoch=318 step=33480 loss=0.478932 loss_full=0.203520 loss_base=0.275412 ms_per_step=3343.83
[epoch-end] epoch=318 elapsed=356.71s step=33495
[epoch-start] epoch=319 step=33495 steps_per_epoch=105
task=ci2v epoch=319 step=33500 loss=0.497768 loss_full=0.201083 loss_base=0.296685 ms_per_step=3433.34
[profile] window=341.27s updates=100 data_wait=2.0ms h2d=0.9ms stage1=206.1ms dit=637.8ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=319 step=33520 loss=0.597110 loss_full=0.264817 loss_base=0.332294 ms_per_step=3363.06
task=ci2v epoch=319 step=33540 loss=0.481194 loss_full=0.196243 loss_base=0.284952 ms_per_step=3307.95
task=ci2v epoch=319 step=33560 loss=0.473703 loss_full=0.190052 loss_base=0.283651 ms_per_step=3330.02
task=ci2v epoch=319 step=33580 loss=0.410942 loss_full=0.162084 loss_base=0.248858 ms_per_step=3333.24
task=ci2v epoch=319 step=33600 loss=0.430528 loss_full=0.169136 loss_base=0.261392 ms_per_step=3305.82
[profile] window=333.07s updates=100 data_wait=0.9ms h2d=0.8ms stage1=188.7ms dit=636.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=319 elapsed=352.06s step=33600
[epoch-start] epoch=320 step=33600 steps_per_epoch=105
task=ci2v epoch=320 step=33620 loss=0.580528 loss_full=0.255332 loss_base=0.325196 ms_per_step=3364.51
task=ci2v epoch=320 step=33640 loss=0.403283 loss_full=0.162852 loss_base=0.240431 ms_per_step=3392.32
task=ci2v epoch=320 step=33660 loss=0.529097 loss_full=0.221679 loss_base=0.307418 ms_per_step=3332.64
task=ci2v epoch=320 step=33680 loss=0.537426 loss_full=0.225795 loss_base=0.311631 ms_per_step=3308.58
task=ci2v epoch=320 step=33700 loss=0.465646 loss_full=0.197428 loss_base=0.268218 ms_per_step=3323.99
[profile] window=334.72s updates=100 data_wait=1.5ms h2d=0.8ms stage1=192.6ms dit=635.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=320 elapsed=351.02s step=33705
[epoch-start] epoch=321 step=33705 steps_per_epoch=105
task=ci2v epoch=321 step=33720 loss=0.496653 loss_full=0.207884 loss_base=0.288768 ms_per_step=3380.35
task=ci2v epoch=321 step=33740 loss=0.517845 loss_full=0.216630 loss_base=0.301215 ms_per_step=3337.34
task=ci2v epoch=321 step=33760 loss=0.494330 loss_full=0.205996 loss_base=0.288334 ms_per_step=3345.47
task=ci2v epoch=321 step=33780 loss=0.542676 loss_full=0.217255 loss_base=0.325421 ms_per_step=3329.50
task=ci2v epoch=321 step=33800 loss=0.491970 loss_full=0.209695 loss_base=0.282275 ms_per_step=3333.95
[profile] window=334.80s updates=100 data_wait=1.5ms h2d=0.8ms stage1=191.1ms dit=637.3ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=321 elapsed=351.80s step=33810
[epoch-start] epoch=322 step=33810 steps_per_epoch=105
task=ci2v epoch=322 step=33820 loss=0.522197 loss_full=0.218739 loss_base=0.303458 ms_per_step=3343.86
task=ci2v epoch=322 step=33840 loss=0.478836 loss_full=0.199538 loss_base=0.279298 ms_per_step=3421.92
task=ci2v epoch=322 step=33860 loss=0.581212 loss_full=0.247290 loss_base=0.333922 ms_per_step=3346.03
task=ci2v epoch=322 step=33880 loss=0.427256 loss_full=0.168793 loss_base=0.258464 ms_per_step=3433.28
task=ci2v epoch=322 step=33900 loss=0.561055 loss_full=0.243759 loss_base=0.317296 ms_per_step=3356.27
[profile] window=338.30s updates=100 data_wait=1.7ms h2d=0.9ms stage1=199.3ms dit=637.6ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=322 elapsed=355.13s step=33915
[epoch-start] epoch=323 step=33915 steps_per_epoch=105
task=ci2v epoch=323 step=33920 loss=0.505552 loss_full=0.210967 loss_base=0.294585 ms_per_step=3369.42
task=ci2v epoch=323 step=33940 loss=0.517295 loss_full=0.217784 loss_base=0.299511 ms_per_step=3339.82
task=ci2v epoch=323 step=33960 loss=0.511579 loss_full=0.216719 loss_base=0.294860 ms_per_step=3454.05
task=ci2v epoch=323 step=33980 loss=0.497820 loss_full=0.206423 loss_base=0.291396 ms_per_step=3326.48
task=ci2v epoch=323 step=34000 loss=0.487258 loss_full=0.200884 loss_base=0.286374 ms_per_step=3585.87
[profile] window=341.78s updates=100 data_wait=1.6ms h2d=1.0ms stage1=206.9ms dit=638.8ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=34000 elapsed=3.05s
[checkpoint] step=34000 elapsed=36.34s
task=ci2v epoch=323 step=34020 loss=0.471525 loss_full=0.198623 loss_base=0.272902 ms_per_step=3835.74
[epoch-end] epoch=323 elapsed=408.32s step=34020
[epoch-start] epoch=324 step=34020 steps_per_epoch=105
task=ci2v epoch=324 step=34040 loss=0.499915 loss_full=0.203688 loss_base=0.296227 ms_per_step=3913.86
task=ci2v epoch=324 step=34060 loss=0.612380 loss_full=0.276659 loss_base=0.335720 ms_per_step=3854.69
task=ci2v epoch=324 step=34080 loss=0.497150 loss_full=0.207876 loss_base=0.289273 ms_per_step=3727.78
task=ci2v epoch=324 step=34100 loss=0.501980 loss_full=0.207049 loss_base=0.294931 ms_per_step=3575.23
[profile] window=378.40s updates=100 data_wait=2.2ms h2d=1.6ms stage1=284.5ms dit=651.5ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=324 step=34120 loss=0.479359 loss_full=0.197733 loss_base=0.281627 ms_per_step=3565.87
[epoch-end] epoch=324 elapsed=389.42s step=34125
[epoch-start] epoch=325 step=34125 steps_per_epoch=105
task=ci2v epoch=325 step=34140 loss=0.447353 loss_full=0.180890 loss_base=0.266463 ms_per_step=3493.83
task=ci2v epoch=325 step=34160 loss=0.402366 loss_full=0.151764 loss_base=0.250602 ms_per_step=3343.25
task=ci2v epoch=325 step=34180 loss=0.523938 loss_full=0.229520 loss_base=0.294418 ms_per_step=3322.71
task=ci2v epoch=325 step=34200 loss=0.562212 loss_full=0.239145 loss_base=0.323067 ms_per_step=3479.03
[profile] window=344.37s updates=100 data_wait=1.7ms h2d=1.0ms stage1=214.8ms dit=637.3ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=325 step=34220 loss=0.478052 loss_full=0.199059 loss_base=0.278993 ms_per_step=3357.68
[epoch-end] epoch=325 elapsed=356.69s step=34230
[epoch-start] epoch=326 step=34230 steps_per_epoch=105
task=ci2v epoch=326 step=34240 loss=0.574674 loss_full=0.249226 loss_base=0.325448 ms_per_step=3348.87
task=ci2v epoch=326 step=34260 loss=0.505966 loss_full=0.218592 loss_base=0.287374 ms_per_step=3302.83
task=ci2v epoch=326 step=34280 loss=0.528474 loss_full=0.223565 loss_base=0.304909 ms_per_step=3327.57
task=ci2v epoch=326 step=34300 loss=0.532085 loss_full=0.220782 loss_base=0.311303 ms_per_step=3315.23
[profile] window=333.31s updates=100 data_wait=1.6ms h2d=0.8ms stage1=190.6ms dit=634.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=326 step=34320 loss=0.466888 loss_full=0.188301 loss_base=0.278587 ms_per_step=3297.03
[epoch-end] epoch=326 elapsed=350.10s step=34335
[epoch-start] epoch=327 step=34335 steps_per_epoch=105
task=ci2v epoch=327 step=34340 loss=0.599338 loss_full=0.259374 loss_base=0.339964 ms_per_step=3412.65
task=ci2v epoch=327 step=34360 loss=0.498061 loss_full=0.211009 loss_base=0.287052 ms_per_step=3380.10
task=ci2v epoch=327 step=34380 loss=0.566785 loss_full=0.251744 loss_base=0.315041 ms_per_step=3355.87
task=ci2v epoch=327 step=34400 loss=0.603532 loss_full=0.259507 loss_base=0.344025 ms_per_step=3326.02
[profile] window=335.70s updates=100 data_wait=1.8ms h2d=0.8ms stage1=193.3ms dit=637.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=327 step=34420 loss=0.667880 loss_full=0.303565 loss_base=0.364315 ms_per_step=3487.47
task=ci2v epoch=327 step=34440 loss=0.477309 loss_full=0.194024 loss_base=0.283285 ms_per_step=3376.82
[epoch-end] epoch=327 elapsed=356.53s step=34440
[epoch-start] epoch=328 step=34440 steps_per_epoch=105
task=ci2v epoch=328 step=34460 loss=0.462608 loss_full=0.180652 loss_base=0.281956 ms_per_step=3381.15
task=ci2v epoch=328 step=34480 loss=0.519182 loss_full=0.217629 loss_base=0.301553 ms_per_step=3326.60
task=ci2v epoch=328 step=34500 loss=0.495520 loss_full=0.206851 loss_base=0.288669 ms_per_step=3322.20
[profile] window=338.15s updates=100 data_wait=1.6ms h2d=0.9ms stage1=198.1ms dit=638.6ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=328 step=34520 loss=0.619190 loss_full=0.288077 loss_base=0.331113 ms_per_step=3466.79
task=ci2v epoch=328 step=34540 loss=0.664929 loss_full=0.302448 loss_base=0.362481 ms_per_step=3371.09
[epoch-end] epoch=328 elapsed=354.05s step=34545
[epoch-start] epoch=329 step=34545 steps_per_epoch=105
task=ci2v epoch=329 step=34560 loss=0.513804 loss_full=0.215745 loss_base=0.298059 ms_per_step=3352.74
task=ci2v epoch=329 step=34580 loss=0.542233 loss_full=0.229759 loss_base=0.312474 ms_per_step=3336.88
task=ci2v epoch=329 step=34600 loss=0.435888 loss_full=0.175313 loss_base=0.260575 ms_per_step=3355.37
[profile] window=337.92s updates=100 data_wait=1.9ms h2d=0.9ms stage1=198.4ms dit=637.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=329 step=34620 loss=0.591792 loss_full=0.257610 loss_base=0.334182 ms_per_step=3372.83
task=ci2v epoch=329 step=34640 loss=0.508321 loss_full=0.208862 loss_base=0.299458 ms_per_step=3351.85
[epoch-end] epoch=329 elapsed=352.46s step=34650
[epoch-start] epoch=330 step=34650 steps_per_epoch=105
task=ci2v epoch=330 step=34660 loss=0.575403 loss_full=0.262088 loss_base=0.313315 ms_per_step=3355.57
task=ci2v epoch=330 step=34680 loss=0.539153 loss_full=0.225257 loss_base=0.313896 ms_per_step=3309.05
task=ci2v epoch=330 step=34700 loss=0.518868 loss_full=0.217740 loss_base=0.301129 ms_per_step=3375.14
[profile] window=335.56s updates=100 data_wait=1.6ms h2d=0.8ms stage1=193.1ms dit=637.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=330 step=34720 loss=0.529193 loss_full=0.224925 loss_base=0.304268 ms_per_step=3356.45
task=ci2v epoch=330 step=34740 loss=0.503796 loss_full=0.215313 loss_base=0.288483 ms_per_step=3315.43
[epoch-end] epoch=330 elapsed=351.17s step=34755
[epoch-start] epoch=331 step=34755 steps_per_epoch=105
task=ci2v epoch=331 step=34760 loss=0.445508 loss_full=0.184996 loss_base=0.260513 ms_per_step=3344.06
task=ci2v epoch=331 step=34780 loss=0.510435 loss_full=0.206208 loss_base=0.304227 ms_per_step=3313.25
task=ci2v epoch=331 step=34800 loss=0.585072 loss_full=0.265303 loss_base=0.319769 ms_per_step=3336.79
[profile] window=333.59s updates=100 data_wait=1.6ms h2d=0.8ms stage1=190.5ms dit=634.9ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=331 step=34820 loss=0.556122 loss_full=0.238615 loss_base=0.317507 ms_per_step=3308.41
task=ci2v epoch=331 step=34840 loss=0.432753 loss_full=0.165387 loss_base=0.267366 ms_per_step=3526.15
task=ci2v epoch=331 step=34860 loss=0.536582 loss_full=0.228913 loss_base=0.307669 ms_per_step=3348.60
[epoch-end] epoch=331 elapsed=354.44s step=34860
[epoch-start] epoch=332 step=34860 steps_per_epoch=105
task=ci2v epoch=332 step=34880 loss=0.652607 loss_full=0.287923 loss_base=0.364684 ms_per_step=3386.26
task=ci2v epoch=332 step=34900 loss=0.589023 loss_full=0.260383 loss_base=0.328640 ms_per_step=3400.93
[profile] window=339.68s updates=100 data_wait=1.4ms h2d=1.0ms stage1=205.7ms dit=634.8ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=332 step=34920 loss=0.451464 loss_full=0.182076 loss_base=0.269388 ms_per_step=3305.06
task=ci2v epoch=332 step=34940 loss=0.425153 loss_full=0.166866 loss_base=0.258287 ms_per_step=3322.81
task=ci2v epoch=332 step=34960 loss=0.505625 loss_full=0.207547 loss_base=0.298079 ms_per_step=3473.73
[epoch-end] epoch=332 elapsed=356.26s step=34965
[epoch-start] epoch=333 step=34965 steps_per_epoch=105
task=ci2v epoch=333 step=34980 loss=0.507535 loss_full=0.205922 loss_base=0.301613 ms_per_step=3464.62
task=ci2v epoch=333 step=35000 loss=0.504929 loss_full=0.209746 loss_base=0.295182 ms_per_step=3519.47
[profile] window=341.98s updates=100 data_wait=1.6ms h2d=1.0ms stage1=209.1ms dit=637.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=35000 elapsed=2.55s
[checkpoint] step=35000 elapsed=19.43s
task=ci2v epoch=333 step=35020 loss=0.516931 loss_full=0.216445 loss_base=0.300486 ms_per_step=3338.01
task=ci2v epoch=333 step=35040 loss=0.497789 loss_full=0.212666 loss_base=0.285123 ms_per_step=3347.86
task=ci2v epoch=333 step=35060 loss=0.448357 loss_full=0.191256 loss_base=0.257101 ms_per_step=3433.97
[epoch-end] epoch=333 elapsed=379.74s step=35070
[epoch-start] epoch=334 step=35070 steps_per_epoch=105
task=ci2v epoch=334 step=35080 loss=0.569906 loss_full=0.247113 loss_base=0.322793 ms_per_step=3359.32
task=ci2v epoch=334 step=35100 loss=0.526423 loss_full=0.222184 loss_base=0.304239 ms_per_step=3512.00
[profile] window=340.10s updates=100 data_wait=1.6ms h2d=0.9ms stage1=203.6ms dit=638.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=334 step=35120 loss=0.426072 loss_full=0.165539 loss_base=0.260534 ms_per_step=3494.17
task=ci2v epoch=334 step=35140 loss=0.441770 loss_full=0.181076 loss_base=0.260694 ms_per_step=3401.38
task=ci2v epoch=334 step=35160 loss=0.574257 loss_full=0.248375 loss_base=0.325881 ms_per_step=3692.61
[epoch-end] epoch=334 elapsed=367.48s step=35175
[epoch-start] epoch=335 step=35175 steps_per_epoch=105
task=ci2v epoch=335 step=35180 loss=0.617449 loss_full=0.275886 loss_base=0.341563 ms_per_step=3421.02
task=ci2v epoch=335 step=35200 loss=0.445315 loss_full=0.188952 loss_base=0.256363 ms_per_step=3317.87
[profile] window=346.81s updates=100 data_wait=1.6ms h2d=1.0ms stage1=218.8ms dit=639.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=335 step=35220 loss=0.496605 loss_full=0.196502 loss_base=0.300103 ms_per_step=3310.15
task=ci2v epoch=335 step=35240 loss=0.660138 loss_full=0.296212 loss_base=0.363925 ms_per_step=3494.80
task=ci2v epoch=335 step=35260 loss=0.620928 loss_full=0.279365 loss_base=0.341563 ms_per_step=3387.57
task=ci2v epoch=335 step=35280 loss=0.553918 loss_full=0.243596 loss_base=0.310322 ms_per_step=3315.77
[epoch-end] epoch=335 elapsed=354.10s step=35280
[epoch-start] epoch=336 step=35280 steps_per_epoch=105
task=ci2v epoch=336 step=35300 loss=0.498844 loss_full=0.218342 loss_base=0.280502 ms_per_step=3503.87
[profile] window=340.53s updates=100 data_wait=1.5ms h2d=1.0ms stage1=206.9ms dit=635.2ms update=29.1ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=336 step=35320 loss=0.512426 loss_full=0.207870 loss_base=0.304556 ms_per_step=3400.29
task=ci2v epoch=336 step=35340 loss=0.514266 loss_full=0.211942 loss_base=0.302325 ms_per_step=3402.60
task=ci2v epoch=336 step=35360 loss=0.536213 loss_full=0.234849 loss_base=0.301364 ms_per_step=3320.94
task=ci2v epoch=336 step=35380 loss=0.425188 loss_full=0.160521 loss_base=0.264667 ms_per_step=3353.86
[epoch-end] epoch=336 elapsed=356.75s step=35385
[epoch-start] epoch=337 step=35385 steps_per_epoch=105
task=ci2v epoch=337 step=35400 loss=0.447074 loss_full=0.186207 loss_base=0.260868 ms_per_step=3388.38
[profile] window=337.59s updates=100 data_wait=1.7ms h2d=0.9ms stage1=199.7ms dit=635.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=337 step=35420 loss=0.614337 loss_full=0.274421 loss_base=0.339915 ms_per_step=3597.08
task=ci2v epoch=337 step=35440 loss=0.517959 loss_full=0.219729 loss_base=0.298229 ms_per_step=3355.41
task=ci2v epoch=337 step=35460 loss=0.462007 loss_full=0.185318 loss_base=0.276689 ms_per_step=3357.13
task=ci2v epoch=337 step=35480 loss=0.546800 loss_full=0.236892 loss_base=0.309908 ms_per_step=3378.18
[epoch-end] epoch=337 elapsed=359.89s step=35490
[epoch-start] epoch=338 step=35490 steps_per_epoch=105
task=ci2v epoch=338 step=35500 loss=0.475484 loss_full=0.191786 loss_base=0.283698 ms_per_step=3456.20
[profile] window=343.15s updates=100 data_wait=1.4ms h2d=1.0ms stage1=212.5ms dit=636.4ms update=29.2ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=338 step=35520 loss=0.491462 loss_full=0.198436 loss_base=0.293027 ms_per_step=3314.44
task=ci2v epoch=338 step=35540 loss=0.493948 loss_full=0.206472 loss_base=0.287477 ms_per_step=3374.91
task=ci2v epoch=338 step=35560 loss=0.530333 loss_full=0.230812 loss_base=0.299521 ms_per_step=3447.38
task=ci2v epoch=338 step=35580 loss=0.494997 loss_full=0.203579 loss_base=0.291418 ms_per_step=3468.29
[epoch-end] epoch=338 elapsed=362.61s step=35595
[epoch-start] epoch=339 step=35595 steps_per_epoch=105
task=ci2v epoch=339 step=35600 loss=0.478867 loss_full=0.196920 loss_base=0.281946 ms_per_step=3813.11
[profile] window=348.63s updates=100 data_wait=2.1ms h2d=1.0ms stage1=224.6ms dit=637.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=339 step=35620 loss=0.585196 loss_full=0.254458 loss_base=0.330739 ms_per_step=3879.19
task=ci2v epoch=339 step=35640 loss=0.566090 loss_full=0.246537 loss_base=0.319553 ms_per_step=3838.24
task=ci2v epoch=339 step=35660 loss=0.406528 loss_full=0.158689 loss_base=0.247839 ms_per_step=3746.97
task=ci2v epoch=339 step=35680 loss=0.483313 loss_full=0.201078 loss_base=0.282235 ms_per_step=3541.64
task=ci2v epoch=339 step=35700 loss=0.468643 loss_full=0.197083 loss_base=0.271560 ms_per_step=3778.71
[profile] window=375.95s updates=100 data_wait=1.2ms h2d=1.5ms stage1=283.8ms dit=647.1ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=339 elapsed=396.69s step=35700
[epoch-start] epoch=340 step=35700 steps_per_epoch=105
task=ci2v epoch=340 step=35720 loss=0.457667 loss_full=0.188596 loss_base=0.269071 ms_per_step=3874.57
task=ci2v epoch=340 step=35740 loss=0.530736 loss_full=0.223227 loss_base=0.307509 ms_per_step=3877.97
task=ci2v epoch=340 step=35760 loss=0.613211 loss_full=0.268537 loss_base=0.344674 ms_per_step=3898.94
task=ci2v epoch=340 step=35780 loss=0.522691 loss_full=0.222310 loss_base=0.300382 ms_per_step=3861.25
task=ci2v epoch=340 step=35800 loss=0.501230 loss_full=0.221645 loss_base=0.279585 ms_per_step=3770.62
[profile] window=385.91s updates=100 data_wait=2.3ms h2d=1.7ms stage1=300.3ms dit=653.6ms update=30.6ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=340 elapsed=402.52s step=35805
[epoch-start] epoch=341 step=35805 steps_per_epoch=105
task=ci2v epoch=341 step=35820 loss=0.543367 loss_full=0.229215 loss_base=0.314152 ms_per_step=3668.44
task=ci2v epoch=341 step=35840 loss=0.454032 loss_full=0.184150 loss_base=0.269882 ms_per_step=3503.42
task=ci2v epoch=341 step=35860 loss=0.495297 loss_full=0.205924 loss_base=0.289373 ms_per_step=3448.25
task=ci2v epoch=341 step=35880 loss=0.517828 loss_full=0.216104 loss_base=0.301724 ms_per_step=3465.60
task=ci2v epoch=341 step=35900 loss=0.569066 loss_full=0.243408 loss_base=0.325658 ms_per_step=3321.30
[profile] window=348.38s updates=100 data_wait=1.8ms h2d=1.1ms stage1=219.9ms dit=642.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=341 elapsed=369.38s step=35910
[epoch-start] epoch=342 step=35910 steps_per_epoch=105
task=ci2v epoch=342 step=35920 loss=0.483604 loss_full=0.196458 loss_base=0.287146 ms_per_step=3812.70
task=ci2v epoch=342 step=35940 loss=0.647190 loss_full=0.292528 loss_base=0.354661 ms_per_step=3782.84
task=ci2v epoch=342 step=35960 loss=0.479848 loss_full=0.200342 loss_base=0.279506 ms_per_step=3620.25
task=ci2v epoch=342 step=35980 loss=0.448176 loss_full=0.179558 loss_base=0.268618 ms_per_step=3594.00
task=ci2v epoch=342 step=36000 loss=0.510845 loss_full=0.215181 loss_base=0.295664 ms_per_step=3353.99
[profile] window=363.54s updates=100 data_wait=2.1ms h2d=1.4ms stage1=252.5ms dit=646.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=36000 elapsed=2.53s
[checkpoint] step=36000 elapsed=24.96s
[epoch-end] epoch=342 elapsed=404.75s step=36015
[epoch-start] epoch=343 step=36015 steps_per_epoch=105
task=ci2v epoch=343 step=36020 loss=0.503213 loss_full=0.206573 loss_base=0.296640 ms_per_step=3428.65
task=ci2v epoch=343 step=36040 loss=0.445683 loss_full=0.175102 loss_base=0.270581 ms_per_step=3554.23
task=ci2v epoch=343 step=36060 loss=0.571458 loss_full=0.255467 loss_base=0.315991 ms_per_step=3699.65
task=ci2v epoch=343 step=36080 loss=0.524565 loss_full=0.220883 loss_base=0.303682 ms_per_step=3457.91
task=ci2v epoch=343 step=36100 loss=0.477469 loss_full=0.191663 loss_base=0.285805 ms_per_step=3621.39
[profile] window=355.49s updates=100 data_wait=1.9ms h2d=1.2ms stage1=236.3ms dit=643.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=343 step=36120 loss=0.559981 loss_full=0.248278 loss_base=0.311704 ms_per_step=3639.87
[epoch-end] epoch=343 elapsed=377.48s step=36120
[epoch-start] epoch=344 step=36120 steps_per_epoch=105
task=ci2v epoch=344 step=36140 loss=0.436877 loss_full=0.171433 loss_base=0.265444 ms_per_step=3776.30
task=ci2v epoch=344 step=36160 loss=0.674299 loss_full=0.304931 loss_base=0.369369 ms_per_step=3516.75
task=ci2v epoch=344 step=36180 loss=0.588573 loss_full=0.254908 loss_base=0.333665 ms_per_step=3682.17
task=ci2v epoch=344 step=36200 loss=0.498329 loss_full=0.213118 loss_base=0.285212 ms_per_step=3493.10
[profile] window=362.42s updates=100 data_wait=2.0ms h2d=1.3ms stage1=252.3ms dit=644.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=344 step=36220 loss=0.476304 loss_full=0.201790 loss_base=0.274514 ms_per_step=3725.83
[epoch-end] epoch=344 elapsed=382.47s step=36225
[epoch-start] epoch=345 step=36225 steps_per_epoch=105
task=ci2v epoch=345 step=36240 loss=0.471073 loss_full=0.192690 loss_base=0.278383 ms_per_step=3495.21
task=ci2v epoch=345 step=36260 loss=0.569017 loss_full=0.254413 loss_base=0.314604 ms_per_step=3358.15
task=ci2v epoch=345 step=36280 loss=0.427410 loss_full=0.171469 loss_base=0.255942 ms_per_step=3592.96
task=ci2v epoch=345 step=36300 loss=0.538993 loss_full=0.230103 loss_base=0.308889 ms_per_step=3525.63
[profile] window=354.22s updates=100 data_wait=2.0ms h2d=1.2ms stage1=233.2ms dit=643.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=345 step=36320 loss=0.473489 loss_full=0.196151 loss_base=0.277338 ms_per_step=3713.89
[epoch-end] epoch=345 elapsed=373.43s step=36330
[epoch-start] epoch=346 step=36330 steps_per_epoch=105
task=ci2v epoch=346 step=36340 loss=0.613986 loss_full=0.273271 loss_base=0.340715 ms_per_step=3606.85
task=ci2v epoch=346 step=36360 loss=0.450088 loss_full=0.181024 loss_base=0.269064 ms_per_step=3712.53
task=ci2v epoch=346 step=36380 loss=0.531640 loss_full=0.236496 loss_base=0.295144 ms_per_step=3524.59
task=ci2v epoch=346 step=36400 loss=0.490455 loss_full=0.205638 loss_base=0.284817 ms_per_step=3697.09
[profile] window=365.36s updates=100 data_wait=2.1ms h2d=1.4ms stage1=255.8ms dit=647.8ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=346 step=36420 loss=0.468195 loss_full=0.195675 loss_base=0.272519 ms_per_step=3362.88
[epoch-end] epoch=346 elapsed=371.90s step=36435
[epoch-start] epoch=347 step=36435 steps_per_epoch=105
task=ci2v epoch=347 step=36440 loss=0.457443 loss_full=0.179947 loss_base=0.277496 ms_per_step=3546.41
task=ci2v epoch=347 step=36460 loss=0.508684 loss_full=0.213090 loss_base=0.295594 ms_per_step=3605.09
task=ci2v epoch=347 step=36480 loss=0.412547 loss_full=0.158259 loss_base=0.254288 ms_per_step=3340.50
task=ci2v epoch=347 step=36500 loss=0.517550 loss_full=0.218534 loss_base=0.299016 ms_per_step=3328.50
[profile] window=343.94s updates=100 data_wait=1.9ms h2d=1.0ms stage1=211.5ms dit=639.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=347 step=36520 loss=0.448016 loss_full=0.182731 loss_base=0.265285 ms_per_step=3625.36
task=ci2v epoch=347 step=36540 loss=0.454122 loss_full=0.185059 loss_base=0.269063 ms_per_step=3391.67
[epoch-end] epoch=347 elapsed=366.17s step=36540
[epoch-start] epoch=348 step=36540 steps_per_epoch=105
task=ci2v epoch=348 step=36560 loss=0.474840 loss_full=0.193846 loss_base=0.280994 ms_per_step=3756.22
task=ci2v epoch=348 step=36580 loss=0.506027 loss_full=0.205342 loss_base=0.300685 ms_per_step=3767.59
task=ci2v epoch=348 step=36600 loss=0.523229 loss_full=0.223213 loss_base=0.300017 ms_per_step=3734.88
[profile] window=365.76s updates=100 data_wait=1.7ms h2d=1.4ms stage1=257.7ms dit=647.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=348 step=36620 loss=0.417149 loss_full=0.166233 loss_base=0.250916 ms_per_step=3816.88
task=ci2v epoch=348 step=36640 loss=0.538794 loss_full=0.222945 loss_base=0.315849 ms_per_step=3800.54
[epoch-end] epoch=348 elapsed=396.07s step=36645
[epoch-start] epoch=349 step=36645 steps_per_epoch=105
task=ci2v epoch=349 step=36660 loss=0.531946 loss_full=0.218069 loss_base=0.313877 ms_per_step=3787.48
task=ci2v epoch=349 step=36680 loss=0.456460 loss_full=0.188873 loss_base=0.267587 ms_per_step=3680.84
task=ci2v epoch=349 step=36700 loss=0.504523 loss_full=0.215196 loss_base=0.289327 ms_per_step=3626.49
[profile] window=374.50s updates=100 data_wait=2.2ms h2d=1.6ms stage1=279.2ms dit=647.2ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=349 step=36720 loss=0.479320 loss_full=0.195761 loss_base=0.283559 ms_per_step=3801.11
task=ci2v epoch=349 step=36740 loss=0.517604 loss_full=0.212773 loss_base=0.304831 ms_per_step=3809.90
[epoch-end] epoch=349 elapsed=392.12s step=36750
[epoch-start] epoch=350 step=36750 steps_per_epoch=105
task=ci2v epoch=350 step=36760 loss=0.651802 loss_full=0.295110 loss_base=0.356692 ms_per_step=3509.35
task=ci2v epoch=350 step=36780 loss=0.589138 loss_full=0.251784 loss_base=0.337354 ms_per_step=3544.59
task=ci2v epoch=350 step=36800 loss=0.580643 loss_full=0.257141 loss_base=0.323502 ms_per_step=3545.54
[profile] window=364.48s updates=100 data_wait=1.8ms h2d=1.4ms stage1=256.7ms dit=644.8ms update=29.6ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=350 step=36820 loss=0.548638 loss_full=0.240850 loss_base=0.307787 ms_per_step=3521.45
task=ci2v epoch=350 step=36840 loss=0.421625 loss_full=0.170294 loss_base=0.251331 ms_per_step=3451.51
[epoch-end] epoch=350 elapsed=371.39s step=36855
[epoch-start] epoch=351 step=36855 steps_per_epoch=105
task=ci2v epoch=351 step=36860 loss=0.522173 loss_full=0.220285 loss_base=0.301888 ms_per_step=3629.81
task=ci2v epoch=351 step=36880 loss=0.428512 loss_full=0.172738 loss_base=0.255774 ms_per_step=3543.26
task=ci2v epoch=351 step=36900 loss=0.427365 loss_full=0.175607 loss_base=0.251758 ms_per_step=3618.65
[profile] window=355.56s updates=100 data_wait=1.8ms h2d=1.2ms stage1=236.4ms dit=643.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=351 step=36920 loss=0.533877 loss_full=0.234363 loss_base=0.299514 ms_per_step=3495.74
task=ci2v epoch=351 step=36940 loss=0.513669 loss_full=0.215375 loss_base=0.298294 ms_per_step=3355.62
task=ci2v epoch=351 step=36960 loss=0.401784 loss_full=0.161328 loss_base=0.240456 ms_per_step=3698.99
[epoch-end] epoch=351 elapsed=371.93s step=36960
[epoch-start] epoch=352 step=36960 steps_per_epoch=105
task=ci2v epoch=352 step=36980 loss=0.734123 loss_full=0.353053 loss_base=0.381070 ms_per_step=3648.78
task=ci2v epoch=352 step=37000 loss=0.457569 loss_full=0.180933 loss_base=0.276636 ms_per_step=3378.25
[profile] window=351.83s updates=100 data_wait=1.7ms h2d=1.1ms stage1=228.4ms dit=642.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=37000 elapsed=2.39s
[checkpoint] step=37000 elapsed=29.81s
task=ci2v epoch=352 step=37020 loss=0.590713 loss_full=0.257000 loss_base=0.333713 ms_per_step=3348.95
task=ci2v epoch=352 step=37040 loss=0.542604 loss_full=0.235121 loss_base=0.307483 ms_per_step=3345.54
task=ci2v epoch=352 step=37060 loss=0.545951 loss_full=0.226084 loss_base=0.319867 ms_per_step=3359.44
[epoch-end] epoch=352 elapsed=390.68s step=37065
[epoch-start] epoch=353 step=37065 steps_per_epoch=105
task=ci2v epoch=353 step=37080 loss=0.574017 loss_full=0.256214 loss_base=0.317803 ms_per_step=3534.67
task=ci2v epoch=353 step=37100 loss=0.544970 loss_full=0.232723 loss_base=0.312248 ms_per_step=3364.03
[profile] window=339.32s updates=100 data_wait=1.5ms h2d=0.9ms stage1=201.5ms dit=638.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=353 step=37120 loss=0.593790 loss_full=0.271661 loss_base=0.322129 ms_per_step=3499.24
task=ci2v epoch=353 step=37140 loss=0.459533 loss_full=0.185874 loss_base=0.273659 ms_per_step=3576.33
task=ci2v epoch=353 step=37160 loss=0.476376 loss_full=0.190877 loss_base=0.285500 ms_per_step=3638.45
[epoch-end] epoch=353 elapsed=373.23s step=37170
[epoch-start] epoch=354 step=37170 steps_per_epoch=105
task=ci2v epoch=354 step=37180 loss=0.470622 loss_full=0.192188 loss_base=0.278434 ms_per_step=3756.67
task=ci2v epoch=354 step=37200 loss=0.526489 loss_full=0.220936 loss_base=0.305553 ms_per_step=3507.45
[profile] window=359.81s updates=100 data_wait=1.9ms h2d=1.3ms stage1=245.0ms dit=645.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=354 step=37220 loss=0.559590 loss_full=0.240487 loss_base=0.319103 ms_per_step=3327.48
task=ci2v epoch=354 step=37240 loss=0.530925 loss_full=0.216271 loss_base=0.314654 ms_per_step=3334.91
task=ci2v epoch=354 step=37260 loss=0.597112 loss_full=0.263021 loss_base=0.334091 ms_per_step=3547.66
[epoch-end] epoch=354 elapsed=363.58s step=37275
[epoch-start] epoch=355 step=37275 steps_per_epoch=105
task=ci2v epoch=355 step=37280 loss=0.551936 loss_full=0.242969 loss_base=0.308967 ms_per_step=3407.71
task=ci2v epoch=355 step=37300 loss=0.453417 loss_full=0.187111 loss_base=0.266306 ms_per_step=3490.66
[profile] window=342.43s updates=100 data_wait=1.7ms h2d=1.0ms stage1=209.9ms dit=637.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=355 step=37320 loss=0.496044 loss_full=0.212647 loss_base=0.283398 ms_per_step=3553.50
task=ci2v epoch=355 step=37340 loss=0.487755 loss_full=0.201773 loss_base=0.285981 ms_per_step=3567.83
task=ci2v epoch=355 step=37360 loss=0.470655 loss_full=0.198594 loss_base=0.272061 ms_per_step=3587.15
task=ci2v epoch=355 step=37380 loss=0.543377 loss_full=0.230257 loss_base=0.313120 ms_per_step=3578.06
[epoch-end] epoch=355 elapsed=373.34s step=37380
[epoch-start] epoch=356 step=37380 steps_per_epoch=105
task=ci2v epoch=356 step=37400 loss=0.579266 loss_full=0.258230 loss_base=0.321036 ms_per_step=3509.74
[profile] window=356.19s updates=100 data_wait=1.6ms h2d=1.2ms stage1=239.9ms dit=640.7ms update=30.9ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=356 step=37420 loss=0.432504 loss_full=0.179727 loss_base=0.252777 ms_per_step=3763.53
task=ci2v epoch=356 step=37440 loss=0.574216 loss_full=0.256409 loss_base=0.317807 ms_per_step=3654.24
task=ci2v epoch=356 step=37460 loss=0.606565 loss_full=0.261480 loss_base=0.345085 ms_per_step=3342.11
task=ci2v epoch=356 step=37480 loss=0.511652 loss_full=0.217472 loss_base=0.294180 ms_per_step=3344.14
[epoch-end] epoch=356 elapsed=368.95s step=37485
[epoch-start] epoch=357 step=37485 steps_per_epoch=105
task=ci2v epoch=357 step=37500 loss=0.573433 loss_full=0.242547 loss_base=0.330886 ms_per_step=3330.54
[profile] window=348.96s updates=100 data_wait=1.8ms h2d=1.1ms stage1=223.0ms dit=640.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=357 step=37520 loss=0.518831 loss_full=0.215799 loss_base=0.303033 ms_per_step=3615.77
task=ci2v epoch=357 step=37540 loss=0.509123 loss_full=0.213972 loss_base=0.295152 ms_per_step=3798.93
task=ci2v epoch=357 step=37560 loss=0.555657 loss_full=0.247789 loss_base=0.307868 ms_per_step=3345.59
task=ci2v epoch=357 step=37580 loss=0.436653 loss_full=0.172277 loss_base=0.264376 ms_per_step=3434.47
[epoch-end] epoch=357 elapsed=368.19s step=37590
[epoch-start] epoch=358 step=37590 steps_per_epoch=105
task=ci2v epoch=358 step=37600 loss=0.512771 loss_full=0.218118 loss_base=0.294653 ms_per_step=3427.27
[profile] window=352.70s updates=100 data_wait=1.6ms h2d=1.1ms stage1=231.5ms dit=641.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=358 step=37620 loss=0.625944 loss_full=0.277146 loss_base=0.348797 ms_per_step=3617.06
task=ci2v epoch=358 step=37640 loss=0.523093 loss_full=0.223840 loss_base=0.299253 ms_per_step=3509.69
task=ci2v epoch=358 step=37660 loss=0.646925 loss_full=0.283730 loss_base=0.363195 ms_per_step=3343.53
task=ci2v epoch=358 step=37680 loss=0.533208 loss_full=0.223153 loss_base=0.310055 ms_per_step=3458.79
[epoch-end] epoch=358 elapsed=366.07s step=37695
[epoch-start] epoch=359 step=37695 steps_per_epoch=105
task=ci2v epoch=359 step=37700 loss=0.495271 loss_full=0.200350 loss_base=0.294921 ms_per_step=3477.02
[profile] window=348.38s updates=100 data_wait=1.6ms h2d=1.1ms stage1=222.0ms dit=640.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=359 step=37720 loss=0.509859 loss_full=0.203743 loss_base=0.306115 ms_per_step=3403.63
task=ci2v epoch=359 step=37740 loss=0.523718 loss_full=0.230779 loss_base=0.292939 ms_per_step=3390.18
task=ci2v epoch=359 step=37760 loss=0.482018 loss_full=0.205532 loss_base=0.276486 ms_per_step=3459.30
task=ci2v epoch=359 step=37780 loss=0.501463 loss_full=0.201115 loss_base=0.300348 ms_per_step=3386.82
task=ci2v epoch=359 step=37800 loss=0.468337 loss_full=0.188712 loss_base=0.279625 ms_per_step=3468.15
[profile] window=342.42s updates=100 data_wait=1.0ms h2d=1.0ms stage1=209.7ms dit=638.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=359 elapsed=359.98s step=37800
[epoch-start] epoch=360 step=37800 steps_per_epoch=105
task=ci2v epoch=360 step=37820 loss=0.567611 loss_full=0.243393 loss_base=0.324219 ms_per_step=3426.84
task=ci2v epoch=360 step=37840 loss=0.638696 loss_full=0.289108 loss_base=0.349588 ms_per_step=3632.18
task=ci2v epoch=360 step=37860 loss=0.574788 loss_full=0.243210 loss_base=0.331578 ms_per_step=3618.85
task=ci2v epoch=360 step=37880 loss=0.497551 loss_full=0.210608 loss_base=0.286943 ms_per_step=3564.08
task=ci2v epoch=360 step=37900 loss=0.451333 loss_full=0.184539 loss_base=0.266794 ms_per_step=3784.36
[profile] window=360.78s updates=100 data_wait=1.7ms h2d=1.3ms stage1=249.5ms dit=642.8ms update=29.5ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=360 elapsed=377.94s step=37905
[epoch-start] epoch=361 step=37905 steps_per_epoch=105
task=ci2v epoch=361 step=37920 loss=0.494693 loss_full=0.210143 loss_base=0.284550 ms_per_step=3455.32
task=ci2v epoch=361 step=37940 loss=0.373034 loss_full=0.144499 loss_base=0.228535 ms_per_step=3368.80
task=ci2v epoch=361 step=37960 loss=0.425236 loss_full=0.164285 loss_base=0.260951 ms_per_step=3333.16
task=ci2v epoch=361 step=37980 loss=0.504677 loss_full=0.210262 loss_base=0.294415 ms_per_step=3496.74
task=ci2v epoch=361 step=38000 loss=0.498227 loss_full=0.206894 loss_base=0.291332 ms_per_step=3456.22
[profile] window=342.48s updates=100 data_wait=1.9ms h2d=1.0ms stage1=209.8ms dit=637.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=38000 elapsed=2.78s
[checkpoint] step=38000 elapsed=35.10s
[epoch-end] epoch=361 elapsed=396.52s step=38010
[epoch-start] epoch=362 step=38010 steps_per_epoch=105
task=ci2v epoch=362 step=38020 loss=0.505906 loss_full=0.219980 loss_base=0.285926 ms_per_step=3431.71
task=ci2v epoch=362 step=38040 loss=0.590367 loss_full=0.266314 loss_base=0.324053 ms_per_step=3566.68
task=ci2v epoch=362 step=38060 loss=0.461354 loss_full=0.191922 loss_base=0.269432 ms_per_step=3585.11
task=ci2v epoch=362 step=38080 loss=0.525869 loss_full=0.222667 loss_base=0.303202 ms_per_step=3481.03
task=ci2v epoch=362 step=38100 loss=0.431826 loss_full=0.168353 loss_base=0.263473 ms_per_step=3742.00
[profile] window=356.40s updates=100 data_wait=1.7ms h2d=1.2ms stage1=240.0ms dit=642.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=362 elapsed=377.28s step=38115
[epoch-start] epoch=363 step=38115 steps_per_epoch=105
task=ci2v epoch=363 step=38120 loss=0.502420 loss_full=0.210046 loss_base=0.292374 ms_per_step=3567.06
task=ci2v epoch=363 step=38140 loss=0.375530 loss_full=0.143648 loss_base=0.231881 ms_per_step=3337.98
task=ci2v epoch=363 step=38160 loss=0.507271 loss_full=0.220791 loss_base=0.286479 ms_per_step=3407.70
task=ci2v epoch=363 step=38180 loss=0.444077 loss_full=0.177926 loss_base=0.266151 ms_per_step=3388.49
task=ci2v epoch=363 step=38200 loss=0.482407 loss_full=0.202669 loss_base=0.279739 ms_per_step=3550.56
[profile] window=345.30s updates=100 data_wait=1.6ms h2d=1.0ms stage1=216.6ms dit=638.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=363 step=38220 loss=0.489962 loss_full=0.204089 loss_base=0.285873 ms_per_step=3339.54
[epoch-end] epoch=363 elapsed=358.40s step=38220
[epoch-start] epoch=364 step=38220 steps_per_epoch=105
task=ci2v epoch=364 step=38240 loss=0.448598 loss_full=0.191072 loss_base=0.257526 ms_per_step=3387.41
task=ci2v epoch=364 step=38260 loss=0.539402 loss_full=0.237545 loss_base=0.301857 ms_per_step=3341.95
task=ci2v epoch=364 step=38280 loss=0.539407 loss_full=0.230152 loss_base=0.309255 ms_per_step=3339.48
task=ci2v epoch=364 step=38300 loss=0.486450 loss_full=0.194723 loss_base=0.291727 ms_per_step=3591.51
[profile] window=340.25s updates=100 data_wait=1.7ms h2d=0.9ms stage1=203.6ms dit=638.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=364 step=38320 loss=0.525727 loss_full=0.219346 loss_base=0.306381 ms_per_step=3777.62
[epoch-end] epoch=364 elapsed=367.68s step=38325
[epoch-start] epoch=365 step=38325 steps_per_epoch=105
task=ci2v epoch=365 step=38340 loss=0.501419 loss_full=0.214799 loss_base=0.286619 ms_per_step=3799.58
task=ci2v epoch=365 step=38360 loss=0.603591 loss_full=0.272301 loss_base=0.331290 ms_per_step=3375.24
task=ci2v epoch=365 step=38380 loss=0.498811 loss_full=0.206435 loss_base=0.292376 ms_per_step=3354.53
task=ci2v epoch=365 step=38400 loss=0.481743 loss_full=0.203501 loss_base=0.278242 ms_per_step=3350.50
[profile] window=353.41s updates=100 data_wait=1.9ms h2d=1.1ms stage1=230.7ms dit=643.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=365 step=38420 loss=0.519567 loss_full=0.209402 loss_base=0.310165 ms_per_step=3358.82
[epoch-end] epoch=365 elapsed=359.98s step=38430
[epoch-start] epoch=366 step=38430 steps_per_epoch=105
task=ci2v epoch=366 step=38440 loss=0.564062 loss_full=0.242124 loss_base=0.321938 ms_per_step=3559.64
task=ci2v epoch=366 step=38460 loss=0.569566 loss_full=0.250917 loss_base=0.318650 ms_per_step=3517.80
task=ci2v epoch=366 step=38480 loss=0.474704 loss_full=0.191607 loss_base=0.283097 ms_per_step=3687.75
task=ci2v epoch=366 step=38500 loss=0.506878 loss_full=0.212318 loss_base=0.294560 ms_per_step=3479.64
[profile] window=352.32s updates=100 data_wait=1.6ms h2d=1.1ms stage1=230.0ms dit=641.1ms update=30.9ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=366 step=38520 loss=0.494178 loss_full=0.199112 loss_base=0.295066 ms_per_step=3346.78
[epoch-end] epoch=366 elapsed=372.66s step=38535
[epoch-start] epoch=367 step=38535 steps_per_epoch=105
task=ci2v epoch=367 step=38540 loss=0.543934 loss_full=0.230780 loss_base=0.313155 ms_per_step=3611.29
task=ci2v epoch=367 step=38560 loss=0.470530 loss_full=0.188785 loss_base=0.281744 ms_per_step=3358.74
task=ci2v epoch=367 step=38580 loss=0.561088 loss_full=0.235325 loss_base=0.325763 ms_per_step=3644.68
task=ci2v epoch=367 step=38600 loss=0.436946 loss_full=0.174368 loss_base=0.262578 ms_per_step=3673.64
[profile] window=352.96s updates=100 data_wait=2.1ms h2d=1.1ms stage1=230.8ms dit=642.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=367 step=38620 loss=0.468225 loss_full=0.193516 loss_base=0.274709 ms_per_step=3425.88
task=ci2v epoch=367 step=38640 loss=0.517702 loss_full=0.218498 loss_base=0.299204 ms_per_step=3408.30
[epoch-end] epoch=367 elapsed=369.07s step=38640
[epoch-start] epoch=368 step=38640 steps_per_epoch=105
task=ci2v epoch=368 step=38660 loss=0.511534 loss_full=0.217898 loss_base=0.293636 ms_per_step=3657.11
task=ci2v epoch=368 step=38680 loss=0.457894 loss_full=0.193117 loss_base=0.264777 ms_per_step=3814.76
task=ci2v epoch=368 step=38700 loss=0.523651 loss_full=0.221142 loss_base=0.302509 ms_per_step=3652.00
[profile] window=359.43s updates=100 data_wait=1.6ms h2d=1.2ms stage1=244.9ms dit=644.6ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=368 step=38720 loss=0.622687 loss_full=0.278234 loss_base=0.344453 ms_per_step=3468.59
task=ci2v epoch=368 step=38740 loss=0.523222 loss_full=0.214347 loss_base=0.308875 ms_per_step=3596.81
[epoch-end] epoch=368 elapsed=382.39s step=38745
[epoch-start] epoch=369 step=38745 steps_per_epoch=105
task=ci2v epoch=369 step=38760 loss=0.450344 loss_full=0.186002 loss_base=0.264342 ms_per_step=3847.14
task=ci2v epoch=369 step=38780 loss=0.625353 loss_full=0.282049 loss_base=0.343303 ms_per_step=3716.22
task=ci2v epoch=369 step=38800 loss=0.572690 loss_full=0.240709 loss_base=0.331981 ms_per_step=3608.72
[profile] window=365.01s updates=100 data_wait=2.1ms h2d=1.4ms stage1=257.5ms dit=645.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=369 step=38820 loss=0.428444 loss_full=0.170686 loss_base=0.257759 ms_per_step=3551.61
task=ci2v epoch=369 step=38840 loss=0.477953 loss_full=0.192745 loss_base=0.285208 ms_per_step=3476.93
[epoch-end] epoch=369 elapsed=382.89s step=38850
[epoch-start] epoch=370 step=38850 steps_per_epoch=105
task=ci2v epoch=370 step=38860 loss=0.581064 loss_full=0.246360 loss_base=0.334704 ms_per_step=3714.01
task=ci2v epoch=370 step=38880 loss=0.521960 loss_full=0.220676 loss_base=0.301285 ms_per_step=3408.14
task=ci2v epoch=370 step=38900 loss=0.567039 loss_full=0.239262 loss_base=0.327778 ms_per_step=3378.07
[profile] window=350.84s updates=100 data_wait=1.6ms h2d=1.1ms stage1=227.2ms dit=641.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=370 step=38920 loss=0.507493 loss_full=0.210667 loss_base=0.296827 ms_per_step=3359.62
task=ci2v epoch=370 step=38940 loss=0.493400 loss_full=0.206315 loss_base=0.287084 ms_per_step=3525.65
[epoch-end] epoch=370 elapsed=367.44s step=38955
[epoch-start] epoch=371 step=38955 steps_per_epoch=105
task=ci2v epoch=371 step=38960 loss=0.470599 loss_full=0.194989 loss_base=0.275610 ms_per_step=3699.34
task=ci2v epoch=371 step=38980 loss=0.602865 loss_full=0.263540 loss_base=0.339326 ms_per_step=3396.44
task=ci2v epoch=371 step=39000 loss=0.477407 loss_full=0.192057 loss_base=0.285350 ms_per_step=3370.70
[profile] window=347.29s updates=100 data_wait=1.9ms h2d=1.0ms stage1=219.9ms dit=639.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=39000 elapsed=2.79s
[checkpoint] step=39000 elapsed=39.80s
task=ci2v epoch=371 step=39020 loss=0.638208 loss_full=0.280361 loss_base=0.357847 ms_per_step=3660.80
task=ci2v epoch=371 step=39040 loss=0.575983 loss_full=0.246818 loss_base=0.329166 ms_per_step=3479.13
task=ci2v epoch=371 step=39060 loss=0.418302 loss_full=0.165772 loss_base=0.252530 ms_per_step=3696.39
[epoch-end] epoch=371 elapsed=413.04s step=39060
[epoch-start] epoch=372 step=39060 steps_per_epoch=105
task=ci2v epoch=372 step=39080 loss=0.443728 loss_full=0.177309 loss_base=0.266419 ms_per_step=3877.78
task=ci2v epoch=372 step=39100 loss=0.470333 loss_full=0.192363 loss_base=0.277970 ms_per_step=3420.21
[profile] window=362.95s updates=100 data_wait=2.2ms h2d=1.3ms stage1=249.7ms dit=647.8ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=372 step=39120 loss=0.506367 loss_full=0.215756 loss_base=0.290611 ms_per_step=3345.55
task=ci2v epoch=372 step=39140 loss=0.559574 loss_full=0.247193 loss_base=0.312382 ms_per_step=3353.77
task=ci2v epoch=372 step=39160 loss=0.425639 loss_full=0.166010 loss_base=0.259630 ms_per_step=3436.15
[epoch-end] epoch=372 elapsed=365.70s step=39165
[epoch-start] epoch=373 step=39165 steps_per_epoch=105
task=ci2v epoch=373 step=39180 loss=0.445217 loss_full=0.178108 loss_base=0.267109 ms_per_step=3717.94
task=ci2v epoch=373 step=39200 loss=0.459082 loss_full=0.192327 loss_base=0.266755 ms_per_step=3465.25
[profile] window=346.64s updates=100 data_wait=1.8ms h2d=1.0ms stage1=215.8ms dit=642.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=373 step=39220 loss=0.523787 loss_full=0.223105 loss_base=0.300682 ms_per_step=3391.48
task=ci2v epoch=373 step=39240 loss=0.482733 loss_full=0.202603 loss_base=0.280130 ms_per_step=3416.16
task=ci2v epoch=373 step=39260 loss=0.597871 loss_full=0.261607 loss_base=0.336264 ms_per_step=3306.56
[epoch-end] epoch=373 elapsed=363.98s step=39270
[epoch-start] epoch=374 step=39270 steps_per_epoch=105
task=ci2v epoch=374 step=39280 loss=0.494001 loss_full=0.202939 loss_base=0.291062 ms_per_step=3562.76
task=ci2v epoch=374 step=39300 loss=0.528637 loss_full=0.234170 loss_base=0.294467 ms_per_step=3431.20
[profile] window=342.43s updates=100 data_wait=1.7ms h2d=0.9ms stage1=208.4ms dit=638.9ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=374 step=39320 loss=0.469916 loss_full=0.190570 loss_base=0.279346 ms_per_step=3528.72
task=ci2v epoch=374 step=39340 loss=0.440694 loss_full=0.174363 loss_base=0.266331 ms_per_step=3345.21
task=ci2v epoch=374 step=39360 loss=0.455149 loss_full=0.177962 loss_base=0.277187 ms_per_step=3374.14
[epoch-end] epoch=374 elapsed=363.44s step=39375
[epoch-start] epoch=375 step=39375 steps_per_epoch=105
task=ci2v epoch=375 step=39380 loss=0.491277 loss_full=0.194581 loss_base=0.296695 ms_per_step=3508.78
task=ci2v epoch=375 step=39400 loss=0.494170 loss_full=0.200856 loss_base=0.293314 ms_per_step=3526.48
[profile] window=345.93s updates=100 data_wait=1.6ms h2d=1.0ms stage1=216.8ms dit=639.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=375 step=39420 loss=0.457399 loss_full=0.183660 loss_base=0.273740 ms_per_step=3477.99
task=ci2v epoch=375 step=39440 loss=0.490487 loss_full=0.205417 loss_base=0.285070 ms_per_step=3443.03
task=ci2v epoch=375 step=39460 loss=0.446980 loss_full=0.187628 loss_base=0.259352 ms_per_step=3517.51
task=ci2v epoch=375 step=39480 loss=0.499307 loss_full=0.206694 loss_base=0.292613 ms_per_step=3468.53
[epoch-end] epoch=375 elapsed=366.63s step=39480
[epoch-start] epoch=376 step=39480 steps_per_epoch=105
task=ci2v epoch=376 step=39500 loss=0.537347 loss_full=0.226719 loss_base=0.310628 ms_per_step=3698.37
[profile] window=352.37s updates=100 data_wait=1.7ms h2d=1.1ms stage1=230.3ms dit=641.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=376 step=39520 loss=0.504198 loss_full=0.212069 loss_base=0.292128 ms_per_step=3583.91
task=ci2v epoch=376 step=39540 loss=0.564444 loss_full=0.246097 loss_base=0.318347 ms_per_step=3717.80
task=ci2v epoch=376 step=39560 loss=0.419049 loss_full=0.166113 loss_base=0.252935 ms_per_step=3549.06
task=ci2v epoch=376 step=39580 loss=0.593718 loss_full=0.260924 loss_base=0.332794 ms_per_step=3423.34
[epoch-end] epoch=376 elapsed=378.04s step=39585
[epoch-start] epoch=377 step=39585 steps_per_epoch=105
task=ci2v epoch=377 step=39600 loss=0.513692 loss_full=0.208812 loss_base=0.304880 ms_per_step=3624.89
[profile] window=358.24s updates=100 data_wait=2.6ms h2d=1.2ms stage1=241.4ms dit=644.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=377 step=39620 loss=0.513487 loss_full=0.210429 loss_base=0.303058 ms_per_step=3454.58
task=ci2v epoch=377 step=39640 loss=0.513896 loss_full=0.215960 loss_base=0.297936 ms_per_step=3499.03
task=ci2v epoch=377 step=39660 loss=0.517292 loss_full=0.217014 loss_base=0.300278 ms_per_step=3402.94
task=ci2v epoch=377 step=39680 loss=0.498709 loss_full=0.207383 loss_base=0.291326 ms_per_step=3444.19
[epoch-end] epoch=377 elapsed=363.83s step=39690
[epoch-start] epoch=378 step=39690 steps_per_epoch=105
task=ci2v epoch=378 step=39700 loss=0.485089 loss_full=0.198682 loss_base=0.286407 ms_per_step=3374.79
[profile] window=343.77s updates=100 data_wait=1.6ms h2d=1.0ms stage1=210.9ms dit=639.9ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=378 step=39720 loss=0.482577 loss_full=0.192869 loss_base=0.289708 ms_per_step=3514.40
task=ci2v epoch=378 step=39740 loss=0.440293 loss_full=0.178308 loss_base=0.261984 ms_per_step=3638.53
task=ci2v epoch=378 step=39760 loss=0.518779 loss_full=0.221070 loss_base=0.297709 ms_per_step=3729.13
task=ci2v epoch=378 step=39780 loss=0.427446 loss_full=0.173644 loss_base=0.253802 ms_per_step=3380.53
[epoch-end] epoch=378 elapsed=371.59s step=39795
[epoch-start] epoch=379 step=39795 steps_per_epoch=105
task=ci2v epoch=379 step=39800 loss=0.521427 loss_full=0.215698 loss_base=0.305729 ms_per_step=3459.35
[profile] window=354.71s updates=100 data_wait=1.7ms h2d=1.2ms stage1=236.0ms dit=641.8ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=379 step=39820 loss=0.443243 loss_full=0.179853 loss_base=0.263390 ms_per_step=3428.65
task=ci2v epoch=379 step=39840 loss=0.559539 loss_full=0.245874 loss_base=0.313666 ms_per_step=3599.98
task=ci2v epoch=379 step=39860 loss=0.519047 loss_full=0.225990 loss_base=0.293058 ms_per_step=3346.94
task=ci2v epoch=379 step=39880 loss=0.481985 loss_full=0.199369 loss_base=0.282616 ms_per_step=3495.12
task=ci2v epoch=379 step=39900 loss=0.462416 loss_full=0.193650 loss_base=0.268766 ms_per_step=3548.66
[profile] window=348.65s updates=100 data_wait=1.0ms h2d=1.1ms stage1=222.5ms dit=640.6ms update=29.6ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=379 elapsed=366.45s step=39900
[epoch-start] epoch=380 step=39900 steps_per_epoch=105
task=ci2v epoch=380 step=39920 loss=0.586340 loss_full=0.242678 loss_base=0.343662 ms_per_step=3405.07
task=ci2v epoch=380 step=39940 loss=0.452737 loss_full=0.183182 loss_base=0.269556 ms_per_step=3357.72
task=ci2v epoch=380 step=39960 loss=0.517504 loss_full=0.225176 loss_base=0.292327 ms_per_step=3373.59
task=ci2v epoch=380 step=39980 loss=0.510502 loss_full=0.222250 loss_base=0.288252 ms_per_step=3662.21
task=ci2v epoch=380 step=40000 loss=0.508130 loss_full=0.214592 loss_base=0.293539 ms_per_step=3488.22
[profile] window=345.99s updates=100 data_wait=1.8ms h2d=1.0ms stage1=217.4ms dit=638.6ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=40000 elapsed=2.45s
[checkpoint] step=40000 elapsed=24.91s
[epoch-end] epoch=380 elapsed=390.25s step=40005
[epoch-start] epoch=381 step=40005 steps_per_epoch=105
task=ci2v epoch=381 step=40020 loss=0.476684 loss_full=0.193105 loss_base=0.283579 ms_per_step=3533.44
task=ci2v epoch=381 step=40040 loss=0.471416 loss_full=0.194047 loss_base=0.277370 ms_per_step=3650.05
task=ci2v epoch=381 step=40060 loss=0.503532 loss_full=0.208581 loss_base=0.294951 ms_per_step=3380.98
task=ci2v epoch=381 step=40080 loss=0.510840 loss_full=0.219895 loss_base=0.290945 ms_per_step=3495.41
task=ci2v epoch=381 step=40100 loss=0.509336 loss_full=0.206052 loss_base=0.303285 ms_per_step=3497.55
[profile] window=351.41s updates=100 data_wait=2.1ms h2d=1.1ms stage1=227.9ms dit=641.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=381 elapsed=369.66s step=40110
[epoch-start] epoch=382 step=40110 steps_per_epoch=105
task=ci2v epoch=382 step=40120 loss=0.501294 loss_full=0.203481 loss_base=0.297814 ms_per_step=3517.42
task=ci2v epoch=382 step=40140 loss=0.500076 loss_full=0.208926 loss_base=0.291149 ms_per_step=3429.86
task=ci2v epoch=382 step=40160 loss=0.482431 loss_full=0.196891 loss_base=0.285539 ms_per_step=3516.66
task=ci2v epoch=382 step=40180 loss=0.492230 loss_full=0.207285 loss_base=0.284945 ms_per_step=3625.88
task=ci2v epoch=382 step=40200 loss=0.551092 loss_full=0.239751 loss_base=0.311341 ms_per_step=3472.61
[profile] window=351.52s updates=100 data_wait=1.8ms h2d=1.1ms stage1=228.5ms dit=641.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=382 elapsed=367.02s step=40215
[epoch-start] epoch=383 step=40215 steps_per_epoch=105
task=ci2v epoch=383 step=40220 loss=0.519216 loss_full=0.218670 loss_base=0.300546 ms_per_step=3535.46
task=ci2v epoch=383 step=40240 loss=0.697406 loss_full=0.311933 loss_base=0.385474 ms_per_step=3518.13
task=ci2v epoch=383 step=40260 loss=0.418079 loss_full=0.164608 loss_base=0.253471 ms_per_step=3555.88
task=ci2v epoch=383 step=40280 loss=0.484813 loss_full=0.200201 loss_base=0.284612 ms_per_step=3688.89
task=ci2v epoch=383 step=40300 loss=0.394054 loss_full=0.153266 loss_base=0.240788 ms_per_step=3471.07
[profile] window=355.65s updates=100 data_wait=2.0ms h2d=1.2ms stage1=236.1ms dit=643.3ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=383 step=40320 loss=0.442953 loss_full=0.173110 loss_base=0.269843 ms_per_step=3516.51
[epoch-end] epoch=383 elapsed=375.86s step=40320
[epoch-start] epoch=384 step=40320 steps_per_epoch=105
task=ci2v epoch=384 step=40340 loss=0.473463 loss_full=0.186612 loss_base=0.286851 ms_per_step=3618.90
task=ci2v epoch=384 step=40360 loss=0.527551 loss_full=0.207994 loss_base=0.319557 ms_per_step=3442.03
task=ci2v epoch=384 step=40380 loss=0.483219 loss_full=0.197183 loss_base=0.286035 ms_per_step=3531.84
task=ci2v epoch=384 step=40400 loss=0.455603 loss_full=0.180019 loss_base=0.275583 ms_per_step=3619.66
[profile] window=354.83s updates=100 data_wait=2.0ms h2d=1.2ms stage1=234.1ms dit=643.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=384 step=40420 loss=0.565959 loss_full=0.241255 loss_base=0.324704 ms_per_step=3338.66
[epoch-end] epoch=384 elapsed=368.77s step=40425
[epoch-start] epoch=385 step=40425 steps_per_epoch=105
task=ci2v epoch=385 step=40440 loss=0.636638 loss_full=0.274813 loss_base=0.361826 ms_per_step=3616.09
task=ci2v epoch=385 step=40460 loss=0.591673 loss_full=0.255402 loss_base=0.336271 ms_per_step=3667.43
task=ci2v epoch=385 step=40480 loss=0.431508 loss_full=0.172679 loss_base=0.258830 ms_per_step=3503.00
task=ci2v epoch=385 step=40500 loss=0.504423 loss_full=0.208417 loss_base=0.296005 ms_per_step=3712.44
[profile] window=357.02s updates=100 data_wait=1.6ms h2d=1.2ms stage1=239.5ms dit=644.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=385 step=40520 loss=0.464461 loss_full=0.193292 loss_base=0.271169 ms_per_step=3369.34
[epoch-end] epoch=385 elapsed=373.50s step=40530
[epoch-start] epoch=386 step=40530 steps_per_epoch=105
task=ci2v epoch=386 step=40540 loss=0.503250 loss_full=0.212288 loss_base=0.290962 ms_per_step=3621.52
task=ci2v epoch=386 step=40560 loss=0.617027 loss_full=0.273409 loss_base=0.343617 ms_per_step=3515.57
task=ci2v epoch=386 step=40580 loss=0.563153 loss_full=0.243296 loss_base=0.319857 ms_per_step=3329.90
task=ci2v epoch=386 step=40600 loss=0.452344 loss_full=0.187419 loss_base=0.264924 ms_per_step=3455.96
[profile] window=346.11s updates=100 data_wait=2.0ms h2d=1.0ms stage1=216.8ms dit=639.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=386 step=40620 loss=0.500620 loss_full=0.206370 loss_base=0.294249 ms_per_step=3496.89
[epoch-end] epoch=386 elapsed=365.63s step=40635
[epoch-start] epoch=387 step=40635 steps_per_epoch=105
task=ci2v epoch=387 step=40640 loss=0.589365 loss_full=0.264298 loss_base=0.325067 ms_per_step=3400.23
task=ci2v epoch=387 step=40660 loss=0.598783 loss_full=0.255781 loss_base=0.343002 ms_per_step=3353.43
task=ci2v epoch=387 step=40680 loss=0.504108 loss_full=0.212590 loss_base=0.291519 ms_per_step=3442.08
task=ci2v epoch=387 step=40700 loss=0.509300 loss_full=0.213180 loss_base=0.296120 ms_per_step=3531.87
[profile] window=344.76s updates=100 data_wait=1.7ms h2d=1.0ms stage1=214.5ms dit=638.6ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=387 step=40720 loss=0.460193 loss_full=0.184140 loss_base=0.276053 ms_per_step=3509.68
task=ci2v epoch=387 step=40740 loss=0.475059 loss_full=0.201502 loss_base=0.273557 ms_per_step=3459.90
[epoch-end] epoch=387 elapsed=364.39s step=40740
[epoch-start] epoch=388 step=40740 steps_per_epoch=105
task=ci2v epoch=388 step=40760 loss=0.433822 loss_full=0.175609 loss_base=0.258213 ms_per_step=3391.04
task=ci2v epoch=388 step=40780 loss=0.495599 loss_full=0.196399 loss_base=0.299200 ms_per_step=3511.64
task=ci2v epoch=388 step=40800 loss=0.439949 loss_full=0.179106 loss_base=0.260843 ms_per_step=3428.09
[profile] window=346.27s updates=100 data_wait=1.6ms h2d=1.0ms stage1=217.1ms dit=639.8ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=388 step=40820 loss=0.524904 loss_full=0.229839 loss_base=0.295065 ms_per_step=3451.84
task=ci2v epoch=388 step=40840 loss=0.587047 loss_full=0.256581 loss_base=0.330466 ms_per_step=3347.51
[epoch-end] epoch=388 elapsed=359.50s step=40845
[epoch-start] epoch=389 step=40845 steps_per_epoch=105
task=ci2v epoch=389 step=40860 loss=0.444303 loss_full=0.188433 loss_base=0.255869 ms_per_step=3356.62
task=ci2v epoch=389 step=40880 loss=0.523084 loss_full=0.221789 loss_base=0.301295 ms_per_step=3367.12
task=ci2v epoch=389 step=40900 loss=0.503354 loss_full=0.203570 loss_base=0.299784 ms_per_step=3432.86
[profile] window=339.38s updates=100 data_wait=1.6ms h2d=0.9ms stage1=202.9ms dit=637.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=389 step=40920 loss=0.427702 loss_full=0.167068 loss_base=0.260634 ms_per_step=3520.03
task=ci2v epoch=389 step=40940 loss=0.502374 loss_full=0.214476 loss_base=0.287898 ms_per_step=3382.96
[epoch-end] epoch=389 elapsed=358.53s step=40950
[epoch-start] epoch=390 step=40950 steps_per_epoch=105
task=ci2v epoch=390 step=40960 loss=0.550426 loss_full=0.229243 loss_base=0.321184 ms_per_step=3392.43
task=ci2v epoch=390 step=40980 loss=0.488563 loss_full=0.204374 loss_base=0.284189 ms_per_step=3417.02
task=ci2v epoch=390 step=41000 loss=0.471151 loss_full=0.188687 loss_base=0.282463 ms_per_step=3334.58
[profile] window=341.21s updates=100 data_wait=1.6ms h2d=0.9ms stage1=207.2ms dit=637.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=41000 elapsed=2.47s
[checkpoint] step=41000 elapsed=24.92s
task=ci2v epoch=390 step=41020 loss=0.456454 loss_full=0.189051 loss_base=0.267404 ms_per_step=3376.43
task=ci2v epoch=390 step=41040 loss=0.538913 loss_full=0.235902 loss_base=0.303011 ms_per_step=3360.69
[epoch-end] epoch=390 elapsed=382.01s step=41055
[epoch-start] epoch=391 step=41055 steps_per_epoch=105
task=ci2v epoch=391 step=41060 loss=0.564193 loss_full=0.245265 loss_base=0.318928 ms_per_step=3383.51
task=ci2v epoch=391 step=41080 loss=0.585985 loss_full=0.245325 loss_base=0.340660 ms_per_step=3463.93
task=ci2v epoch=391 step=41100 loss=0.436289 loss_full=0.177999 loss_base=0.258290 ms_per_step=3610.11
[profile] window=344.17s updates=100 data_wait=1.7ms h2d=0.9ms stage1=209.8ms dit=641.9ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=391 step=41120 loss=0.462760 loss_full=0.188732 loss_base=0.274028 ms_per_step=3378.47
task=ci2v epoch=391 step=41140 loss=0.520899 loss_full=0.225067 loss_base=0.295833 ms_per_step=3398.93
task=ci2v epoch=391 step=41160 loss=0.498468 loss_full=0.202782 loss_base=0.295686 ms_per_step=3376.80
[epoch-end] epoch=391 elapsed=362.59s step=41160
[epoch-start] epoch=392 step=41160 steps_per_epoch=105
task=ci2v epoch=392 step=41180 loss=0.592906 loss_full=0.271450 loss_base=0.321455 ms_per_step=3793.53
task=ci2v epoch=392 step=41200 loss=0.488782 loss_full=0.200676 loss_base=0.288106 ms_per_step=3547.97
[profile] window=350.18s updates=100 data_wait=2.0ms h2d=1.0ms stage1=222.5ms dit=643.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=392 step=41220 loss=0.552051 loss_full=0.258797 loss_base=0.293254 ms_per_step=3496.48
task=ci2v epoch=392 step=41240 loss=0.476549 loss_full=0.193475 loss_base=0.283074 ms_per_step=3467.45
task=ci2v epoch=392 step=41260 loss=0.599861 loss_full=0.267133 loss_base=0.332728 ms_per_step=3325.90
[epoch-end] epoch=392 elapsed=369.40s step=41265
[epoch-start] epoch=393 step=41265 steps_per_epoch=105
task=ci2v epoch=393 step=41280 loss=0.550736 loss_full=0.234364 loss_base=0.316372 ms_per_step=3409.10
task=ci2v epoch=393 step=41300 loss=0.562704 loss_full=0.242662 loss_base=0.320041 ms_per_step=3338.23
[profile] window=341.02s updates=100 data_wait=1.6ms h2d=0.9ms stage1=207.1ms dit=636.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=393 step=41320 loss=0.597950 loss_full=0.249285 loss_base=0.348664 ms_per_step=3392.27
task=ci2v epoch=393 step=41340 loss=0.568487 loss_full=0.246776 loss_base=0.321711 ms_per_step=3495.42
task=ci2v epoch=393 step=41360 loss=0.601191 loss_full=0.253101 loss_base=0.348090 ms_per_step=3357.56
[epoch-end] epoch=393 elapsed=356.65s step=41370
[epoch-start] epoch=394 step=41370 steps_per_epoch=105
task=ci2v epoch=394 step=41380 loss=0.488263 loss_full=0.200408 loss_base=0.287854 ms_per_step=3341.58
task=ci2v epoch=394 step=41400 loss=0.473714 loss_full=0.192306 loss_base=0.281409 ms_per_step=3329.56
[profile] window=338.60s updates=100 data_wait=1.5ms h2d=0.9ms stage1=202.3ms dit=635.0ms update=29.9ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=394 step=41420 loss=0.489116 loss_full=0.199858 loss_base=0.289258 ms_per_step=3334.69
task=ci2v epoch=394 step=41440 loss=0.538606 loss_full=0.235233 loss_base=0.303373 ms_per_step=3331.13
task=ci2v epoch=394 step=41460 loss=0.552477 loss_full=0.251865 loss_base=0.300612 ms_per_step=3309.25
[epoch-end] epoch=394 elapsed=351.71s step=41475
[epoch-start] epoch=395 step=41475 steps_per_epoch=105
task=ci2v epoch=395 step=41480 loss=0.582010 loss_full=0.254417 loss_base=0.327592 ms_per_step=3429.19
task=ci2v epoch=395 step=41500 loss=0.436565 loss_full=0.175852 loss_base=0.260713 ms_per_step=3489.88
[profile] window=338.15s updates=100 data_wait=1.4ms h2d=0.9ms stage1=201.5ms dit=635.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=395 step=41520 loss=0.512228 loss_full=0.213067 loss_base=0.299161 ms_per_step=3328.07
task=ci2v epoch=395 step=41540 loss=0.481668 loss_full=0.204589 loss_base=0.277079 ms_per_step=3372.91
task=ci2v epoch=395 step=41560 loss=0.513962 loss_full=0.218006 loss_base=0.295956 ms_per_step=3330.77
task=ci2v epoch=395 step=41580 loss=0.501832 loss_full=0.214294 loss_base=0.287537 ms_per_step=3336.31
[epoch-end] epoch=395 elapsed=354.94s step=41580
[epoch-start] epoch=396 step=41580 steps_per_epoch=105
task=ci2v epoch=396 step=41600 loss=0.535432 loss_full=0.222879 loss_base=0.312553 ms_per_step=3388.41
[profile] window=335.41s updates=100 data_wait=1.4ms h2d=0.8ms stage1=193.3ms dit=636.6ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=396 step=41620 loss=0.461787 loss_full=0.186236 loss_base=0.275551 ms_per_step=3406.02
task=ci2v epoch=396 step=41640 loss=0.480621 loss_full=0.200359 loss_base=0.280262 ms_per_step=3403.99
task=ci2v epoch=396 step=41660 loss=0.476768 loss_full=0.190194 loss_base=0.286573 ms_per_step=3324.43
task=ci2v epoch=396 step=41680 loss=0.522407 loss_full=0.222653 loss_base=0.299755 ms_per_step=3316.73
[epoch-end] epoch=396 elapsed=353.53s step=41685
[epoch-start] epoch=397 step=41685 steps_per_epoch=105
task=ci2v epoch=397 step=41700 loss=0.461407 loss_full=0.179210 loss_base=0.282197 ms_per_step=3340.20
[profile] window=336.10s updates=100 data_wait=1.5ms h2d=0.9ms stage1=194.6ms dit=637.1ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=397 step=41720 loss=0.464525 loss_full=0.188213 loss_base=0.276313 ms_per_step=3443.62
task=ci2v epoch=397 step=41740 loss=0.495866 loss_full=0.218763 loss_base=0.277104 ms_per_step=3322.45
task=ci2v epoch=397 step=41760 loss=0.403018 loss_full=0.158723 loss_base=0.244296 ms_per_step=3325.29
task=ci2v epoch=397 step=41780 loss=0.486698 loss_full=0.200733 loss_base=0.285965 ms_per_step=3335.63
[epoch-end] epoch=397 elapsed=354.10s step=41790
[epoch-start] epoch=398 step=41790 steps_per_epoch=105
task=ci2v epoch=398 step=41800 loss=0.495508 loss_full=0.202838 loss_base=0.292670 ms_per_step=3534.63
[profile] window=339.50s updates=100 data_wait=1.8ms h2d=1.0ms stage1=203.7ms dit=635.1ms update=30.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=398 step=41820 loss=0.534663 loss_full=0.224437 loss_base=0.310226 ms_per_step=3384.52
task=ci2v epoch=398 step=41840 loss=0.576457 loss_full=0.259727 loss_base=0.316731 ms_per_step=3336.23
task=ci2v epoch=398 step=41860 loss=0.498431 loss_full=0.210614 loss_base=0.287817 ms_per_step=3328.32
task=ci2v epoch=398 step=41880 loss=0.578671 loss_full=0.258807 loss_base=0.319864 ms_per_step=3306.53
[epoch-end] epoch=398 elapsed=354.94s step=41895
[epoch-start] epoch=399 step=41895 steps_per_epoch=105
task=ci2v epoch=399 step=41900 loss=0.627954 loss_full=0.268358 loss_base=0.359596 ms_per_step=3499.72
[profile] window=337.37s updates=100 data_wait=2.1ms h2d=0.9ms stage1=197.5ms dit=636.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=399 step=41920 loss=0.505716 loss_full=0.212908 loss_base=0.292808 ms_per_step=3325.10
task=ci2v epoch=399 step=41940 loss=0.418341 loss_full=0.168669 loss_base=0.249673 ms_per_step=3438.19
task=ci2v epoch=399 step=41960 loss=0.517792 loss_full=0.226532 loss_base=0.291260 ms_per_step=3314.15
task=ci2v epoch=399 step=41980 loss=0.487382 loss_full=0.206325 loss_base=0.281057 ms_per_step=3636.80
task=ci2v epoch=399 step=42000 loss=0.518962 loss_full=0.225194 loss_base=0.293768 ms_per_step=3417.18
[profile] window=342.90s updates=100 data_wait=0.9ms h2d=1.0ms stage1=211.9ms dit=637.3ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=42000 elapsed=2.49s
[checkpoint] step=42000 elapsed=15.88s
[epoch-end] epoch=399 elapsed=379.92s step=42000
[epoch-start] epoch=400 step=42000 steps_per_epoch=105
task=ci2v epoch=400 step=42020 loss=0.454995 loss_full=0.188053 loss_base=0.266942 ms_per_step=3629.17
task=ci2v epoch=400 step=42040 loss=0.518255 loss_full=0.217079 loss_base=0.301175 ms_per_step=3375.56
task=ci2v epoch=400 step=42060 loss=0.587229 loss_full=0.259011 loss_base=0.328218 ms_per_step=3321.07
task=ci2v epoch=400 step=42080 loss=0.515967 loss_full=0.212871 loss_base=0.303096 ms_per_step=3323.00
task=ci2v epoch=400 step=42100 loss=0.609316 loss_full=0.257576 loss_base=0.351740 ms_per_step=3345.01
[profile] window=340.15s updates=100 data_wait=1.6ms h2d=0.9ms stage1=206.1ms dit=635.6ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=400 elapsed=356.57s step=42105
[epoch-start] epoch=401 step=42105 steps_per_epoch=105
task=ci2v epoch=401 step=42120 loss=0.507985 loss_full=0.211046 loss_base=0.296939 ms_per_step=3326.15
task=ci2v epoch=401 step=42140 loss=0.551917 loss_full=0.238620 loss_base=0.313297 ms_per_step=3353.54
task=ci2v epoch=401 step=42160 loss=0.470380 loss_full=0.188940 loss_base=0.281440 ms_per_step=3467.46
task=ci2v epoch=401 step=42180 loss=0.538233 loss_full=0.226694 loss_base=0.311539 ms_per_step=3454.59
task=ci2v epoch=401 step=42200 loss=0.548456 loss_full=0.237267 loss_base=0.311189 ms_per_step=3364.55
[profile] window=339.59s updates=100 data_wait=1.4ms h2d=0.9ms stage1=204.4ms dit=636.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=401 elapsed=356.57s step=42210
[epoch-start] epoch=402 step=42210 steps_per_epoch=105
task=ci2v epoch=402 step=42220 loss=0.431134 loss_full=0.178136 loss_base=0.252999 ms_per_step=3352.22
task=ci2v epoch=402 step=42240 loss=0.495625 loss_full=0.202933 loss_base=0.292692 ms_per_step=3402.31
task=ci2v epoch=402 step=42260 loss=0.487478 loss_full=0.200914 loss_base=0.286564 ms_per_step=3310.55
task=ci2v epoch=402 step=42280 loss=0.536257 loss_full=0.230631 loss_base=0.305626 ms_per_step=3353.28
task=ci2v epoch=402 step=42300 loss=0.479142 loss_full=0.195329 loss_base=0.283814 ms_per_step=3304.17
[profile] window=334.72s updates=100 data_wait=1.5ms h2d=0.9ms stage1=193.3ms dit=634.9ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=402 elapsed=351.47s step=42315
[epoch-start] epoch=403 step=42315 steps_per_epoch=105
task=ci2v epoch=403 step=42320 loss=0.465271 loss_full=0.184593 loss_base=0.280678 ms_per_step=3346.34
task=ci2v epoch=403 step=42340 loss=0.445552 loss_full=0.181893 loss_base=0.263659 ms_per_step=3584.18
task=ci2v epoch=403 step=42360 loss=0.603882 loss_full=0.266597 loss_base=0.337285 ms_per_step=3318.44
task=ci2v epoch=403 step=42380 loss=0.500498 loss_full=0.221984 loss_base=0.278514 ms_per_step=3306.78
task=ci2v epoch=403 step=42400 loss=0.423183 loss_full=0.165920 loss_base=0.257262 ms_per_step=3309.00
[profile] window=337.57s updates=100 data_wait=1.6ms h2d=0.9ms stage1=198.1ms dit=637.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=403 step=42420 loss=0.491099 loss_full=0.199844 loss_base=0.291255 ms_per_step=3323.98
[epoch-end] epoch=403 elapsed=354.45s step=42420
[epoch-start] epoch=404 step=42420 steps_per_epoch=105
task=ci2v epoch=404 step=42440 loss=0.608615 loss_full=0.277928 loss_base=0.330687 ms_per_step=3333.46
task=ci2v epoch=404 step=42460 loss=0.464794 loss_full=0.193873 loss_base=0.270921 ms_per_step=3299.69
task=ci2v epoch=404 step=42480 loss=0.458827 loss_full=0.182895 loss_base=0.275932 ms_per_step=3367.85
task=ci2v epoch=404 step=42500 loss=0.511728 loss_full=0.204181 loss_base=0.307546 ms_per_step=3573.59
[profile] window=338.24s updates=100 data_wait=1.5ms h2d=0.9ms stage1=202.2ms dit=634.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=404 step=42520 loss=0.473832 loss_full=0.191931 loss_base=0.281902 ms_per_step=3375.33
[epoch-end] epoch=404 elapsed=357.05s step=42525
[epoch-start] epoch=405 step=42525 steps_per_epoch=105
task=ci2v epoch=405 step=42540 loss=0.502587 loss_full=0.210932 loss_base=0.291655 ms_per_step=3420.03
task=ci2v epoch=405 step=42560 loss=0.462616 loss_full=0.186655 loss_base=0.275961 ms_per_step=3325.57
task=ci2v epoch=405 step=42580 loss=0.484397 loss_full=0.200391 loss_base=0.284006 ms_per_step=3318.23
task=ci2v epoch=405 step=42600 loss=0.489603 loss_full=0.203833 loss_base=0.285770 ms_per_step=3309.91
[profile] window=335.25s updates=100 data_wait=1.8ms h2d=0.9ms stage1=195.1ms dit=634.3ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=405 step=42620 loss=0.453929 loss_full=0.180916 loss_base=0.273013 ms_per_step=3410.85
[epoch-end] epoch=405 elapsed=351.88s step=42630
[epoch-start] epoch=406 step=42630 steps_per_epoch=105
task=ci2v epoch=406 step=42640 loss=0.518192 loss_full=0.226288 loss_base=0.291905 ms_per_step=3514.37
task=ci2v epoch=406 step=42660 loss=0.444415 loss_full=0.172313 loss_base=0.272103 ms_per_step=3379.04
task=ci2v epoch=406 step=42680 loss=0.499045 loss_full=0.218779 loss_base=0.280265 ms_per_step=3427.56
task=ci2v epoch=406 step=42700 loss=0.485003 loss_full=0.200190 loss_base=0.284813 ms_per_step=3401.95
[profile] window=342.94s updates=100 data_wait=1.7ms h2d=1.0ms stage1=211.6ms dit=636.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=406 step=42720 loss=0.438461 loss_full=0.174829 loss_base=0.263632 ms_per_step=3385.96
[epoch-end] epoch=406 elapsed=358.54s step=42735
[epoch-start] epoch=407 step=42735 steps_per_epoch=105
task=ci2v epoch=407 step=42740 loss=0.475845 loss_full=0.196525 loss_base=0.279320 ms_per_step=3342.50
task=ci2v epoch=407 step=42760 loss=0.565875 loss_full=0.243818 loss_base=0.322057 ms_per_step=3363.66
task=ci2v epoch=407 step=42780 loss=0.485513 loss_full=0.197113 loss_base=0.288400 ms_per_step=3315.07
task=ci2v epoch=407 step=42800 loss=0.510661 loss_full=0.219352 loss_base=0.291309 ms_per_step=3315.26
[profile] window=334.72s updates=100 data_wait=1.6ms h2d=0.8ms stage1=193.7ms dit=634.6ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=407 step=42820 loss=0.497725 loss_full=0.207428 loss_base=0.290296 ms_per_step=3383.32
task=ci2v epoch=407 step=42840 loss=0.489741 loss_full=0.205761 loss_base=0.283980 ms_per_step=3309.23
[epoch-end] epoch=407 elapsed=351.54s step=42840
[epoch-start] epoch=408 step=42840 steps_per_epoch=105
task=ci2v epoch=408 step=42860 loss=0.440002 loss_full=0.178558 loss_base=0.261445 ms_per_step=3408.06
task=ci2v epoch=408 step=42880 loss=0.457804 loss_full=0.192933 loss_base=0.264871 ms_per_step=3361.51
task=ci2v epoch=408 step=42900 loss=0.554118 loss_full=0.236545 loss_base=0.317573 ms_per_step=3420.35
[profile] window=337.91s updates=100 data_wait=1.7ms h2d=0.9ms stage1=198.6ms dit=637.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=408 step=42920 loss=0.513903 loss_full=0.215339 loss_base=0.298565 ms_per_step=3327.87
task=ci2v epoch=408 step=42940 loss=0.470784 loss_full=0.202710 loss_base=0.268074 ms_per_step=3646.89
[epoch-end] epoch=408 elapsed=360.27s step=42945
[epoch-start] epoch=409 step=42945 steps_per_epoch=105
task=ci2v epoch=409 step=42960 loss=0.412923 loss_full=0.169506 loss_base=0.243417 ms_per_step=3370.83
task=ci2v epoch=409 step=42980 loss=0.492131 loss_full=0.206075 loss_base=0.286055 ms_per_step=3329.25
task=ci2v epoch=409 step=43000 loss=0.469236 loss_full=0.193656 loss_base=0.275580 ms_per_step=3356.64
[profile] window=340.90s updates=100 data_wait=1.5ms h2d=0.9ms stage1=204.3ms dit=639.3ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=43000 elapsed=2.52s
[checkpoint] step=43000 elapsed=24.09s
task=ci2v epoch=409 step=43020 loss=0.592369 loss_full=0.268334 loss_base=0.324035 ms_per_step=3328.58
task=ci2v epoch=409 step=43040 loss=0.549286 loss_full=0.231680 loss_base=0.317606 ms_per_step=3318.48
[epoch-end] epoch=409 elapsed=377.69s step=43050
[epoch-start] epoch=410 step=43050 steps_per_epoch=105
task=ci2v epoch=410 step=43060 loss=0.649260 loss_full=0.289515 loss_base=0.359745 ms_per_step=3349.89
task=ci2v epoch=410 step=43080 loss=0.458894 loss_full=0.192859 loss_base=0.266035 ms_per_step=3338.61
task=ci2v epoch=410 step=43100 loss=0.459725 loss_full=0.185164 loss_base=0.274561 ms_per_step=3342.08
[profile] window=333.82s updates=100 data_wait=1.6ms h2d=0.8ms stage1=190.2ms dit=635.2ms update=29.9ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=410 step=43120 loss=0.525798 loss_full=0.218712 loss_base=0.307086 ms_per_step=3309.67
task=ci2v epoch=410 step=43140 loss=0.529143 loss_full=0.213642 loss_base=0.315501 ms_per_step=3304.00
[epoch-end] epoch=410 elapsed=351.36s step=43155
[epoch-start] epoch=411 step=43155 steps_per_epoch=105
task=ci2v epoch=411 step=43160 loss=0.528413 loss_full=0.223812 loss_base=0.304601 ms_per_step=3472.39
task=ci2v epoch=411 step=43180 loss=0.426506 loss_full=0.166747 loss_base=0.259759 ms_per_step=3333.51
task=ci2v epoch=411 step=43200 loss=0.535167 loss_full=0.228344 loss_base=0.306823 ms_per_step=3375.92
[profile] window=336.18s updates=100 data_wait=1.9ms h2d=0.9ms stage1=196.6ms dit=634.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=411 step=43220 loss=0.518590 loss_full=0.222366 loss_base=0.296224 ms_per_step=3396.66
task=ci2v epoch=411 step=43240 loss=0.538957 loss_full=0.222578 loss_base=0.316379 ms_per_step=3313.74
task=ci2v epoch=411 step=43260 loss=0.578286 loss_full=0.246066 loss_base=0.332220 ms_per_step=3385.52
[epoch-end] epoch=411 elapsed=354.67s step=43260
[epoch-start] epoch=412 step=43260 steps_per_epoch=105
task=ci2v epoch=412 step=43280 loss=0.495008 loss_full=0.212523 loss_base=0.282485 ms_per_step=3382.25
task=ci2v epoch=412 step=43300 loss=0.526780 loss_full=0.220310 loss_base=0.306470 ms_per_step=3397.73
[profile] window=337.78s updates=100 data_wait=1.5ms h2d=0.9ms stage1=199.8ms dit=636.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=412 step=43320 loss=0.537866 loss_full=0.231219 loss_base=0.306648 ms_per_step=3464.66
task=ci2v epoch=412 step=43340 loss=0.589869 loss_full=0.264786 loss_base=0.325082 ms_per_step=3411.06
task=ci2v epoch=412 step=43360 loss=0.485414 loss_full=0.204364 loss_base=0.281050 ms_per_step=3452.95
[epoch-end] epoch=412 elapsed=358.98s step=43365
[epoch-start] epoch=413 step=43365 steps_per_epoch=105
task=ci2v epoch=413 step=43380 loss=0.577453 loss_full=0.261590 loss_base=0.315863 ms_per_step=3361.68
task=ci2v epoch=413 step=43400 loss=0.428583 loss_full=0.171596 loss_base=0.256987 ms_per_step=3512.41
[profile] window=344.32s updates=100 data_wait=1.5ms h2d=1.0ms stage1=214.1ms dit=638.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=413 step=43420 loss=0.468675 loss_full=0.198081 loss_base=0.270594 ms_per_step=3401.06
task=ci2v epoch=413 step=43440 loss=0.486314 loss_full=0.210420 loss_base=0.275894 ms_per_step=3419.42
task=ci2v epoch=413 step=43460 loss=0.511357 loss_full=0.209179 loss_base=0.302178 ms_per_step=3313.59
[epoch-end] epoch=413 elapsed=357.28s step=43470
[epoch-start] epoch=414 step=43470 steps_per_epoch=105
task=ci2v epoch=414 step=43480 loss=0.574811 loss_full=0.247935 loss_base=0.326876 ms_per_step=3338.57
task=ci2v epoch=414 step=43500 loss=0.544563 loss_full=0.244247 loss_base=0.300317 ms_per_step=3384.12
[profile] window=337.40s updates=100 data_wait=1.6ms h2d=0.9ms stage1=197.7ms dit=637.3ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=414 step=43520 loss=0.446109 loss_full=0.174964 loss_base=0.271145 ms_per_step=3335.51
task=ci2v epoch=414 step=43540 loss=0.541800 loss_full=0.223946 loss_base=0.317854 ms_per_step=3387.95
task=ci2v epoch=414 step=43560 loss=0.415860 loss_full=0.159308 loss_base=0.256552 ms_per_step=3640.80
[epoch-end] epoch=414 elapsed=362.88s step=43575
[epoch-start] epoch=415 step=43575 steps_per_epoch=105
task=ci2v epoch=415 step=43580 loss=0.505436 loss_full=0.204287 loss_base=0.301150 ms_per_step=3555.30
task=ci2v epoch=415 step=43600 loss=0.585937 loss_full=0.238309 loss_base=0.347628 ms_per_step=3397.83
[profile] window=346.61s updates=100 data_wait=1.6ms h2d=1.1ms stage1=219.0ms dit=638.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=415 step=43620 loss=0.457603 loss_full=0.184752 loss_base=0.272852 ms_per_step=3415.75
task=ci2v epoch=415 step=43640 loss=0.507216 loss_full=0.213162 loss_base=0.294054 ms_per_step=3419.32
task=ci2v epoch=415 step=43660 loss=0.458008 loss_full=0.187582 loss_base=0.270426 ms_per_step=3364.89
task=ci2v epoch=415 step=43680 loss=0.530095 loss_full=0.233047 loss_base=0.297048 ms_per_step=3449.51
[epoch-end] epoch=415 elapsed=358.51s step=43680
[epoch-start] epoch=416 step=43680 steps_per_epoch=105
task=ci2v epoch=416 step=43700 loss=0.532942 loss_full=0.223986 loss_base=0.308956 ms_per_step=3378.77
[profile] window=340.83s updates=100 data_wait=1.5ms h2d=0.9ms stage1=206.2ms dit=637.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=416 step=43720 loss=0.440163 loss_full=0.185447 loss_base=0.254716 ms_per_step=3345.58
task=ci2v epoch=416 step=43740 loss=0.452405 loss_full=0.178200 loss_base=0.274205 ms_per_step=3341.58
task=ci2v epoch=416 step=43760 loss=0.416086 loss_full=0.160806 loss_base=0.255281 ms_per_step=3342.30
task=ci2v epoch=416 step=43780 loss=0.460800 loss_full=0.182757 loss_base=0.278043 ms_per_step=3444.65
[epoch-end] epoch=416 elapsed=353.76s step=43785
[epoch-start] epoch=417 step=43785 steps_per_epoch=105
task=ci2v epoch=417 step=43800 loss=0.572611 loss_full=0.267427 loss_base=0.305184 ms_per_step=3355.74
[profile] window=336.87s updates=100 data_wait=1.6ms h2d=0.9ms stage1=197.6ms dit=636.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=417 step=43820 loss=0.501753 loss_full=0.208095 loss_base=0.293658 ms_per_step=3318.68
task=ci2v epoch=417 step=43840 loss=0.510190 loss_full=0.217619 loss_base=0.292571 ms_per_step=3387.09
task=ci2v epoch=417 step=43860 loss=0.465488 loss_full=0.197539 loss_base=0.267949 ms_per_step=3535.61
task=ci2v epoch=417 step=43880 loss=0.641455 loss_full=0.297802 loss_base=0.343653 ms_per_step=3369.34
[epoch-end] epoch=417 elapsed=358.67s step=43890
[epoch-start] epoch=418 step=43890 steps_per_epoch=105
task=ci2v epoch=418 step=43900 loss=0.581554 loss_full=0.248487 loss_base=0.333068 ms_per_step=3455.24
[profile] window=341.59s updates=100 data_wait=1.5ms h2d=1.0ms stage1=209.2ms dit=636.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=418 step=43920 loss=0.550834 loss_full=0.240585 loss_base=0.310249 ms_per_step=3329.51
task=ci2v epoch=418 step=43940 loss=0.473110 loss_full=0.194648 loss_base=0.278461 ms_per_step=3322.00
task=ci2v epoch=418 step=43960 loss=0.584471 loss_full=0.259051 loss_base=0.325420 ms_per_step=3383.02
task=ci2v epoch=418 step=43980 loss=0.526613 loss_full=0.234238 loss_base=0.292375 ms_per_step=3358.35
[epoch-end] epoch=418 elapsed=353.79s step=43995
[epoch-start] epoch=419 step=43995 steps_per_epoch=105
task=ci2v epoch=419 step=44000 loss=0.559289 loss_full=0.241327 loss_base=0.317962 ms_per_step=3596.93
[profile] window=340.07s updates=100 data_wait=2.0ms h2d=0.9ms stage1=204.1ms dit=636.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=44000 elapsed=2.81s
[checkpoint] step=44000 elapsed=36.79s
task=ci2v epoch=419 step=44020 loss=0.623593 loss_full=0.276804 loss_base=0.346788 ms_per_step=3366.55
task=ci2v epoch=419 step=44040 loss=0.467055 loss_full=0.189328 loss_base=0.277726 ms_per_step=3341.75
task=ci2v epoch=419 step=44060 loss=0.440754 loss_full=0.171074 loss_base=0.269680 ms_per_step=3329.57
task=ci2v epoch=419 step=44080 loss=0.494889 loss_full=0.210972 loss_base=0.283918 ms_per_step=3404.15
task=ci2v epoch=419 step=44100 loss=0.516128 loss_full=0.217698 loss_base=0.298430 ms_per_step=3294.12
[profile] window=334.99s updates=100 data_wait=0.8ms h2d=0.9ms stage1=195.3ms dit=634.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=419 elapsed=395.00s step=44100
[epoch-start] epoch=420 step=44100 steps_per_epoch=105
task=ci2v epoch=420 step=44120 loss=0.372137 loss_full=0.141183 loss_base=0.230954 ms_per_step=3368.67
task=ci2v epoch=420 step=44140 loss=0.551612 loss_full=0.234003 loss_base=0.317609 ms_per_step=3310.22
task=ci2v epoch=420 step=44160 loss=0.445430 loss_full=0.174924 loss_base=0.270506 ms_per_step=3347.07
task=ci2v epoch=420 step=44180 loss=0.607988 loss_full=0.257803 loss_base=0.350185 ms_per_step=3313.84
task=ci2v epoch=420 step=44200 loss=0.549792 loss_full=0.232042 loss_base=0.317750 ms_per_step=3412.17
[profile] window=335.31s updates=100 data_wait=1.7ms h2d=0.9ms stage1=195.4ms dit=634.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=420 elapsed=351.80s step=44205
[epoch-start] epoch=421 step=44205 steps_per_epoch=105
task=ci2v epoch=421 step=44220 loss=0.584675 loss_full=0.270242 loss_base=0.314433 ms_per_step=3445.71
task=ci2v epoch=421 step=44240 loss=0.489848 loss_full=0.202171 loss_base=0.287677 ms_per_step=3388.79
task=ci2v epoch=421 step=44260 loss=0.494420 loss_full=0.209989 loss_base=0.284431 ms_per_step=3557.84
task=ci2v epoch=421 step=44280 loss=0.539076 loss_full=0.218981 loss_base=0.320095 ms_per_step=3350.09
task=ci2v epoch=421 step=44300 loss=0.520948 loss_full=0.221889 loss_base=0.299059 ms_per_step=3347.19
[profile] window=342.05s updates=100 data_wait=1.5ms h2d=1.0ms stage1=209.5ms dit=637.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=421 elapsed=359.16s step=44310
[epoch-start] epoch=422 step=44310 steps_per_epoch=105
task=ci2v epoch=422 step=44320 loss=0.427201 loss_full=0.174125 loss_base=0.253076 ms_per_step=3426.50
task=ci2v epoch=422 step=44340 loss=0.491032 loss_full=0.197938 loss_base=0.293095 ms_per_step=3305.05
task=ci2v epoch=422 step=44360 loss=0.472486 loss_full=0.195215 loss_base=0.277271 ms_per_step=3487.70
task=ci2v epoch=422 step=44380 loss=0.516419 loss_full=0.214966 loss_base=0.301453 ms_per_step=3312.58
task=ci2v epoch=422 step=44400 loss=0.465175 loss_full=0.189658 loss_base=0.275516 ms_per_step=3362.75
[profile] window=338.16s updates=100 data_wait=1.9ms h2d=0.9ms stage1=200.9ms dit=635.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=422 elapsed=354.53s step=44415
[epoch-start] epoch=423 step=44415 steps_per_epoch=105
task=ci2v epoch=423 step=44420 loss=0.453130 loss_full=0.188482 loss_base=0.264648 ms_per_step=3361.39
task=ci2v epoch=423 step=44440 loss=0.487593 loss_full=0.198265 loss_base=0.289329 ms_per_step=3372.59
task=ci2v epoch=423 step=44460 loss=0.544855 loss_full=0.229541 loss_base=0.315314 ms_per_step=3331.73
task=ci2v epoch=423 step=44480 loss=0.577444 loss_full=0.241094 loss_base=0.336349 ms_per_step=3316.17
task=ci2v epoch=423 step=44500 loss=0.590813 loss_full=0.262702 loss_base=0.328110 ms_per_step=3315.70
[profile] window=334.22s updates=100 data_wait=1.4ms h2d=0.8ms stage1=192.9ms dit=634.2ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=423 step=44520 loss=0.442597 loss_full=0.173296 loss_base=0.269301 ms_per_step=3305.84
[epoch-end] epoch=423 elapsed=350.84s step=44520
[epoch-start] epoch=424 step=44520 steps_per_epoch=105
task=ci2v epoch=424 step=44540 loss=0.496605 loss_full=0.208387 loss_base=0.288218 ms_per_step=3428.90
task=ci2v epoch=424 step=44560 loss=0.524262 loss_full=0.214839 loss_base=0.309423 ms_per_step=3423.91
task=ci2v epoch=424 step=44580 loss=0.502317 loss_full=0.218047 loss_base=0.284270 ms_per_step=3389.55
task=ci2v epoch=424 step=44600 loss=0.409544 loss_full=0.163319 loss_base=0.246224 ms_per_step=3482.22
[profile] window=340.87s updates=100 data_wait=1.6ms h2d=0.9ms stage1=206.8ms dit=636.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=424 step=44620 loss=0.494426 loss_full=0.215123 loss_base=0.279303 ms_per_step=3451.45
[epoch-end] epoch=424 elapsed=360.39s step=44625
[epoch-start] epoch=425 step=44625 steps_per_epoch=105
task=ci2v epoch=425 step=44640 loss=0.496608 loss_full=0.201446 loss_base=0.295162 ms_per_step=3357.63
task=ci2v epoch=425 step=44660 loss=0.522557 loss_full=0.213606 loss_base=0.308951 ms_per_step=3307.84
task=ci2v epoch=425 step=44680 loss=0.499479 loss_full=0.211080 loss_base=0.288400 ms_per_step=3328.53
task=ci2v epoch=425 step=44700 loss=0.534524 loss_full=0.219451 loss_base=0.315072 ms_per_step=3352.17
[profile] window=336.23s updates=100 data_wait=1.7ms h2d=0.9ms stage1=196.2ms dit=635.8ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=425 step=44720 loss=0.503751 loss_full=0.210815 loss_base=0.292936 ms_per_step=3332.39
[epoch-end] epoch=425 elapsed=352.29s step=44730
[epoch-start] epoch=426 step=44730 steps_per_epoch=105
task=ci2v epoch=426 step=44740 loss=0.607731 loss_full=0.259837 loss_base=0.347895 ms_per_step=3433.68
task=ci2v epoch=426 step=44760 loss=0.471780 loss_full=0.195737 loss_base=0.276043 ms_per_step=3448.62
task=ci2v epoch=426 step=44780 loss=0.586238 loss_full=0.257114 loss_base=0.329123 ms_per_step=3325.78
task=ci2v epoch=426 step=44800 loss=0.422854 loss_full=0.170873 loss_base=0.251981 ms_per_step=3385.38
[profile] window=338.78s updates=100 data_wait=1.6ms h2d=0.9ms stage1=202.7ms dit=635.5ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=426 step=44820 loss=0.533334 loss_full=0.228692 loss_base=0.304642 ms_per_step=3537.45
[epoch-end] epoch=426 elapsed=357.76s step=44835
[epoch-start] epoch=427 step=44835 steps_per_epoch=105
task=ci2v epoch=427 step=44840 loss=0.454673 loss_full=0.188198 loss_base=0.266476 ms_per_step=3330.04
task=ci2v epoch=427 step=44860 loss=0.464670 loss_full=0.196234 loss_base=0.268436 ms_per_step=3383.30
task=ci2v epoch=427 step=44880 loss=0.536245 loss_full=0.232551 loss_base=0.303694 ms_per_step=3768.62
task=ci2v epoch=427 step=44900 loss=0.452197 loss_full=0.178187 loss_base=0.274010 ms_per_step=3756.13
[profile] window=355.78s updates=100 data_wait=1.5ms h2d=1.3ms stage1=241.3ms dit=639.2ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=427 step=44920 loss=0.445562 loss_full=0.177335 loss_base=0.268227 ms_per_step=3754.64
task=ci2v epoch=427 step=44940 loss=0.482953 loss_full=0.194928 loss_base=0.288025 ms_per_step=3709.40
[epoch-end] epoch=427 elapsed=384.95s step=44940
[epoch-start] epoch=428 step=44940 steps_per_epoch=105
task=ci2v epoch=428 step=44960 loss=0.527968 loss_full=0.228265 loss_base=0.299703 ms_per_step=3853.29
task=ci2v epoch=428 step=44980 loss=0.429309 loss_full=0.170780 loss_base=0.258529 ms_per_step=3795.67
task=ci2v epoch=428 step=45000 loss=0.466088 loss_full=0.189991 loss_base=0.276097 ms_per_step=3781.94
[profile] window=378.16s updates=100 data_wait=2.1ms h2d=1.7ms stage1=288.7ms dit=646.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=45000 elapsed=2.76s
[checkpoint] step=45000 elapsed=42.03s
task=ci2v epoch=428 step=45020 loss=0.555788 loss_full=0.242895 loss_base=0.312893 ms_per_step=3803.21
task=ci2v epoch=428 step=45040 loss=0.468389 loss_full=0.196158 loss_base=0.272231 ms_per_step=3833.75
[epoch-end] epoch=428 elapsed=444.52s step=45045
[epoch-start] epoch=429 step=45045 steps_per_epoch=105
task=ci2v epoch=429 step=45060 loss=0.413847 loss_full=0.160998 loss_base=0.252848 ms_per_step=3869.17
task=ci2v epoch=429 step=45080 loss=0.469470 loss_full=0.198348 loss_base=0.271122 ms_per_step=3532.62
task=ci2v epoch=429 step=45100 loss=0.566954 loss_full=0.246190 loss_base=0.320763 ms_per_step=3339.95
[profile] window=367.82s updates=100 data_wait=2.1ms h2d=1.4ms stage1=266.8ms dit=643.0ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=429 step=45120 loss=0.569423 loss_full=0.248026 loss_base=0.321397 ms_per_step=3392.31
task=ci2v epoch=429 step=45140 loss=0.497648 loss_full=0.204024 loss_base=0.293624 ms_per_step=3333.76
[epoch-end] epoch=429 elapsed=364.33s step=45150
[epoch-start] epoch=430 step=45150 steps_per_epoch=105
task=ci2v epoch=430 step=45160 loss=0.441977 loss_full=0.172397 loss_base=0.269579 ms_per_step=3332.27
task=ci2v epoch=430 step=45180 loss=0.361620 loss_full=0.140594 loss_base=0.221026 ms_per_step=3338.73
task=ci2v epoch=430 step=45200 loss=0.582806 loss_full=0.258286 loss_base=0.324520 ms_per_step=3326.39
[profile] window=334.74s updates=100 data_wait=1.6ms h2d=0.8ms stage1=193.9ms dit=634.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=430 step=45220 loss=0.572064 loss_full=0.253118 loss_base=0.318946 ms_per_step=3345.37
task=ci2v epoch=430 step=45240 loss=0.402890 loss_full=0.162646 loss_base=0.240244 ms_per_step=3771.29
[epoch-end] epoch=430 elapsed=366.17s step=45255
[epoch-start] epoch=431 step=45255 steps_per_epoch=105
task=ci2v epoch=431 step=45260 loss=0.507726 loss_full=0.197918 loss_base=0.309809 ms_per_step=3810.90
task=ci2v epoch=431 step=45280 loss=0.493881 loss_full=0.207308 loss_base=0.286572 ms_per_step=3777.47
task=ci2v epoch=431 step=45300 loss=0.506148 loss_full=0.213756 loss_base=0.292392 ms_per_step=3771.69
[profile] window=369.79s updates=100 data_wait=2.1ms h2d=1.5ms stage1=270.6ms dit=643.9ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=431 step=45320 loss=0.502877 loss_full=0.207771 loss_base=0.295106 ms_per_step=3817.62
task=ci2v epoch=431 step=45340 loss=0.509264 loss_full=0.206663 loss_base=0.302601 ms_per_step=3806.09
task=ci2v epoch=431 step=45360 loss=0.591114 loss_full=0.248162 loss_base=0.342952 ms_per_step=3764.50
[epoch-end] epoch=431 elapsed=399.25s step=45360
[epoch-start] epoch=432 step=45360 steps_per_epoch=105
task=ci2v epoch=432 step=45380 loss=0.530559 loss_full=0.226504 loss_base=0.304055 ms_per_step=3874.84
task=ci2v epoch=432 step=45400 loss=0.488191 loss_full=0.196667 loss_base=0.291524 ms_per_step=3882.62
[profile] window=383.17s updates=100 data_wait=2.1ms h2d=1.7ms stage1=296.6ms dit=651.3ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=432 step=45420 loss=0.445333 loss_full=0.183515 loss_base=0.261818 ms_per_step=3791.97
task=ci2v epoch=432 step=45440 loss=0.483645 loss_full=0.197260 loss_base=0.286384 ms_per_step=4019.14
task=ci2v epoch=432 step=45460 loss=0.416286 loss_full=0.165027 loss_base=0.251259 ms_per_step=3791.67
[epoch-end] epoch=432 elapsed=405.68s step=45465
[epoch-start] epoch=433 step=45465 steps_per_epoch=105
task=ci2v epoch=433 step=45480 loss=0.393099 loss_full=0.151575 loss_base=0.241524 ms_per_step=4144.62
task=ci2v epoch=433 step=45500 loss=0.525826 loss_full=0.217332 loss_base=0.308494 ms_per_step=3837.98
[profile] window=391.96s updates=100 data_wait=2.2ms h2d=1.7ms stage1=310.6ms dit=658.6ms update=29.1ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=433 step=45520 loss=0.466489 loss_full=0.186512 loss_base=0.279977 ms_per_step=3793.70
task=ci2v epoch=433 step=45540 loss=0.465131 loss_full=0.189985 loss_base=0.275147 ms_per_step=3615.31
task=ci2v epoch=433 step=45560 loss=0.484658 loss_full=0.203081 loss_base=0.281577 ms_per_step=3780.35
[epoch-end] epoch=433 elapsed=402.91s step=45570
[epoch-start] epoch=434 step=45570 steps_per_epoch=105
task=ci2v epoch=434 step=45580 loss=0.427030 loss_full=0.172291 loss_base=0.254739 ms_per_step=3847.24
task=ci2v epoch=434 step=45600 loss=0.493535 loss_full=0.199928 loss_base=0.293607 ms_per_step=3791.67
[profile] window=376.81s updates=100 data_wait=2.2ms h2d=1.6ms stage1=284.7ms dit=647.4ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=434 step=45620 loss=0.586703 loss_full=0.257828 loss_base=0.328875 ms_per_step=3782.89
task=ci2v epoch=434 step=45640 loss=0.577777 loss_full=0.251575 loss_base=0.326201 ms_per_step=3794.46
task=ci2v epoch=434 step=45660 loss=0.560267 loss_full=0.237236 loss_base=0.323031 ms_per_step=3804.31
[epoch-end] epoch=434 elapsed=403.08s step=45675
[epoch-start] epoch=435 step=45675 steps_per_epoch=105
task=ci2v epoch=435 step=45680 loss=0.475485 loss_full=0.199766 loss_base=0.275719 ms_per_step=3991.11
task=ci2v epoch=435 step=45700 loss=0.528371 loss_full=0.223577 loss_base=0.304794 ms_per_step=3812.19
[profile] window=383.95s updates=100 data_wait=2.1ms h2d=1.9ms stage1=297.9ms dit=651.7ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=435 step=45720 loss=0.607529 loss_full=0.262770 loss_base=0.344758 ms_per_step=3798.60
task=ci2v epoch=435 step=45740 loss=0.581257 loss_full=0.258672 loss_base=0.322585 ms_per_step=3824.96
task=ci2v epoch=435 step=45760 loss=0.531809 loss_full=0.224404 loss_base=0.307405 ms_per_step=3798.22
task=ci2v epoch=435 step=45780 loss=0.482674 loss_full=0.206791 loss_base=0.275883 ms_per_step=3754.38
[epoch-end] epoch=435 elapsed=400.44s step=45780
[epoch-start] epoch=436 step=45780 steps_per_epoch=105
task=ci2v epoch=436 step=45800 loss=0.449527 loss_full=0.183480 loss_base=0.266047 ms_per_step=3823.43
[profile] window=380.24s updates=100 data_wait=2.0ms h2d=1.7ms stage1=291.2ms dit=649.4ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=436 step=45820 loss=0.555340 loss_full=0.240086 loss_base=0.315254 ms_per_step=3787.07
task=ci2v epoch=436 step=45840 loss=0.484173 loss_full=0.198604 loss_base=0.285569 ms_per_step=3871.28
task=ci2v epoch=436 step=45860 loss=0.446303 loss_full=0.178365 loss_base=0.267938 ms_per_step=3786.09
task=ci2v epoch=436 step=45880 loss=0.498147 loss_full=0.204511 loss_base=0.293637 ms_per_step=3791.16
[epoch-end] epoch=436 elapsed=400.07s step=45885
[epoch-start] epoch=437 step=45885 steps_per_epoch=105
task=ci2v epoch=437 step=45900 loss=0.428723 loss_full=0.170808 loss_base=0.257915 ms_per_step=3868.38
[profile] window=382.34s updates=100 data_wait=2.0ms h2d=1.7ms stage1=293.8ms dit=651.3ms update=31.0ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=437 step=45920 loss=0.449844 loss_full=0.187798 loss_base=0.262046 ms_per_step=3873.89
task=ci2v epoch=437 step=45940 loss=0.530332 loss_full=0.234546 loss_base=0.295786 ms_per_step=3779.06
task=ci2v epoch=437 step=45960 loss=0.541338 loss_full=0.232353 loss_base=0.308985 ms_per_step=3797.69
task=ci2v epoch=437 step=45980 loss=0.451607 loss_full=0.181694 loss_base=0.269913 ms_per_step=3812.59
[epoch-end] epoch=437 elapsed=402.31s step=45990
[epoch-start] epoch=438 step=45990 steps_per_epoch=105
task=ci2v epoch=438 step=46000 loss=0.471267 loss_full=0.194679 loss_base=0.276588 ms_per_step=3800.00
[profile] window=381.52s updates=100 data_wait=2.2ms h2d=1.7ms stage1=291.2ms dit=651.7ms update=31.1ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
wandb: Encoding video...
[sample] step=46000 elapsed=2.53s
[checkpoint] step=46000 elapsed=37.25s
task=ci2v epoch=438 step=46020 loss=0.429959 loss_full=0.181100 loss_base=0.248859 ms_per_step=3855.17
task=ci2v epoch=438 step=46040 loss=0.415543 loss_full=0.163284 loss_base=0.252259 ms_per_step=3864.65
task=ci2v epoch=438 step=46060 loss=0.547611 loss_full=0.238320 loss_base=0.309291 ms_per_step=3792.55
task=ci2v epoch=438 step=46080 loss=0.596156 loss_full=0.276378 loss_base=0.319777 ms_per_step=3786.37
[epoch-end] epoch=438 elapsed=440.65s step=46095
[epoch-start] epoch=439 step=46095 steps_per_epoch=105
task=ci2v epoch=439 step=46100 loss=0.446284 loss_full=0.181088 loss_base=0.265196 ms_per_step=3810.48
[profile] window=382.43s updates=100 data_wait=1.9ms h2d=1.7ms stage1=294.6ms dit=651.6ms update=27.7ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
task=ci2v epoch=439 step=46120 loss=0.542682 loss_full=0.230185 loss_base=0.312497 ms_per_step=3838.73
task=ci2v epoch=439 step=46140 loss=0.520880 loss_full=0.228332 loss_base=0.292548 ms_per_step=3808.94
task=ci2v epoch=439 step=46160 loss=0.521792 loss_full=0.223641 loss_base=0.298151 ms_per_step=3819.70
task=ci2v epoch=439 step=46180 loss=0.493744 loss_full=0.202776 loss_base=0.290968 ms_per_step=3815.03
task=ci2v epoch=439 step=46200 loss=0.427626 loss_full=0.174895 loss_base=0.252731 ms_per_step=3802.40
[profile] window=381.94s updates=100 data_wait=1.2ms h2d=1.7ms stage1=298.2ms dit=647.9ms update=27.8ms allocated=29.0GiB reserved=39.6GiB peak=38.0GiB
[epoch-end] epoch=439 elapsed=402.02s step=46200
[epoch-start] epoch=440 step=46200 steps_per_epoch=105
task=ci2v epoch=440 step=46220 loss=0.495942 loss_full=0.196469 loss_base=0.299473 ms_per_step=3895.81
W0907 11:20:31.316000 34878 site-packages/torch/distributed/elastic/agent/server/api.py:739] Received Signals.SIGTERM death signal, shutting down workers
W0907 11:20:31.336000 34878 site-packages/torch/distributed/elastic/multiprocessing/api.py:1010] Sending process 34953 closing signal SIGTERM
Traceback (most recent call last):
  File "/zsh/miniconda3/envs/vrae/bin/torchrun", line 6, in <module>
    sys.exit(main())
  File "/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 362, in wrapper
    return f(*args, **kwargs)
  File "/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/torch/distributed/run.py", line 991, in main
    run(args)
  File "/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/torch/distributed/run.py", line 982, in run
    elastic_launch(
  File "/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 170, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 308, in launch_agent
    result = agent.run()
  File "/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/torch/distributed/elastic/metrics/api.py", line 134, in wrapper
    result = f(*args, **kwargs)
  File "/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/torch/distributed/elastic/agent/server/api.py", line 731, in run
    result = self._invoke_run(role)
  File "/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/torch/distributed/elastic/agent/server/api.py", line 908, in _invoke_run
    time.sleep(monitor_interval)
  File "/zsh/miniconda3/envs/vrae/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/api.py", line 86, in _terminate_process_handler
    raise SignalException(f"Process {os.getpid()} got signal: {sigval}", sigval=sigval)
torch.distributed.elastic.multiprocessing.api.SignalException: Process 34878 got signal: 15
