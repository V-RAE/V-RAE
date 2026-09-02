from unittest import TestCase

import torch

from vrae.training.common.profiler import TrainingProfiler, run_fine_profile


class TrainingProfilerTest(TestCase):
    def test_cpu_summary_reports_wait_and_throughput(self) -> None:
        profiler = TrainingProfiler(torch.device("cpu"))
        profiler.record_data_wait(0.01)
        profiler.record_data_wait(0.03)
        profiler.begin_batch()
        profiler.finish_h2d()
        profiler.finish_stage1()
        profiler.finish_backward()
        profiler.finish_update()

        metrics = profiler.summarize(
            window_seconds=2.0,
            updates=4,
            global_batch_size=8,
        )

        self.assertAlmostEqual(metrics["train/data_wait_ms_mean"], 20.0)
        self.assertAlmostEqual(metrics["train/data_wait_ms_max"], 30.0)
        self.assertAlmostEqual(metrics["train/updates_per_second"], 2.0)
        self.assertAlmostEqual(metrics["train/samples_per_second"], 16.0)

    def test_fine_profile_disabled_runs_callback_without_trace(self) -> None:
        calls: list[int] = []
        run_fine_profile(calls.append, steps=3)
        self.assertEqual(calls, [0, 1, 2])
