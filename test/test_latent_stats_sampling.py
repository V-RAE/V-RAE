from types import SimpleNamespace
from unittest import TestCase

from vrae.training.common.engine import episode_start_indices


class EpisodeStartIndicesTest(TestCase):
    def test_selects_one_clip_per_episode(self) -> None:
        dataset = SimpleNamespace(
            meta=SimpleNamespace(
                episodes=[
                    {"dataset_from_index": 0, "dataset_to_index": 214},
                    {"dataset_from_index": 214, "dataset_to_index": 350},
                    {"dataset_from_index": 350, "dataset_to_index": 499},
                ]
            )
        )

        self.assertEqual(episode_start_indices(dataset), [0, 214, 350])
