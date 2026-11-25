import unittest

import torch

from tfc_benchmark.models import MODEL_NAMES, build_model


class TestTFCModels(unittest.TestCase):
    def _batch(self, batch_size=2):
        return {
            "tongue_image": torch.randn(batch_size, 3, 64, 64),
            "face_image": torch.randn(batch_size, 3, 64, 64),
            "clinical": torch.randn(batch_size, 10),
            "modality_mask": torch.tensor([[1, 1, 1], [1, 0, 1]]).float(),
            "targets": torch.randint(0, 2, (batch_size, 3)).float(),
            "target_mask": torch.ones(batch_size, 3),
        }

    def test_all_models_forward(self):
        batch = self._batch()
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                model = build_model(
                    name,
                    hidden_dim=32,
                    image_backbone="tiny",
                    pretrained=False,
                    dropout=0.1,
                )
                output = model(batch)
                self.assertEqual(tuple(output.logits.shape), (2, 3))
                self.assertEqual(output.aux_loss.ndim, 0)
                self.assertTrue(torch.isfinite(output.logits).all())
                self.assertTrue(torch.isfinite(output.aux_loss))


if __name__ == "__main__":
    unittest.main()

