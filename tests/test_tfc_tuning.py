import unittest

import numpy as np
import yaml

from tfc_benchmark.tuning import _native_params


class TestTFCTuning(unittest.TestCase):
    def test_optimizer_values_are_yaml_safe(self):
        params = _native_params(
            ["num_experts", "top_k", "weight"],
            [np.int64(8), np.int32(2), np.float64(0.125)],
        )

        self.assertIs(type(params["num_experts"]), int)
        self.assertIs(type(params["top_k"]), int)
        self.assertIs(type(params["weight"]), float)
        dumped = yaml.safe_dump({"model_params": params})
        self.assertIn("num_experts: 8", dumped)


if __name__ == "__main__":
    unittest.main()
