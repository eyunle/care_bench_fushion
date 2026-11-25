import unittest

import numpy as np
import pandas as pd
from PIL import Image

from tfc_benchmark.data import fit_clinical_stats, pad_face_to_square, transform_clinical
from tfc_benchmark.manifest import grouped_multilabel_split


class TestTFCData(unittest.TestCase):
    def test_face_padding(self):
        image = Image.new("RGB", (384, 576), (0, 0, 0))
        padded = pad_face_to_square(image)
        self.assertEqual(padded.size, (576, 576))

    def test_clinical_vector_has_values_and_masks(self):
        frame = pd.DataFrame(
            {
                "age": [40.0, 50.0],
                "HEIGHT": [170.0, 160.0],
                "WEIGHT": [60.0, np.nan],
                "gender": [0, 1],
                "waist": [70.0, 80.0],
            }
        )
        stats = fit_clinical_stats(frame)
        vector = transform_clinical(frame.iloc[1], stats)
        self.assertEqual(vector.shape, (10,))
        self.assertEqual(vector[7], 1.0)

    def test_patient_split_is_disjoint(self):
        rows = []
        for patient in range(30):
            rows.append(
                {
                    "id_register": str(patient),
                    "visit_year": str(2022 + patient % 3),
                    "影像学脂肪肝": float(patient % 2),
                    "影像学脂肪肝_mask": 1,
                    "糖尿病前期": float(patient % 3 == 0),
                    "糖尿病前期_mask": 1,
                    "MAFLD": float(patient % 4 == 0),
                    "MAFLD_mask": 1,
                }
            )
        frame = pd.DataFrame(rows)
        assignments = grouped_multilabel_split(frame)
        self.assertEqual(len(assignments), 30)
        self.assertEqual(set(assignments.values()), {"train", "val", "test"})


if __name__ == "__main__":
    unittest.main()

