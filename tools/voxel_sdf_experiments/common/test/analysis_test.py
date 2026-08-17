import unittest

from tools.voxel_sdf_experiments.common import analysis


class AnalysisTest(unittest.TestCase):
    @staticmethod
    def _rows(values, *, settled=None):
        if settled is None:
            settled = [True] * len(values)
        widths = [4.0 / (2**index) for index in range(len(values))]
        return [
            {
                "scene": "test_scene",
                "representation": "test_representation",
                "voxel_width_m": str(width),
                "error": str(value),
                "settled": str(is_settled),
            }
            for width, value, is_settled in zip(
                widths, values, settled, strict=True
            )
        ]

    @staticmethod
    def _value(row, metric_name):
        return float(row[metric_name])

    def test_clean_second_order_series(self):
        fits, classifications = analysis.order_fits(
            self._rows([16.0, 4.0, 1.0]),
            (analysis.Metric("error", 0.0, "error"),),
            self._value,
        )
        self.assertEqual(classifications, [])
        self.assertEqual(len(fits), 1)
        self.assertAlmostEqual(fits[0]["order"], 2.0)
        self.assertAlmostEqual(fits[0]["log_residual_rms"], 0.0)

    def test_identically_zero_series(self):
        fits, classifications = analysis.order_fits(
            self._rows([0.0, 0.0, 0.0]),
            (analysis.Metric("error", 1e-12, "error"),),
            self._value,
        )
        self.assertEqual(fits, [])
        self.assertEqual(classifications[0]["reason"], "identically_zero")

    def test_series_at_noise_floor(self):
        fits, classifications = analysis.order_fits(
            self._rows([0.8, 0.4, 0.2]),
            (analysis.Metric("error", 1.0, "error"),),
            self._value,
        )
        self.assertEqual(fits, [])
        self.assertEqual(classifications[0]["reason"], "at_noise_floor")

    def test_partially_below_floor_series(self):
        fits, classifications = analysis.order_fits(
            self._rows([16.0, 4.0, 0.5]),
            (analysis.Metric("error", 1.0, "error"),),
            self._value,
        )
        self.assertEqual(len(fits), 1)
        self.assertAlmostEqual(fits[0]["order"], 2.0)
        self.assertEqual(classifications[0]["reason"], "partially_below_floor")
        self.assertIn("fit uses the rest", classifications[0]["detail"])

    def test_too_few_settled_rungs(self):
        fits, classifications = analysis.order_fits(
            self._rows([16.0, 4.0, 1.0], settled=[True, False, False]),
            (analysis.Metric("error", 0.0, "error"),),
            self._value,
            row_gate=analysis.RowGate(
                "settled", lambda row: row["settled"] == "True"
            ),
        )
        self.assertEqual(fits, [])
        self.assertEqual(classifications[0]["reason"], "too_few_settled_rungs")

    def test_no_settled_rungs(self):
        fits, classifications = analysis.order_fits(
            self._rows([16.0, 4.0], settled=[False, False]),
            (analysis.Metric("error", 0.0, "error"),),
            self._value,
            row_gate=analysis.RowGate(
                "settled", lambda row: row["settled"] == "True"
            ),
        )
        self.assertEqual(fits, [])
        self.assertEqual(classifications[0]["reason"], "no_settled_rungs")

    def test_unsettled_rung_is_excluded_and_reported(self):
        fits, classifications = analysis.order_fits(
            self._rows([16.0, 4.0, 1.0e12], settled=[True, True, False]),
            (analysis.Metric("error", 0.0, "error"),),
            self._value,
            row_gate=analysis.RowGate(
                "settled", lambda row: row["settled"] == "True"
            ),
        )
        self.assertEqual(len(fits), 1)
        self.assertEqual(fits[0]["samples"], 2)
        self.assertEqual(fits[0]["eligible_rungs"], 2)
        self.assertAlmostEqual(fits[0]["order"], 2.0)
        self.assertEqual(len(classifications), 1)
        self.assertEqual(
            classifications[0]["reason"],
            "excluded_non_settled_rungs",
        )
        self.assertIn("excluded from the fit", classifications[0]["detail"])

    def test_saturated_series(self):
        fits, classifications = analysis.order_fits(
            self._rows([0.9, 0.8, 0.7]),
            (analysis.Metric("error", 0.0, "error", ceiling=0.5),),
            self._value,
        )
        self.assertEqual(fits, [])
        self.assertEqual(classifications[0]["reason"], "saturated")
