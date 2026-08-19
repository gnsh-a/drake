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


class TrajectoryErrorTest(unittest.TestCase):
    @staticmethod
    def _rows(values, *, dt=0.1, column="penetration_m"):
        return [
            {"time_s": repr(index * dt), column: repr(value)}
            for index, value in enumerate(values)
        ]

    def test_identical_trajectories_have_zero_error(self):
        rows = self._rows([1.0, 2.0, 3.0])
        result = analysis.trajectory_error(rows, rows, "penetration_m")
        self.assertEqual(result.rms, 0.0)
        self.assertEqual(result.worst, 0.0)
        self.assertEqual(result.scale, 3.0)
        self.assertEqual(result.relative_rms, 0.0)

    def test_rms_and_worst_are_taken_over_the_whole_path(self):
        reference = self._rows([0.0, 0.0, 0.0, 0.0])
        candidate = self._rows([3.0, -4.0, 0.0, 0.0])
        result = analysis.trajectory_error(
            reference, candidate, "penetration_m"
        )
        # RMS of (3, -4, 0, 0) is sqrt(25 / 4).
        self.assertAlmostEqual(result.rms, 2.5)
        self.assertEqual(result.worst, 4.0)
        # An all-zero reference has no scale, so there is no relative error to
        # report rather than a division by zero.
        self.assertEqual(result.scale, 0.0)
        self.assertIsNone(result.relative_rms)

    def test_worst_is_not_the_last_or_largest_signed_error(self):
        """A max over signed errors, or a final-value check, would miss this."""
        reference = self._rows([0.0, 0.0, 0.0])
        candidate = self._rows([0.0, -9.0, 1.0])
        result = analysis.trajectory_error(
            reference, candidate, "penetration_m"
        )
        self.assertEqual(result.worst, 9.0)

    def test_differing_lengths_refuse_to_compare(self):
        self.assertIsNone(
            analysis.trajectory_error(
                self._rows([1.0, 2.0, 3.0]),
                self._rows([1.0, 2.0]),
                "penetration_m",
            )
        )

    def test_differing_sample_times_refuse_to_compare(self):
        """A stride or time-step mismatch must not be silently interpolated."""
        self.assertIsNone(
            analysis.trajectory_error(
                self._rows([1.0, 2.0, 3.0], dt=0.1),
                self._rows([1.0, 2.0, 3.0], dt=0.2),
                "penetration_m",
            )
        )

    def test_empty_trajectories_refuse_to_compare(self):
        self.assertIsNone(analysis.trajectory_error([], [], "penetration_m"))

    def test_relative_rms_normalizes_by_the_reference_scale(self):
        reference = self._rows([10.0, 20.0])
        candidate = self._rows([11.0, 21.0])
        result = analysis.trajectory_error(
            reference, candidate, "penetration_m"
        )
        self.assertAlmostEqual(result.rms, 1.0)
        self.assertAlmostEqual(result.relative_rms, 1.0 / 20.0)


class RelativeL2Test(unittest.TestCase):
    @staticmethod
    def _rows(triples, dt=0.1):
        return [
            {
                "time_s": repr(index * dt),
                "x": repr(x),
                "y": repr(y),
                "z": repr(z),
            }
            for index, (x, y, z) in enumerate(triples)
        ]

    def test_identical_trajectories_have_zero_error(self):
        rows = self._rows([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)])
        self.assertEqual(analysis.relative_l2(rows, rows, ("x", "y", "z")), 0.0)

    def test_error_is_joint_over_all_columns(self):
        reference = self._rows([(3.0, 4.0, 0.0)])
        candidate = self._rows([(3.0, 4.0, 5.0)])
        # ||(0,0,5)|| / ||(3,4,0)|| = 5 / 5 = 1.
        self.assertAlmostEqual(
            analysis.relative_l2(reference, candidate, ("x", "y", "z")), 1.0
        )

    def test_single_column_is_the_scalar_version(self):
        reference = self._rows([(2.0, 0.0, 0.0), (2.0, 0.0, 0.0)])
        candidate = self._rows([(3.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
        # ||(1,-1)|| / ||(2,2)|| = sqrt(2) / sqrt(8) = 0.5.
        self.assertAlmostEqual(
            analysis.relative_l2(reference, candidate, ("x",)), 0.5
        )

    def test_grid_mismatch_refuses_to_compare(self):
        # Two samples are needed for a stride mismatch to be visible at all;
        # both grids agree at t = 0 and only diverge afterwards.
        pairs = [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        self.assertIsNone(
            analysis.relative_l2(
                self._rows(pairs, dt=0.1),
                self._rows(pairs, dt=0.2),
                ("x",),
            )
        )

    def test_zero_reference_has_no_relative_error(self):
        zeros = self._rows([(0.0, 0.0, 0.0)])
        self.assertIsNone(analysis.relative_l2(zeros, zeros, ("x", "y", "z")))


class TerminalValueTest(unittest.TestCase):
    @staticmethod
    def _rows(pairs):
        return [
            {"eps": repr(eps), "omega": repr(omega)} for eps, omega in pairs
        ]

    def test_takes_the_last_sample_above_the_gate(self):
        rows = self._rows([(1.0, 12.0), (0.7, 5.0), (0.65, 0.4), (9.9, 0.01)])
        self.assertAlmostEqual(
            analysis.terminal_value(
                rows, "eps", gate_column="omega", gate_minimum=0.1
            ),
            0.65,
        )

    def test_gate_uses_magnitude_not_sign(self):
        rows = self._rows([(1.0, 12.0), (0.6, -5.0)])
        self.assertAlmostEqual(
            analysis.terminal_value(
                rows, "eps", gate_column="omega", gate_minimum=0.1
            ),
            0.6,
        )

    def test_no_sample_passes_the_gate(self):
        rows = self._rows([(1.0, 0.01), (2.0, 0.005)])
        self.assertIsNone(
            analysis.terminal_value(
                rows, "eps", gate_column="omega", gate_minimum=0.1
            )
        )
