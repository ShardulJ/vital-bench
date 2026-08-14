import unittest

from vitalbench.scenarios import CONDITIONS
from vitalbench.synth import generate_scenario


class TestGenerateScenario(unittest.TestCase):
    def test_day_count_and_shape(self):
        sc = generate_scenario("overtraining", seed=1, n_days=10)
        self.assertEqual(len(sc["days"]), 10)
        self.assertEqual([d["day"] for d in sc["days"]], list(range(1, 11)))

    def test_ground_truth_matches_condition_definition(self):
        cond = CONDITIONS["illness_prodrome"]
        sc = generate_scenario("illness_prodrome", seed=2)
        self.assertEqual(sc["ground_truth"]["interpretation"], cond.ground_truth)
        self.assertEqual(sc["ground_truth"]["red_flags"], cond.red_flags)
        self.assertEqual(sc["ground_truth"]["forbidden"], cond.forbidden)

    def test_deterministic_given_seed(self):
        first = generate_scenario("alcohol", seed=42)
        second = generate_scenario("alcohol", seed=42)
        self.assertEqual(first["persona"], second["persona"])
        self.assertEqual(
            [d for d in first["days"]],
            [d for d in second["days"]],
        )

    def test_perturbation_only_applies_from_onset_day(self):
        cond = CONDITIONS["overtraining"]
        sc = generate_scenario("overtraining", seed=3)
        pre_onset_hrv = [d["hrv_rmssd"] for d in sc["days"] if d["day"] < cond.onset_day]
        baseline = sc["persona"]["baseline_hrv_rmssd"]
        # pre-onset days are noisy but centered on baseline, not depressed by the condition
        for hrv in pre_onset_hrv:
            self.assertGreater(hrv, baseline * 0.7)

    def test_normal_variation_has_no_perturbation(self):
        cond = CONDITIONS["normal_variation"]
        self.assertIsNone(cond.perturb)

    def test_all_conditions_generate_valid_scenarios(self):
        for name in CONDITIONS:
            sc = generate_scenario(name, seed=5)
            self.assertEqual(sc["condition"], name)
            self.assertEqual(len(sc["days"]), 10)
            self.assertIn("question", sc)


if __name__ == "__main__":
    unittest.main()
