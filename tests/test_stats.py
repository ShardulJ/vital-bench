import unittest

from vitalbench.stats import bootstrap_ci, cohens_kappa, judge_validation


class TestBootstrapCi(unittest.TestCase):
    def test_mean_and_bounds(self):
        values = [3, 4, 4, 5, 3, 4, 5, 4]
        mean, lo, hi = bootstrap_ci(values, n_boot=2000, seed=0)
        self.assertAlmostEqual(mean, sum(values) / len(values))
        self.assertLessEqual(lo, mean)
        self.assertLessEqual(mean, hi)

    def test_constant_values_zero_width_ci(self):
        mean, lo, hi = bootstrap_ci([5, 5, 5, 5], n_boot=500, seed=0)
        self.assertEqual((mean, lo, hi), (5.0, 5.0, 5.0))

    def test_deterministic_given_seed(self):
        values = [1, 2, 3, 4, 5]
        first = bootstrap_ci(values, n_boot=1000, seed=7)
        second = bootstrap_ci(values, n_boot=1000, seed=7)
        self.assertEqual(first, second)


class TestCohensKappa(unittest.TestCase):
    def test_perfect_agreement(self):
        labels = [1, 2, 3, 4, 5, 1, 2, 3]
        self.assertAlmostEqual(cohens_kappa(labels, labels), 1.0)

    def test_systematic_disagreement_below_one(self):
        a = [1, 1, 1, 2, 2, 2]
        b = [2, 2, 2, 1, 1, 1]
        self.assertLess(cohens_kappa(a, b), 1.0)

    def test_all_same_label_is_perfect_by_convention(self):
        self.assertEqual(cohens_kappa([3, 3, 3], [3, 3, 3]), 1.0)


class TestJudgeValidation(unittest.TestCase):
    def test_only_common_scenarios_counted(self):
        judge = {"a": 5, "b": 4, "c": 3}
        human = {"a": 5, "b": 4, "d": 1}
        result = judge_validation(judge, human)
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["exact_agreement"], 1.0)

    def test_within_one_tolerance(self):
        judge = {"a": 4, "b": 2}
        human = {"a": 5, "b": 5}
        result = judge_validation(judge, human)
        self.assertEqual(result["exact_agreement"], 0.0)
        self.assertEqual(result["within_one"], 0.5)


if __name__ == "__main__":
    unittest.main()
