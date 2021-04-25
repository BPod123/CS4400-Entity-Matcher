""" This is a series of tests that splits the training data and uses part of it for training, and the other part as
a validation set.
"""
import unittest
from os.path import abspath, join
import pandas as pd
from src.evaluate import calculate_scores, generate_output
from src.data_handler import split_table, trim_tables

train, ltable, rtable = trim_tables(pd.read_csv(join(abspath('../data'), "train.csv")),
                                    pd.read_csv(join(abspath('../data'), "ltable.csv")),
                                    pd.read_csv(join(abspath('../data'), "rtable.csv")))


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.percentage = 0
        self.random = True
        self.string = []

    def test_5_95(self):
        self.percentage = 0.05
        self.check_passed()

    def test_10_90(self):
        self.percentage = 0.10
        self.check_passed()

    def test_15_85(self):
        self.percentage = 0.15
        self.check_passed()

    def test_20_80(self):
        self.percentage = 0.20
        self.check_passed()

    def test_25_75(self):
        self.percentage = 0.25
        self.check_passed()

    def test_40_60(self):
        self.percentage = 0.4
        self.check_passed()

    def test_50_50(self):
        self.percentage = 0.5
        self.check_passed()


    def test_60_40(self):
        self.percentage = 0.6
        self.check_passed()

    def test_75_25(self):
        self.percentage = 0.75
        self.check_passed()

    def test_80_20(self):
        self.percentage = 0.80
        self.check_passed()

    def test_85_15(self):
        self.percentage = 0.15
        self.check_passed()

    def test_90_10(self):
        self.percentage = 0.90
        self.check_passed()

    def test_95_5(self):
        self.percentage = 0.95
        self.check_passed()

    def check_passed(self):
        training_set, validation_set = split_table(train, self.percentage, self.random)
        generate_output(training_set, ltable, rtable, "label")
        precision, recall, F1 = calculate_scores(validation_set)
        n1 = self.percentage * 100
        n2 = 100 - n1
        order_string = "random" if self.random else "ordered"
        string_addition = "{0}-{1} {2} split:\tPrecision: {3}\tRecall: {4}\tF1: {5}".format(n1, n2, order_string, precision, recall,
                                                                              F1)
        self.string.append(string_addition)
        print(string_addition)
        self.assertGreater(F1, 0)
        # self.assertGreaterEqual(F1, 0.032)

    def tearDown(self) -> None:
        from os.path import abspath
        f = open(abspath("../out/log.txt"), "a")
        for s in self.string:
            f.write("{0}\n".format(s))
            # print(s)
        f.close()












if __name__ == '__main__':
    unittest.main()
