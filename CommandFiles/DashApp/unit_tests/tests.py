import unittest
from DashApp.func import Classifier
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

df = pd.DataFrame()
class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_prepare_new_text(self):
        df = pd.DataFrame()
        test_text = 'Adam iż Maciek. lubi a jeść –Kokosy nj\nd ajn'
        expected_text1 = 'adam iż maciek lubi a jeśćkokosy nj d ajn'
        expected_text2 = 'adam maciek lubi jeśćkokosy nj d ajn'


        boss = Classifier(MultinomialNB(), 1, df, False)


        self.assertEqual(expected_text1, Classifier(MultinomialNB(), 1, df, True).prepare_new_text(test_text))
        self.assertEqual(expected_text2, Classifier(MultinomialNB(), 2, df, True).prepare_new_text(test_text))
        self.assertEqual(expected_text2, Classifier(MultinomialNB(), 1, df, False).prepare_new_text(test_text))
        self.assertEqual(expected_text2, Classifier(MultinomialNB(), 2, df, False).prepare_new_text(test_text))
        self.assertEqual(expected_text2, Classifier(MultinomialNB(), 2, df, False).prepare_new_text(test_text))
        self.assertEqual(expected_text2, Classifier(MultinomialNB(), 2, df, False).prepare_new_text(test_text))
        self.assertEqual(expected_text2, Classifier(MultinomialNB(), 2, df, False).prepare_new_text(test_text))
        self.assertEqual(expected_text2, Classifier(MultinomialNB(), 2, df, False).prepare_new_text(test_text))
        self.assertEqual(expected_text2, Classifier(MultinomialNB(), 2, df, False).prepare_new_text(test_text))
        self.assertEqual(expected_text2, Classifier(MultinomialNB(), 2, df, False).prepare_new_text(test_text))
if __name__ == '__main__':
    unittest.main()
