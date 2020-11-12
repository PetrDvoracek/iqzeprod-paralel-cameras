import unittest
from explore_yolo import build_is_c_k_onlyc_onlyk

from pandas import DataFrame

class TestIsCKOnlyCOnlyK(unittest.TestCase):
    is_crack, is_knot, is_only_crack, is_only_knot = build_is_c_k_onlyc_onlyk('error_kind')
    
    def test_is_crack(self):
        df = DataFrame({'error_kind': ['Crack', 'Knot_OK']})
        self.assertTrue(TestIsCKOnlyCOnlyK.is_crack(df))
    
    def test_is_knot(self):
        df = DataFrame({'error_kind': ['Crack', 'Knot_OK']})
        self.assertTrue(TestIsCKOnlyCOnlyK.is_knot(df))
    
    def test_is_only_crack(self):
        df = DataFrame({'error_kind': ['Crack', 'Knot_OK']})
        self.assertFalse(TestIsCKOnlyCOnlyK.is_only_crack(df))
        df = DataFrame({'error_kind': ['Crack', 'random']})
        self.assertTrue(TestIsCKOnlyCOnlyK.is_only_crack(df))
        df = DataFrame({'error_kind': ['Crack', 'knot_with_crack', 'random']})
        self.assertFalse(TestIsCKOnlyCOnlyK.is_only_crack(df))
    
    def test_is_only_knot(self):
        df = DataFrame({'error_kind': ['Crack', 'Knot_OK']})
        self.assertFalse(TestIsCKOnlyCOnlyK.is_only_knot(df))
        df = DataFrame({'error_kind': ['Knot_OK', 'random']})
        self.assertTrue(TestIsCKOnlyCOnlyK.is_only_knot(df))
        df = DataFrame({'error_kind': ['Knot_black', 'random']})
        self.assertTrue(TestIsCKOnlyCOnlyK.is_only_knot(df))

if __name__ == '__main__':
    unittest.main()

