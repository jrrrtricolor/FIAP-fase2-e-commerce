import importlib
import unittest


class EcommerceRecommenderImportTest(unittest.TestCase):
    def test_package_imports_successfully(self):
        package = importlib.import_module("ecommerce_recommender")

        self.assertIsNotNone(package)


if __name__ == "__main__":
    unittest.main()
