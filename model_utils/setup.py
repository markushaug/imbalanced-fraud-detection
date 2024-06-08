# setup.py
from setuptools import setup, find_packages

VERSION='0.1.0'

setup(
    name="model_utils",
    version=VERSION,
    packages=find_packages(),
    description="Local Model Utils",
    author="Markus Haug",
    author_email="mh@haugmarkus.com",
    keywords=["machine learning", "helpers", "ml utils", "model evaluation"],
    install_requires=[
        "torch",
        "pandas",
        "scikit-learn",
		"scikeras",
		"xgboost",
        "numpy",
        "imblearn",
        "matplotlib",
		"keras",
		"imbalanced-ensemble",
		"imbalanced-ensemble",
		"imblearn"
    ]
)
