from setuptools import setup

setup(
    name="naive_bayes_plus",
    version="1.0",
    author="Kalafinaian",
    author_email="Kalafinaian@outlook.com",
    keywords="naive bayes classification classifier",
    packages=["naivebayes", "simple"],
    description="Naive Bayes Text Classification",
    install_requires=[
        "numpy",
    ],
)