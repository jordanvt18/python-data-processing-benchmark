from setuptools import setup, find_packages

setup(
    name="data-processing-benchmark",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'pandas>=2.0.0',
        'polars>=0.19.0',
        'datatable>=1.0.0',
        'numpy>=1.20.0',
        'jupyter>=1.0.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
    ],
    author="Jordan V",
    author_email="jordanviion@gmail.com",
    description="Benchmark comparativo entre Pandas, Polars y Data.table",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jordanvt18/python-data-processing-benchmark",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
