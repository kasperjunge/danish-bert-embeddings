import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="danish-bert-embeddings", 
    version="0.0.8",
    author="Kasper Junge",
    author_email="kasperjuunge@gmail.com",
    description="A package to create danish word/sentence embeddings with BERT.",
    url="https://github.com/KasperJuunge/danish-bert-embeddings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'transformers',
        'torch'
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    include_package_data=True
)


