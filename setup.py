import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="danish-bert-embeddings", 
    version="0.0.1",
    author="Kasper Junge",
    author_email="kasperjuunge@gmail.com",
    description="A package to create danish word/sentence embeddings with BERT.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KasperJuunge/danish-bert-embeddings",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
    python_requires='>=3.8.5',
)


