import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gl-coarsener",  # Replace with your own username
    version="1.0.0",
    author="Reza Namazi",
    author_email="rezanmz@ymail.com",
    description="GL-Coarsener, a graph learning based coarsening method",
    keywords=['multigrid', 'gl-coarsener', 'graph', 'machine learning',
              ' graph embedding', 'representation learning'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rezanmz/GL-Coarsener",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    python_requires='>=3.6',
)
