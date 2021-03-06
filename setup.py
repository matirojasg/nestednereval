import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name='nestednereval',
    version='0.0.2',
    author='Matias Rojas',
    author_email='matias.rojas.g@ug.uchile.cl',
    description='Python framework for evaluating nested NER systems.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/matirojasg/nestednereval',
    project_urls = {
        "Bug Tracker": "https://github.com/matirojasg/nestednereval/issues"
    },
    license='MIT',
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=install_requires,
    include_package_data=True,
)