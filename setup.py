import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='nestednereval',
    version='0.0.1',
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
    packages=['nestednereval'],
    install_requires=['requests'],
)