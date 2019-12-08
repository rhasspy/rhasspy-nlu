import os
import setuptools

this_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_dir, "README.md"), "r") as readme_file:
    long_description = readme_file.read()

setuptools.setup(
    name="rhasspy-nlu",
    version="0.1",
    author="Michael Hansen",
    author_email="hansen.mike@gmail.com",
    url="https://github.com/synesthesiam/rhasspy-nlu",
    packages=setuptools.find_packages(),
    install_requires=["attrs==19.1.0", "networkx==2.3"],
    classifiers=["Programming Language :: Python :: 3"],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
