from setuptools import setup, find_packages
from pathlib import Path

version = {}
with open("netmedgpt/version.py") as f:
    exec(f.read(), version)

root = Path(__file__).parent
requirements = (root / "requirements.txt").read_text().splitlines()


setup(
    name="netmedgpt",
    version=version["__version__"],
    description="NetMedGPT: a foundation model for network medicine",
    long_description=(root / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="NetMedGPT Team",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
)

