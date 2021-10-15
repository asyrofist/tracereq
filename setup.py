import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tracereq",
    version="0.0.4",
    author="Asyrofist (Rakha Asyrofi)",
    author_email="rakhasyrofist@gmail.com",
    description="Berikut ini adalah deskripsi singkat bagaimana program tracebility requirement dibuat",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/asyrofist/Simple-Traceability-SRS-Document",
    project_urls={
        "Bug Tracker": "https://github.com/asyrofist/tracereqissues",
        "Documentation": "https://asyrofist.github.io/tracereq/",
        "Source Code": "https://github.com/asyrofist/tracereq",
        "Changelog": "https://asyrofist.github.io/tracereq/CHANGELOG/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)