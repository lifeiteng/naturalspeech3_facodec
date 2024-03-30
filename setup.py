from setuptools import find_packages, setup

setup(
    name="ns3_codec",
    version="0.2.0",
    packages=find_packages(where="codec"),
    package_dir={"": "codec"},
    package_data={"": []},
    description="ns3_codec",
    author="The ns3_codec Development Team",
    long_description=open("ns3_codec/README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="ns3_codec/README.md",
    python_requires=">=3.8",
    install_requires=[],
    include_package_data=True,
)
