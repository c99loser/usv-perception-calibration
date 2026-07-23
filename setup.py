from setuptools import find_packages, setup


package_name = "competition_user"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("tests",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/competition_user"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="competition user",
    maintainer_email="user@example.invalid",
    description="USVSim perception entry point.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "perception_user = competition_user.perception_user:main",
        ],
    },
)
