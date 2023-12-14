import setuptools

EXTRAS_REQUIRE = {
    "fmu": ["FMPy>0.2.23"],
    "ml": ["keras>=2.6.0", "tensorflow>=2.6.0", "scikit-learn"],
}
FULL_REQUIRES = []
for OPTIONAL_REQUIRES in EXTRAS_REQUIRE.values():
    FULL_REQUIRES.extend(OPTIONAL_REQUIRES)
EXTRAS_REQUIRE.update({"full": FULL_REQUIRES})

INSTALL_REQUIRES = [
    "numpy>=1.17.4",
    "pandas>=1.1.0",
    "scipy>=1.5.2",
    "simpy>=4.0.1",
    "pydantic>=1.10.4",
    "casadi>=3.5.5",
    "matplotlib>=3.5.1",
    "rwth-CD-colors",
    "matplotlib",
    "attrs>=22.2.0",
    "agentlib>=0.6.0",
    'orjson>=3.9.5',
    'agentlib>=0.6.0',
]

setuptools.setup(
    name="agentlib_mpc",
    version="0.4.0",
    author="Associates of the AGENT project",
    author_email="AGENT.Projekt@eonerc.rwth-aachen.de",
    description="Plugin for the agentlib. Includes tools to run model predictive "
    "controllers.",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    extras_require=EXTRAS_REQUIRE,
    install_requires=INSTALL_REQUIRES,
)
