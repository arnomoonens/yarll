import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="yarll",
    version="0.0.5",
    author="Arno Moonens",
    author_email="arno.moonens@outlook.com",
    description="Yet Another Reinforcement Learning Library",
    license="MIT",
    keywords="deep learning reinforcement learning a3c ddpg sac ppo machine neural networks",
    python_requires=">=3.4",
    install_requires=[
        "numpy",
        "gym>=0.8.0",
        "tensorflow>=1.6.0",
        "matplotlib",
        "scipy",
        "opencv-contrib-python"
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arnomoonens/YARLL",
    packages=setuptools.find_packages(),
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering',
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    zip_safe=False
)
