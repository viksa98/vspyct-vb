from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Adding uncertainty estimation in oblique PCTs with GP leaves',
    author='Viktor',
    license='MIT',
    install_requires=[
        'torch>=1.9.0',
        'pyro-ppl>=1.8.0',
        'gpytorch>=1.9.0',
        'numpy>=1.19.0',
        'scikit-learn>=0.24.0',
        'tqdm>=4.60.0',
    ],
)
