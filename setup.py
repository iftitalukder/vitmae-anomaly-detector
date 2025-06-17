from setuptools import setup, find_packages

setup(
    name='vitmae_anomaly',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'timm',
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'scikit-image',
        'torchvision'
    ],
    author='Khalid Hossen',
    description='ViT-MAE based unsupervised anomaly detection for time series',
    license='MIT',
)
