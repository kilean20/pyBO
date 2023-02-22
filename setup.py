from setuptools import setup, find_packages

# version = {}
# with open('pyGPGO/version.py') as fp:
#     exec(fp.read(), version)

# def readme():
#     with open('README.md') as f:
#         return f.read()
    
setup(name='pyBO',
    version=0.0,
    description='Bayesian Optimization tools in Python',
    classifiers=[
    'Development Status :: 2 - Pre-Alpha',
    'Programming Language :: Python :: 3.6',
    'Topic :: Scientific/Engineering :: Mathematics'
    ],
    keywords = ['machine-learning', 'optimization', 'bayesian'],
    url='https://github.com/kilean20/pyBO',
    author='Kilean Hwang',
    author_email='hwang@frib.msu.edu',
#     license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
#         'mkl',
        'scipy',
#         'joblib',
#         'scikit-learn',
#         'Theano-PyMC',
#         'pyMC3'
    ],
    zip_safe=False)
