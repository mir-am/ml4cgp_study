from src import __version__
from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt', 'r') as req:
        content = req.read()
        requirements = content.split('\n')
    return requirements

setup(
    name='ml4cgp_study',
    version=__version__,
    description='An Empirical Study on the Effectiveness of Machine Learning-based Call Graph Pruning',
    packages=find_packages(), 
    install_requires=read_requirements(),
    url='https://github.com/mir-am/ml4cgp_study',
    author='Amir M. Mir (TU Delft)', # The main author
    author_email='mir-am@hotmail.com',
    license="GNU General Public License v3.0",
    classifiers=[
        'Environment :: Console',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: Unix',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    python_requries='>=3.9',
)
