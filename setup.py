from setuptools import setup
from glob import glob

setup(
    name='adas',
    use_scm_version=True,
    version="1.1.2",
    packages=['adas'],
    package_dir={'': 'src'},
    include_package_data=True,
    zip_safe=False,
    url='https://github.com/mahdihosseini/AdaS',
    python_requires='>=3.6',
    install_requires=[
        "numpy>=1.20.2",
        "scipy>=1.6.2",
        "torch>=1.8.1",
    ],
    extras_require={
        'docs': [
            'sphinx',
            'sphinx_rtd_theme',
            'recommonmark',
        ],
    },
    dependency_links=[
    ],
    setup_requires=[
        # 'setuptools_scm',  # for git-based versioning
    ],
    # DO NOT do tests_require; just call pytest or python -m pytest.
    license='License :: Other/Proprietary License',
    author='Mahdi Hosseini and Mathieu Tuli',
    author_email='mahdi.hosseini@mail.utoronto.ca',
    description='Python package for AdaS: Adaptive Scheduling of Stochastic Gradients',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: Other/Proprietary License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
    ],
    scripts=glob('bin/*'),
)
