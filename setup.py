from setuptools import setup
from glob import glob

setup(
    name='adas',
    use_scm_version=True,
    packages=[''],
    package_dir={'': 'src'},
    include_package_data=True,
    zip_safe=False,
    # url='https://github.com/mahdihosseini/AdaS',
    python_requires='~=3.7',
    install_requires=[
        # 'attrs==19.3.0',
        # 'coverage==5.1',
        # 'cycler==0.10.0',
        # 'et-xmlfile==1.0.1',
        # 'future==0.18.2',
        # 'importlib-metadata==1.6.1',
        # 'jdcal==1.4.1',
        # 'kiwisolver==1.2.0',
        # 'matplotlib==3.2.1',
        # 'more-itertools==8.3.0',
        # 'numpy==1.18.5',
        # 'openpyxl==3.0.3',
        # 'packaging==20.4',
        # 'pandas==1.0.4',
        # 'Pillow==7.1.2',
        # 'pluggy==0.13.1',
        # 'py==1.8.1',
        # 'pyparsing==2.4.7',
        # 'pytest==5.4.3',
        # 'pytest-cov==2.9.0',
        # 'pytest-cover==3.0.0',
        # 'pytest-coverage==0.0',
        # 'python-dateutil==2.8.1',
        # 'pytz==2020.1',
        # 'PyYAML==5.3.1',
        # 'scipy==1.4.1',
        # 'six==1.15.0',
        # 'torch==1.5.0',
        # 'torchvision==0.6.0',
        # 'wcwidth==0.2.4',
        # 'xlrd==1.2.0',
        # 'zipp==3.1.0'
    ],
    extras_require={
        'docs': [
            'sphinx',
            'sphinx_rtd_theme',
            'recommonmark',
        ],
    },
    dependency_links=[
        # Use SSH-based GitHub auth! HTTPS not acceptable for production use,
        # due to less flexible authentication.
        # Yes, you need to add that #egg=… bit, else pip doesn't know which
        # package in that repo you want to install. A repo may have many or no
        # packages
    ],
    setup_requires=[
        # 'setuptools_scm',  # for git-based versioning
    ],
    # DO NOT do tests_require; just call pytest or python -m pytest.
    license='License :: Other/Proprietary License',
    author='Mahdi Hosseini',
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
