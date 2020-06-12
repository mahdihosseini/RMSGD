from setuptools import setup
from glob import glob

setup(
    name='adas',
    use_scm_version=True,
    packages=[''],
    package_dir={'': 'src'},
    include_package_data=True,
    zip_safe=False,
    url='https://github.com/mahdihosseini/AdaS',
    python_requires='~=3.7',
    install_requires=[
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
        'setuptools_scm',  # for git-based versioning
    ],
    # DO NOT do tests_require; just call pytest or python -m pytest.
    license='License :: Other/Proprietary License',
    author='Mahdi Hosseini, Mathieu Tuli',
    author_email='mahdi.hosseini@mail.utoronto.ca, tuli.mathieu@gmail.com',
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
