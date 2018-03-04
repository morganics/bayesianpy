from distutils.core import setup

setup(
    name='bayesianpy',
    version='0.1',
    packages=['bayesianpy'],
    url='',
    license='',
    author='morganics',
    author_email='',
    description='', 
	install_requires=['pandas', 'sqlalchemy',
                      'numpy',
                      'jpype1>=0.6.1', 'pathos', 'scikit-learn',
                      'scipy',
                      'dask[complete]==0.16',
                      'multiprocess'
                      ],
    include_package_data=True
)
