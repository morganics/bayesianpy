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
	install_requires=['pandas', 'sqlalchemy', 'networkx', 'numpy', 'jpype1', 'pathos', 'sklearn', 'dask']#, 'matplotlib', 'seaborn']
)
