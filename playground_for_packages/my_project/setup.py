from setuptools import setup

setup(
    name='my_project', #project name
    version='0.1.0',
    description='my project description',
    #url
    author='Brando Miranda',
    author_email='brando90@mit.edu',
    license='MIT',
    packages=['pkg_1','pkg_2'],
    install_requires=['numpy>=1.11.0']
)
