from setuptools import setup

setup(
    name='my_tf_lib',
    version='0.1.0',
    description='my research library for deep learning',
    #url
    author='Brando Miranda',
    author_email='brando90@mit.edu',
    license='MIT',
    packages=['my_tf_lib'],
    install_requires=['numpy>=1.11.0']
)
