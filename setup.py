from setuptools import setup


def get_version(filename):
    import ast
    version = None
    with open(filename) as f:
        for line in f:
            if line.startswith('__version__'):
                version = ast.parse(line).body[0].value.s
                break
        else:
            raise ValueError('No version found in %r.' % filename)
    if version is None:
        raise ValueError(filename)
    return version


version = get_version(filename='gym_duckietown/__init__.py')

setup(
        name='gym_duckietown',
        version=version,
        keywords='duckietown, environment, agent, rl, openaigym, openai-gym, gym',
        install_requires=[
            'gym>=0.9.0',
            'numpy>=1.10.0',
            'pyglet',
            'pyzmq>=16.0.0',
            'scikit-image>=0.13.1',
            'opencv-python>=3.4',
            'pyyaml>=3.11',
            'cloudpickle',
            'duckietown_slimremote>=2018.8.2', 
            'pygeometry',
            'dataclasses'
        ],
        entry_points={
            'console_scripts': [
                'duckietown-start-gym=gym_duckietown.launcher:main',
            ],
        },
)
