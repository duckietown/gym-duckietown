from setuptools import setup

setup(
    name='gym_duckietown',
    version='2018.8.2',
    keywords='duckietown, environment, agent, rl, openaigym, openai-gym, gym',
    install_requires=[
        'gym>=0.9.0',
        'numpy>=1.10.0',
        'pyglet',
        'pyzmq>=16.0.0',
        'scikit-image>=0.13.1',
        'opencv-python>=3.4',
        'pyyaml>=3.12',
        'cloudpickle',
        'duckietown_slimremote>=2018.8.2'
    ],
    entry_points={
        'console_scripts': [
            'duckietown-start-gym=gym_duckietown.launcher:main',
        ],
    },
)
