from setuptools import setup

setup(
    name='gym_duckietown',
    version='0.0.1',
    keywords='duckietown, environment, agent, rl, openaigym, openai-gym, gym',
    install_requires=[
        'gym>=0.9.0',
        'numpy>=1.10.0',
        'pyglet',
        'pyzmq>=16.0.0',
        'scikit-image>=0.13.1',
        'opencv-python>=3.4',
    ]
)
