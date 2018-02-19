from setuptools import setup

setup(
    name='gym_duckietown',
    version='0.0.1',
    keywords='duckietown, environment, agent, rl, openaigym, openai-gym, gym',
    install_requires=[
        'gym>=0.9.0',
        'numpy>=1.10.0',
        'opencv-python'
        'pyzmq>=16.0.0',
        'pyglet'
    ]
)
