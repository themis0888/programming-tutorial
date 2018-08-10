#nsml: neptuneml/tensorflow-1.4-gpu-py3:2.8.3

from distutils.core import setup
setup(
        name='nsml example 10 ladder_network',
        version='1.0',
        description='ns-ml',
        install_requires=[
            'matplotlib',
            'tqdm',
            'pillow',
            'imageio'
        ]
)
