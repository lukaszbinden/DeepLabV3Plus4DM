import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='DeepLabV3Plus4DM',
    version='1.0',
    author='VainF',
    author_email='vainf@hotmail.com',
    description='Testing installation of Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/VainF/DeepLabV3Plus-Pytorch',
    project_urls={
        "Bug Tracker": "https://github.com/VainF/DeepLabV3Plus-Pytorch/issues"
    },
    license='MIT',
    packages=['deeplabv3plus', 'deeplabv3plus.datasets', 'deeplabv3plus.metrics', 'deeplabv3plus.network', 'deeplabv3plus.network.backbone',
              'deeplabv3plus.utils'],
    install_requires=['torch', 'torchvision', 'numpy', 'pillow',
                      'scikit-learn', 'tqdm', 'matplotlib', 'visdom']
)
