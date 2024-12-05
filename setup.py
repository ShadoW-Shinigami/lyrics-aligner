from setuptools import setup, find_packages

setup(
    name="lyrics_aligner",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'faster-whisper',
        'numpy',
        'tqdm'
    ]
)