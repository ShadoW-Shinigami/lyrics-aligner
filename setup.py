from setuptools import setup, find_packages

setup(
    name="lyrics_aligner",
    version="0.1",
    packages=find_packages(),
    python_requires='>=3.8',
    
    # Dependencies
    install_requires=[
        'numpy<2.0.0',  # Pin to avoid NaN issue
        'tqdm>=4.65.0',
        'transformers>=4.33.0',
        'ffmpeg-python>=0.2.0',
        'pandas>=2.0.0',
        'librosa>=0.10.0',
        'whisperx @ git+https://github.com/m-bain/whisperx.git',
        'pyannote.audio==3.1.1'  # Pin specific version
    ],
    
    # PyTorch dependencies with CUDA support
    dependency_links=[
        'https://download.pytorch.org/whl/cu121'
    ],
    
    # Additional PyTorch packages
    extras_require={
        'cuda': [
            'torch>=2.0.0',
            'torchaudio>=2.0.0',
            'torchvision>=0.15.0'
        ],
        'cpu': [
            'torch>=2.0.0',
            'torchaudio>=2.0.0',
            'torchvision>=0.15.0'
        ]
    },
    
    # Metadata
    description="A tool for aligning lyrics with audio using WhisperX",
    author="Harsha",
    author_email="psy2day@pm.me",

    # Entry points
    entry_points={
        'console_scripts': [
            'lyrics-aligner=lyrics_aligner.main:main',
        ],
    },
)