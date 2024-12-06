# Lyrics Aligner
It's a Python tool that creates accurately timed SRT subtitle files from audio and lyrics. It uses WhisperX for high-precision audio-to-text alignment with CUDA acceleration support. I made this to make creation of lyric videos easier for my Suno generated tracks. Personal Plug :P - https://suno.com/@shadowshinigami

## Key Features:
- CUDA-accelerated processing with float16/int8 support
- Word-level timing precision
- Chorus pattern detection
- Detailed debug logging
- Configurable timing parameters

## Installation:
- Clone Repo
- `cd lyrics-aligner`
- Install PyTorch with CUDA (recommended) - `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 (Tested with 2.5)`
- `pip install -e ".[cuda]"`

## Usage:
- Modify settings.txt with your paths
- Place audio files (MP3/WAV) in your input folder
- Create matching .txt files with lyrics (same name as audio)
    - Input Format: Audio: MP3 or WAV files Lyrics: Plain text files (.txt) One line per lyric in text file Files must share the same name (e.g., song.mp3 and song.txt)
    - Output: SRT subtitle files Optional debug logs
- Run with command `python lyrics_aligner\main.py`
