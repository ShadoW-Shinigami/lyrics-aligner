from typing import List, Dict
import os
from pathlib import Path
from lyrics_aligner.src.core.models import AlignedLine

def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    millisecs = int((secs - int(secs)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{millisecs:03d}"

def read_reference_lyrics(file_path: str) -> List[str]:
    """Read reference lyrics file and return as list of lines."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = []
        for line in f.readlines():
            line = line.strip()
            if line and not (line.startswith('[') and line.endswith(']')):
                lines.append(line)
        return lines

def ensure_directory_exists(file_path: str) -> None:
    """Ensure the directory for the given file path exists."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

def get_output_paths(audio_path: str, output_dir: str = None) -> Dict[str, str]:
    """Generate output paths for various files."""
    audio_dir = os.path.dirname(audio_path)
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    base_dir = output_dir or audio_dir
    
    return {
        'lyrics': os.path.join(audio_dir, f"{audio_name}.txt"),
        'srt': os.path.join(base_dir, f"{audio_name}.srt"),
        'log_dir': os.path.join(base_dir, "logs"),
        'audio_name': audio_name
    }

def write_srt_file(aligned_lyrics: List[AlignedLine], output_path: str) -> None:
    """Write aligned lyrics to SRT format file."""
    ensure_directory_exists(output_path)
    
    srt_output = []
    for i, line in enumerate(aligned_lyrics, 1):
        srt_output.extend([
            str(i),
            f"{format_timestamp(line.start)} --> {format_timestamp(line.end)}",
            line.text,
            ""
        ])
    
    with open(output_path, "w", encoding="utf-8") as srt_file:
        srt_file.write("\n".join(srt_output))