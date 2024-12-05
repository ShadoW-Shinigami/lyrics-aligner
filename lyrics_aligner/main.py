from pathlib import Path
import sys
from typing import Dict, List
from tqdm import tqdm
from lyrics_aligner.src.core.aligner import LyricsAligner
from lyrics_aligner.src.config.config_loader import ConfigLoader

def find_audio_files(folder: Path) -> List[Path]:
    """Find all audio files in the specified folder."""
    return list(folder.glob("*.mp3")) + list(folder.glob("*.wav"))

def process_single_file(audio_file: Path, aligner: LyricsAligner, settings: Dict[str, str], pbar: tqdm) -> bool:
    """Process a single audio file."""
    try:
        return aligner.process_audio(
            str(audio_file),
            output_dir=settings['OUTPUT_FOLDER'],
            debug=settings.get('DEBUG', '').lower() == 'yes'
        )
    except Exception as e:
        print(f"\n✗ Error processing {audio_file.name}: {e}")
        return False

def validate_settings(settings: Dict[str, str]) -> bool:
    """Validate required settings are present and valid."""
    required = ['AUDIO_FOLDER', 'OUTPUT_FOLDER']
    
    # Check required settings exist
    for key in required:
        if key not in settings:
            print(f"Error: Missing required setting: {key}")
            return False
        if not settings[key]:
            print(f"Error: {key} cannot be empty")
            return False
    
    # Validate paths exist
    for key in ['AUDIO_FOLDER', 'OUTPUT_FOLDER']:
        path = Path(settings[key])
        if not path.exists():
            print(f"Error: {key} path does not exist: {path}")
            return False
    
    return True

def main():
    """Main function to process all audio files."""
    print("\nLyrics-SRT Aligner")
    print("=================")
    
    try:
        # Read and validate settings using ConfigLoader
        settings = ConfigLoader.read_settings()
        if not validate_settings(settings):
            input("\nPress Enter to exit...")
            return
        
        # Initialize aligner
        aligner = LyricsAligner()
        
        # Find audio files
        audio_files = find_audio_files(Path(settings['AUDIO_FOLDER']))
        if not audio_files:
            print(f"\nNo audio files found in: {settings['AUDIO_FOLDER']}")
            input("\nPress Enter to exit...")
            return
        
        print(f"\nFound {len(audio_files)} audio file(s)")
        print("\nStarting processing...")
        
        # Process files with progress bar
        successful = 0
        with tqdm(total=len(audio_files), unit='file', ncols=80, 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            for audio_file in audio_files:
                if process_single_file(audio_file, aligner, settings, pbar):
                    successful += 1
                pbar.update(1)
        
        # Summary
        print("\nProcessing Complete!")
        print(f"Successfully processed: {successful}/{len(audio_files)} files")
        if successful < len(audio_files):
            print("Some files were skipped or failed. Check the messages above for details.")
        
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    
    finally:
        print("\nThank you for using Lyrics-SRT Aligner!")
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        input("\nPress Enter to exit...")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        input("\nPress Enter to exit...")