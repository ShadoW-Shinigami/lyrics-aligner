from pathlib import Path
import sys
import torch
from typing import Dict, List
from tqdm import tqdm
from lyrics_aligner.src.core.aligner import LyricsAligner
from lyrics_aligner.src.config.config_loader import ConfigLoader

def check_system_requirements() -> Dict[str, any]:
    """Check if system meets requirements for WhisperX and return device info."""
    print("\nChecking system capabilities:")
    
    device_info = {
        "has_gpu": torch.cuda.is_available(),
        "compute_type": "int8"  # default to int8 for better memory efficiency
    }
    
    if device_info["has_gpu"]:
        print(f"CUDA available: Yes")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        
        # Check GPU capabilities
        if torch.cuda.get_device_capability()[0] >= 7:
            # Let user choose between float16 and int8 based on available VRAM
            total_memory = torch.cuda.get_device_properties(0).total_memory
            if total_memory >= 8 * (1024**3):  # 8GB or more VRAM
                device_info["compute_type"] = "float16"
                print("GPU has sufficient VRAM, using float16 for better accuracy")
            else:
                print("GPU has limited VRAM, using int8 for better memory efficiency")
        else:
            print("GPU does not support efficient float16, using int8")
    else:
        print("CUDA not available, using CPU with int8")
    
    print(f"Using compute type: {device_info['compute_type']}")
    return device_info

def find_audio_files(folder: Path) -> List[Path]:
    """Find all audio files in the specified folder."""
    audio_files = list(folder.glob("*.mp3")) + list(folder.glob("*.wav"))
    
    # Sort files for consistent processing order
    audio_files.sort()
    return audio_files

def process_single_file(audio_file: Path, aligner: LyricsAligner, settings: Dict[str, str], pbar: tqdm) -> bool:
    """Process a single audio file."""
    try:
        # Update progress description
        pbar.set_description(f"Processing {audio_file.name}")
        
        # Process the file
        success = aligner.process_audio(
            str(audio_file),
            output_dir=settings['OUTPUT_FOLDER'],
            debug=settings.get('DEBUG', '').lower() == 'yes'
        )
        
        # Log result
        if success:
            print(f"\n✓ Successfully processed {audio_file.name}")
        else:
            print(f"\n✗ Failed to process {audio_file.name}")
        
        return success
        
    except torch.cuda.OutOfMemoryError:
        print(f"\n✗ GPU out of memory while processing {audio_file.name}")
        print("  Try reducing batch_size in config or using int8 compute_type")
        return False
    except Exception as e:
        print(f"\n✗ Error processing {audio_file.name}: {str(e)}")
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
        
        # Check if output directory is writable
        if key == 'OUTPUT_FOLDER':
            try:
                test_file = path / '.write_test'
                test_file.touch()
                test_file.unlink()
            except Exception:
                print(f"Error: {key} is not writable")
                return False
    
    return True

def main():
    """Main entry point for the lyrics aligner."""
    print("\nLyrics-SRT Aligner")
    print("=================")
    
    try:
        # Check system requirements
        device_info = check_system_requirements()
        
        # Load settings
        settings = ConfigLoader.read_settings()
        if not validate_settings(settings):
            sys.exit(1)
        
        # Initialize aligner with appropriate device
        config = ConfigLoader.load_config()
        config.compute_type = device_info["compute_type"]
        
        try:
            aligner = LyricsAligner(config)
        except Exception as e:
            print(f"Failed to initialize aligner: {str(e)}")
            print("Try using a different compute_type or check your CUDA installation")
            sys.exit(1)
        
        # Find audio files
        audio_folder = Path(settings['AUDIO_FOLDER'])
        audio_files = find_audio_files(audio_folder)
        
        if not audio_files:
            print(f"\nNo audio files found in {audio_folder}")
            print("Supported formats: .mp3, .wav")
            sys.exit(1)
        
        print(f"\nFound {len(audio_files)} audio file(s)")
        
        # Process files with progress bar
        success_count = 0
        with tqdm(total=len(audio_files), desc="Processing", unit="file") as pbar:
            for audio_file in audio_files:
                if process_single_file(audio_file, aligner, settings, pbar):
                    success_count += 1
                pbar.update(1)
        
        # Print summary
        print("\nProcessing Summary")
        print("=================")
        print(f"Total files: {len(audio_files)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {len(audio_files) - success_count}")
        
        if success_count < len(audio_files):
            print("\nCheck the logs for details on failed files")
            if device_info["compute_type"] != "int8":
                print("Try using int8 compute_type if you're experiencing memory issues")
        
        sys.exit(0 if success_count == len(audio_files) else 1)
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()