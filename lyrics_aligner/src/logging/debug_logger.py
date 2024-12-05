import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from lyrics_aligner.src.core.utils import format_timestamp
from lyrics_aligner.src.core.models import AlignedLine, MatchResult

class DebugLogger:
    def __init__(self, log_dir: str, audio_name: str, debug_mode: bool = False, console_output: bool = True):
        """Initialize the debug logger."""
        self.debug_mode = debug_mode
        self.logger = None
        
        if not self.debug_mode:
            return
            
        # Create log directory if it doesn't exist
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(log_dir) / f"{audio_name}_{timestamp}.log"
        
        self.logger = logging.getLogger(f"lyrics_aligner_{audio_name}")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # Console handler (optional)
        if console_output:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def section_header(self, title: str):
        """Log a section header."""
        if self.debug_mode and self.logger:
            self.logger.info("\n" + "="*50)
            self.logger.info(title)
            self.logger.info("="*50)

    def log_reference_lyrics(self, lyrics: List[str]):
        """Log the reference lyrics."""
        if self.debug_mode and self.logger:
            self.logger.info("\nReference Lyrics:")
            for i, line in enumerate(lyrics, 1):
                self.logger.info(f"{i:3d}: {line}")

    def log_transcription_results(self, segments: List[Dict]):
        """Log the transcription results."""
        if self.debug_mode and self.logger:
            self.logger.info("\nTranscription Results:")
            for seg in segments:
                self.logger.info(f"{format_timestamp(seg.start)} -> {format_timestamp(seg.end)}: {seg.text}")

    def log_alignment_progress(self, reference: str, matches: List[MatchResult], best_match: Optional[MatchResult]):
        """Log the alignment progress for a line."""
        if self.debug_mode and self.logger:
            self.logger.info("\nAlignment Progress:")
            self.logger.info(f"Reference Line: {reference}")
            if matches:
                self.logger.info(f"Found {len(matches)} potential matches")
                if best_match:
                    self.logger.info(f"Best Match Score: {best_match.score:.3f}")
                    self.logger.info(f"Best Match Timing: {format_timestamp(best_match.timing[0])} -> {format_timestamp(best_match.timing[1])}")
            else:
                self.logger.info("No matches found")

    def log_final_results(self, aligned_lyrics: List[AlignedLine]):
        """Log the final alignment results."""
        if self.debug_mode and self.logger:
            self.logger.info("\nFinal Alignment Results:")
            for i, line in enumerate(aligned_lyrics, 1):
                self.logger.info(f"\nLine {i}:")
                self.logger.info(f"Text: {line.text}")
                self.logger.info(f"Time: {format_timestamp(line.start)} -> {format_timestamp(line.end)}")
                self.logger.info(f"Confidence: {line.confidence:.3f}")
                if line.is_chorus:
                    self.logger.info("(Chorus Line)")
                if line.manual_timing:
                    self.logger.info("(Manual Timing Applied)")