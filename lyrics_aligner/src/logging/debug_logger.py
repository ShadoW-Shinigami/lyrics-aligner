import logging
import os
from pathlib import Path
from typing import List, Optional
from lyrics_aligner.src.core.models import AlignedLine, MatchResult
from lyrics_aligner.src.core.utils import format_timestamp

class DebugLogger:
    """Handles debug logging for the alignment process."""

    def __init__(self, audio_path: str):
        """Initialize debug logger for an audio file."""
        self.debug_mode = True
        self.logger = None
        self._setup_logger(audio_path)

    def _setup_logger(self, audio_path: str):
        """Set up logging configuration."""
        try:
            # Create logs directory
            log_dir = Path(audio_path).parent / "logs"
            log_dir.mkdir(exist_ok=True)
            
            # Set up log file path
            audio_name = Path(audio_path).stem
            log_file = log_dir / f"{audio_name}_alignment.log"
            
            # Configure logger
            self.logger = logging.getLogger(f"aligner_{audio_name}")
            self.logger.setLevel(logging.DEBUG)
            
            # File handler
            fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            
            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter('%(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            # Add handlers
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
            
        except Exception as e:
            print(f"Failed to set up logger: {e}")
            self.debug_mode = False

    def log_transcription_result(self, result: dict):
        """Log WhisperX transcription result."""
        if self.debug_mode and self.logger:
            self.logger.info("\nWhisperX Transcription Result:")
            for segment in result["segments"]:
                self.logger.info(f"\nSegment {segment.get('id', '?')}:")
                self.logger.info(f"Text: {segment.get('text', '')}")
                if "words" in segment:
                    self.logger.info("Words:")
                    for word in segment["words"]:
                        self.logger.info(
                            f"  {word['word']:<20} "
                            f"Time: {word['start']:.2f} -> {word['end']:.2f} "
                            f"Score: {word.get('score', 0.0):.2f}"
                        )

    def log_match_result(self, reference: str, match: MatchResult):
        """Log matching result for a line."""
        if self.debug_mode and self.logger:
            self.logger.info(f"\nMatch found for: {reference}")
            self.logger.info(f"Score: {match.score:.3f}")
            self.logger.info(f"Confidence: {match.confidence:.3f}")
            self.logger.info(f"Timing: {format_timestamp(match.timing[0])} -> {format_timestamp(match.timing[1])}")
            self.logger.info("Matched words:")
            for word in match.words:
                self.logger.info(
                    f"  {word.text:<20} "
                    f"Time: {word.start:.2f} -> {word.end:.2f} "
                    f"Probability: {word.probability:.2f}"
                )

    def log_alignment_refinement(self, line: AlignedLine, refined_start: float, refined_end: float):
        """Log forced alignment refinement results."""
        if self.debug_mode and self.logger:
            self.logger.info(f"\nRefinement for line: {line.text}")
            self.logger.info(f"Original timing: {format_timestamp(line.start)} -> {format_timestamp(line.end)}")
            self.logger.info(f"Refined timing: {format_timestamp(refined_start)} -> {format_timestamp(refined_end)}")

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

    def log_error(self, message: str):
        """Log error message."""
        if self.debug_mode and self.logger:
            self.logger.error(f"\nERROR: {message}")