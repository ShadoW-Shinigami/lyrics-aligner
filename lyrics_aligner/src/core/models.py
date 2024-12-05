from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch

@dataclass
class TimingConfig:
    """Configuration for timing parameters."""
    # Timing parameters
    min_line_duration: float = 0.5
    max_line_duration: float = 8.0
    min_gap_between_lines: float = 0.05
    chorus_gap: float = 0.2
    
    # Matching parameters
    base_similarity_threshold: float = 0.3
    chorus_similarity_threshold: float = 0.4
    word_window_buffer: int = 5
    
    # Enhanced matching parameters
    position_penalty_weight: float = 0.5
    length_penalty_weight: float = 0.3
    timing_confidence_weight: float = 0.3
    word_match_min_confidence: float = 0.8
    expected_words_per_second: float = 3.0
    
    # WhisperX specific
    batch_size: int = 16
    compute_type: str = "float16"  # Using float16 for CUDA
    language_code: str = "en"
    use_vad: bool = True  # Added VAD configuration
    vad_onset: float = 0.5
    vad_offset: float = 0.363
    min_word_duration: float = 0.1
    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate compute_type
        valid_compute_types = ["int8", "float16", "float32"]
        if self.compute_type not in valid_compute_types:
            raise ValueError(f"compute_type must be one of {valid_compute_types}")
        
        # Validate numeric ranges
        if self.min_line_duration <= 0:
            raise ValueError("min_line_duration must be positive")
        if self.max_line_duration <= self.min_line_duration:
            raise ValueError("max_line_duration must be greater than min_line_duration")
        if self.min_gap_between_lines < 0:
            raise ValueError("min_gap_between_lines cannot be negative")
        if self.chorus_gap < 0:
            raise ValueError("chorus_gap cannot be negative")
        
        # Validate thresholds and weights
        if not 0 <= self.base_similarity_threshold <= 1:
            raise ValueError("base_similarity_threshold must be between 0 and 1")
        if not 0 <= self.chorus_similarity_threshold <= 1:
            raise ValueError("chorus_similarity_threshold must be between 0 and 1")
        if not 0 <= self.word_match_min_confidence <= 1:
            raise ValueError("word_match_min_confidence must be between 0 and 1")
        if not 0 <= self.position_penalty_weight <= 1:
            raise ValueError("position_penalty_weight must be between 0 and 1")
        if not 0 <= self.length_penalty_weight <= 1:
            raise ValueError("length_penalty_weight must be between 0 and 1")
        if not 0 <= self.timing_confidence_weight <= 1:
            raise ValueError("timing_confidence_weight must be between 0 and 1")
        
        # Validate VAD parameters
        if not 0 <= self.vad_onset <= 1:
            raise ValueError("vad_onset must be between 0 and 1")
        if not 0 <= self.vad_offset <= 1:
            raise ValueError("vad_offset must be between 0 and 1")
        if self.min_word_duration <= 0:
            raise ValueError("min_word_duration must be positive")

@dataclass
class WordSegment:
    """Represents a word-level segment from WhisperX."""
    text: str
    start: float
    end: float
    probability: float

    def __post_init__(self):
        """Validate word segment parameters."""
        if not self.text:
            raise ValueError("text cannot be empty")
        if self.start < 0:
            raise ValueError("start time cannot be negative")
        if self.end <= self.start:
            raise ValueError("end time must be greater than start time")
        if not 0 <= self.probability <= 1:
            raise ValueError("probability must be between 0 and 1")

@dataclass
class AlignedLine:
    """Represents an aligned lyric line with timing information."""
    text: str
    start: float
    end: float
    confidence: float
    is_chorus: bool = False
    manual_timing: Optional[Tuple[float, float]] = None
    words: Optional[List[WordSegment]] = None

    def __post_init__(self):
        """Validate aligned line parameters."""
        if not self.text:
            raise ValueError("text cannot be empty")
        if self.start < 0:
            raise ValueError("start time cannot be negative")
        if self.end <= self.start:
            raise ValueError("end time must be greater than start time")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if self.manual_timing:
            start, end = self.manual_timing
            if start < 0 or end <= start:
                raise ValueError("invalid manual timing values")

@dataclass
class MatchResult:
    """Represents a match result between reference and transcribed text."""
    score: float
    start_idx: int
    end_idx: int
    timing: Tuple[float, float]
    confidence: float
    words: List[WordSegment]

    def __post_init__(self):
        """Validate match result parameters."""
        if not 0 <= self.score <= 1:
            raise ValueError("score must be between 0 and 1")
        if self.start_idx < 0:
            raise ValueError("start_idx cannot be negative")
        if self.end_idx <= self.start_idx:
            raise ValueError("end_idx must be greater than start_idx")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        start, end = self.timing
        if start < 0 or end <= start:
            raise ValueError("invalid timing values")
        if not self.words:
            raise ValueError("words list cannot be empty")