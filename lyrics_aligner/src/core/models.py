from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class TimingConfig:
    """Configuration for timing parameters."""
    min_line_duration: float = 0.5
    max_line_duration: float = 8.0
    min_gap_between_lines: float = 0.05
    chorus_gap: float = 0.2
    base_similarity_threshold: float = 0.3
    chorus_similarity_threshold: float = 0.4
    word_window_buffer: int = 5

@dataclass
class AlignedLine:
    """Represents an aligned lyric line with timing information."""
    text: str
    start: float
    end: float
    confidence: float
    is_chorus: bool = False
    manual_timing: Optional[Tuple[float, float]] = None

@dataclass
class WordSegment:
    """Represents a word-level segment from Whisper."""
    text: str
    start: float
    end: float
    probability: float

@dataclass
class MatchResult:
    """Represents a matching result between reference and transcribed text."""
    score: float
    start_idx: int
    end_idx: int
    timing: Tuple[float, float]
    confidence: float
    words: list[WordSegment]