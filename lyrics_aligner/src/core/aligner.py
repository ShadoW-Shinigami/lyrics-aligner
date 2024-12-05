from faster_whisper import WhisperModel
from typing import List, Dict, Optional, Tuple, Set
import numpy as np
from collections import defaultdict
from pathlib import Path
import os
import re

from lyrics_aligner.src.core.models import TimingConfig, AlignedLine, WordSegment, MatchResult
from lyrics_aligner.src.core.utils import (
    similarity_score, 
    read_reference_lyrics, 
    format_timestamp,
    get_output_paths,
    write_srt_file
)
from lyrics_aligner.src.logging.debug_logger import DebugLogger
from lyrics_aligner.src.config.config_loader import ConfigLoader

class LyricsAligner:
    def __init__(self, model_size: str = "base"):
        """Initialize the LyricsAligner with specified Whisper model size."""
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.config = ConfigLoader.load_config()
        self.debug_logger = None
        self.chorus_patterns: Set[str] = set()

    def process_audio(self, audio_path: str, output_dir: str = None, debug: bool = False) -> bool:
        """Process a single audio file and generate SRT output."""
        try:
            # Get paths
            paths = get_output_paths(audio_path, output_dir)
            
            # Initialize debug logger if needed
            if debug:
                self.debug_logger = DebugLogger(paths['log_dir'], paths['audio_name'], debug_mode=True)
                self.debug_logger.section_header("STARTING ALIGNMENT PROCESS")
                self.debug_logger.logger.info(f"Processing: {paths['audio_name']}")

            # Read reference lyrics
            reference_lyrics = read_reference_lyrics(paths['lyrics'])
            if not reference_lyrics:
                print(f"✗ No valid lyrics found in: {paths['lyrics']}")
                return False

            if self.debug_logger:
                self.debug_logger.log_reference_lyrics(reference_lyrics)

            # Pre-process lyrics to identify chorus patterns
            self._identify_chorus_patterns(reference_lyrics)

            # Transcribe audio
            segments, _ = self.model.transcribe(
                audio_path,
                word_timestamps=True,
                initial_prompt="[Music] Lyrics:",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            # Convert segments to word segments
            word_segments = self._extract_word_segments(segments)
            if self.debug_logger:
                self.debug_logger.log_transcription_results(word_segments)

            # Align lyrics with transcription
            aligned_lyrics = self._align_lyrics(reference_lyrics, word_segments)
            if not aligned_lyrics:
                print(f"✗ Failed to align lyrics for: {paths['audio_name']}")
                return False

            # Post-process alignments
            aligned_lyrics = self._post_process_alignments(aligned_lyrics)

            # Write output
            write_srt_file(aligned_lyrics, paths['srt'])
            print(f"✓ Successfully processed: {paths['audio_name']}")
            return True

        except Exception as e:
            print(f"✗ Error processing {Path(audio_path).name}: {str(e)}")
            return False

    def _identify_chorus_patterns(self, lyrics: List[str]) -> None:
        """Identify potential chorus patterns in lyrics."""
        self.chorus_patterns.clear()
        line_counts = defaultdict(int)
        
        # Count occurrences of each line
        for line in lyrics:
            normalized = self._normalize_text(line)
            line_counts[normalized] += 1
        
        # Lines appearing multiple times might be chorus
        for line, count in line_counts.items():
            if count > 1:
                self.chorus_patterns.add(line)

        if self.debug_logger:
            self.debug_logger.logger.info(f"\nIdentified {len(self.chorus_patterns)} potential chorus patterns")

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Remove extra whitespace
        return ' '.join(text.split())

    def _extract_word_segments(self, segments) -> List[WordSegment]:
        """Extract word-level segments from Whisper output."""
        word_segments = []
        for segment in segments:
            for word in segment.words:
                # Skip non-speech segments
                if '[' in word.word or ']' in word.word:
                    continue
                    
                word_segments.append(WordSegment(
                    text=word.word.strip(),
                    start=word.start,
                    end=word.end,
                    probability=word.probability
                ))
        return word_segments

    def _align_lyrics(self, reference_lyrics: List[str], word_segments: List[WordSegment]) -> List[AlignedLine]:
        """Align reference lyrics with transcribed word segments."""
        aligned_lines = []
        word_idx = 0
        
        for ref_line in reference_lyrics:
            if self.debug_logger:
                self.debug_logger.logger.info(f"\nProcessing line: {ref_line}")

            # Check if this is a chorus line
            is_chorus = self._normalize_text(ref_line) in self.chorus_patterns

            # Find best matching segment
            best_match = self._find_best_match(
                ref_line, 
                word_segments[word_idx:],
                threshold=self.config.chorus_similarity_threshold if is_chorus else self.config.base_similarity_threshold
            )
            
            if not best_match:
                if self.debug_logger:
                    self.debug_logger.logger.warning(f"No match found for line: {ref_line}")
                continue

            # Create aligned line
            aligned_line = AlignedLine(
                text=ref_line,
                start=best_match.timing[0],
                end=best_match.timing[1],
                confidence=best_match.confidence,
                is_chorus=is_chorus
            )
            aligned_lines.append(aligned_line)
            
            # Update word index
            word_idx += best_match.end_idx

            if self.debug_logger:
                self.debug_logger.log_alignment_progress(ref_line, [best_match], best_match)

        return aligned_lines

    def _find_best_match(self, reference: str, word_segments: List[WordSegment], threshold: float) -> Optional[MatchResult]:
        """Find best matching segment for a reference line."""
        best_match = None
        best_score = threshold
        window_size = len(reference.split()) + self.config.word_window_buffer

        for start_idx in range(len(word_segments)):
            for end_idx in range(start_idx + 1, min(start_idx + window_size, len(word_segments) + 1)):
                segment_words = word_segments[start_idx:end_idx]
                
                # Skip if segment duration is outside acceptable range
                duration = segment_words[-1].end - segment_words[0].start
                if duration < self.config.min_line_duration or duration > self.config.max_line_duration:
                    continue
                
                segment_text = " ".join(word.text for word in segment_words)
                score = similarity_score(reference, segment_text)
                
                if score > best_score:
                    best_score = score
                    best_match = MatchResult(
                        score=score,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        timing=(segment_words[0].start, segment_words[-1].end),
                        confidence=np.mean([w.probability for w in segment_words]),
                        words=segment_words
                    )

        return best_match

    def _post_process_alignments(self, aligned_lyrics: List[AlignedLine]) -> List[AlignedLine]:
        """Post-process aligned lyrics to ensure proper timing and gaps."""
        if not aligned_lyrics:
            return aligned_lyrics

        # Sort by start time
        aligned_lyrics.sort(key=lambda x: x.start)

        # Adjust timings to ensure minimum gaps and handle overlaps
        for i in range(1, len(aligned_lyrics)):
            prev_line = aligned_lyrics[i-1]
            curr_line = aligned_lyrics[i]

            # If lines overlap or gap is too small
            if curr_line.start < prev_line.end + self.config.min_gap_between_lines:
                # For chorus lines, use larger gap
                min_gap = self.config.chorus_gap if curr_line.is_chorus else self.config.min_gap_between_lines
                curr_line.start = prev_line.end + min_gap

                # Ensure minimum duration is maintained
                if curr_line.end < curr_line.start + self.config.min_line_duration:
                    curr_line.end = curr_line.start + self.config.min_line_duration

        if self.debug_logger:
            self.debug_logger.log_final_results(aligned_lyrics)

        return aligned_lyrics