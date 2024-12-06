import whisperx
from typing import List, Dict, Optional, Tuple, Set
import numpy as np
from collections import defaultdict
from pathlib import Path
import os
import torch

from lyrics_aligner.src.core.models import TimingConfig, AlignedLine, WordSegment, MatchResult
from lyrics_aligner.src.core.utils import (
    read_reference_lyrics, 
    format_timestamp,
    get_output_paths,
    write_srt_file
)
from lyrics_aligner.src.core.text_matcher import enhanced_similarity_score, normalize_text
from lyrics_aligner.src.logging.debug_logger import DebugLogger
from lyrics_aligner.src.config.config_loader import ConfigLoader

class LyricsAligner:
    """Main class for aligning lyrics with audio using WhisperX."""
    
    def __init__(self, config: Optional[TimingConfig] = None):
        """Initialize the aligner with optional configuration."""
        self.config = config or ConfigLoader.load_config()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # Initialize Whisper model with compute type
            self.model = whisperx.load_model(
                "large-v2",
                self.device,
                compute_type=self.config.compute_type,
                language="en"
            )
            
            # Initialize alignment model
            self.alignment_model = whisperx.load_align_model(
                language_code=self.config.language_code,
                device=self.device
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize models: {str(e)}")
        
        self.chorus_patterns: Set[str] = set()
        self.debug_logger: Optional[DebugLogger] = None

    def process_audio(self, audio_path: str, output_dir: str, debug: bool = False) -> bool:
        """Process an audio file and generate aligned SRT."""
        try:
            if debug:
                self.debug_logger = DebugLogger(audio_path)
                self.debug_logger.logger.info("\nSTARTING ALIGNMENT PROCESS")

            # Get corresponding lyrics file
            lyrics_file = self._get_lyrics_file(audio_path)
            if not lyrics_file or not os.path.exists(lyrics_file):
                raise FileNotFoundError(f"No lyrics file found for {audio_path}")

            # Read reference lyrics
            reference_lyrics = read_reference_lyrics(lyrics_file)
            if not reference_lyrics:
                raise ValueError("No valid lyrics found in lyrics file")

            self._identify_chorus_patterns(reference_lyrics)

            # Load and process audio
            audio = whisperx.load_audio(audio_path)
            
            # Transcribe
            result = self.model.transcribe(
                audio,
                batch_size=self.config.batch_size,
                language="en"
            )

            # Align transcription
            aligned_results = whisperx.align(
                transcript=result["segments"],
                model=self.alignment_model[0],  # The model itself
                align_model_metadata=self.alignment_model[1],  # The metadata
                audio=audio,
                device=self.device,
                return_char_alignments=False
            )

            # Extract word segments
            word_segments = self._extract_word_segments(aligned_results["segments"])
            if not word_segments:
                raise ValueError("No words detected in audio")

            # Align lyrics with transcription
            aligned_lyrics = self._align_lyrics(reference_lyrics, word_segments)
            if not aligned_lyrics:
                raise ValueError("Failed to align any lyrics")

            # Refine alignments
            aligned_lyrics = self._refine_alignments(audio, aligned_lyrics)
            aligned_lyrics = self._post_process_alignments(aligned_lyrics)

            # Write output
            srt_path = os.path.join(output_dir, Path(audio_path).stem + ".srt")
            write_srt_file(aligned_lyrics, srt_path)

            if self.debug_logger:
                self.debug_logger.logger.info("\nAlignment process completed successfully")

            return True

        except Exception as e:
            if self.debug_logger:
                self.debug_logger.logger.error(f"Error processing {audio_path}: {str(e)}")
            raise

    def _get_lyrics_file(self, audio_path: str) -> Optional[str]:
        """Get the corresponding lyrics file path."""
        base_path = Path(audio_path).with_suffix('.txt')
        return str(base_path) if base_path.exists() else None

    def _identify_chorus_patterns(self, lyrics: List[str]) -> None:
        """Identify potential chorus patterns in lyrics."""
        self.chorus_patterns.clear()
        line_counts = defaultdict(int)
        
        for line in lyrics:
            normalized = normalize_text(line)
            line_counts[normalized] += 1
        
        for line, count in line_counts.items():
            if count > 1:
                self.chorus_patterns.add(line)

    def _extract_word_segments(self, segments: List[Dict]) -> List[WordSegment]:
        """Extract word segments from WhisperX output."""
        word_segments = []
        for segment in segments:
            for word in segment.get("words", []):
                word_segments.append(WordSegment(
                    text=word["word"],
                    start=word["start"],
                    end=word["end"],
                    probability=word.get("score", 0.0)
                ))
        return word_segments

    def _align_lyrics(self, reference_lyrics: List[str], word_segments: List[WordSegment]) -> List[AlignedLine]:
        """Align reference lyrics with transcribed segments."""
        aligned_lines = []
        last_end_time = 0
        
        for line in reference_lyrics:
            is_chorus = normalize_text(line) in self.chorus_patterns
            threshold = (
                self.config.chorus_similarity_threshold if is_chorus 
                else self.config.base_similarity_threshold
            )
            
            best_match = self._find_best_match(line, word_segments, threshold)
            
            if best_match:
                aligned_lines.append(AlignedLine(
                    text=line,
                    start=best_match.timing[0],
                    end=best_match.timing[1],
                    confidence=best_match.confidence,
                    is_chorus=is_chorus,
                    words=best_match.words
                ))
                last_end_time = best_match.timing[1]
            else:
                # Fallback timing estimation
                estimated_duration = len(line.split()) / self.config.expected_words_per_second
                start_time = last_end_time + self.config.min_gap_between_lines
                
                aligned_lines.append(AlignedLine(
                    text=line,
                    start=start_time,
                    end=start_time + estimated_duration,
                    confidence=0.0,
                    is_chorus=is_chorus
                ))
                last_end_time = start_time + estimated_duration

        return aligned_lines

    def _refine_alignments(self, audio: np.ndarray, aligned_lines: List[AlignedLine]) -> List[AlignedLine]:
        """Refine alignments using forced alignment."""
        refined_lines = []
        
        for line in aligned_lines:
            if self.debug_logger:
                self.debug_logger.logger.info(f"\nRefining alignment for: {line.text}")
            
            try:
                # Extract audio segment with buffer
                buffer = 0.5  # 500ms buffer
                start_sample = max(0, int((line.start - buffer) * 16000))
                end_sample = min(len(audio), int((line.end + buffer) * 16000))
                segment = audio[start_sample:end_sample]
                
                # Create alignment model input
                segments = [{
                    "text": line.text,
                    "start": buffer,
                    "end": (end_sample - start_sample) / 16000 - buffer
                }]
                
                # Use whisperx.align with correct parameter order
                result = whisperx.align(
                    transcript=segments,
                    model=self.alignment_model[0],  # The model itself
                    align_model_metadata=self.alignment_model[1],  # The metadata
                    audio=segment,
                    device=self.device,
                    return_char_alignments=False
                )
                
                if result and "segments" in result and result["segments"]:
                    aligned_segment = result["segments"][0]
                    if "words" in aligned_segment:
                        words = aligned_segment["words"]
                        # Adjust timing to account for the buffer
                        start_offset = line.start - buffer if line.start > buffer else 0
                        
                        refined_line = AlignedLine(
                            text=line.text,
                            start=words[0]["start"] + start_offset,
                            end=words[-1]["end"] + start_offset,
                            confidence=max(line.confidence, aligned_segment.get("confidence", 0.0)),
                            is_chorus=line.is_chorus,
                            words=[WordSegment(
                                text=w["word"],
                                start=w["start"] + start_offset,
                                end=w["end"] + start_offset,
                                probability=w.get("score", 0.0)
                            ) for w in words]
                        )
                        refined_lines.append(refined_line)
                        continue
                
                refined_lines.append(line)  # Keep original if refinement fails
                
            except Exception as e:
                if self.debug_logger:
                    self.debug_logger.logger.warning(f"Refinement failed for line: {line.text}. Error: {str(e)}")
                refined_lines.append(line)
        
        return refined_lines

    def _find_best_match(self, reference: str, word_segments: List[WordSegment], threshold: float) -> Optional[MatchResult]:
        """Find best matching segment using enhanced matching."""
        best_match = None
        best_score = threshold
        window_size = len(reference.split()) + self.config.word_window_buffer
        expected_words = len(reference.split())
        expected_duration = expected_words / self.config.expected_words_per_second

        for start_idx in range(len(word_segments)):
            for end_idx in range(start_idx + 1, min(start_idx + window_size, len(word_segments) + 1)):
                segment_words = word_segments[start_idx:end_idx]
                duration = segment_words[-1].end - segment_words[0].start
                
                if duration < self.config.min_line_duration or duration > self.config.max_line_duration:
                    continue
                
                segment_text = " ".join(word.text for word in segment_words)
                similarity, text_confidence = enhanced_similarity_score(reference, segment_text)
                
                timing_diff = abs(duration - expected_duration)
                timing_confidence = 1.0 - min(timing_diff / expected_duration, 1.0)
                
                prob_confidence = np.mean([w.probability for w in segment_words])
                
                confidence = (
                    text_confidence * (1 - self.config.timing_confidence_weight) +
                    timing_confidence * self.config.timing_confidence_weight
                ) * prob_confidence
                
                score = similarity * 0.7 + confidence * 0.3
                
                if score > best_score:
                    best_score = score
                    best_match = MatchResult(
                        score=score,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        timing=(segment_words[0].start, segment_words[-1].end),
                        confidence=confidence,
                        words=segment_words
                    )

        return best_match

    def _post_process_alignments(self, aligned_lyrics: List[AlignedLine]) -> List[AlignedLine]:
        """Post-process aligned lyrics to ensure proper timing and gaps."""
        if not aligned_lyrics:
            return aligned_lyrics

        aligned_lyrics.sort(key=lambda x: x.start)

        for i in range(1, len(aligned_lyrics)):
            prev_line = aligned_lyrics[i-1]
            curr_line = aligned_lyrics[i]

            min_gap = self.config.chorus_gap if curr_line.is_chorus else self.config.min_gap_between_lines
            
            if curr_line.start < prev_line.end + min_gap:
                curr_line.start = prev_line.end + min_gap
                if curr_line.end < curr_line.start + self.config.min_line_duration:
                    curr_line.end = curr_line.start + self.config.min_line_duration

            if curr_line.end - curr_line.start > self.config.max_line_duration:
                curr_line.end = curr_line.start + self.config.max_line_duration

        return aligned_lyrics