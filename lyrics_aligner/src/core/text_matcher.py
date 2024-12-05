from difflib import SequenceMatcher
from typing import Tuple
import re

def normalize_text(text: str) -> str:
    """Normalize text for comparison by removing punctuation and extra whitespace."""
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Remove extra whitespace
    return ' '.join(text.split())

def word_level_similarity(word1: str, word2: str) -> float:
    """Calculate similarity between two words."""
    # Exact match
    if word1 == word2:
        return 1.0
        
    # Substring match
    if word1 in word2 or word2 in word1:
        return 0.8
        
    # Character-level similarity with higher threshold for short words
    char_sim = SequenceMatcher(None, word1, word2).ratio()
    if len(word1) <= 3 or len(word2) <= 3:
        return char_sim if char_sim > 0.9 else 0.0
    return char_sim

def enhanced_similarity_score(reference: str, transcribed: str) -> Tuple[float, float]:
    """
    Calculate enhanced similarity score between reference and transcribed text.
    Returns (similarity_score, confidence_score)
    """
    ref_words = normalize_text(reference).split()
    trans_words = normalize_text(transcribed).split()
    
    if not ref_words or not trans_words:
        return 0.0, 0.0
    
    # Word-level matching with position awareness
    word_matches = []
    matched_positions = []
    
    for i, ref_word in enumerate(ref_words):
        best_word_score = 0
        best_position = -1
        
        for j, trans_word in enumerate(trans_words):
            if j in matched_positions:
                continue
                
            word_sim = word_level_similarity(ref_word, trans_word)
            
            # Apply position penalty (words further apart are less likely to match)
            position_penalty = abs(i - j) / max(len(ref_words), len(trans_words))
            adjusted_sim = word_sim * (1 - position_penalty * 0.5)
            
            if adjusted_sim > best_word_score:
                best_word_score = adjusted_sim
                best_position = j
        
        word_matches.append(best_word_score)
        if best_position >= 0:
            matched_positions.append(best_position)
    
    # Calculate overall similarity
    similarity = sum(word_matches) / len(word_matches)
    
    # Calculate confidence based on multiple factors:
    # 1. Overall similarity
    # 2. Length difference penalty
    # 3. Word order consistency
    len_diff_penalty = abs(len(ref_words) - len(trans_words)) / max(len(ref_words), len(trans_words))
    order_consistency = len(matched_positions) / max(len(ref_words), len(trans_words))
    
    confidence = similarity * (1 - len_diff_penalty * 0.3) * (0.7 + order_consistency * 0.3)
    
    return similarity, confidence