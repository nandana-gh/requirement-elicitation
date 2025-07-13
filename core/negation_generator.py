import re
from typing import List, Dict, Tuple
from utils.logic_utils import LogicUtils
from utils.nlp_utils import NLPUtils

class NegationGenerator:
    def __init__(self):
        self.logic_utils = LogicUtils()
        self.nlp_utils = NLPUtils()
        self.inference_id_counter = 1
        
    def generate_negations(self, propositions: List[Dict]) -> Dict[str, any]:
        """Generate negations for all propositions and analyze their validity."""
        self.inference_id_counter = 1  # Reset inference counter
        negations = []
        inferences = []
        
        for proposition in propositions:
            negation_data = self._generate_proposition_negation(proposition)
            negations.append(negation_data)
            
            # Generate inference about the negation
            inference = self._analyze_negation_inference(proposition, negation_data)
            if inference:
                inference['inference_id'] = f'INF{self.inference_id_counter}'
                inference['proposition_id'] = proposition['proposition_id']
                inferences.append(inference)
                self.inference_id_counter += 1
        
        return {
            'negations': negations,
            'inferences': inferences,
            'statistics': self._get_negation_statistics(negations)
        }
    
    def _generate_proposition_negation(self, proposition: Dict) -> Dict[str, any]:
        """Generate negation for a single proposition."""
        statement = proposition['proposition_statement']
        
        # Generate basic negation
        basic_negation = self.logic_utils.generate_negation(statement)
        
        # Generate alternative negations
        alternative_negations = self._generate_alternative_negations(statement)
        
        # Select the most relevant negation
        best_negation = self._select_best_negation(statement, basic_negation, alternative_negations)
        
        # Validate the negation
        validation = self._validate_negation(statement, best_negation)
        
        return {
            'proposition_id': proposition['proposition_id'],
            'original_statement': statement,
            'negation': best_negation,
            'alternative_negations': alternative_negations,
            'validation': validation,
            'semantic_similarity': self._calculate_semantic_similarity(statement, best_negation)
        }
    
    def _generate_alternative_negations(self, statement: str) -> List[str]:
        """Generate alternative forms of negation."""
        alternatives = []
        statement_lower = statement.lower()
        
        # Method 1: Add "not" before main verb
        words = statement_lower.split()
        for i, word in enumerate(words):
            if word in ['is', 'are', 'was', 'were', 'will', 'shall', 'must', 'should', 'can', 'may']:
                new_words = words.copy()
                new_words.insert(i + 1, 'not')
                alternatives.append(' '.join(new_words))
                break
        
        # Method 2: Replace positive terms with negative equivalents
        positive_negative_pairs = [
            ('enable', 'disable'),
            ('allow', 'prevent'),
            ('require', 'forbid'),
            ('accept', 'reject'),
            ('include', 'exclude'),
            ('support', 'not support'),
            ('provide', 'not provide'),
            ('implement', 'not implement'),
            ('process', 'not process'),
            ('handle', 'not handle')
        ]
        
        for positive, negative in positive_negative_pairs:
            if positive in statement_lower:
                alt_negation = statement_lower.replace(positive, negative)
                alternatives.append(alt_negation)
        
        return list(set(alternatives))  # Remove duplicates
    
    def _select_best_negation(self, original: str, basic_negation: str, alternatives: List[str]) -> str:
        """Select the most relevant negation from alternatives."""
        if not alternatives:
            return basic_negation
        
        # Score each negation based on relevance
        scored_negations = []
        
        for negation in [basic_negation] + alternatives:
            score = self._score_negation_relevance(original, negation)
            scored_negations.append((negation, score))
        
        # Return the negation with highest score
        best_negation = max(scored_negations, key=lambda x: x[1])
        return best_negation[0]
    
    def _score_negation_relevance(self, original: str, negation: str) -> float:
        """Score how relevant a negation is to the original statement."""
        score = 0.0
        
        # Check for logical correctness
        if self._is_logically_correct_negation(original, negation):
            score += 0.4
        
        # Check for semantic similarity (should be high but not too high)
        similarity = self._calculate_semantic_similarity(original, negation)
        if 0.3 <= similarity <= 0.7:  # Good balance
            score += 0.3
        
        # Check for grammatical correctness
        if self._is_grammatically_correct(negation):
            score += 0.2
        
        # Check for clarity
        if self._is_clear_negation(negation):
            score += 0.1
        
        return score
    
    def _is_logically_correct_negation(self, original: str, negation: str) -> bool:
        """Check if negation is logically correct."""
        original_lower = original.lower()
        negation_lower = negation.lower()
        
        # Check for contradiction patterns
        contradiction_patterns = [
            ('shall', 'shall not'),
            ('must', 'must not'),
            ('will', 'will not'),
            ('should', 'should not'),
            ('can', 'cannot'),
            ('enable', 'disable'),
            ('allow', 'prevent')
        ]
        
        for positive, negative in contradiction_patterns:
            if positive in original_lower and negative in negation_lower:
                return True
        
        # Check for "not" presence
        if 'not' in negation_lower and 'not' not in original_lower:
            return True
        
        return False
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        keywords1 = set(self.nlp_utils.extract_keywords(text1))
        keywords2 = set(self.nlp_utils.extract_keywords(text2))
        
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = keywords1 & keywords2
        union = keywords1 | keywords2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _is_grammatically_correct(self, negation: str) -> bool:
        """Check if negation is grammatically correct."""
        words = negation.split()
        if len(words) < 2:
            return False
        
        not_count = negation.lower().count('not')
        if not_count > 2:
            return False
        
        return True
    
    def _is_clear_negation(self, negation: str) -> bool:
        """Check if negation is clear and unambiguous."""
        vague_terms = ['appropriate', 'suitable', 'adequate', 'reasonable']
        negation_lower = negation.lower()
        
        for term in vague_terms:
            if term in negation_lower:
                return False
        
        return True
    
    def _validate_negation(self, original: str, negation: str) -> Dict[str, any]:
        """Validate the quality of a negation."""
        validation = {
            'is_valid': True,
            'issues': [],
            'strengths': []
        }
        
        if not self._is_logically_correct_negation(original, negation):
            validation['is_valid'] = False
            validation['issues'].append('Not a proper logical negation')
        
        if not self._is_grammatically_correct(negation):
            validation['is_valid'] = False
            validation['issues'].append('Grammatical issues detected')
        
        if not self._is_clear_negation(negation):
            validation['issues'].append('Negation may be unclear or ambiguous')
        
        similarity = self._calculate_semantic_similarity(original, negation)
        if similarity < 0.2:
            validation['issues'].append('Negation may be too different from original')
        elif similarity > 0.8:
            validation['issues'].append('Negation may be too similar to original')
        else:
            validation['strengths'].append('Good semantic balance')
        
        return validation
    
    def _analyze_negation_inference(self, proposition: Dict, negation_data: Dict) -> Dict[str, any]:
        """Analyze and create inference about the negation."""
        original = proposition['proposition_statement']
        negation = negation_data['negation']
        validation = negation_data['validation']
        
        inference = {
            'type': 'negation_analysis',
            'proposition_statement': original,
            'negation_statement': negation,
            'analysis': []
        }
        
        if self.logic_utils.check_contradiction(original, negation):
            inference['analysis'].append('Original and negation are logically contradictory (expected)')
        else:
            inference['analysis'].append('Original and negation may not be properly contradictory')
        
        if validation['is_valid']:
            inference['analysis'].append('Negation is logically and grammatically valid')
        else:
            inference['analysis'].extend([f'Issue: {issue}' for issue in validation['issues']])
        
        similarity = negation_data['semantic_similarity']
        if 0.3 <= similarity <= 0.7:
            inference['analysis'].append('Negation maintains appropriate semantic distance from original')
        else:
            inference['analysis'].append(f'Semantic similarity ({similarity:.2f}) may be inappropriate')
        
        if self._can_coexist(original, negation):
            inference['analysis'].append('Original and negation can coexist in different contexts')
        else:
            inference['analysis'].append('Original and negation cannot coexist (mutually exclusive)')
        
        return inference
    
    def _can_coexist(self, original: str, negation: str) -> bool:
        """Check if original and negation can coexist without contradiction."""
        original_lower = original.lower()
        negation_lower = negation.lower()
        
        absolute_contradictions = [
            ('shall', 'shall not'),
            ('must', 'must not'),
            ('will', 'will not'),
            ('enable', 'disable'),
            ('allow', 'prevent')
        ]
        
        for pos, neg in absolute_contradictions:
            if pos in original_lower and neg in negation_lower:
                return False
        
        return True
    
    def _get_negation_statistics(self, negations: List[Dict]) -> Dict[str, any]:
        """Get statistics about the generated negations."""
        stats = {
            'total_negations': len(negations),
            'valid_negations': 0,
            'invalid_negations': 0,
            'average_similarity': 0.0
        }
        
        total_similarity = 0.0
        
        for negation in negations:
            if negation['validation']['is_valid']:
                stats['valid_negations'] += 1
            else:
                stats['invalid_negations'] += 1
            
            total_similarity += negation['semantic_similarity']
        
        if len(negations) > 0:
            stats['average_similarity'] = total_similarity / len(negations)
        
        return stats 