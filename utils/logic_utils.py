import re
from typing import List, Dict, Tuple, Set
from sympy.logic import simplify_logic, to_cnf, to_dnf
from sympy import symbols, And, Or, Not, Implies, Equivalent
import pandas as pd

class LogicUtils:
    def __init__(self):
        self.logical_operators = {
            'and': '&',
            'or': '|',
            'not': '~',
            'implies': '>>',
            'equivalent': '==',
            'if': '>>',
            'then': '>>',
            'only if': '>>',
            'if and only if': '=='
        }
        
        self.negation_patterns = {
            'shall': 'shall not',
            'must': 'must not',
            'will': 'will not',
            'should': 'should not',
            'can': 'cannot',
            'enable': 'disable',
            'allow': 'prevent',
            'require': 'forbid',
            'accept': 'reject',
            'include': 'exclude',
            'support': 'not support',
            'provide': 'not provide'
        }
    
    def extract_propositional_form(self, sentence: str) -> str:
        """Convert natural language sentence to propositional form."""
        # Basic conversion - this is a simplified version
        # In a real implementation, this would use more sophisticated NLP
        
        sentence_lower = sentence.lower()
        
        # Replace logical connectors
        for word, symbol in self.logical_operators.items():
            sentence_lower = sentence_lower.replace(word, symbol)
        
        # Extract the main proposition (simplified)
        # Remove common requirement words
        proposition = re.sub(r'\b(shall|must|will|should|can|may)\b', '', sentence_lower)
        proposition = re.sub(r'\b(the|a|an|is|are|be|been|being)\b', '', proposition)
        
        return proposition.strip()
    
    def generate_negation(self, proposition: str) -> str:
        """Generate the most relevant negation of a proposition."""
        if not proposition:
            return ""
        
        proposition_lower = proposition.lower()
        
        # Check for direct negation patterns
        for positive, negative in self.negation_patterns.items():
            if positive in proposition_lower:
                return proposition_lower.replace(positive, negative)
        
        # If no direct pattern, add "not" before the main verb
        # This is a simplified approach
        words = proposition_lower.split()
        for i, word in enumerate(words):
            if word in ['is', 'are', 'was', 'were', 'will', 'shall', 'must', 'should']:
                words.insert(i + 1, 'not')
                break
        else:
            # If no auxiliary verb found, add "does not" or "do not"
            if words and words[0] in ['the', 'a', 'an']:
                words.insert(1, 'does not')
            else:
                words.insert(0, 'does not')
        
        return ' '.join(words)
    
    def check_contradiction(self, prop1: str, prop2: str) -> bool:
        """Check if two propositions are contradictory."""
        # Simplified contradiction check
        prop1_lower = prop1.lower()
        prop2_lower = prop2.lower()
        
        # Check for direct contradictions
        contradictions = [
            ('shall', 'shall not'),
            ('must', 'must not'),
            ('will', 'will not'),
            ('should', 'should not'),
            ('enable', 'disable'),
            ('allow', 'prevent'),
            ('require', 'forbid'),
            ('accept', 'reject')
        ]
        
        for pos, neg in contradictions:
            if pos in prop1_lower and neg in prop2_lower:
                return True
            if pos in prop2_lower and neg in prop1_lower:
                return True
        
        return False
    
    def analyze_logical_relation(self, prop1: str, prop2: str) -> Dict[str, any]:
        """Analyze the logical relation between two propositions."""
        relation = {
            'type': 'unknown',
            'confidence': 0.0,
            'reasoning': ''
        }
        
        prop1_lower = prop1.lower()
        prop2_lower = prop2.lower()
        
        # Check for contradiction
        if self.check_contradiction(prop1, prop2):
            relation['type'] = 'contradictory'
            relation['confidence'] = 0.9
            relation['reasoning'] = 'Direct contradiction detected'
            return relation
        
        # Check for equivalence (similar meaning)
        if self._check_equivalence(prop1_lower, prop2_lower):
            relation['type'] = 'equivalent'
            relation['confidence'] = 0.8
            relation['reasoning'] = 'Similar meaning detected'
            return relation
        
        # Check for implication
        if self._check_implication(prop1_lower, prop2_lower):
            relation['type'] = 'implies'
            relation['confidence'] = 0.7
            relation['reasoning'] = 'Implication relationship detected'
            return relation
        
        # Check for dependency
        if self._check_dependency(prop1_lower, prop2_lower):
            relation['type'] = 'dependent'
            relation['confidence'] = 0.6
            relation['reasoning'] = 'Dependency relationship detected'
            return relation
        
        # Check for disjointness
        if self._check_disjoint(prop1_lower, prop2_lower):
            relation['type'] = 'disjoint'
            relation['confidence'] = 0.5
            relation['reasoning'] = 'Disjoint relationship detected'
            return relation
        
        return relation
    
    def _check_equivalence(self, prop1: str, prop2: str) -> bool:
        """Check if two propositions are equivalent."""
        # Simplified equivalence check
        # In a real implementation, this would use semantic similarity
        
        # Check for same keywords
        words1 = set(prop1.split())
        words2 = set(prop2.split())
        
        common_words = words1 & words2
        total_words = words1 | words2
        
        if len(total_words) > 0:
            similarity = len(common_words) / len(total_words)
            return similarity > 0.7
        
        return False
    
    def _check_implication(self, prop1: str, prop2: str) -> bool:
        """Check if prop1 implies prop2."""
        # Simplified implication check
        # Look for conditional words
        conditional_words = ['if', 'when', 'then', 'implies', 'requires']
        
        for word in conditional_words:
            if word in prop1 and word in prop2:
                return True
        
        return False
    
    def _check_dependency(self, prop1: str, prop2: str) -> bool:
        """Check if two propositions are dependent."""
        # Simplified dependency check
        # Look for shared subjects or objects
        
        # Extract nouns (simplified)
        nouns1 = [word for word in prop1.split() if word.endswith(('ion', 'ment', 'ness', 'ity'))]
        nouns2 = [word for word in prop2.split() if word.endswith(('ion', 'ment', 'ness', 'ity'))]
        
        common_nouns = set(nouns1) & set(nouns2)
        return len(common_nouns) > 0
    
    def _check_disjoint(self, prop1: str, prop2: str) -> bool:
        """Check if two propositions are disjoint."""
        # Simplified disjoint check
        # Look for mutually exclusive terms
        
        exclusive_pairs = [
            ('enable', 'disable'),
            ('allow', 'prevent'),
            ('include', 'exclude'),
            ('accept', 'reject')
        ]
        
        for term1, term2 in exclusive_pairs:
            if (term1 in prop1 and term2 in prop2) or (term1 in prop2 and term2 in prop1):
                return True
        
        return False
    
    def validate_proposition(self, proposition: str) -> Dict[str, any]:
        """Validate if a proposition is well-formed."""
        validation = {
            'is_valid': True,
            'issues': [],
            'suggestions': []
        }
        
        if not proposition or len(proposition.strip()) < 5:
            validation['is_valid'] = False
            validation['issues'].append('Proposition too short or empty')
            return validation
        
        prop_lower = proposition.lower()
        
        # Check for common issues
        if 'shall' in prop_lower and not any(word in prop_lower for word in ['system', 'component', 'function', 'feature']):
            validation['suggestions'].append('Consider specifying what shall be done')
        
        if any(word in prop_lower for word in ['it', 'this', 'that']) and not any(word in prop_lower for word in ['system', 'component']):
            validation['issues'].append('Unclear reference - specify what "it" refers to')
            validation['is_valid'] = False
        
        if any(word in prop_lower for word in ['appropriate', 'suitable', 'adequate']):
            validation['suggestions'].append('Consider using more specific terms instead of "appropriate"')
        
        return validation
    
    def create_truth_table(self, propositions: List[str]) -> pd.DataFrame:
        """Create a truth table for the given propositions."""
        # This is a simplified version
        # In a real implementation, this would create a proper truth table
        
        n_props = len(propositions)
        rows = 2 ** n_props
        
        # Create all possible combinations
        combinations = []
        for i in range(rows):
            row = []
            for j in range(n_props):
                row.append(bool((i >> j) & 1))
            combinations.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(combinations, columns=[f'P{i+1}' for i in range(n_props)])
        
        return df 