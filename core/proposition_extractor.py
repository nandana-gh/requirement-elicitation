import re
from typing import List, Dict, Tuple
from utils.nlp_utils import NLPUtils
from utils.logic_utils import LogicUtils

class PropositionExtractor:
    def __init__(self):
        self.nlp_utils = NLPUtils()
        self.logic_utils = LogicUtils()
        self.proposition_id_counter = 1
        
    def extract_propositions(self, requirements: List[str]) -> Dict[str, any]:
        """Extract atomic propositions from requirement statements."""
        self.proposition_id_counter = 1
        all_propositions = []
        requirement_propositions = {}
        
        for req_id, requirement in enumerate(requirements, 1):
            # Clean the requirement
            cleaned_req = self.nlp_utils.clean_text(requirement)
            
            # Extract sentences from the requirement
            sentences = self.nlp_utils.extract_sentences(cleaned_req)
            
            # Convert each sentence to atomic propositions
            propositions = []
            for sentence in sentences:
                atomic_props = self._split_into_atomic_propositions(sentence)
                for prop in atomic_props:
                    prop_data = {
                        'proposition_id': f'P{self.proposition_id_counter}',
                        'requirement_id': f'R{req_id}',
                        'proposition_statement': prop,
                        'proposition_type': self.nlp_utils.identify_proposition_type(prop),
                        'validation': self.logic_utils.validate_proposition(prop),
                        'ambiguity': self.nlp_utils.detect_ambiguity(prop)
                    }
                    propositions.append(prop_data)
                    all_propositions.append(prop_data)
                    self.proposition_id_counter += 1
            
            requirement_propositions[f'R{req_id}'] = {
                'requirement_statement': requirement,
                'propositions': propositions
            }
        
        return {
            'all_propositions': all_propositions,
            'requirement_propositions': requirement_propositions
        }
    
    def _split_into_atomic_propositions(self, sentence: str) -> List[str]:
        """Split a complex sentence into atomic propositions."""
        atomic_propositions = []
        
        # Remove common requirement prefixes
        sentence = re.sub(r'^(The system|The software|The application|It)\s+', '', sentence)
        
        # Split by common conjunctions
        conjunctions = [' and ', ' or ', ' but ', ' however ', ' while ', ' whereas ']
        
        # First, try to split by conjunctions
        parts = [sentence]
        for conj in conjunctions:
            new_parts = []
            for part in parts:
                if conj in part:
                    split_parts = part.split(conj)
                    new_parts.extend(split_parts)
                else:
                    new_parts.append(part)
            parts = new_parts
        
        # Further split complex parts
        for part in parts:
            atomic_parts = self._further_split_proposition(part.strip())
            atomic_propositions.extend(atomic_parts)
        
        # Filter out empty or very short propositions
        atomic_propositions = [prop for prop in atomic_propositions if len(prop.strip()) > 3]
        
        return atomic_propositions
    
    def _further_split_proposition(self, proposition: str) -> List[str]:
        """Further split a proposition into atomic parts."""
        parts = []
        
        # Split by conditional clauses
        if ' if ' in proposition:
            parts = proposition.split(' if ')
        elif ' when ' in proposition:
            parts = proposition.split(' when ')
        elif ' unless ' in proposition:
            parts = proposition.split(' unless ')
        else:
            parts = [proposition]
        
        # Clean up each part
        cleaned_parts = []
        for part in parts:
            part = part.strip()
            if part:
                # Remove trailing punctuation
                part = re.sub(r'[.,;:]$', '', part)
                if len(part) > 3:
                    cleaned_parts.append(part)
        
        return cleaned_parts
    
    def classify_proposition_type(self, proposition: str) -> str:
        """Classify the type of proposition."""
        return self.nlp_utils.identify_proposition_type(proposition)
    
    def validate_proposition(self, proposition: str) -> Dict[str, any]:
        """Validate a proposition for quality issues."""
        validation = self.logic_utils.validate_proposition(proposition)
        
        # Add NLP-based validation
        ambiguity = self.nlp_utils.detect_ambiguity(proposition)
        if ambiguity:
            validation['ambiguity_issues'] = ambiguity
        
        return validation
    
    def get_proposition_statistics(self, propositions: List[Dict]) -> Dict[str, any]:
        """Get statistics about the extracted propositions."""
        stats = {
            'total_propositions': len(propositions),
            'by_type': {},
            'validation_issues': 0,
            'ambiguity_issues': 0
        }
        
        for prop in propositions:
            prop_type = prop['proposition_type']
            if prop_type not in stats['by_type']:
                stats['by_type'][prop_type] = 0
            stats['by_type'][prop_type] += 1
            
            if not prop['validation']['is_valid']:
                stats['validation_issues'] += 1
            
            if prop.get('ambiguity'):
                stats['ambiguity_issues'] += 1
        
        return stats 