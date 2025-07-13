from typing import List, Dict, Tuple
from utils.logic_utils import LogicUtils
from utils.nlp_utils import NLPUtils
import pandas as pd

class RelationAnalyzer:
    def __init__(self):
        self.logic_utils = LogicUtils()
        self.nlp_utils = NLPUtils()
        self.relation_id_counter = 1
        
    def analyze_relations(self, propositions: List[Dict]) -> Dict[str, any]:
        """Analyze logical relations between all propositions."""
        self.relation_id_counter = 1  # Reset relation counter
        relations = []
        relation_matrix = {}
        
        # Create all possible pairs of propositions
        for i, prop1 in enumerate(propositions):
            for j, prop2 in enumerate(propositions[i+1:], i+1):
                sim = self.nlp_utils.semantic_similarity(prop1['proposition_statement'], prop2['proposition_statement'])
                subj1 = self.nlp_utils.get_subject(prop1['proposition_statement'])
                subj2 = self.nlp_utils.get_subject(prop2['proposition_statement'])
                # Heuristic for group reference
                if "both of them" in prop2['proposition_statement'].lower():
                    relation_type = 'related'
                    confidence = sim
                    reasoning = f"'both of them' likely refers to previous subjects"
                elif sim > 0.3:
                    relation_type = 'related'
                    confidence = sim
                    reasoning = f"Semantic similarity ({sim:.2f})"
                else:
                    relation_type = 'unknown'
                if relation_type != 'unknown':
                    relations.append({
                        'relation_id': f'REL{self.relation_id_counter}',
                        'proposition_id_1': prop1['proposition_id'],
                        'proposition_id_2': prop2['proposition_id'],
                        'type': relation_type,
                        'confidence': confidence,
                        'reasoning': reasoning
                    })
                    self.relation_id_counter += 1
                    
                    # Store in matrix for easy lookup
                    key = (prop1['proposition_id'], prop2['proposition_id'])
                    relation_matrix[key] = {
                        'relation_id': f'REL{self.relation_id_counter - 1}',
                        'proposition_id_1': prop1['proposition_id'],
                        'proposition_id_2': prop2['proposition_id'],
                        'type': relation_type,
                        'confidence': confidence,
                        'reasoning': reasoning
                    }
        
        return {
            'relations': relations,
            'relation_matrix': relation_matrix,
            'statistics': self._get_relation_statistics(relations)
        }
    
    def _analyze_proposition_pair(self, prop1: Dict, prop2: Dict) -> Dict[str, any]:
        """Analyze the logical relation between two propositions."""
        statement1 = prop1['proposition_statement']
        statement2 = prop2['proposition_statement']
        
        # Get basic logical relation
        relation = self.logic_utils.analyze_logical_relation(statement1, statement2)
        
        # Add semantic analysis
        semantic_relation = self._analyze_semantic_relation(prop1, prop2)
        if semantic_relation['type'] != 'unknown':
            relation.update(semantic_relation)
        
        # Add keyword-based analysis
        keyword_relation = self._analyze_keyword_relation(prop1, prop2)
        if keyword_relation['type'] != 'unknown':
            relation.update(keyword_relation)
        
        return relation
    
    def _analyze_semantic_relation(self, prop1: Dict, prop2: Dict) -> Dict[str, any]:
        """Analyze semantic relation between propositions."""
        # Extract keywords from both propositions
        keywords1 = self.nlp_utils.extract_keywords(prop1['proposition_statement'])
        keywords2 = self.nlp_utils.extract_keywords(prop2['proposition_statement'])
        
        # Calculate semantic similarity
        common_keywords = set(keywords1) & set(keywords2)
        total_keywords = set(keywords1) | set(keywords2)
        
        if len(total_keywords) > 0:
            similarity = len(common_keywords) / len(total_keywords)
            
            if similarity > 0.8:
                return {
                    'type': 'equivalent',
                    'confidence': similarity,
                    'reasoning': f'Semantic similarity: {similarity:.2f}'
                }
            elif similarity > 0.5:
                return {
                    'type': 'related',
                    'confidence': similarity,
                    'reasoning': f'Semantic similarity: {similarity:.2f}'
                }
        
        return {'type': 'unknown', 'confidence': 0.0, 'reasoning': ''}
    
    def _analyze_keyword_relation(self, prop1: Dict, prop2: Dict) -> Dict[str, any]:
        """Analyze relation based on specific keywords."""
        statement1 = prop1['proposition_statement'].lower()
        statement2 = prop2['proposition_statement'].lower()
        
        # Check for functional dependencies
        if self._check_functional_dependency(statement1, statement2):
            return {
                'type': 'functional_dependency',
                'confidence': 0.7,
                'reasoning': 'Functional dependency detected'
            }
        
        # Check for temporal dependencies
        if self._check_temporal_dependency(statement1, statement2):
            return {
                'type': 'temporal_dependency',
                'confidence': 0.6,
                'reasoning': 'Temporal dependency detected'
            }
        
        # Check for resource dependencies
        if self._check_resource_dependency(statement1, statement2):
            return {
                'type': 'resource_dependency',
                'confidence': 0.6,
                'reasoning': 'Resource dependency detected'
            }
        
        return {'type': 'unknown', 'confidence': 0.0, 'reasoning': ''}
    
    def _check_functional_dependency(self, statement1: str, statement2: str) -> bool:
        """Check if there's a functional dependency between statements."""
        # Look for input/output relationships
        input_output_pairs = [
            ('input', 'output'),
            ('receive', 'send'),
            ('accept', 'provide'),
            ('read', 'write'),
            ('collect', 'process')
        ]
        
        for input_term, output_term in input_output_pairs:
            if input_term in statement1 and output_term in statement2:
                return True
            if input_term in statement2 and output_term in statement1:
                return True
        
        return False
    
    def _check_temporal_dependency(self, statement1: str, statement2: str) -> bool:
        """Check if there's a temporal dependency between statements."""
        temporal_words = ['before', 'after', 'during', 'while', 'when', 'then']
        
        for word in temporal_words:
            if word in statement1 or word in statement2:
                return True
        
        return False
    
    def _check_resource_dependency(self, statement1: str, statement2: str) -> bool:
        """Check if there's a resource dependency between statements."""
        resource_words = ['memory', 'cpu', 'storage', 'network', 'file', 'database']
        
        for word in resource_words:
            if word in statement1 and word in statement2:
                return True
        
        return False
    
    def _get_relation_statistics(self, relations: List[Dict]) -> Dict[str, any]:
        """Get statistics about the detected relations."""
        stats = {
            'total_relations': len(relations),
            'by_type': {},
            'average_confidence': 0.0
        }
        
        total_confidence = 0.0
        
        for relation in relations:
            rel_type = relation['type']
            if rel_type not in stats['by_type']:
                stats['by_type'][rel_type] = 0
            stats['by_type'][rel_type] += 1
            
            total_confidence += relation.get('confidence', 0.0)
        
        if len(relations) > 0:
            stats['average_confidence'] = total_confidence / len(relations)
        
        return stats
    
    def find_contradictions(self, propositions: List[Dict]) -> List[Dict]:
        """Find contradictory propositions."""
        contradictions = []
        
        for i, prop1 in enumerate(propositions):
            for j, prop2 in enumerate(propositions[i+1:], i+1):
                if self.logic_utils.check_contradiction(
                    prop1['proposition_statement'], 
                    prop2['proposition_statement']
                ):
                    contradictions.append({
                        'proposition1': prop1,
                        'proposition2': prop2,
                        'type': 'contradiction',
                        'reason': 'Direct contradiction detected'
                    })
        
        return contradictions
    
    def find_circular_dependencies(self, relations: List[Dict]) -> List[List[str]]:
        """Find circular dependencies in relations."""
        # This is a simplified implementation
        # In a real implementation, this would use graph algorithms
        
        circular_deps = []
        
        # Look for A -> B -> C -> A patterns
        for rel1 in relations:
            for rel2 in relations:
                if rel1['proposition_id_2'] == rel2['proposition_id_1']:
                    for rel3 in relations:
                        if rel2['proposition_id_2'] == rel3['proposition_id_1'] and \
                           rel3['proposition_id_2'] == rel1['proposition_id_1']:
                            circular_deps.append([
                                rel1['proposition_id_1'],
                                rel1['proposition_id_2'],
                                rel2['proposition_id_2']
                            ])
        
        return circular_deps 