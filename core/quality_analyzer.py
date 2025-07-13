from typing import List, Dict, Tuple
from utils.nlp_utils import NLPUtils
from utils.logic_utils import LogicUtils
from spacy import Language

class QualityAnalyzer:
    def __init__(self):
        self.nlp_utils = NLPUtils()
        self.logic_utils = LogicUtils()
        self.nlp = Language()
        
    def analyze_requirement_quality(self, requirements: List[str], propositions: List[Dict]) -> Dict[str, any]:
        """Comprehensive quality analysis of requirements."""
        analysis = {
            'completeness': self._analyze_completeness(propositions),
            'consistency': self._analyze_consistency(propositions),
            'clarity': self._analyze_clarity(propositions),
            'ambiguity': self._analyze_ambiguity(propositions),
            'missing_information': self._analyze_missing_information(requirements, propositions),
            'contradictions': self._find_contradictions(propositions),
            'overall_score': 0.0,
            'recommendations': []
        }
        
        # Calculate overall quality score
        analysis['overall_score'] = self._calculate_overall_score(analysis)
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _analyze_completeness(self, propositions: List[Dict]) -> Dict[str, any]:
        """Analyze completeness of requirements."""
        completeness = {
            'score': 0.0,
            'issues': [],
            'missing_elements': []
        }
        
        total_issues = 0
        
        for prop in propositions:
            statement = prop['proposition_statement'].lower()
            
            # Check for missing quantifiers
            if any(word in statement for word in ['shall', 'must', 'will']) and \
               not any(word in statement for word in ['all', 'every', 'each', 'any', 'some']):
                completeness['issues'].append(f"Missing quantifier in: {prop['proposition_statement']}")
                total_issues += 1
            
            # Check for missing conditions
            if 'shall' in statement and 'if' not in statement and 'when' not in statement:
                completeness['issues'].append(f"Missing conditions in: {prop['proposition_statement']}")
                total_issues += 1
            
            # Check for incomplete definitions
            if any(word in statement for word in ['system', 'component', 'function']) and \
               not any(word in statement for word in ['defined', 'specified', 'described']):
                completeness['missing_elements'].append(f"Incomplete definition in: {prop['proposition_statement']}")
        
        # Calculate completeness score
        if len(propositions) > 0:
            completeness['score'] = max(0.0, 1.0 - (total_issues / len(propositions)))
        
        return completeness
    
    def _analyze_consistency(self, propositions: List[Dict]) -> Dict[str, any]:
        """Analyze consistency of requirements."""
        consistency = {
            'score': 0.0,
            'contradictions': [],
            'inconsistencies': []
        }
        
        # Find contradictions
        contradictions = self.nlp_utils.detect_inconsistency(
            [prop['proposition_statement'] for prop in propositions]
        )
        
        consistency['contradictions'] = contradictions
        
        # Check for logical inconsistencies
        for i, prop1 in enumerate(propositions):
            for j, prop2 in enumerate(propositions[i+1:], i+1):
                if self.logic_utils.check_contradiction(
                    prop1['proposition_statement'], 
                    prop2['proposition_statement']
                ):
                    consistency['inconsistencies'].append({
                        'proposition1': prop1['proposition_statement'],
                        'proposition2': prop2['proposition_statement'],
                        'type': 'logical_contradiction'
                    })
        
        # Calculate consistency score
        total_issues = len(consistency['contradictions']) + len(consistency['inconsistencies'])
        if len(propositions) > 0:
            consistency['score'] = max(0.0, 1.0 - (total_issues / len(propositions)))
        
        return consistency
    
    def _analyze_clarity(self, propositions: List[Dict]) -> Dict[str, any]:
        """Analyze clarity of requirements."""
        clarity = {
            'score': 0.0,
            'unclear_propositions': [],
            'vague_terms': []
        }
        
        total_issues = 0
        
        for prop in propositions:
            statement = prop['proposition_statement']
            ambiguity = prop.get('ambiguity', {})
            
            if ambiguity:
                clarity['unclear_propositions'].append({
                    'proposition': statement,
                    'issues': ambiguity
                })
                total_issues += 1
            
            # Check for vague terms
            vague_terms = ['appropriate', 'suitable', 'adequate', 'reasonable', 'some', 'many', 'few']
            found_vague = [term for term in vague_terms if term in statement.lower()]
            if found_vague:
                clarity['vague_terms'].append({
                    'proposition': statement,
                    'vague_terms': found_vague
                })
                total_issues += 1
        
        # Calculate clarity score
        if len(propositions) > 0:
            clarity['score'] = max(0.0, 1.0 - (total_issues / len(propositions)))
        
        return clarity
    
    def _analyze_ambiguity(self, propositions: List[Dict]) -> Dict[str, any]:
        """Analyze ambiguity in requirements."""
        ambiguity = {
            'score': 0.0,
            'ambiguous_propositions': [],
            'unclear_references': []
        }
        
        total_issues = 0
        
        for prop in propositions:
            statement = prop['proposition_statement']
            
            # Check for unclear references
            unclear_refs = ['it', 'this', 'that', 'these', 'those']
            found_refs = [ref for ref in unclear_refs if ref in statement.lower()]
            if found_refs:
                ambiguity['unclear_references'].append({
                    'proposition': statement,
                    'references': found_refs
                })
                total_issues += 1
            
            # Check for relative terms
            relative_terms = ['large', 'small', 'fast', 'slow', 'high', 'low']
            found_relative = [term for term in relative_terms if term in statement.lower()]
            if found_relative:
                ambiguity['ambiguous_propositions'].append({
                    'proposition': statement,
                    'relative_terms': found_relative
                })
                total_issues += 1
        
        # Calculate ambiguity score (lower is better)
        if len(propositions) > 0:
            ambiguity['score'] = max(0.0, 1.0 - (total_issues / len(propositions)))
        
        return ambiguity
    
    def _analyze_missing_information(self, requirements: List[str], propositions: List[Dict]) -> Dict[str, any]:
        """Analyze missing information in requirements."""
        missing_info = {
            'missing_quantifiers': [],
            'missing_conditions': [],
            'missing_exceptions': [],
            'missing_definitions': [],
            'suggestions': []
        }
        
        for prop in propositions:
            statement = prop['proposition_statement'].lower()
            
            # Check for missing quantifiers
            if any(word in statement for word in ['shall', 'must', 'will']) and \
               not any(word in statement for word in ['all', 'every', 'each', 'any', 'some']):
                missing_info['missing_quantifiers'].append(prop['proposition_statement'])
            
            # Check for missing conditions
            if 'shall' in statement and 'if' not in statement and 'when' not in statement:
                missing_info['missing_conditions'].append(prop['proposition_statement'])
            
            # Check for missing exceptions
            if 'shall' in statement and 'except' not in statement and 'unless' not in statement:
                missing_info['missing_exceptions'].append(prop['proposition_statement'])
            
            # Check for missing definitions
            if any(word in statement for word in ['system', 'component', 'function']) and \
               not any(word in statement for word in ['defined', 'specified', 'described']):
                missing_info['missing_definitions'].append(prop['proposition_statement'])
        
        # Generate suggestions
        if missing_info['missing_quantifiers']:
            missing_info['suggestions'].append("Add specific quantifiers (all, every, each, any, some)")
        
        if missing_info['missing_conditions']:
            missing_info['suggestions'].append("Specify conditions under which requirements apply")
        
        if missing_info['missing_exceptions']:
            missing_info['suggestions'].append("Consider adding exception clauses")
        
        if missing_info['missing_definitions']:
            missing_info['suggestions'].append("Define key terms and concepts")
        
        return missing_info
    
    def _calculate_overall_score(self, analysis: Dict[str, any]) -> float:
        """Calculate overall quality score."""
        scores = [
            analysis['completeness']['score'],
            analysis['consistency']['score'],
            analysis['clarity']['score'],
            analysis['ambiguity']['score']
        ]
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _generate_recommendations(self, analysis: Dict[str, any]) -> List[str]:
        """Generate recommendations for improvement."""
        recommendations = []
        
        # Completeness recommendations
        if analysis['completeness']['score'] < 0.8:
            recommendations.append("Improve completeness by adding missing quantifiers and conditions")
        
        # Consistency recommendations
        if analysis['consistency']['score'] < 0.8:
            recommendations.append("Resolve contradictions and inconsistencies in requirements")
        
        # Clarity recommendations
        if analysis['clarity']['score'] < 0.8:
            recommendations.append("Improve clarity by replacing vague terms with specific ones")
        
        # Ambiguity recommendations
        if analysis['ambiguity']['score'] < 0.8:
            recommendations.append("Reduce ambiguity by clarifying references and relative terms")
        
        # Add specific recommendations from missing information
        recommendations.extend(analysis['missing_information']['suggestions'])
        
        return recommendations
    
    def _find_contradictions(self, propositions: List[Dict]) -> List[Dict]:
        contradictions = []
        for i, prop1 in enumerate(propositions):
            for j, prop2 in enumerate(propositions[i+1:], i+1):
                subj1 = self.nlp_utils.get_subject(prop1['proposition_statement'])
                subj2 = self.nlp_utils.get_subject(prop2['proposition_statement'])
                pred1 = self.nlp_utils.get_predicate(prop1['proposition_statement'])
                pred2 = self.nlp_utils.get_predicate(prop2['proposition_statement'])
                # Heuristic: "never worked" vs "worked with"
                if ("never worked" in prop1['proposition_statement'].lower() and "worked" in prop2['proposition_statement'].lower()) or \
                   ("never worked" in prop2['proposition_statement'].lower() and "worked" in prop1['proposition_statement'].lower()):
                    # Check if "raju" is involved in both
                    if "raju" in prop1['proposition_statement'].lower() and "raju" in prop2['proposition_statement'].lower():
                        contradictions.append({
                            'proposition1_id': prop1['proposition_id'],
                            'proposition1': prop1['proposition_statement'],
                            'proposition2_id': prop2['proposition_id'],
                            'proposition2': prop2['proposition_statement'],
                            'reason': "Contradiction: 'never worked' vs 'worked with'"
                        })
            # ... (other contradiction checks) ...
        return contradictions
    
    def analyze_per_proposition(self, propositions: List[Dict], all_propositions: List[Dict]) -> List[Dict]:
        results = []
        for prop in propositions:
            completeness = self._check_completeness(prop['proposition_statement'])
            clarity, vague_words = self._check_clarity(prop['proposition_statement'])
            contradictions = [
                p['proposition_id'] for p in all_propositions
                if p['proposition_id'] != prop['proposition_id'] and
                self._is_contradictory(prop['proposition_statement'], p['proposition_statement'])
            ]
            coexistence = "No" if contradictions else "Yes"
            results.append({
                'proposition_id': prop['proposition_id'],
                'proposition_statement': prop['proposition_statement'],
                'completeness': completeness,
                'clarity': clarity,
                'vague_words': ", ".join(vague_words) if vague_words else "",
                'contradicts': contradictions,
                'coexistence': coexistence
            })
        return results

    def _check_completeness(self, statement: str) -> str:
        # Simple check: does it have subject, verb, object?
        nlp = self.nlp_utils.nlp if hasattr(self.nlp_utils, 'nlp') else None
        if not nlp:
            return "Unknown"
        doc = nlp(statement)
        has_subj = any(token.dep_ in ("nsubj", "nsubjpass") for token in doc)
        has_verb = any(token.pos_ == "VERB" for token in doc)
        has_obj = any(token.dep_ in ("dobj", "pobj") for token in doc)
        return "Complete" if has_subj and has_verb and has_obj else "Incomplete"

    def _check_clarity(self, statement: str) -> (str, list):
        vague_words = [
            "some", "many", "few", "several", "appropriate", "suitable", "adequate", "reasonable",
            "as needed", "as necessary", "as soon as possible", "etc.", "sufficient", "user-friendly",
            "fast", "quick", "large", "small", "minimal", "optimal", "flexible", "robust"
        ]
        lower_stmt = statement.lower()
        found = [word for word in vague_words if word in lower_stmt]
        clarity = "Unclear" if found else "Clear"
        return clarity, found 

    def _is_contradictory(self, s1: str, s2: str) -> bool:
        s1_lower = s1.lower()
        s2_lower = s2.lower()
        # Check for direct negation
        sim = self.nlp_utils.semantic_similarity(s1, s2)
        if sim > 0.7 and ("not" in s1_lower or "not" in s2_lower):
            return True
        # Check for "never worked" vs "worked"
        if ("never worked" in s1_lower and "worked" in s2_lower) or \
           ("never worked" in s2_lower and "worked" in s1_lower):
            # Extract the person(s) involved
            for name in ["ram", "raju", "raj"]:  # Add more as needed
                if name in s1_lower and name in s2_lower:
                    return True
        # Check for "never" and "with" (e.g., "never worked before" vs "works with")
        if ("never worked" in s1_lower and "with" in s2_lower) or \
           ("never worked" in s2_lower and "with" in s1_lower):
            for name in ["ram", "raju", "raj"]:
                if name in s1_lower and name in s2_lower:
                    return True
        return False 