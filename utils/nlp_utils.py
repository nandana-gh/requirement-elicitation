import torch
import nltk
import spacy
from textblob import TextBlob
import re
from typing import List, Dict, Tuple, Set
import pandas as pd
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer, util

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Please install spaCy model: python -m spacy download en_core_web_sm")
    nlp = None

class NLPUtils:
    def __init__(self):
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.proposition_keywords = {
            'telecommand': ['command', 'send', 'transmit', 'execute', 'control', 'activate'],
            'telemetry': ['measure', 'monitor', 'collect', 'data', 'sensor', 'reading'],
            'software_function': ['function', 'process', 'compute', 'calculate', 'algorithm', 'method'],
            'hardware_function': ['hardware', 'device', 'component', 'physical', 'mechanical', 'electrical']
        }
        # Force model to use CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.st_model = SentenceTransformer('paraphrase-MiniLM-L3-v2',device="cpu")
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:]', '', text)
        return text
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        if not nlp:
            return text.split('.')
        
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        return [s for s in sentences if s.strip()]
    
    def identify_proposition_type(self, proposition: str) -> str:
        """Identify the type of proposition based on keywords."""
        if not proposition:
            return "unknown"
        
        proposition_lower = proposition.lower()
        
        # Check for each type
        for prop_type, keywords in self.proposition_keywords.items():
            for keyword in keywords:
                if keyword in proposition_lower:
                    return prop_type
        
        # Default classification based on sentence structure
        if any(word in proposition_lower for word in ['shall', 'must', 'will', 'should']):
            return "requirement"
        elif any(word in proposition_lower for word in ['if', 'when', 'then']):
            return "conditional"
        else:
            return "statement"
    
    def detect_ambiguity(self, text: str) -> Dict[str, any]:
        """Detect ambiguous language in text."""
        ambiguity_indicators = {
            'vague_terms': ['some', 'many', 'few', 'several', 'appropriate', 'suitable'],
            'relative_terms': ['large', 'small', 'fast', 'slow', 'high', 'low'],
            'unclear_references': ['it', 'this', 'that', 'these', 'those'],
            'conditional_terms': ['if possible', 'when appropriate', 'as needed']
        }
        
        text_lower = text.lower()
        found_issues = {}
        
        for issue_type, terms in ambiguity_indicators.items():
            found_terms = [term for term in terms if term in text_lower]
            if found_terms:
                found_issues[issue_type] = found_terms
        
        return found_issues
    
    def detect_inconsistency(self, propositions: List[str]) -> List[Dict]:
        """Detect inconsistencies between propositions."""
        inconsistencies = []
        
        # Check for contradictory statements
        contradictions = [
            ('must', 'must not'),
            ('shall', 'shall not'),
            ('will', 'will not'),
            ('should', 'should not'),
            ('enable', 'disable'),
            ('allow', 'prevent'),
            ('require', 'forbid')
        ]
        
        for i, prop1 in enumerate(propositions):
            for j, prop2 in enumerate(propositions[i+1:], i+1):
                prop1_lower = prop1.lower()
                prop2_lower = prop2.lower()
                
                for pos_term, neg_term in contradictions:
                    if pos_term in prop1_lower and neg_term in prop2_lower:
                        # Check if they refer to the same subject
                        if self._similar_subject(prop1, prop2):
                            inconsistencies.append({
                                'type': 'contradiction',
                                'proposition1': prop1,
                                'proposition2': prop2,
                                'reason': f"Contradictory terms: {pos_term} vs {neg_term}"
                            })
        
        return inconsistencies
    
    def _similar_subject(self, prop1: str, prop2: str) -> bool:
        """Check if two propositions have similar subjects."""
        if not nlp:
            return True  # Default to True if spaCy not available
        
        doc1 = nlp(prop1)
        doc2 = nlp(prop2)
        
        # Extract nouns from both propositions
        nouns1 = [token.text.lower() for token in doc1 if token.pos_ in ['NOUN', 'PROPN']]
        nouns2 = [token.text.lower() for token in doc2 if token.pos_ in ['NOUN', 'PROPN']]
        
        # Check for common nouns
        common_nouns = set(nouns1) & set(nouns2)
        return len(common_nouns) > 0
    
    def analyze_completeness(self, propositions: List[str]) -> Dict[str, any]:
        """Analyze completeness of requirements."""
        completeness_issues = {
            'missing_quantifiers': [],
            'missing_conditions': [],
            'missing_exceptions': [],
            'incomplete_definitions': []
        }
        
        for prop in propositions:
            prop_lower = prop.lower()
            
            # Check for missing quantifiers
            if any(word in prop_lower for word in ['shall', 'must', 'will']) and \
               not any(word in prop_lower for word in ['all', 'every', 'each', 'any', 'some']):
                completeness_issues['missing_quantifiers'].append(prop)
            
            # Check for missing conditions
            if 'shall' in prop_lower and 'if' not in prop_lower and 'when' not in prop_lower:
                completeness_issues['missing_conditions'].append(prop)
            
            # Check for incomplete definitions
            if any(word in prop_lower for word in ['system', 'component', 'function']) and \
               not any(word in prop_lower for word in ['defined', 'specified', 'described']):
                completeness_issues['incomplete_definitions'].append(prop)
        
        return completeness_issues
    
    def get_sentiment_score(self, text: str) -> float:
        """Get sentiment score of text."""
        blob = TextBlob(text)
        return blob.sentiment.polarity
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        if not nlp:
            return text.split()
        
        doc = nlp(text)
        keywords = []
        
        for token in doc:
            if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and not token.is_stop:
                keywords.append(token.text.lower())
        
        return keywords

    def get_subject(self, text: str) -> str:
        """Extract the main subject from a sentence using spaCy."""
        if not nlp:
            return ""
        doc = nlp(text)
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass"):
                return token.text.lower()
        # fallback: first noun
        for token in doc:
            if token.pos_ == "NOUN":
                return token.text.lower()
        return ""

    def get_main_adjective(self, text: str) -> str:
        """Extract the main adjective from a sentence using spaCy."""
        if not nlp:
            return ""
        doc = nlp(text)
        for token in doc:
            if token.pos_ == "ADJ":
                return token.text.lower()
        return ""

    def are_antonyms(self, word1: str, word2: str) -> bool:
        """Check if two words are antonyms using WordNet."""
        antonyms = set()
        for syn in wordnet.synsets(word1, pos=wordnet.ADJ):
            for lemma in syn.lemmas():
                for ant in lemma.antonyms():
                    antonyms.add(ant.name())
        return word2 in antonyms

    def get_predicate(self, text: str) -> str:
        """Extract the main verb (predicate) from a sentence using spaCy."""
        if not nlp:
            return ""
        doc = nlp(text)
        for token in doc:
            if token.pos_ == "VERB":
                return token.lemma_.lower()
        return ""

    def predicate_similarity(self, pred1: str, pred2: str) -> float:
        """Simple similarity: 1 if same lemma, else 0."""
        return 1.0 if pred1 and pred2 and pred1 == pred2 else 0.0

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity using sentence-transformers."""
        emb1 = self.st_model.encode(text1, convert_to_tensor=True, batch_size=8)
        emb2 = self.st_model.encode(text2, convert_to_tensor=True, batch_size=8)
        return float(util.pytorch_cos_sim(emb1, emb2)[0][0]) 