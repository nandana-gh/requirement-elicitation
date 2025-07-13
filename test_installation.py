#!/usr/bin/env python3
"""
Test script to verify RET tool installation and dependencies.
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported."""
    required_packages = [
        'streamlit',
        'pandas',
        'nltk',
        'spacy',
        'textblob',
        'sympy',
        'scikit-learn',
        'numpy',
        'openpyxl',
        'plotly',
        'networkx',
        'matplotlib',
        'seaborn'
    ]
    
    print("Testing package imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please install missing packages using: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True

def test_nltk_data():
    """Test if NLTK data is available."""
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/stopwords')
        print("✅ NLTK data available")
        return True
    except LookupError as e:
        print(f"❌ NLTK data missing: {e}")
        print("Please run: python -c \"import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords')\"")
        return False

def test_spacy_model():
    """Test if spaCy model is available."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy model available")
        return True
    except OSError:
        print("❌ spaCy model missing")
        print("Please run: python -m spacy download en_core_web_sm")
        return False

def test_ret_modules():
    """Test if RET modules can be imported."""
    try:
        from core.proposition_extractor import PropositionExtractor
        from core.relation_analyzer import RelationAnalyzer
        from core.quality_analyzer import QualityAnalyzer
        from core.database_manager import DatabaseManager
        from utils.nlp_utils import NLPUtils
        from utils.logic_utils import LogicUtils
        print("✅ RET modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ RET module import failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🔍 Testing RET Tool Installation")
    print("=" * 40)
    
    all_passed = True
    
    # Test package imports
    if not test_imports():
        all_passed = False
    
    print()
    
    # Test NLTK data
    if not test_nltk_data():
        all_passed = False
    
    print()
    
    # Test spaCy model
    if not test_spacy_model():
        all_passed = False
    
    print()
    
    # Test RET modules
    if not test_ret_modules():
        all_passed = False
    
    print()
    print("=" * 40)
    
    if all_passed:
        print("🎉 All tests passed! RET tool is ready to use.")
        print("Run the application with: streamlit run app.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 