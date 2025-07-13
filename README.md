# Requirement Elicitation Tool (RET)

A comprehensive tool for analyzing and validating software requirements using propositional logic and natural language processing.

## Features

- **Automatic Proposition Extraction**: Converts requirement statements into atomic proposition statements
- **Relation Detection**: Automatically infers logical relations between propositions
- **Negation Analysis**: Generates and validates relevant negations for each proposition
- **Quality Assessment**: Analyzes requirements for completeness, consistency, and clarity
- **Interactive GUI**: User-friendly interface built with Streamlit
- **Comprehensive Reporting**: Detailed analysis reports in CSV/Excel format

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Download NLTK data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords')"
```

3. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

Run the application:
```bash
streamlit run app.py
```

## Project Structure

```
ret/
├── app.py                 # Main Streamlit application
├── core/
│   ├── __init__.py
│   ├── proposition_extractor.py    # Extracts propositions from requirements
│   ├── relation_analyzer.py        # Analyzes logical relations
│   ├── negation_generator.py       # Generates negations
│   ├── quality_analyzer.py         # Analyzes requirement quality
│   └── database_manager.py         # Manages data tables
├── utils/
│   ├── __init__.py
│   ├── nlp_utils.py               # NLP helper functions
│   └── logic_utils.py             # Logic helper functions
└── data/
    └── output/                    # Generated reports
```

## Output Tables

1. **Requirement ID Table**: requirement_id, requirement_statement
2. **Proposition Table**: proposition_id (PK), requirement_id (FK), proposition_statement, proposition_type
3. **Relation Table**: relation_id (PK), proposition_id_1 (FK), proposition_id_2 (FK), relation_type
4. **Inference Table**: inference_id (PK), relation_id (FK), inference_description

## Analysis Features

- **Proposition Classification**: Auto-detects telecommand, telemetry, software functions, hardware functions
- **Logical Relations**: Identifies dependent, disjoint, biconditional requirements
- **Negation Validation**: Checks for contradictions and logical consistency
- **Quality Metrics**: Completeness, clarity, consistency, and ambiguity analysis 