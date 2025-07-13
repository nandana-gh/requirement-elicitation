import pandas as pd
import os
from typing import List, Dict, Tuple
from datetime import datetime

class DatabaseManager:
    def __init__(self):
        self.output_dir = "data/output"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def create_requirement_table(self, requirements: List[str]) -> pd.DataFrame:
        """Create the Requirement ID table."""
        data = []
        for i, requirement in enumerate(requirements, 1):
            data.append({
                'requirement_id': f'R{i}',
                'requirement_statement': requirement
            })
        
        return pd.DataFrame(data)
    
    def create_proposition_table(self, propositions: List[Dict]) -> pd.DataFrame:
        """Create the Proposition table."""
        data = []
        for prop in propositions:
            data.append({
                'proposition_id': prop['proposition_id'],
                'requirement_id': prop['requirement_id'],
                'proposition_statement': prop['proposition_statement'],
                'proposition_type': prop['proposition_type']
            })
        
        return pd.DataFrame(data)
    
    def create_relation_table(self, relations: List[Dict]) -> pd.DataFrame:
        """Create the Relation table."""
        data = []
        for relation in relations:
            data.append({
                'relation_id': relation['relation_id'],
                'proposition_id_1': relation['proposition_id_1'],
                'proposition_id_2': relation['proposition_id_2'],
                'relation_type': relation['type'],
                'confidence': relation.get('confidence', 0.0),
                'reasoning': relation.get('reasoning', '')
            })
        
        return pd.DataFrame(data)
    
    def create_inference_table(self, inferences: List[Dict]) -> pd.DataFrame:
        """Create the Inference table."""
        data = []
        for inference in inferences:
            data.append({
                'inference_id': inference['inference_id'],
                'relation_id': inference.get('relation_id', ''),
                'inference_description': self._format_inference_description(inference)
            })
        
        return pd.DataFrame(data)
    
    def _format_inference_description(self, inference: Dict) -> str:
        """Format inference description for the table."""
        if inference['type'] == 'negation_analysis':
            return f"Negation Analysis: {inference.get('negation_statement', '')} - {'; '.join(inference.get('analysis', []))}"
        else:
            return inference.get('reasoning', '')
    
    def create_quality_report(self, quality_analysis: Dict[str, any]) -> pd.DataFrame:
        """Create a comprehensive quality report."""
        report_data = []
        
        # Overall quality metrics
        report_data.append({
            'metric': 'Overall Quality Score',
            'value': f"{quality_analysis['overall_score']:.2f}",
            'status': self._get_status(quality_analysis['overall_score'])
        })
        
        # Individual quality metrics
        for metric, data in quality_analysis.items():
            if isinstance(data, dict) and 'score' in data:
                report_data.append({
                    'metric': metric.title(),
                    'value': f"{data['score']:.2f}",
                    'status': self._get_status(data['score'])
                })
        
        return pd.DataFrame(report_data)
    
    def _get_status(self, score: float) -> str:
        """Get status based on score."""
        if score >= 0.8:
            return "Good"
        elif score >= 0.6:
            return "Fair"
        else:
            return "Poor"
    
    def create_negation_report(self, negations: List[Dict]) -> pd.DataFrame:
        """Create a negation analysis report."""
        data = []
        for negation in negations:
            data.append({
                'proposition_id': negation['proposition_id'],
                'original_statement': negation['original_statement'],
                'negation': negation['negation'],
                'is_valid': negation['validation']['is_valid'],
                'issues': '; '.join(negation['validation']['issues']),
                'semantic_similarity': f"{negation['semantic_similarity']:.2f}"
            })
        
        return pd.DataFrame(data)
    
    def create_comprehensive_report(self, all_data: Dict[str, any]) -> pd.DataFrame:
        """Create a comprehensive analysis report."""
        report_sections = []
        
        # Executive Summary
        report_sections.append({
            'section': 'Executive Summary',
            'content': f"Total Requirements: {len(all_data.get('requirements', []))}, "
                      f"Total Propositions: {len(all_data.get('propositions', []))}, "
                      f"Total Relations: {len(all_data.get('relations', []))}, "
                      f"Quality Score: {all_data.get('quality_analysis', {}).get('overall_score', 0):.2f}"
        })
        
        # Quality Analysis
        quality = all_data.get('quality_analysis', {})
        if quality:
            report_sections.append({
                'section': 'Quality Analysis',
                'content': f"Completeness: {quality.get('completeness', {}).get('score', 0):.2f}, "
                          f"Consistency: {quality.get('consistency', {}).get('score', 0):.2f}, "
                          f"Clarity: {quality.get('clarity', {}).get('score', 0):.2f}, "
                          f"Ambiguity: {quality.get('ambiguity', {}).get('score', 0):.2f}"
            })
        
        # Issues Summary
        issues = []
        if quality.get('completeness', {}).get('issues'):
            issues.append(f"Completeness Issues: {len(quality['completeness']['issues'])}")
        if quality.get('consistency', {}).get('contradictions'):
            issues.append(f"Contradictions: {len(quality['consistency']['contradictions'])}")
        if quality.get('clarity', {}).get('unclear_propositions'):
            issues.append(f"Unclear Propositions: {len(quality['clarity']['unclear_propositions'])}")
        
        if issues:
            report_sections.append({
                'section': 'Issues Summary',
                'content': '; '.join(issues)
            })
        
        # Recommendations
        recommendations = quality.get('recommendations', [])
        if recommendations:
            report_sections.append({
                'section': 'Recommendations',
                'content': '; '.join(recommendations)
            })
        
        return pd.DataFrame(report_sections)
    
    def export_to_excel(self, data: Dict[str, any], filename: str = None) -> str:
        """Export all data to Excel file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ret_analysis_{timestamp}.xlsx"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Create all tables
            if 'requirements' in data:
                req_df = self.create_requirement_table(data['requirements'])
                req_df.to_excel(writer, sheet_name='Requirements', index=False)
            
            if 'propositions' in data:
                prop_df = self.create_proposition_table(data['propositions'])
                prop_df.to_excel(writer, sheet_name='Propositions', index=False)
            
            if 'relations' in data:
                rel_df = self.create_relation_table(data['relations'])
                rel_df.to_excel(writer, sheet_name='Relations', index=False)
            
            if 'inferences' in data:
                inf_df = self.create_inference_table(data['inferences'])
                inf_df.to_excel(writer, sheet_name='Inferences', index=False)
            
            if 'quality_analysis' in data:
                qual_df = self.create_quality_report(data['quality_analysis'])
                qual_df.to_excel(writer, sheet_name='Quality_Analysis', index=False)
            
            if 'negations' in data:
                neg_df = self.create_negation_report(data['negations'])
                neg_df.to_excel(writer, sheet_name='Negation_Analysis', index=False)
            
            # Comprehensive report
            comp_df = self.create_comprehensive_report(data)
            comp_df.to_excel(writer, sheet_name='Comprehensive_Report', index=False)
        
        return filepath
    
    def export_to_csv(self, data: Dict[str, any], base_filename: str = None) -> List[str]:
        """Export all data to CSV files."""
        if base_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"ret_analysis_{timestamp}"
        
        exported_files = []
        
        # Export each table to separate CSV
        if 'requirements' in data:
            req_df = self.create_requirement_table(data['requirements'])
            filename = f"{base_filename}_requirements.csv"
            filepath = os.path.join(self.output_dir, filename)
            req_df.to_csv(filepath, index=False)
            exported_files.append(filepath)
        
        if 'propositions' in data:
            prop_df = self.create_proposition_table(data['propositions'])
            filename = f"{base_filename}_propositions.csv"
            filepath = os.path.join(self.output_dir, filename)
            prop_df.to_csv(filepath, index=False)
            exported_files.append(filepath)
        
        if 'relations' in data:
            rel_df = self.create_relation_table(data['relations'])
            filename = f"{base_filename}_relations.csv"
            filepath = os.path.join(self.output_dir, filename)
            rel_df.to_csv(filepath, index=False)
            exported_files.append(filepath)
        
        if 'inferences' in data:
            inf_df = self.create_inference_table(data['inferences'])
            filename = f"{base_filename}_inferences.csv"
            filepath = os.path.join(self.output_dir, filename)
            inf_df.to_csv(filepath, index=False)
            exported_files.append(filepath)
        
        if 'quality_analysis' in data:
            qual_df = self.create_quality_report(data['quality_analysis'])
            filename = f"{base_filename}_quality_analysis.csv"
            filepath = os.path.join(self.output_dir, filename)
            qual_df.to_csv(filepath, index=False)
            exported_files.append(filepath)
        
        return exported_files 