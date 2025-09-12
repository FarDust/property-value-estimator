#!/usr/bin/env python3
"""
Data Quality Analysis Script

Analyzes CSV data files for duplicates, missing values, and other quality issues.
Generates a detailed markdown report.
"""
import pandas as pd
import typer
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

app = typer.Typer(help="Analyze data quality in CSV files")


def analyze_duplicates(df: pd.DataFrame, id_column: str, file_name: str) -> Dict[str, Any]:
    """Analyze duplicate records in a DataFrame."""
    total_records = len(df)
    
    analysis = {
        'file_name': file_name,
        'total_records': total_records,
        'duplicate_records_count': 0,
        'unique_duplicate_ids': 0,
        'duplicate_percentage': 0.0,
        'duplicate_details': [],
        'has_id_column': id_column in df.columns if id_column else False
    }
    
    # If no ID column specified or doesn't exist, check for row-level duplicates
    if not id_column or id_column not in df.columns:
        # Check for completely duplicate rows
        duplicated_mask = df.duplicated(keep=False)
        duplicate_records = df[duplicated_mask].copy()
        
        analysis.update({
            'duplicate_records_count': len(duplicate_records),
            'duplicate_percentage': (len(duplicate_records) / total_records) * 100 if total_records > 0 else 0,
            'analysis_type': 'row_level_duplicates'
        })
        
        if len(duplicate_records) > 0:
            # Group by all columns to find duplicate sets
            duplicate_groups = duplicate_records.groupby(list(df.columns))
            for group_key, group_data in duplicate_groups:
                if len(group_data) > 1:
                    duplicate_detail = {
                        'group_identifier': str(hash(str(group_key))),  # Hash of the row values
                        'count': len(group_data),
                        'are_identical': True,  # By definition, row-level duplicates are identical
                        'differing_columns': [],
                        'records': group_data.to_dict('records')[:5]  # Limit to 5 examples
                    }
                    analysis['duplicate_details'].append(duplicate_detail)
        
        return analysis
    
    # Original ID-based analysis for files with ID columns
    duplicated_mask = df.duplicated(subset=[id_column], keep=False)
    duplicate_records = df[duplicated_mask].copy()
    unique_duplicate_ids = df[duplicated_mask][id_column].nunique()
    
    analysis.update({
        'duplicate_records_count': len(duplicate_records),
        'unique_duplicate_ids': unique_duplicate_ids,
        'duplicate_percentage': (len(duplicate_records) / total_records) * 100 if total_records > 0 else 0,
        'analysis_type': 'id_based_duplicates'
    })
    
    if len(duplicate_records) > 0:
        # Group by ID to analyze each duplicate set
        for duplicate_id in duplicate_records[id_column].unique():
            id_records = df[df[id_column] == duplicate_id].copy()
            
            # Check if all duplicates are identical
            first_record = id_records.iloc[0]
            are_identical = True
            differing_columns = []
            
            for _, row in id_records.iloc[1:].iterrows():
                for col in df.columns:
                    if col != id_column:  # Skip the ID column
                        try:
                            # Handle NaN comparisons
                            if pd.isna(first_record[col]) and pd.isna(row[col]):
                                continue
                            elif first_record[col] != row[col]:
                                are_identical = False
                                if col not in differing_columns:
                                    differing_columns.append(col)
                        except:
                            # Handle comparison errors (e.g., different types)
                            are_identical = False
                            if col not in differing_columns:
                                differing_columns.append(col)
            
            duplicate_detail = {
                'id': str(duplicate_id),
                'count': len(id_records),
                'are_identical': are_identical,
                'differing_columns': differing_columns,
                'records': id_records.to_dict('records')
            }
            
            analysis['duplicate_details'].append(duplicate_detail)
    
    return analysis


def analyze_missing_values(df: pd.DataFrame, file_name: str) -> Dict[str, Any]:
    """Analyze missing values in a DataFrame."""
    total_records = len(df)
    missing_analysis = {
        'file_name': file_name,
        'total_records': total_records,
        'columns_with_missing': [],
        'total_missing_cells': df.isnull().sum().sum()
    }
    
    for column in df.columns:
        missing_count = df[column].isnull().sum()
        if missing_count > 0:
            missing_analysis['columns_with_missing'].append({
                'column': column,
                'missing_count': missing_count,
                'missing_percentage': (missing_count / total_records) * 100
            })
    
    return missing_analysis


def generate_markdown_report(analyses: Dict[str, Dict[str, Any]]) -> str:
    """Generate a markdown report from the analyses."""
    
    report = f"""# Data Quality Analysis Report

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

"""
    
    # Summary table
    total_files = len(analyses)
    total_records = sum(analysis.get('duplicate_analysis', {}).get('total_records', 0) for analysis in analyses.values())
    total_duplicates = sum(analysis.get('duplicate_analysis', {}).get('duplicate_records_count', 0) for analysis in analyses.values())
    
    report += f"""
| Metric | Value |
|--------|-------|
| Files Analyzed | {total_files} |
| Total Records | {total_records:,} |
| Total Duplicate Records | {total_duplicates:,} |
| Overall Duplicate Rate | {(total_duplicates / total_records * 100):.2f}% |

"""
    
    # Detailed analysis for each file
    for file_path, analysis in analyses.items():
        report += f"\n## Analysis: {Path(file_path).name}\n\n"
        
        # Duplicate analysis
        if 'duplicate_analysis' in analysis:
            dup_analysis = analysis['duplicate_analysis']
            report += f"### Duplicate Records Analysis\n\n"
            report += f"- **Total Records:** {dup_analysis['total_records']:,}\n"
            report += f"- **Duplicate Records:** {dup_analysis['duplicate_records_count']:,}\n"
            report += f"- **Unique IDs with Duplicates:** {dup_analysis['unique_duplicate_ids']:,}\n"
            report += f"- **Duplicate Percentage:** {dup_analysis['duplicate_percentage']:.2f}%\n\n"
            
            if dup_analysis['duplicate_details']:
                report += "#### Duplicate Details\n\n"
                
                identical_count = sum(1 for detail in dup_analysis['duplicate_details'] if detail['are_identical'])
                different_count = len(dup_analysis['duplicate_details']) - identical_count
                
                report += f"- **Identical Duplicates:** {identical_count} sets\n"
                report += f"- **Different Duplicates:** {different_count} sets\n\n"
                
                if different_count > 0:
                    report += "##### Records with Differences\n\n"
                    for detail in dup_analysis['duplicate_details']:
                        if not detail['are_identical']:
                            report += f"**ID: {detail['id']}** ({detail['count']} records)\n"
                            report += f"- Differing columns: {', '.join(detail['differing_columns'])}\n"
                            
                            # Show the actual differing values
                            for i, record in enumerate(detail['records']):
                                report += f"  - Record {i+1}: "
                                diff_values = []
                                for col in detail['differing_columns']:
                                    diff_values.append(f"{col}={record.get(col, 'N/A')}")
                                report += ", ".join(diff_values) + "\n"
                            report += "\n"
                
                # Sample of first few identical duplicates
                identical_samples = [detail for detail in dup_analysis['duplicate_details'] if detail['are_identical']][:5]
                if identical_samples:
                    report += "##### Sample Identical Duplicates\n\n"
                    for detail in identical_samples:
                        report += f"- **ID: {detail['id']}** ({detail['count']} identical records)\n"
                    if len(identical_samples) < identical_count:
                        report += f"- ... and {identical_count - len(identical_samples)} more\n"
                    report += "\n"
        
        # Missing values analysis
        if 'missing_analysis' in analysis:
            miss_analysis = analysis['missing_analysis']
            report += f"### Missing Values Analysis\n\n"
            report += f"- **Total Missing Cells:** {miss_analysis['total_missing_cells']:,}\n"
            
            if miss_analysis['columns_with_missing']:
                report += "\n#### Columns with Missing Values\n\n"
                report += "| Column | Missing Count | Missing % |\n"
                report += "|--------|---------------|----------|\n"
                for col_info in miss_analysis['columns_with_missing']:
                    report += f"| {col_info['column']} | {col_info['missing_count']:,} | {col_info['missing_percentage']:.2f}% |\n"
            else:
                report += "- **No missing values found** ✅\n"
            
            report += "\n"
    
    # Recommendations
    report += "## Recommendations\n\n"
    
    for file_path, analysis in analyses.items():
        if 'duplicate_analysis' in analysis:
            dup_analysis = analysis['duplicate_analysis']
            file_name = Path(file_path).name
            
            if dup_analysis['duplicate_records_count'] > 0:
                identical_count = sum(1 for detail in dup_analysis['duplicate_details'] if detail['are_identical'])
                different_count = len(dup_analysis['duplicate_details']) - identical_count
                
                report += f"### {file_name}\n\n"
                
                if identical_count > 0:
                    report += f"- ✅ **Safe to remove {identical_count} sets of identical duplicates** - these are exact copies and can be deduplicated by keeping the first occurrence.\n"
                
                if different_count > 0:
                    report += f"- ⚠️ **Careful review needed for {different_count} sets of differing duplicates** - these may represent legitimate different records or data quality issues.\n"
                    report += "  - Consider which record to keep based on data completeness, recency, or business rules.\n"
                    report += "  - May need manual review or domain expert input.\n"
                
                if identical_count > 0 and different_count == 0:
                    report += f"- 🎯 **Recommended action**: Use `df.drop_duplicates(subset=['id'], keep='first')` to remove {dup_analysis['duplicate_records_count']} duplicate records.\n"
                
                report += "\n"
            else:
                report += f"### {file_name}\n\n- ✅ **No duplicates found** - data is clean.\n\n"
    
    return report


@app.command()
def analyze_house_sales(
    csv_path: Path = typer.Option(
        "mle-project-challenge-2/data/kc_house_data.csv",
        help="Path to house sales CSV file"
    )
):
    """Analyze house sales data for duplicates and quality issues."""
    if not csv_path.exists():
        typer.echo(f"❌ CSV file not found: {csv_path}")
        raise typer.Exit(1)
    
    typer.echo(f"📊 Analyzing house sales data: {csv_path}")
    
    # Read data
    df = pd.read_csv(csv_path, dtype={'zipcode': str})
    
    # Perform analyses
    duplicate_analysis = analyze_duplicates(df, 'id', str(csv_path))
    missing_analysis = analyze_missing_values(df, str(csv_path))
    
    analyses = {
        str(csv_path): {
            'duplicate_analysis': duplicate_analysis,
            'missing_analysis': missing_analysis
        }
    }
    
    # Generate report
    report = generate_markdown_report(analyses)
    
    # Save report to reports folder
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / "data_quality_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    typer.echo(f"✅ Analysis complete! Report saved to: {report_path}")
    typer.echo(f"📈 Found {duplicate_analysis['duplicate_records_count']} duplicate records out of {duplicate_analysis['total_records']} total records")


@app.command()
def analyze_all():
    """Analyze all CSV files for duplicates and quality issues."""
    csv_configs = [
        {
            "file": "mle-project-challenge-2/data/kc_house_data.csv",
            "id_column": "id",  # Has explicit ID column
            "dtype": {'zipcode': str}
        },
        {
            "file": "mle-project-challenge-2/data/zipcode_demographics.csv", 
            "id_column": "zipcode",  # Uses zipcode as natural key
            "dtype": {'zipcode': str}
        },
        {
            "file": "mle-project-challenge-2/data/future_unseen_examples.csv",
            "id_column": None,  # No ID column - analyze row-level duplicates
            "dtype": {'zipcode': str}
        }
    ]
    
    analyses = {}
    
    for config in csv_configs:
        csv_path = Path(config["file"])
        if not csv_path.exists():
            typer.echo(f"⚠️ Skipping missing file: {csv_path}")
            continue
        
        typer.echo(f"📊 Analyzing: {csv_path}")
        
        # Read data with appropriate dtypes
        df = pd.read_csv(csv_path, dtype=config["dtype"])
        
        # Perform analyses
        duplicate_analysis = analyze_duplicates(df, config["id_column"], str(csv_path))
        missing_analysis = analyze_missing_values(df, str(csv_path))
        
        analyses[str(csv_path)] = {
            'duplicate_analysis': duplicate_analysis,
            'missing_analysis': missing_analysis
        }
    
    # Generate comprehensive report
    report = generate_markdown_report(analyses)
    
    # Save report to reports folder
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / "comprehensive_data_quality_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    typer.echo(f"✅ Comprehensive analysis complete! Report saved to: {report_path}")


if __name__ == "__main__":
    app()
