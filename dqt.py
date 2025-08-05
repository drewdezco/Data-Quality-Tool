import pandas as pd
import numpy as np
import re
import json
from datetime import datetime
from collections import defaultdict


class DataQualityChecker:
    def __init__(self, df: pd.DataFrame, dataset_name: str = "Unknown Dataset"):
        self.df = df
        self.results = []
        self.dataset_name = dataset_name

    # -------- Dataset Management --------
    def set_dataset_name(self, dataset_name: str):
        """Set or update the dataset name for this checker instance."""
        self.dataset_name = dataset_name
    
    def get_dataset_name(self):
        """Get the current dataset name."""
        return self.dataset_name

    # -------- Expectation Helpers --------
    def _record_result(self, column, rule, success_rate, details):
        self.results.append({
            "column": column,
            "rule": rule,
            "success_rate": success_rate,
            "details": details
        })

    # -------- Expectations --------
    def expect_column_values_to_not_be_null(self, column):
        total = len(self.df[column])
        missing = self.df[column].isnull().sum()
        passed = total - missing
        success_rate = (passed / total) * 100 if total > 0 else 0

        self._record_result(
            column, "not null", success_rate,
            {"total": total, "passed": passed, "failed": missing}
        )

    def expect_column_values_to_be_unique(self, column):
        total = len(self.df[column])
        # Count all instances of duplicated values (not just subsequent ones)
        value_counts = self.df[column].value_counts()
        duplicate_values = value_counts[value_counts > 1]
        failed = duplicate_values.sum()  # Total count of all duplicate instances
        passed = total - failed
        success_rate = (passed / total) * 100 if total > 0 else 0

        self._record_result(
            column, "unique", success_rate,
            {"total": total, "passed": passed, "failed": failed}
        )

    def expect_column_values_to_be_in_set(self, column, allowed_values):
        total = len(self.df[column])
        invalid_mask = ~self.df[column].isin(allowed_values)
        failed = invalid_mask.sum()
        passed = total - failed
        success_rate = (passed / total) * 100 if total > 0 else 0

        self._record_result(
            column, "in allowed set", success_rate,
            {"total": total, "passed": passed, "failed": failed, "allowed_values": allowed_values}
        )

    def expect_column_values_to_match_regex(self, column, pattern):
        regex = re.compile(pattern)
        total = len(self.df[column])  # Include all rows, including nulls
        # Count null values as failed regex matches
        null_count = self.df[column].isnull().sum()
        non_matching = self.df[column].dropna().apply(lambda x: not bool(regex.match(str(x)))).sum()
        failed = null_count + non_matching
        passed = total - failed
        success_rate = (passed / total) * 100 if total > 0 else 0

        self._record_result(
            column, "matches regex", success_rate,
            {"total": total, "passed": passed, "failed": failed, "pattern": pattern}
        )

    def expect_column_values_to_be_in_range(self, column, min_val, max_val):
        total = len(self.df[column])
        # Count null values as failed range checks
        null_count = self.df[column].isnull().sum()
        out_of_range = self.df[column].dropna().apply(lambda x: not (min_val <= x <= max_val)).sum()
        failed = null_count + out_of_range
        passed = total - failed
        success_rate = (passed / total) * 100 if total > 0 else 0

        self._record_result(
            column, f"in range {min_val}-{max_val}", success_rate,
            {"total": total, "passed": passed, "failed": failed, "range": (min_val, max_val)}
        )

    def expect_column_values_to_be_in_date_range(self, column, min_date, max_date):
        dates = pd.to_datetime(self.df[column], errors="coerce")
        total = dates.notnull().sum()
        out_of_range = ((dates < pd.to_datetime(min_date)) | (dates > pd.to_datetime(max_date))).sum()
        passed = total - out_of_range
        success_rate = (passed / total) * 100 if total > 0 else 0

        self._record_result(
            column, "in date range", success_rate,
            {"total": total, "passed": passed, "failed": out_of_range, "range": (min_date, max_date)}
        )

    # -------- JSON Rules Loader --------
    def run_rules_from_json(self, rules):
        for expectation, configs in rules.items():
            method = getattr(self, expectation, None)
            if not method:
                print(f"Warning: {expectation} not implemented.")
                continue
            for config in configs:
                method(**config)

    # -------- Results --------
    def get_results(self):
        return pd.DataFrame(self.results)
    
    def get_comprehensive_results(self, title: str = "Data Quality Report"):
        """
        Get comprehensive data quality snapshot including all metrics for historical analysis.
        Returns a structured dictionary with all key information.
        
        Args:
            title (str): Title for the data quality report
        """
        if not hasattr(self, 'df') or self.df is None:
            return {"error": "No dataframe available for analysis"}
        
        # Basic results dataframe
        df_results = pd.DataFrame(self.results)
        
        # Calculate basic metrics
        total_checks = len(df_results)
        avg_pass_rate = df_results["success_rate"].mean() if total_checks > 0 else 0
        
        # Dataset overview metrics
        total_rows = len(self.df)
        total_columns = len(self.df.columns)
        total_cells = total_rows * total_columns
        null_cells = self.df.isnull().sum().sum()
        completeness_rate = ((total_cells - null_cells) / total_cells) * 100 if total_cells > 0 else 0
        
        # Helper functions (copied from generate_data_docs)
        def classify_data_type(col_data):
            """Classify data type with user-friendly names"""
            if pd.api.types.is_datetime64_any_dtype(col_data):
                return "Date/Time"
            elif pd.api.types.is_bool_dtype(col_data):
                return "Boolean"
            elif pd.api.types.is_numeric_dtype(col_data):
                if 'int' in str(col_data.dtype).lower():
                    return "Integer"
                elif 'float' in str(col_data.dtype).lower():
                    return "Decimal"
                else:
                    return "Numeric"
            elif pd.api.types.is_object_dtype(col_data):
                sample = col_data.dropna().head(100)
                if len(sample) > 0:
                    bool_count = sum(1 for x in sample if isinstance(x, bool))
                    if bool_count == len(sample):
                        return "Boolean"
                    bool_string_count = sum(1 for x in sample if str(x).lower() in ['true', 'false', '1', '0'])
                    if bool_string_count == len(sample):
                        return "Boolean"
                    if all(isinstance(x, str) for x in sample):
                        return "Text/String"
                return "Text/String"
            else:
                dtype_str = str(col_data.dtype).lower()
                if 'category' in dtype_str:
                    return "Category"
                else:
                    return dtype_str.title()

        def calculate_quality_scores(col_data):
            """Calculate quality scores for a column"""
            total_count = len(col_data)
            null_count = col_data.isnull().sum()
            distinct_count = col_data.nunique()
            
            # Completeness Score
            completeness = ((total_count - null_count) / total_count) * 100 if total_count > 0 else 0
            
            # Uniqueness Score
            uniqueness = (distinct_count / total_count) * 100 if total_count > 0 else 0
            
            # Enhanced Consistency Score (matching the field summary logic)
            consistency = 100.0
            non_null_data = col_data.dropna()
            
            if len(non_null_data) == 0:
                consistency = 0.0
            elif pd.api.types.is_object_dtype(col_data):
                sample = non_null_data.head(200)
                issues_count = 0
                total_checks = len(sample)
                
                if len(sample) > 0:
                    # Check for mixed data types
                    type_counts = {}
                    for item in sample:
                        item_type = type(item).__name__
                        type_counts[item_type] = type_counts.get(item_type, 0) + 1
                    
                    if len(type_counts) > 1:
                        minority_types = sum(count for type_name, count in type_counts.items() 
                                           if type_name != max(type_counts, key=type_counts.get))
                        if minority_types / len(sample) > 0.1:
                            issues_count += minority_types
                    
                    # Check for email format issues
                    if any('email' in str(col_data.name).lower() or '@' in str(val) for val in sample[:5]):
                        import re
                        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
                        for val in sample:
                            if '@' in str(val) and not email_pattern.match(str(val)):
                                issues_count += 1
                
                consistency = max(0, 100 - (issues_count / total_checks * 100))
                
            elif pd.api.types.is_numeric_dtype(col_data):
                if len(non_null_data) > 2:
                    try:
                        Q1 = non_null_data.quantile(0.25)
                        Q3 = non_null_data.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outliers = non_null_data[(non_null_data < lower_bound) | (non_null_data > upper_bound)]
                        outlier_percentage = len(outliers) / len(non_null_data) * 100
                        consistency = max(70, 100 - outlier_percentage * 2)
                    except Exception:
                        consistency = 95.0
            else:
                consistency = 95.0
            
            return {
                'completeness': round(completeness, 1),
                'uniqueness': round(uniqueness, 1),
                'consistency': round(consistency, 1)
            }

        def is_critical_data_element(col_data):
            """Determine if a column is a critical data element"""
            total_count = len(col_data)
            null_count = col_data.isnull().sum()
            completeness = ((total_count - null_count) / total_count) * 100 if total_count > 0 else 0
            data_density = (total_count - null_count) / total_count if total_count > 0 else 0
            is_critical = (completeness >= 80 and data_density >= 0.7 and null_count < total_count * 0.5)
            return is_critical
        
        # Analyze columns and separate critical vs other
        critical_columns = []
        other_columns = []
        all_column_details = {}
        
        for column in self.df.columns:
            col_data = self.df[column]
            data_type = classify_data_type(col_data)
            quality_scores = calculate_quality_scores(col_data)
            is_critical = is_critical_data_element(col_data)
            
            column_info = {
                'name': column,
                'data_type': data_type,
                'total_count': len(col_data),
                'null_count': col_data.isnull().sum(),
                'distinct_count': col_data.nunique(),
                'completeness': quality_scores['completeness'],
                'uniqueness': quality_scores['uniqueness'],
                'consistency': quality_scores['consistency'],
                'is_critical': is_critical
            }
            
            # Add numeric statistics if applicable
            if pd.api.types.is_numeric_dtype(col_data):
                column_info.update({
                    'mean': round(col_data.mean(), 2) if not col_data.empty else None,
                    'median': round(col_data.median(), 2) if not col_data.empty else None,
                    'std': round(col_data.std(), 2) if not col_data.empty else None,
                    'min': col_data.min() if not col_data.empty else None,
                    'max': col_data.max() if not col_data.empty else None
                })
            
            all_column_details[column] = column_info
            
            if is_critical:
                critical_columns.append(column_info)
            else:
                other_columns.append(column_info)
        
        # Calculate overall data quality metrics
        completeness_scores = [info['completeness'] for info in all_column_details.values()]
        uniqueness_scores = [info['uniqueness'] for info in all_column_details.values()]
        consistency_scores = [info['consistency'] for info in all_column_details.values()]
        
        overall_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0
        overall_uniqueness = sum(uniqueness_scores) / len(uniqueness_scores) if uniqueness_scores else 0
        overall_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0
        
        # Calculate column type distribution
        type_distribution = {}
        for column_info in all_column_details.values():
            data_type = column_info['data_type']
            type_distribution[data_type] = type_distribution.get(data_type, 0) + 1
        
        # Health status classification
        def get_health_status(rate):
            if rate >= 80:
                return "Healthy"
            elif rate >= 60:
                return "Degraded"
            else:
                return "Critical"
        
        # Rule success distribution
        rule_health_distribution = {"healthy": 0, "degraded": 0, "critical": 0}
        for rate in df_results["success_rate"]:
            if rate >= 80:
                rule_health_distribution["healthy"] += 1
            elif rate >= 60:
                rule_health_distribution["degraded"] += 1
            else:
                rule_health_distribution["critical"] += 1
        
        # Compile comprehensive snapshot
        comprehensive_snapshot = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "title": title,
                "dataset_name": self.dataset_name
            },
            "key_metrics": {
                "total_records": total_rows,
                "total_columns": total_columns,
                "total_cells": total_cells,
                "null_cells": null_cells,
                "data_completeness_rate": round(completeness_rate, 1),
                "total_rules_executed": total_checks,
                "overall_health_score": round(avg_pass_rate, 1),
                "overall_health_status": get_health_status(avg_pass_rate)
            },
            "overall_data_quality": {
                "completeness": round(overall_completeness, 1),
                "uniqueness": round(overall_uniqueness, 1),
                "consistency": round(overall_consistency, 1),
                "combined_score": round((overall_completeness + overall_uniqueness + overall_consistency) / 3, 1)
            },
            "critical_data_elements": {
                "count": len(critical_columns),
                "columns": critical_columns
            },
            "other_fields": {
                "count": len(other_columns),
                "columns": other_columns
            },
            "column_type_distribution": type_distribution,
            "rule_execution_summary": {
                "total_rules": total_checks,
                "healthy_rules": rule_health_distribution["healthy"],
                "degraded_rules": rule_health_distribution["degraded"],
                "critical_rules": rule_health_distribution["critical"],
                "health_percentages": {
                    "healthy": round((rule_health_distribution["healthy"] / total_checks) * 100, 1) if total_checks > 0 else 0,
                    "degraded": round((rule_health_distribution["degraded"] / total_checks) * 100, 1) if total_checks > 0 else 0,
                    "critical": round((rule_health_distribution["critical"] / total_checks) * 100, 1) if total_checks > 0 else 0
                }
            },
            "detailed_results": df_results.to_dict('records') if not df_results.empty else []
        }
        
        return comprehensive_snapshot
    
    def save_comprehensive_results_to_csv(self, title: str = "Data Quality Report", csv_filename: str = "data_quality_history.csv", include_field_summary: bool = True):
        """
        Save comprehensive results to CSV file for historical analysis.
        Creates a new file or appends to existing one.
        
        Args:
            title (str): Title for the data quality report
            csv_filename (str): Name of the CSV file to save/append to
            include_field_summary (bool): If True, also saves field-level details to a separate CSV
        """
        import csv
        import os
        
        # Get comprehensive results
        results = self.get_comprehensive_results(title=title)
        
        if "error" in results:
            print(f"Error: {results['error']}")
            return
        
        # Flatten the results into a single row
        flattened_row = self._flatten_comprehensive_results(results)
        
        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(csv_filename)
        
        # Write to CSV
        with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = flattened_row.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header only if file is new
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(flattened_row)
        
        print(f"✅ Data quality results saved to: {csv_filename}")
        
        # Optionally save field-level summary to separate CSV
        field_csv_filename = None
        if include_field_summary:
            # Generate field summary filename based on main filename
            base_name = csv_filename.rsplit('.', 1)[0]  # Remove .csv extension
            field_csv_filename = f"{base_name}_field_details.csv"
            self.save_field_summary_to_csv(title=title, csv_filename=field_csv_filename)
        
        return csv_filename, field_csv_filename if include_field_summary else csv_filename
    
    def save_field_summary_to_csv(self, title: str = "Data Quality Report", csv_filename: str = "field_summary_history.csv"):
        """
        Save detailed field-level information to CSV file for historical analysis.
        Creates a new file or appends to existing one. Each column becomes a separate row.
        
        Args:
            title (str): Title for the data quality report
            csv_filename (str): Name of the CSV file to save/append to
            
        Returns:
            str: The filename that was created/updated
        """
        import csv
        import os
        from datetime import datetime
        
        if not hasattr(self, 'df') or self.df is None:
            print("Error: No data available for field summary export.")
            return None
        
        def classify_data_type(col_data):
            """Classify data type with user-friendly names"""
            if pd.api.types.is_datetime64_any_dtype(col_data):
                return "Date/Time"
            elif pd.api.types.is_bool_dtype(col_data):
                return "Boolean"
            elif pd.api.types.is_numeric_dtype(col_data):
                if 'int' in str(col_data.dtype).lower():
                    return "Integer"
                elif 'float' in str(col_data.dtype).lower():
                    return "Decimal"
                else:
                    return "Numeric"
            elif pd.api.types.is_object_dtype(col_data):
                sample = col_data.dropna().head(100)
                if len(sample) > 0:
                    bool_count = sum(1 for x in sample if isinstance(x, bool))
                    if bool_count == len(sample):
                        return "Boolean"
                    bool_string_count = sum(1 for x in sample if str(x).lower() in ['true', 'false', '1', '0'])
                    if bool_string_count == len(sample):
                        return "Boolean"
                    if all(isinstance(x, str) for x in sample):
                        return "Text/String"
                return "Text/String"
            else:
                dtype_str = str(col_data.dtype).lower()
                if 'category' in dtype_str:
                    return "Category"
                else:
                    return dtype_str.title()

        def calculate_quality_scores(col_data):
            """Calculate quality scores for a column"""
            total_count = len(col_data)
            null_count = col_data.isnull().sum()
            distinct_count = col_data.nunique()
            
            # Completeness Score
            completeness = ((total_count - null_count) / total_count) * 100 if total_count > 0 else 0
            
            # Uniqueness Score
            uniqueness = (distinct_count / total_count) * 100 if total_count > 0 else 0
            
            # Enhanced Consistency Score
            consistency = 100.0
            non_null_data = col_data.dropna()
            
            if len(non_null_data) == 0:
                consistency = 0.0  # No data to assess
            elif pd.api.types.is_object_dtype(col_data):
                # For text/object columns, check multiple consistency factors
                sample = non_null_data.head(200)  # Larger sample for better assessment
                issues_count = 0
                total_checks = len(sample)
                
                if len(sample) > 0:
                    # 1. Check for mixed data types
                    type_counts = {}
                    for item in sample:
                        item_type = type(item).__name__
                        type_counts[item_type] = type_counts.get(item_type, 0) + 1
                    
                    # If more than 10% are different types, penalize consistency
                    if len(type_counts) > 1:
                        minority_types = sum(count for type_name, count in type_counts.items() 
                                           if type_name != max(type_counts, key=type_counts.get))
                        if minority_types / len(sample) > 0.1:
                            issues_count += minority_types
                    
                    # 2. Check for potential email format issues (if column might be email)
                    if any('email' in str(col_data.name).lower() or '@' in str(val) for val in sample[:5]):
                        import re
                        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
                        for val in sample:
                            if '@' in str(val) and not email_pattern.match(str(val)):
                                issues_count += 1
                    
                    # 3. Check for inconsistent string lengths (potential formatting issues)
                    if len(sample) > 5:
                        str_lengths = [len(str(val)) for val in sample]
                        mean_length = sum(str_lengths) / len(str_lengths)
                        # Flag values that are significantly different from mean length
                        for length in str_lengths:
                            if abs(length - mean_length) > mean_length * 0.5 and mean_length > 5:
                                issues_count += 0.5  # Partial penalty for length inconsistency
                    
                    # 4. Check for mixed case patterns (if alphabetic)
                    if sample.dtype == 'object':
                        case_patterns = {'upper': 0, 'lower': 0, 'title': 0, 'mixed': 0}
                        for val in sample:
                            str_val = str(val)
                            if str_val.isalpha():
                                if str_val.isupper():
                                    case_patterns['upper'] += 1
                                elif str_val.islower():
                                    case_patterns['lower'] += 1
                                elif str_val.istitle():
                                    case_patterns['title'] += 1
                                else:
                                    case_patterns['mixed'] += 1
                        
                        # If there's significant case inconsistency, penalize
                        total_alpha = sum(case_patterns.values())
                        if total_alpha > 5:
                            patterns_used = sum(1 for count in case_patterns.values() if count > 0)
                            if patterns_used > 2:  # More than 2 different case patterns
                                issues_count += total_alpha * 0.1
                
                # Calculate consistency as percentage of consistent values
                consistency = max(0, 100 - (issues_count / total_checks * 100))
                
            elif pd.api.types.is_numeric_dtype(col_data):
                # For numeric columns, check for outliers and data distribution consistency
                if len(non_null_data) > 2:
                    try:
                        Q1 = non_null_data.quantile(0.25)
                        Q3 = non_null_data.quantile(0.75)
                        IQR = Q3 - Q1
                        
                        # Define outliers using IQR method
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outliers = non_null_data[(non_null_data < lower_bound) | (non_null_data > upper_bound)]
                        
                        # Consistency reduced by percentage of outliers
                        outlier_percentage = len(outliers) / len(non_null_data) * 100
                        consistency = max(70, 100 - outlier_percentage * 2)  # Cap minimum at 70% for numeric
                        
                    except Exception:
                        consistency = 95.0  # Default high score if calculation fails
                        
            else:
                # For other data types (datetime, boolean, etc.)
                consistency = 95.0  # Assume good consistency for properly typed columns
            
            return {
                'completeness': round(completeness, 1),
                'uniqueness': round(uniqueness, 1),
                'consistency': round(consistency, 1)
            }

        def is_critical_data_element(col_data):
            """Determine if a column is a critical data element"""
            total_count = len(col_data)
            null_count = col_data.isnull().sum()
            completeness = ((total_count - null_count) / total_count) * 100 if total_count > 0 else 0
            data_density = (total_count - null_count) / total_count if total_count > 0 else 0
            is_critical = (completeness >= 80 and data_density >= 0.7 and null_count < total_count * 0.5)
            return is_critical
        
        # Generate timestamp
        timestamp = datetime.now().isoformat()
        
        # Prepare field-level data
        field_rows = []
        
        for column in self.df.columns:
            col_data = self.df[column]
            data_type = classify_data_type(col_data)
            quality_scores = calculate_quality_scores(col_data)
            is_critical = is_critical_data_element(col_data)
            
            # Base row data
            row = {
                'timestamp': timestamp,
                'title': title,
                'dataset_name': getattr(self, 'dataset_name', 'Unknown Dataset'),
                'column_name': column,
                'data_type': data_type,
                'total_count': len(col_data),
                'null_count': col_data.isnull().sum(),
                'distinct_count': col_data.nunique(),
                'completeness': quality_scores['completeness'],
                'uniqueness': quality_scores['uniqueness'],
                'consistency': quality_scores['consistency'],
                'is_critical': is_critical,
                'mean': '',
                'median': '',
                'std': '',
                'min': '',
                'max': ''
            }
            
            # Add statistical measures for numeric columns
            if pd.api.types.is_numeric_dtype(col_data):
                try:
                    numeric_data = col_data.dropna()
                    if len(numeric_data) > 0:
                        row['mean'] = round(numeric_data.mean(), 2)
                        row['median'] = round(numeric_data.median(), 2)
                        row['std'] = round(numeric_data.std(), 2)
                        row['min'] = numeric_data.min()
                        row['max'] = numeric_data.max()
                except Exception:
                    pass  # Keep empty values if calculation fails
            
            field_rows.append(row)
        
        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(csv_filename)
        
        # Write to CSV
        with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
            if field_rows:
                fieldnames = field_rows[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write header only if file is new
                if not file_exists:
                    writer.writeheader()
                
                # Write all field rows
                writer.writerows(field_rows)
        
        print(f"✅ Field summary data saved to: {csv_filename}")
        print(f"   📊 Exported details for {len(field_rows)} columns")
        critical_count = sum(1 for row in field_rows if row['is_critical'])
        print(f"   🔑 Critical elements: {critical_count}, Other fields: {len(field_rows) - critical_count}")
        
        return csv_filename
    
    def _flatten_comprehensive_results(self, results):
        """
        Flatten the nested comprehensive results into a single dictionary for CSV export.
        """
        flattened = {}
        
        # Metadata
        flattened.update({
            'timestamp': results['metadata']['timestamp'],
            'title': results['metadata']['title'],
            'dataset_name': results['metadata']['dataset_name']
        })
        
        # Key metrics
        key_metrics = results['key_metrics']
        flattened.update({
            'total_records': key_metrics['total_records'],
            'total_columns': key_metrics['total_columns'],
            'total_cells': key_metrics['total_cells'],
            'null_cells': key_metrics['null_cells'],
            'data_completeness_rate': key_metrics['data_completeness_rate'],
            'total_rules_executed': key_metrics['total_rules_executed'],
            'overall_health_score': key_metrics['overall_health_score'],
            'overall_health_status': key_metrics['overall_health_status']
        })
        
        # Overall data quality
        quality = results['overall_data_quality']
        flattened.update({
            'overall_completeness': quality['completeness'],
            'overall_uniqueness': quality['uniqueness'],
            'overall_consistency': quality['consistency'],
            'combined_quality_score': quality['combined_score']
        })
        
        # Critical data elements and other fields counts
        flattened.update({
            'critical_elements_count': results['critical_data_elements']['count'],
            'other_fields_count': results['other_fields']['count']
        })
        
        # Rule execution summary
        rule_summary = results['rule_execution_summary']
        flattened.update({
            'healthy_rules_count': rule_summary['healthy_rules'],
            'degraded_rules_count': rule_summary['degraded_rules'],
            'critical_rules_count': rule_summary['critical_rules'],
            'healthy_rules_percentage': rule_summary['health_percentages']['healthy'],
            'degraded_rules_percentage': rule_summary['health_percentages']['degraded'],
            'critical_rules_percentage': rule_summary['health_percentages']['critical']
        })
        
        # Column type distribution (flatten into separate columns)
        type_dist = results['column_type_distribution']
        for data_type, count in type_dist.items():
            flattened[f'columns_{data_type.lower().replace("/", "_").replace(" ", "_")}_count'] = count
        
        # Calculate averages for critical and other fields
        if results['critical_data_elements']['columns']:
            critical_completeness = [col['completeness'] for col in results['critical_data_elements']['columns']]
            critical_uniqueness = [col['uniqueness'] for col in results['critical_data_elements']['columns']]
            critical_consistency = [col['consistency'] for col in results['critical_data_elements']['columns']]
            
            flattened.update({
                'critical_avg_completeness': round(sum(critical_completeness) / len(critical_completeness), 1),
                'critical_avg_uniqueness': round(sum(critical_uniqueness) / len(critical_uniqueness), 1),
                'critical_avg_consistency': round(sum(critical_consistency) / len(critical_consistency), 1)
            })
        else:
            flattened.update({
                'critical_avg_completeness': 0,
                'critical_avg_uniqueness': 0,
                'critical_avg_consistency': 0
            })
        
        if results['other_fields']['columns']:
            other_completeness = [col['completeness'] for col in results['other_fields']['columns']]
            other_uniqueness = [col['uniqueness'] for col in results['other_fields']['columns']]
            other_consistency = [col['consistency'] for col in results['other_fields']['columns']]
            
            flattened.update({
                'other_avg_completeness': round(sum(other_completeness) / len(other_completeness), 1),
                'other_avg_uniqueness': round(sum(other_uniqueness) / len(other_uniqueness), 1),
                'other_avg_consistency': round(sum(other_consistency) / len(other_consistency), 1)
            })
        else:
            flattened.update({
                'other_avg_completeness': 0,
                'other_avg_uniqueness': 0,
                'other_avg_consistency': 0
            })
        
        return flattened

    # -------- HTML Report --------
    def generate_data_docs(self, title="Data Quality Report", dataset_name="Dataset"):
        df_results = pd.DataFrame(self.results)

        total_checks = len(df_results)
        avg_pass_rate = df_results["success_rate"].mean() if total_checks > 0 else 0

        # Determine textual health status based on avg_pass_rate
        def get_health_status(rate):
            if rate >= 80:
                return "Healthy"
            elif rate >= 60:
                return "Degraded"
            else:
                return "Critical"

        status_text = get_health_status(avg_pass_rate)

        # Classify each rule's success rate into health categories for the pie chart
        categories = {
            "healthy": 0,
            "degraded": 0,
            "critical": 0
        }

        for rate in df_results["success_rate"]:
            if rate >= 80:
                categories["healthy"] += 1
            elif rate >= 60:
                categories["degraded"] += 1
            else:
                categories["critical"] += 1

        if total_checks == 0:
            health_distribution = [0, 0, 0]
        else:
            health_distribution = [
                round((categories["healthy"] / total_checks) * 100, 2),
                round((categories["degraded"] / total_checks) * 100, 2),
                round((categories["critical"] / total_checks) * 100, 2)
            ]

        # Prepare data for bar chart
        bar_chart_data = {
            'labels': df_results['column'].tolist(),
            'datasets': [{
                'label': 'Success Rate (%)',
                'data': df_results['success_rate'].tolist(),
                'backgroundColor': [self._get_bar_color(rate) for rate in df_results['success_rate']]
            }]
        }

        def badge_color(rate):
            if rate >= 90:
                return "badge-green"
            elif rate >= 50:
                return "badge-yellow"
            else:
                return "badge-red"

        def format_details_human_readable(details):
            """Convert JSON details to human-readable format"""
            if isinstance(details, str):
                try:
                    details = json.loads(details)
                except:
                    return details

            readable_parts = []

            if 'passed' in details:
                readable_parts.append(f"<b>Passed:</b> <span class='passed-green'>{details['passed']}</span>")
            if 'failed' in details:
                readable_parts.append(f"<b>Failed:</b> <span class='failed-red'>{details['failed']}</span>")
            if 'total' in details:
                readable_parts.append(f"<b>Total records:</b> {details['total']}")
            if 'allowed_values' in details:
                readable_parts.append(f"<b>Allowed values:</b> <span class='allowed-set'>{', '.join(map(str, details['allowed_values']))}</span>")
            if 'pattern' in details:
                readable_parts.append(f"<b>Pattern:</b> <span class='pattern-highlight'>{details['pattern']}</span>")
            if 'range' in details:
                readable_parts.append(f"<b>Range:</b> <span class='range-highlight'>{details['range'][0]} to {details['range'][1]}</span>")

            return '<br>'.join(readable_parts)

        # --- Grouped, collapsible tables for Detailed Results ---
        grouped = defaultdict(list)
        for i, row in df_results.iterrows():
            grouped[row['rule']].append((i, row))

        def rule_label(rule):
            # Capitalize and prettify rule names for section headers
            return rule.replace('_', ' ').capitalize()
        
        def get_validation_description(rule):
            """Get a user-friendly description for each validation rule"""
            descriptions = {
                'not null': 'Checks for missing or empty values in the column. Ensures data completeness.',
                'unique': 'Verifies that all values in the column are distinct. Identifies duplicate entries.',
                'matches regex': 'Validates data format using pattern matching. Ensures values follow specific formats (emails, phone numbers, etc.).',
                'in allowed set': 'Confirms values are from a predefined list of acceptable options. Enforces data consistency.',
                'in range': 'Ensures numeric values fall within specified minimum and maximum bounds.',
                'in date range': 'Validates that dates fall within acceptable time periods.',
                'is boolean': 'Checks that values are valid boolean (True/False) data types.',
                'is numeric': 'Verifies that values are valid numbers (integers or decimals).',
                'is datetime': 'Ensures values are properly formatted date and time data.',
                'length': 'Validates that text values meet minimum and maximum character length requirements.',
                'contains': 'Checks if values contain specific required text or characters.'
            }
            
            # Handle range variations
            if 'range' in rule.lower():
                if 'date' in rule.lower():
                    return descriptions.get('in date range', descriptions.get('in range', 'Validates that values fall within acceptable bounds.'))
                else:
                    return descriptions.get('in range', 'Validates that values fall within acceptable bounds.')
            
            # Try exact match first, then partial matches
            if rule in descriptions:
                return descriptions[rule]
            
            for key, desc in descriptions.items():
                if key in rule.lower() or rule.lower() in key:
                    return desc
            
            return 'Validates data quality according to specified business rules.'

        detailed_results_html = ""
        for rule, items in grouped.items():
            table_rows = ""
            for i, row in items:
                details_id = f"details-{rule}-{i}"
                readable_details = format_details_human_readable(row['details'])
                table_rows += f'''
                <tr>
                    <td><strong>{row['column']}</strong></td>
                    <td><span class="badge {badge_color(row['success_rate'])}">{row['success_rate']:.2f}%</span></td>
                    <td>
                        <button class="toggle-btn" onclick="toggleDetails('{details_id}')">Show Details</button>
                        <div id="{details_id}" class="details hidden">{readable_details}</div>
                    </td>
                </tr>
                '''
            # Collapsible section for this rule
            section_id = f"section-{rule.replace(' ', '-').replace('_', '-')}"
            validation_description = get_validation_description(rule)
            detailed_results_html += f'''
            <div class="collapsible-section">
                <button class="collapsible" onclick="toggleSection(this, '{section_id}')"><span class="arrow">&#9654;</span> {rule_label(rule)}</button>
                <div id="{section_id}" class="content-collapsible hidden">
                    <div class="validation-description">
                        <p><em>{validation_description}</em></p>
                    </div>
                    <table>
                        <tr><th>Column</th><th>Pass %</th><th>Details</th></tr>
                        {table_rows}
                    </table>
                </div>
            </div>
            '''

        # Generate field summary data
        field_summary_html = self._generate_field_summary_html()

        # Get list of columns for the filter
        columns_list = list(self.df.columns)
        # Custom dropdown filter HTML
        field_summary_filter_html = f'''
        <div class="field-summary-filter" style="margin-bottom: 18px;">
            <div style="display: flex; gap: 15px; align-items: center; margin-bottom: 15px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-top: 20px;">
                    <label style="font-weight: 600; color: var(--text-primary);">Filter by section:</label>
                    <select id="sectionTypeFilter" style="padding: 4px 8px; border: none; border-radius: 4px; background: var(--header-bg); color: var(--header-text); font-size: 1em; cursor: pointer; box-shadow: var(--card-shadow); transition: background 0.2s, transform 0.2s;">
                        <option value="all">All Sections</option>
                        <option value="critical">Critical Data Elements Only</option>
                        <option value="other">Other Fields Only</option>
                    </select>
                </div>
                <div style="display: flex; align-items: center; gap: 8px; position: relative; margin-top: 20px;">
                    <label style="font-weight: 600; color: var(--text-primary);">Filter by column:</label>
                    <div style="position: relative;">
                        <button id="fieldSummaryDropdownBtn" class="dropdown-btn" style="padding: 4px 8px;">Filter columns &#x25BC;</button>
                        <div id="fieldSummaryDropdown" class="dropdown-content">
                            <div class="dropdown-actions">
                                <button type="button" id="selectAllBtn">Select All</button>
                                <button type="button" id="clearAllBtn">Clear All</button>
                            </div>
                            <div class="dropdown-checkboxes">
                                {''.join([f'<label><input type="checkbox" class="field-checkbox" value="{col}" checked> {col}</label>' for col in columns_list])}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        '''

        # Generate data table HTML
        data_table_html = self._generate_data_table_html()

        # Get validation rules for filtering
        validation_rules = list(set([result['rule'] for result in self.results]))

        # Calculate comprehensive dataset metrics
        total_rows = len(self.df)
        total_columns = len(self.df.columns)
        total_cells = total_rows * total_columns
        null_cells = self.df.isnull().sum().sum()
        completeness_rate = ((total_cells - null_cells) / total_cells) * 100 if total_cells > 0 else 0
        
        # Calculate data type distribution with user-friendly names
        def get_user_friendly_dtype(dtype_name):
            """Convert pandas dtype to user-friendly name"""
            dtype_str = str(dtype_name).lower()
            if 'object' in dtype_str:
                return "Text/String"
            elif 'int' in dtype_str:
                return "Integer"
            elif 'float' in dtype_str:
                return "Decimal"
            elif 'bool' in dtype_str:
                return "Boolean"
            elif 'datetime' in dtype_str:
                return "Date/Time"
            elif 'category' in dtype_str:
                return "Category"
            else:
                return dtype_str.title()
        
        dtype_counts = self.df.dtypes.value_counts()
        dtype_distribution = {get_user_friendly_dtype(dtype): count for dtype, count in dtype_counts.items()}
        
        # Calculate overall uniqueness score
        uniqueness_scores = []
        for col in self.df.columns:
            if self.df[col].nunique() > 0:
                uniqueness = (self.df[col].nunique() / len(self.df[col])) * 100
                uniqueness_scores.append(uniqueness)
        avg_uniqueness = sum(uniqueness_scores) / len(uniqueness_scores) if uniqueness_scores else 0
        
        # Calculate overall consistency score
        consistency_scores = []
        for col in self.df.columns:
            col_data = self.df[col]
            if pd.api.types.is_object_dtype(col_data):
                sample = col_data.dropna().head(100)
                if len(sample) > 0:
                    type_counts = {}
                    for item in sample:
                        item_type = type(item).__name__
                        type_counts[item_type] = type_counts.get(item_type, 0) + 1
                    most_common_type_count = max(type_counts.values()) if type_counts else 0
                    consistency = (most_common_type_count / len(sample)) * 100
                    consistency_scores.append(consistency)
                else:
                    consistency_scores.append(100)
            else:
                consistency_scores.append(100)
        avg_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 100
        
        # Calculate overall quality score
        overall_score = (completeness_rate + avg_uniqueness + avg_consistency) / 3
        
        # Enhanced summary cards data
        summary_cards = [
            {"title": "Total Records", "value": f"{total_rows:,}", "icon": "📊", "color": "#2d3e50"},
            {"title": "Total Columns", "value": f"{total_columns:,}", "icon": "📋", "color": "#34495e"},
            {"title": "Data Completeness", "value": f"{completeness_rate:.1f}%", "icon": "✅", "color": "#27ae60"},
            {"title": "Avg Uniqueness", "value": f"{avg_uniqueness:.1f}%", "icon": "🔍", "color": "#3498db"},
            {"title": "Total Rules", "value": f"{total_checks:,}", "icon": "⚡", "color": "#e74c3c"},
            {"title": "Overall Health", "value": f"{avg_pass_rate:.1f}%", "icon": "🏥", "color": "#f39c12"}
        ]
        
        # Enhanced health distribution with better colors
        health_distribution = [
            round((categories["healthy"] / total_checks) * 100, 2) if total_checks > 0 else 0,
            round((categories["degraded"] / total_checks) * 100, 2) if total_checks > 0 else 0,
            round((categories["critical"] / total_checks) * 100, 2) if total_checks > 0 else 0
        ]
        
        # Enhanced bar chart data with better styling
        bar_chart_data = {
            'labels': df_results['column'].tolist(),
            'datasets': [{
                'label': 'Success Rate (%)',
                'data': df_results['success_rate'].tolist(),
                'backgroundColor': [self._get_enhanced_bar_color(rate) for rate in df_results['success_rate']],
                'borderColor': [self._get_enhanced_border_color(rate) for rate in df_results['success_rate']],
                'borderWidth': 2,
                'borderRadius': 6,
                'borderSkipped': False,
            }]
        }
        
        # Generate enhanced summary cards HTML
        summary_cards_html = ""
        for card in summary_cards:
            summary_cards_html += f"""
            <div class="summary-card" style="background: linear-gradient(135deg, {card['color']}, {self._darken_color(card['color'])})">
                <div class="card-icon">{card['icon']}</div>
                <div class="card-content">
                    <h2>{card['value']}</h2>
                    <p>{card['title']}</p>
                </div>
            </div>
            """
        
        # Generate dataset summary section
        dataset_summary_html = f"""
        <div class="dataset-summary">
            <h3>📊 Overall Data Quality</h3>
            <div class="quality-overview-grid">
                <div class="quality-metric">
                    <div class="metric-icon">✅</div>
                    <div class="metric-content">
                        <div class="metric-label">Completeness</div>
                        <div class="metric-value">{completeness_rate:.1f}%</div>
                    </div>
                </div>
                <div class="quality-metric">
                    <div class="metric-icon">🔍</div>
                    <div class="metric-content">
                        <div class="metric-label">Uniqueness</div>
                        <div class="metric-value">{avg_uniqueness:.1f}%</div>
                    </div>
                </div>
                <div class="quality-metric">
                    <div class="metric-icon">📋</div>
                    <div class="metric-content">
                        <div class="metric-label">Consistency</div>
                        <div class="metric-value">{avg_consistency:.1f}%</div>
                    </div>
                </div>
                <div class="quality-metric overall-score">
                    <div class="metric-icon">🏆</div>
                    <div class="metric-content">
                        <div class="metric-label">Overall Score</div>
                        <div class="metric-value">{overall_score:.1f}%</div>
                    </div>
                </div>
            </div>
        </div>
        """
        
        # Generate data type distribution
        dtype_chart_data = {
            'labels': list(dtype_distribution.keys()),
            'datasets': [{
                'data': list(dtype_distribution.values()),
                'backgroundColor': ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'],
                'borderWidth': 2,
                'borderColor': '#fff'
            }]
        }

        html_template = f"""
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{title} - {dataset_name}</title>
            <style>
                :root {{
                    /* Light theme */
                    --bg-primary: #f7f8fa;
                    --bg-secondary: #fff;
                    --bg-tertiary: #f4f6f8;
                    --text-primary: #2d3e50;
                    --text-secondary: #444;
                    --text-muted: #666;
                    --border-color: #e1e5e9;
                    --header-bg: #2d3e50;
                    --header-text: #fff;
                    --tab-bg: #eee;
                    --tab-active-bg: #fff;
                    --tab-border: #ccc;
                    --card-shadow: 0 2px 6px rgba(0,0,0,0.1);
                    --pattern-bg: #f8f9fa;
                    --pattern-text: #495057;
                    --pattern-border: #dee2e6;
                    --range-bg: #e3f2fd;
                    --range-text: #1976d2;
                    --range-border: #bbdefb;
                }}

                [data-theme="dark"] {{
                    /* Dark theme */
                    --bg-primary: #1a1a1a;
                    --bg-secondary: #2d2d2d;
                    --bg-tertiary: #333333;
                    --text-primary: #ffffff;
                    --text-secondary: #cccccc;
                    --text-muted: #999999;
                    --border-color: #404040;
                    --header-bg: #1a1a1a;
                    --header-text: #ffffff;
                    --tab-bg: #333333;
                    --tab-active-bg: #2d2d2d;
                    --tab-border: #404040;
                    --card-shadow: 0 2px 6px rgba(0,0,0,0.3);
                    --pattern-bg: #404040;
                    --pattern-text: #e0e0e0;
                    --pattern-border: #555555;
                    --range-bg: #2d3748;
                    --range-text: #90cdf4;
                    --range-border: #4a5568;
                }}

                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: var(--bg-primary);
                    color: var(--text-primary);
                    transition: background-color 0.3s, color 0.3s;
                }}
                header {{
                    background-color: var(--header-bg);
                    color: var(--header-text);
                    padding: 20px;
                    text-align: center;
                    position: relative;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    border-bottom: 1px solid var(--border-color);
                }}
                .theme-toggle {{
                    background: none;
                    border: none;
                    color: var(--text-primary);
                    font-size: 1.1em;
                    cursor: pointer;
                    padding: 6px 10px;
                    border-radius: 4px;
                    transition: color 0.3s;
                    margin-left: auto;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    opacity: 0.8;
                    margin-top: -16px;
                }}
                .theme-toggle:hover {{
                    opacity: 1;
                }}
                .theme-toggle .sun-icon {{
                    display: none;
                }}
                .theme-toggle .moon-icon {{
                    display: inline;
                }}
                [data-theme="dark"] .theme-toggle .sun-icon {{
                    display: inline;
                }}
                [data-theme="dark"] .theme-toggle .moon-icon {{
                    display: none;
                }}
                h1 {{
                    margin: 0;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 20px auto;
                    padding: 20px;
                    background: var(--bg-secondary);
                    border-radius: 8px;
                    box-shadow: var(--card-shadow);
                    transition: background-color 0.3s, box-shadow 0.3s;
                }}
                .tabs {{
                    display: flex;
                    border-bottom: 2px solid #ccc;
                    align-items: center;
                }}
                .tab {{
                    padding: 10px 20px;
                    cursor: pointer;
                    border: 1px solid var(--tab-border);
                    border-bottom: none;
                    background: var(--tab-bg);
                    margin-right: 5px;
                    border-radius: 6px 6px 0 0;
                    color: var(--text-primary);
                    transition: background-color 0.3s, color 0.3s;
                }}
                .tab.active {{
                    background: var(--tab-active-bg);
                    border-bottom: 2px solid var(--tab-active-bg);
                    font-weight: bold;
                }}
                .tab-content {{
                    display: none;
                }}
                .tab-content.active {{
                    display: block;
                }}
                .summary {{
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 20px;
                    margin-top:20px;
                    color: white;
                }}
                .card {{
                    background: #2d3e50;
                    border-radius: 8px;
                    padding: 15px;
                    text-align: center;
                    flex: 1;
                    margin: 0 5px;
                }}
                .charts-container {{
                    display: flex;
                    justify-content: space-between;
                    margin: 20px 0;
                }}
                .chart {{
                    flex: 1;
                    margin: 0 10px;
                }}
                canvas {{
                    max-width: 100%;
                    margin: 20px auto;
                    display: block;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 10px;
                }}
                th, td {{
                    padding: 10px;
                    text-align: left;
                    border-bottom: 1px solid var(--border-color);
                    color: var(--text-primary);
                    transition: border-color 0.3s, background-color 0.3s, color 0.3s;
                }}
                th {{
                    background-color: var(--header-bg);
                    color: var(--header-text);
                    cursor: pointer;
                    transition: background-color 0.3s, color 0.3s;
                }}
                th:hover {{
                    background-color: #34495e;
                }}
                [data-theme="dark"] th:hover {{
                    background-color: #404040;
                }}
                tr:nth-child(even) {{
                    background-color: var(--bg-tertiary);
                    transition: background-color 0.3s, color 0.3s;
                }}
                .badge {{
                    padding: 4px 8px;
                    border-radius: 4px;
                    color: #fff;
                    font-weight: bold;
                }}
                .badge-green {{ background-color: #28a745; }}
                .badge-yellow {{ background-color: #ffc107; color: #333; }}
                .badge-red {{ background-color: #dc3545; }}
                .toggle-btn {{
                    background: var(--header-bg);
                    border: none;
                    color: var(--header-text);
                    padding: 5px 10px;
                    cursor: pointer;
                    border-radius: 4px;
                    font-size: 0.85em;
                    transition: background-color 0.3s;
                }}
                .toggle-btn:hover {{ background: #34495e; }}
                .details {{
                    margin-top: 5px;
                    background: var(--bg-tertiary);
                    padding: 6px 10px;
                    border-radius: 4px;
                    font-size: 0.9em;
                    line-height: 1.4;
                    border: 1px solid var(--border-color);
                    transition: background-color 0.3s, border-color 0.3s;
                }}
                .hidden {{ display: none; }}
                .status-text {{
                    font-weight: bold;
                    margin-top: 4px;
                    font-size: 1em;
                    color: #fff; 
                }}
                .field-summary {{
                    margin: 20px 0;
                }}
                .field-section {{
                    margin: 0px 0px 20px 0px;
                    padding: 20px;
                    border: 1px solid var(--border-color);
                    border-radius: 8px;
                    background: var(--bg-secondary);
                    box-shadow: var(--card-shadow);
                    transition: box-shadow 0.2s, border-color 0.2s, background-color 0.3s;
                }}
                .field-section:hover {{
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    border-color: var(--border-color);
                }}
                .field-section h3 {{
                    margin-top: 0;
                    margin-bottom: 20px;
                    color: var(--text-primary);
                    font-size: 1.3em;
                    border-bottom: 2px solid var(--border-color);
                    padding-bottom: 8px;
                    transition: color 0.3s, border-color 0.3s;
                }}
                .field-section p {{
                    margin: 8px 0;
                    line-height: 1.5;
                    color: #4a5568;
                }}
                .field-section strong {{
                    color: #2d3e50;
                    font-weight: 600;
                }}
                .data-table-controls {{
                    margin: 20px 0;
                    padding: 15px;
                    background: var(--header-bg);
                    border-radius: 8px;
                    transition: background-color 0.3s;
                }}
                .filter-row {{
                    display: flex;
                    align-items: center;
                    margin-bottom: 10px;
                }}
                .filter-row:last-child {{
                    margin-bottom: 0;
                }}
                .data-table-controls label {{
                    font-weight: 600;
                    color: var(--header-text);    
                    margin-right: 8px;
                    min-width: 140px;
                    transition: color 0.3s;
                }}
                .data-table-controls select, .data-table-controls input {{
                    margin: 0 10px;
                    padding: 6px 10px;
                    border: 1px solid var(--border-color);
                    border-radius: 4px;
                    font-size: 1em;
                    background: var(--bg-secondary);
                    color: var(--text-primary);
                    transition: background-color 0.3s, border-color 0.3s, color 0.3s;
                    min-width: 150px;
                }}
                .data-table-controls select:focus, .data-table-controls input:focus {{
                    outline: none;
                    border-color: var(--text-primary);
                    box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.1);
                }}
                .data-table-controls select {{
                    cursor: pointer;
                    min-width: 120px;
                }}
                .data-table-controls input {{
                    min-width: 200px;
                }}
                .sortable {{
                    cursor: pointer;
                }}
                .sortable::after {{
                    content: ' ↕';
                    font-size: 0.8em;
                }}
                .sortable.asc::after {{
                    content: ' ↑';
                }}
                .sortable.desc::after {{
                    content: ' ↓';
                }}
                .collapsible-section {{ margin-bottom: 18px; }}
                .collapsible {{
                    background-color: var(--header-bg);
                    color: var(--header-text);
                    cursor: pointer;
                    padding: 10px 18px;
                    width: 100%;
                    border: none;
                    text-align: left;
                    outline: none;
                    font-size: 1.1em;
                    border-radius: 6px;
                    margin-bottom: 4px;
                    box-shadow: var(--card-shadow);
                    display: flex;
                    align-items: center;
                    transition: background 0.2s, box-shadow 0.2s, transform 0.2s;
                    position: relative;
                }}
                .collapsible .arrow {{
                    display: inline-block;
                    margin-right: 12px;
                    font-size: 1.2em;
                    transition: transform 0.25s cubic-bezier(0.4,0,0.2,1);
                }}
                .collapsible.active .arrow {{
                    transform: rotate(90deg);
                }}
                .collapsible:hover {{
                    background-color: #34495e;
                    box-shadow: 0 4px 12px rgba(44,62,80,0.13);
                    transform: translateY(-1px);
                }}
                [data-theme="dark"] .collapsible:hover {{
                    background-color: #404040;
                }}
                .content-collapsible {{
                    padding: 0 24px 18px 24px;
                    display: none;
                    overflow: hidden;
                    background-color: var(--bg-tertiary);
                    border-radius: 0 0 8px 8px;
                    transition: max-height 0.35s cubic-bezier(0.4,0,0.2,1), opacity 0.25s, background-color 0.3s;
                    max-height: 0;
                    opacity: 0;
                }}
                .content-collapsible.active {{
                    display: block;
                    max-height: 2000px;
                    opacity: 1;
                }}
                .validation-description {{
                    margin: 15px 0 20px 0;
                    padding: 12px 16px;
                    background-color: var(--bg-secondary);
                    border-left: 4px solid var(--border-color);
                    border-radius: 4px;
                    transition: background-color 0.3s, border-color 0.3s;
                }}
                .validation-description p {{
                    margin: 0;
                    color: var(--text-secondary);
                    font-size: 0.95em;
                    line-height: 1.4;
                    transition: color 0.3s;
                }}
                .validation-description em {{
                    font-style: italic;
                    color: var(--text-primary);
                    transition: color 0.3s;
                }}
                .passed-green {{
                    color: #28a745;
                    font-weight: bold;
                }}
                .failed-red {{
                    color: #dc3545;
                    font-weight: bold;
                }}
                .allowed-set {{
                    color: #007bff;
                }}
                .pattern-highlight {{
                    background-color: var(--pattern-bg);
                    color: var(--pattern-text);
                    border-radius: 3px;
                    padding: 1px 4px;
                    font-style: italic;
                    border: 1px solid var(--pattern-border);
                    transition: background-color 0.3s, color 0.3s, border-color 0.3s;
                }}
                .range-highlight {{
                    background-color: var(--range-bg);
                    color: var(--range-text);
                    border-radius: 3px;
                    padding: 1px 4px;
                    font-style: italic;
                    border: 1px solid var(--range-border);
                    transition: background-color 0.3s, color 0.3s, border-color 0.3s;
                }}
                .details b {{
                    font-weight: 600;
                }}
                .field-summary-filter {{
                    margin-bottom: 18px;
                }}
                .field-summary-filter label {{
                    font-size: 1.05em;
                    color: var(--text-primary);
                    transition: color 0.3s;
                }}
                .field-summary-filter select {{
                    font-size: 1em;
                    border-radius: 4px;
                    border: none;
                    padding: 4px 8px;
                    background: var(--header-bg);
                    color: var(--header-text);
                    transition: background 0.2s, transform 0.2s;
                    cursor: pointer;
                    box-shadow: var(--card-shadow);
                }}
                .field-summary-filter select:hover {{
                    background: #34495e;
                    transform: translateY(-1px);
                }}
                [data-theme="dark"] .field-summary-filter select:hover {{
                    background: #404040;
                }}
                .field-summary-filter select:focus {{
                    outline: none;
                    box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.1);
                }}
                .field-summary-filter {{
                    margin-bottom: 18px;
                    position: relative;
                    display: inline-block;
                }}
                .dropdown-btn {{
                    background: var(--header-bg);
                    color: var(--header-text);
                    border: none;
                    border-radius: 4px;
                    padding: 8px 18px;
                    font-size: 1em;
                    cursor: pointer;
                    box-shadow: var(--card-shadow);
                    transition: background 0.2s, transform 0.2s;
                }}
                .dropdown-btn:hover {{ 
                    background: #34495e;
                    transform: translateY(-1px);
                }}
                [data-theme="dark"] .dropdown-btn:hover {{ 
                    background: #404040;
                }}
                .dropdown-content {{
                    display: none;
                    position: absolute;
                    left: 0;
                    top: 110%;
                    background: var(--bg-secondary);
                    min-width: 220px;
                    max-height: 320px;
                    overflow-y: auto;
                    border: 1px solid var(--border-color);
                    border-radius: 6px;
                    box-shadow: var(--card-shadow);
                    z-index: 100;
                    padding: 10px 0 10px 0;
                    transition: background-color 0.3s, border-color 0.3s;
                }}
                .dropdown-content.show {{ display: block; }}
                .dropdown-checkboxes label {{
                    display: block;
                    padding: 4px 18px;
                    font-size: 1em;
                    cursor: pointer;
                    user-select: none;
                    color: var(--text-primary);
                    transition: color 0.3s;
                }}
                .dropdown-checkboxes input[type="checkbox"] {{
                    margin-right: 8px;
                }}
                .dropdown-actions {{
                    display: flex;
                    justify-content: space-between;
                    padding: 0 18px 8px 18px;
                    border-bottom: 1px solid var(--border-color);
                    margin-bottom: 6px;
                    transition: border-color 0.3s;
                }}
                .dropdown-actions button {{
                    background: var(--bg-tertiary);
                    border: 1px solid var(--border-color);
                    border-radius: 3px;
                    padding: 3px 10px;
                    font-size: 0.97em;
                    cursor: pointer;
                    transition: background 0.2s, border-color 0.3s;
                    color: var(--text-primary);
                }}
                .dropdown-actions button:hover {{ background: var(--bg-primary); }}
                .data-table-container {{
                    overflow-x: auto;
                    max-width: 100%;
                    border-radius: 8px;
                    box-shadow: var(--card-shadow);
                    transition: box-shadow 0.3s;
                }}
                .data-table-container table {{
                    min-width: 100%;
                    white-space: nowrap;
                }}
                .data-table-container th, .data-table-container td {{
                    padding: 10px 12px;
                    text-align: left;
                    border-bottom: 1px solid var(--border-color);
                    min-width: 120px;
                    color: var(--text-primary);
                    transition: border-color 0.3s, background-color 0.3s, color 0.3s;
                }}
                .data-table-container th {{
                    background-color: var(--header-bg);
                    color: var(--header-text);
                    cursor: pointer;
                    position: sticky;
                    top: 0;
                    z-index: 10;
                    transition: background-color 0.3s, color 0.3s;
                }}
                .data-table-container th:hover {{
                    background-color: #34495e;
                }}
                [data-theme="dark"] .data-table-container th:hover {{
                    background-color: #404040;
                }}
                .data-table-container tr:nth-child(even) {{
                    background-color: var(--bg-tertiary);
                    transition: background-color 0.3s, color 0.3s;
                }}
                .data-table-container tr:hover {{
                    background-color: var(--bg-primary);
                    transition: background-color 0.3s, color 0.3s;
                }}
                .field-stats {{
                    display: flex;
                    flex-direction: column;
                    gap: 12px;
                    margin-bottom: 20px;
                    padding: 18px;
                    background: var(--bg-tertiary);
                    border-radius: 8px;
                    border: 1px solid var(--border-color);
                    transition: background-color 0.3s, border-color 0.3s;
                }}
                .field-stats-header {{
                    margin-bottom: 15px;
                    color: var(--text-primary);
                    font-size: 1.1em;
                    font-weight: 600;
                    border-bottom: 2px solid var(--border-color);
                    padding-bottom: 8px;
                    transition: color 0.3s, border-color 0.3s;
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(5, 1fr);
                    gap: 12px;
                }}
                .stat-row {{
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    padding: 12px 16px;
                    background: var(--bg-secondary);
                    border-radius: 6px;
                    border: 1px solid var(--border-color);
                    text-align: center;
                    min-height: 80px;
                    transition: background-color 0.3s, border-color 0.3s;
                }}
                .stat-row.data-type-row {{
                    background: var(--bg-secondary);
                    border: 1px solid var(--border-color);
                }}
                .stat-label {{
                    font-weight: 500;
                    color: var(--text-secondary);
                    font-size: 0.9em;
                    margin-bottom: 8px;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    transition: color 0.3s;
                }}
                .stat-value {{
                    font-weight: bold;
                    color: var(--text-primary);
                    font-size: 1.6em;
                    text-align: center;
                    word-break: break-word;
                    transition: color 0.3s;
                }}
                .data-type-value {{
                    font-size: 1.6em;
                    font-weight: bold;
                    color: var(--text-primary);
                    text-align: center;
                    transition: color 0.3s;
                }}
                .stat-row.sample-values {{
                    flex-direction: column;
                    align-items: center;
                }}
                .stat-row.sample-values .stat-value {{
                    text-align: center;
                    font-weight: normal;
                    font-size: 0.9em;
                    line-height: 1.4;
                    background: var(--bg-tertiary);
                    padding: 6px 8px;
                    border-radius: 4px;
                    border: 1px solid var(--border-color);
                    width: 100%;
                    box-sizing: border-box;
                    margin-top: 4px;
                    word-wrap: break-word;
                    overflow-wrap: break-word;
                    transition: background-color 0.3s, border-color 0.3s;
                }}
                .quality-scores {{
                    margin-top: 20px;
                    padding: 18px;
                    background: var(--bg-tertiary);
                    border-radius: 8px;
                    border: 1px solid var(--border-color);
                    transition: background-color 0.3s, border-color 0.3s;
                }}
                .quality-scores h4 {{
                    margin-top: 0;
                    margin-bottom: 15px;
                    color: var(--text-primary);
                    font-size: 1.1em;
                    font-weight: 600;
                    border-bottom: 2px solid var(--border-color);
                    padding-bottom: 8px;
                    transition: color 0.3s, border-color 0.3s;
                }}
                .score-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                    gap: 12px;
                }}
                .score-item {{
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    padding: 10px 15px;
                    background: var(--bg-secondary);
                    border-radius: 6px;
                    border: 1px solid var(--border-color);
                    text-align: center;
                    min-height: 80px;
                    transition: background-color 0.3s, border-color 0.3s;
                }}
                .score-label {{
                    font-weight: 500;
                    color: var(--text-secondary);
                    font-size: 0.85em;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    margin-bottom: 8px;
                    transition: color 0.3s;
                }}
                .score-value {{
                    font-weight: bold;
                    font-size: 1.3em;
                    padding: 4px 8px;
                    border-radius: 4px;
                    min-width: 60px;
                    text-align: center;
                }}
                .score-value.score-excellent {{ 
                    color: #28a745; 
                    background: rgba(40, 167, 69, 0.1);
                }}
                .score-value.score-good {{ 
                    color: #28a745; 
                    background: rgba(40, 167, 69, 0.1);
                }}
                .score-value.score-warning {{ 
                    color: #ffc107; 
                    background: rgba(255, 193, 7, 0.1);
                }}
                .score-value.score-poor {{ 
                    color: #dc3545; 
                    background: rgba(220, 53, 69, 0.1);
                }}
                .summary-card {{
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    text-align: center;
                    padding: 15px 20px;
                    border-radius: 10px;
                    color: white;
                    margin-bottom: 15px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                    transition: transform 0.2s, box-shadow 0.2s;
                }}
                .summary-card:hover {{
                    transform: translateY(-3px);
                    box-shadow: 0 6px 16px rgba(0,0,0,0.15);
                }}
                .card-icon {{
                    font-size: 2.5em;
                    margin-bottom: 10px;
                    flex-shrink: 0;
                }}
                .card-content h2 {{
                    margin: 0 0 5px 0;
                    font-size: 1.8em;
                    font-weight: bold;
                    text-align: center;
                }}
                .card-content p {{
                    margin: 0;
                    font-size: 0.9em;
                    opacity: 0.8;
                    text-align: center;
                }}
                
                /* Dashboard Layout */
                .dashboard-section {{
                    margin-bottom: 30px;
                    padding: 25px;
                    background: var(--bg-secondary);
                    border-radius: 12px;
                    box-shadow: var(--card-shadow);
                    border: 1px solid var(--border-color);
                    transition: background-color 0.3s, box-shadow 0.3s, border-color 0.3s;
                }}
                .section-header {{
                    margin-top: 0;
                    margin-bottom: 20px;
                    color: var(--text-primary);
                    font-size: 1.4em;
                    font-weight: 600;
                    border-bottom: 2px solid var(--border-color);
                    padding-bottom: 10px;
                    transition: color 0.3s, border-color 0.3s;
                }}
                .summary-cards-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                    gap: 20px;
                }}
                .charts-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                    gap: 25px;
                }}
                .chart-container {{
                    background: var(--bg-tertiary);
                    border-radius: 8px;
                    padding: 20px;
                    border: 1px solid var(--border-color);
                    height: 400px;
                    display: flex;
                    flex-direction: column;
                    transition: background-color 0.3s, border-color 0.3s;
                }}
                .chart-container h4 {{
                    margin-top: 0;
                    margin-bottom: 15px;
                    color: var(--text-primary);
                    font-size: 1.1em;
                    font-weight: 600;
                    text-align: center;
                    flex-shrink: 0;
                    transition: color 0.3s;
                }}
                .chart-container canvas {{
                    flex: 1;
                    max-width: 100%;
                    height: 100% !important;
                }}
                
                /* Dataset Summary Styling */
                .dataset-summary {{
                    margin-top: 0;
                }}
                .dataset-summary h3 {{
                    margin-top: 0;
                    margin-bottom: 20px;
                    color: var(--text-primary);
                    font-size: 1.3em;
                    border-bottom: 2px solid var(--border-color);
                    padding-bottom: 8px;
                    transition: color 0.3s, border-color 0.3s;
                }}
                .quality-overview-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                }}
                .quality-metric {{
                    background: var(--bg-tertiary);
                    border-radius: 10px;
                    padding: 20px;
                    border: 1px solid var(--border-color);
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    text-align: center;
                    transition: transform 0.2s, box-shadow 0.2s, background-color 0.3s, border-color 0.3s;
                }}
                .quality-metric:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                }}
                .quality-metric.overall-score {{
                    background: linear-gradient(135deg, #4a5568, #2d3748);
                    color: white;
                    border: none;
                }}
                .quality-metric.overall-score .metric-label,
                .quality-metric.overall-score .metric-value {{
                    color: white;
                }}
                .metric-icon {{
                    font-size: 2em;
                    margin-bottom: 10px;
                    flex-shrink: 0;
                }}
                .metric-content {{
                    flex: 1;
                    text-align: center;
                }}
                .metric-label {{
                    font-size: 0.9em;
                    color: var(--text-muted);
                    margin-bottom: 5px;
                    font-weight: 500;
                    transition: color 0.3s;
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 1.8em;
                    font-weight: bold;
                    color: var(--text-primary);
                    transition: color 0.3s;
                    text-align: center;
                }}
                
                /* Responsive Design */
                @media (max-width: 768px) {{
                    .summary-cards-grid {{
                        grid-template-columns: 1fr;
                    }}
                    .charts-grid {{
                        grid-template-columns: 1fr;
                    }}
                    .dashboard-section {{
                        padding: 15px;
                        margin-bottom: 20px;
                    }}
                }}
            </style>
        </head>
        <body>
            <header>
                <h1>{title} - {dataset_name}</h1>
            </header>
            <div class="container">
                <div class="tabs">
                    <div class="tab active" onclick="openTab(event, 'overview')">Overview</div>
                    <div class="tab" onclick="openTab(event, 'details')">Validations</div>
                    <div class="tab" onclick="openTab(event, 'field-summary')">Field Summary</div>
                    <div class="tab" onclick="openTab(event, 'data-table')">Data Table</div>
                    <button class="theme-toggle" onclick="toggleTheme()" title="Toggle dark mode">
                        <span class="sun-icon">☀</span>
                        <span class="moon-icon">☾</span>
                    </button>
                </div>
                <div id="overview" class="tab-content active">
                    <!-- Dataset Summary Section -->
                    <div class="dashboard-section" style="margin-top: 20px;">
                        {dataset_summary_html}
                    </div>
                    
                    <!-- Charts Section -->
                    <div class="dashboard-section">
                        <h3 class="section-header">📈 Data Quality Charts</h3>
                        <div class="charts-grid">
                            <div class="chart-container">
                                <h4>Health Distribution</h4>
                                <canvas id="passFailChart"></canvas>
                            </div>
                            <div class="chart-container">
                                <h4>Success Rate by Column</h4>
                                <canvas id="barChart"></canvas>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Data Type Distribution -->
                    <div class="dashboard-section">
                        <h3 class="section-header">🔍 Data Type Analysis</h3>
                        <div class="chart-container">
                            <h4>Column Types Distribution</h4>
                            <canvas id="dtypeChart"></canvas>
                        </div>
                    </div>
                    
                    <!-- Summary Cards Section -->
                    <div class="dashboard-section">
                        <h3 class="section-header">📊 Key Metrics</h3>
                        <div class="summary-cards-grid">
                            {summary_cards_html}
                        </div>
                    </div>
                </div>
                <div id="details" class="tab-content" style="margin-top: 20px;">
                    {detailed_results_html}
                </div>
                <div id="field-summary" class="tab-content">
                    {field_summary_filter_html}
                    {field_summary_html}
                </div>
                <div id="data-table" class="tab-content">
                    {data_table_html}
                </div>
            </div>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script>
                // Pie Chart
                const ctx = document.getElementById('passFailChart').getContext('2d');
                new Chart(ctx, {{
                    type: 'pie',
                    data: {{
                        labels: ['Healthy', 'Degraded', 'Critical'],
                        datasets: [{{
                            data: {health_distribution},
                            backgroundColor: ['#28a745', '#ffc107', '#dc3545']
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{
                                position: 'bottom'
                            }},
                            title: {{
                                display: true,
                                text: 'Health Distribution'
                            }}
                        }}
                    }}
                }});

                // Bar Chart
                const barCtx = document.getElementById('barChart').getContext('2d');
                new Chart(barCtx, {{
                    type: 'bar',
                    data: {json.dumps(bar_chart_data)},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {{
                            y: {{
                                beginAtZero: true,
                                max: 100,
                                grid: {{
                                    color: 'rgba(0,0,0,0.1)'
                                }}
                            }},
                            x: {{
                                grid: {{
                                    display: false
                                }},
                                ticks: {{
                                    maxRotation: 45,
                                    minRotation: 45,
                                    font: {{
                                        size: 10
                                    }}
                                }}
                            }}
                        }},
                        plugins: {{
                            title: {{
                                display: true,
                                text: 'Success Rate by Column',
                                font: {{
                                    size: 16,
                                    weight: 'bold'
                                }}
                            }},
                            legend: {{
                                display: false
                            }}
                        }},
                        layout: {{
                            padding: {{
                                top: 20,
                                bottom: 20
                            }}
                        }},
                        elements: {{
                            bar: {{
                                borderWidth: 1,
                                borderColor: 'rgba(255,255,255,0.8)'
                            }}
                        }}
                    }}
                }});

                // Data type distribution chart
                const dtypeCtx = document.getElementById('dtypeChart').getContext('2d');
                const dtypeData = {json.dumps(dtype_chart_data)};
                
                // Calculate dynamic max value with safety checks
                let suggestedMax = 10; // Default fallback
                if (dtypeData.datasets && dtypeData.datasets[0] && dtypeData.datasets[0].data && dtypeData.datasets[0].data.length > 0) {{
                    const maxValue = Math.max(...dtypeData.datasets[0].data);
                    suggestedMax = Math.max(Math.ceil(maxValue * 1.1), 1); // Ensure at least 1
                }}
                
                new Chart(dtypeCtx, {{
                    type: 'bar',
                    data: dtypeData,
                    options: {{
                        responsive: true,
                        scales: {{
                            y: {{
                                beginAtZero: true,
                                suggestedMax: suggestedMax
                            }}
                        }},
                        plugins: {{
                            title: {{
                                display: true,
                                text: 'Data Type Distribution'
                            }}
                        }}
                    }}
                }});

                function openTab(evt, tabName) {{
                    const tabs = document.querySelectorAll('.tab');
                    const contents = document.querySelectorAll('.tab-content');
                    tabs.forEach(t => t.classList.remove('active'));
                    contents.forEach(c => c.classList.remove('active'));
                    evt.currentTarget.classList.add('active');
                    document.getElementById(tabName).classList.add('active');
                }}

                function toggleDetails(id) {{
                    const el = document.getElementById(id);
                    el.classList.toggle('hidden');
                }}

                function toggleSection(btn, id) {{
                    var section = document.getElementById(id);
                    if (section.classList.contains('active')) {{
                        section.classList.remove('active');
                        section.classList.add('hidden');
                        btn.classList.remove('active');
                    }} else {{
                        section.classList.add('active');
                        section.classList.remove('hidden');
                        btn.classList.add('active');
                    }}
                }}

                // Data table sorting functionality
                function sortTable(table, column, type) {{
                    const tbody = table.querySelector('tbody');
                    const rows = Array.from(tbody.querySelectorAll('tr'));
                    const header = table.querySelector('th:nth-child(' + (column + 1) + ')');

                    // Remove existing sort classes
                    table.querySelectorAll('th').forEach(th => {{
                        th.classList.remove('asc', 'desc');
                    }});

                    // Add sort class
                    if (header.classList.contains('asc')) {{
                        header.classList.remove('asc');
                        header.classList.add('desc');
                    }} else {{
                        header.classList.remove('desc');
                        header.classList.add('asc');
                    }}

                    rows.sort((a, b) => {{
                        let aVal = a.cells[column].textContent.trim();
                        let bVal = b.cells[column].textContent.trim();

                        if (type === 'number') {{
                            aVal = parseFloat(aVal) || 0;
                            bVal = parseFloat(bVal) || 0;
                        }}

                        if (header.classList.contains('asc')) {{
                            return aVal > bVal ? 1 : -1;
                        }} else {{
                            return aVal < bVal ? 1 : -1;
                        }}
                    }});

                    rows.forEach(row => tbody.appendChild(row));
                }}

                // Filter functionality
                function filterTable() {{
                    const filterValue = document.getElementById('filterInput').value.toLowerCase();
                    const filterColumn = document.getElementById('filterColumn').value;
                    const filterRule = document.getElementById('filterRule').value;
                    const filterStatus = document.getElementById('filterStatus').value;
                    const table = document.getElementById('dataTable');
                    const rows = table.querySelectorAll('tbody tr');

                    // Reset all column visibility first
                    const allHeaders = table.querySelectorAll('thead tr th');
                    allHeaders.forEach(header => {{
                        header.style.display = '';
                    }});
                    rows.forEach(row => {{
                        const allCells = row.querySelectorAll('td');
                        allCells.forEach(cell => {{
                            cell.style.display = '';
                        }});
                    }});

                    rows.forEach(row => {{
                        let showRow = true;
                        
                        // Text filter
                        if (filterValue) {{
                        if (filterColumn === "0") {{
                            // All columns: check every cell
                            const match = Array.from(row.cells).some(cell =>
                                cell.textContent.toLowerCase().includes(filterValue)
                            );
                                showRow = showRow && match;
                        }} else {{
                                // Specific column (adjust for 1-based indexing)
                                const columnIndex = parseInt(filterColumn) - 1;
                                if (columnIndex >= 0 && columnIndex < row.cells.length) {{
                                    const cell = row.cells[columnIndex];
                            const text = cell.textContent.toLowerCase();
                                    showRow = showRow && text.includes(filterValue);
                                }}
                            }}
                        }}
                        
                        // Column and validation filters
                        if (filterColumn !== "0" || filterRule || filterStatus) {{
                            const selectedColumnName = document.getElementById('filterColumn').options[document.getElementById('filterColumn').selectedIndex].text;
                            
                            // Find validation columns for the selected column and rule
                            let validationColumnIndex = -1;
                            if (filterColumn !== "0" && filterRule && filterRule !== "") {{
                                // Look for the specific validation column
                                const validationColumnName = selectedColumnName + '_' + filterRule;
                                for (let i = 0; i < row.cells.length; i++) {{
                                    const headerCell = table.querySelector('thead tr th:nth-child(' + (i + 1) + ')');
                                    if (headerCell && headerCell.textContent.toLowerCase().includes(validationColumnName.toLowerCase().replace('_', ' '))) {{
                                        validationColumnIndex = i;
                                        break;
                                    }}
                                }}
                            }}
                            
                            // Column filter - show/hide data columns based on selection
                            if (filterColumn !== "0") {{
                                // Hide all data columns except the selected one
                                for (let i = 0; i < dataColumnCount; i++) {{
                                    const headerCell = table.querySelector('thead tr th:nth-child(' + (i + 1) + ')');
                                    const dataCell = row.cells[i];
                                    if (headerCell && dataCell) {{
                                        // Compare column names (preserve underscores, convert to lowercase)
                                        const headerText = headerCell.textContent.toLowerCase().trim();
                                        const selectedText = selectedColumnName.toLowerCase().trim();
                                        
                                        if (headerText === selectedText) {{
                                            // Show the selected column
                                            headerCell.style.display = '';
                                            dataCell.style.display = '';
                                        }} else {{
                                            // Hide other data columns
                                            headerCell.style.display = 'none';
                                            dataCell.style.display = 'none';
                                        }}
                                    }}
                                }}
                                
                                // Show validation columns for the selected column
                                for (let i = dataColumnCount; i < row.cells.length; i++) {{
                                    const headerCell = table.querySelector('thead tr th:nth-child(' + (i + 1) + ')');
                                    const validationCell = row.cells[i];
                                    if (headerCell && validationCell) {{
                                        // Compare column names for validation columns (preserve underscores)
                                        const headerText = headerCell.textContent.toLowerCase().trim();
                                        const selectedText = selectedColumnName.toLowerCase().trim();
                                        
                                        if (headerText.includes(selectedText)) {{
                                            // Check if we should show this specific validation column based on rule filter
                                            if (filterRule && filterRule !== "") {{
                                                // Only show the specific validation rule column
                                                const expectedValidationName = selectedText + '_' + filterRule.toLowerCase();
                                                if (headerText.includes(expectedValidationName)) {{
                                                    headerCell.style.display = '';
                                                    validationCell.style.display = '';
                                                }} else {{
                                                    headerCell.style.display = 'none';
                                                    validationCell.style.display = 'none';
                                                }}
                                            }} else {{
                                                // Show all validation columns for this column when no specific rule is selected
                                                headerCell.style.display = '';
                                                validationCell.style.display = '';
                                            }}
                                        }} else {{
                                            // Hide validation columns for other columns
                                            headerCell.style.display = 'none';
                                            validationCell.style.display = 'none';
                                        }}
                                    }}
                                }}
                            }} else {{
                                // Show all columns when "All Columns" is selected
                                for (let i = 0; i < row.cells.length; i++) {{
                                    const headerCell = table.querySelector('thead tr th:nth-child(' + (i + 1) + ')');
                                    const cell = row.cells[i];
                                    if (headerCell && cell) {{
                                        headerCell.style.display = '';
                                        cell.style.display = '';
                                    }}
                                }}
                            }}
                            
                            // Reset all column visibility when "All Columns" is selected
                            if (filterColumn === "0") {{
                                const allHeaders = table.querySelectorAll('thead tr th');
                                const allCells = row.querySelectorAll('td');
                                allHeaders.forEach(header => {{
                                    header.style.display = '';
                                }});
                                allCells.forEach(cell => {{
                                    cell.style.display = '';
                    }});
                }}

                                                        // Validation rule filter - only show rows that have the selected rule
                            if (filterRule && filterRule !== "" && showRow) {{
                                if (validationColumnIndex >= 0) {{
                                    // Check the specific validation column
                                    const validationCell = row.cells[validationColumnIndex];
                                    const cellValue = validationCell.textContent.trim();
                                    showRow = showRow && (cellValue !== "N/A");
                                }} else {{
                                    // Check all validation columns for this rule
                                    let hasRule = false;
                                    for (let i = dataColumnCount; i < row.cells.length; i++) {{
                                        const headerCell = table.querySelector('thead tr th:nth-child(' + (i + 1) + ')');
                                        if (headerCell && headerCell.textContent.toLowerCase().includes(filterRule.replace('_', ' '))) {{
                                            const validationCell = row.cells[i];
                                            const cellValue = validationCell.textContent.trim();
                                            if (cellValue !== "N/A") {{
                                                hasRule = true;
                                                break;
                                            }}
                                        }}
                                    }}
                                    showRow = showRow && hasRule;
                                }}
                            }}
                            
                                                        // Status filter - only show rows that match the selected status
                            if (filterStatus && filterStatus !== "" && showRow) {{
                                let hasMatchingStatus = false;
                                
                                if (validationColumnIndex >= 0) {{
                                    // Check the specific validation column
                                    const validationCell = row.cells[validationColumnIndex];
                                    const cellValue = validationCell.textContent.trim();
                                    hasMatchingStatus = (cellValue.toLowerCase() === filterStatus.toLowerCase());
                                }} else {{
                                    // Check all validation columns for the selected column
                                    for (let i = dataColumnCount; i < row.cells.length; i++) {{
                                        const headerCell = table.querySelector('thead tr th:nth-child(' + (i + 1) + ')');
                                        if (headerCell && headerCell.textContent.toLowerCase().includes(selectedColumnName.toLowerCase())) {{
                                            const validationCell = row.cells[i];
                                            const cellValue = validationCell.textContent.trim();
                                            if (cellValue.toLowerCase() === filterStatus.toLowerCase()) {{
                                                hasMatchingStatus = true;
                                                break;
                                            }}
                                        }}
                                    }}
                                }}
                                
                                showRow = showRow && hasMatchingStatus;
                            }}
                        }}
                        
                        row.style.display = showRow ? '' : 'none';
                    }});
                }}

                // Update validation rules when column changes
                function updateValidationRules() {{
                    const filterColumn = document.getElementById('filterColumn').value;
                    const filterRule = document.getElementById('filterRule');
                    const filterStatus = document.getElementById('filterStatus');
                    
                    // Reset filters
                    filterRule.value = '';
                    filterStatus.value = '';
                    
                    // Get the selected column name
                    const columnSelect = document.getElementById('filterColumn');
                    const selectedColumnName = columnSelect.options[columnSelect.selectedIndex].text;
                    
                    // Update validation rules based on selected column
                    if (filterColumn === "0" || filterColumn === 0) {{
                        // Show all rules for "All Columns"
                        Array.from(filterRule.options).forEach(option => {{
                            option.style.display = '';
                        }});
                    }} else {{
                        // Show only rules that apply to the selected column
                        const availableRules = columnRuleMapping[selectedColumnName] || [];
                        
                        Array.from(filterRule.options).forEach(option => {{
                            if (option.value === '') {{
                                option.style.display = ''; // Always show "All Rules"
                            }} else {{
                                // Show only rules that apply to this column
                                const shouldShow = availableRules.includes(option.value);
                                option.style.display = shouldShow ? '' : 'none';
                            }}
                        }});
                    }}
                }}

                // Theme toggle functionality
                function toggleTheme() {{
                    const body = document.body;
                    const currentTheme = body.getAttribute('data-theme');
                    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
                    
                    body.setAttribute('data-theme', newTheme);
                    localStorage.setItem('theme', newTheme);
                    
                    // Force table repaint to ensure colors update immediately
                    const tables = document.querySelectorAll('table');
                    tables.forEach(table => {{
                        table.style.display = 'none';
                        setTimeout(() => {{
                            table.style.display = '';
                        }}, 10);
                    }});
                }}

                // Load saved theme on page load
                document.addEventListener('DOMContentLoaded', function() {{
                    const savedTheme = localStorage.getItem('theme') || 'light';
                    document.body.setAttribute('data-theme', savedTheme);
                    
                    // Initialize validation rules for data table filtering
                    if (document.getElementById('filterColumn')) {{
                        updateValidationRules();
                    }}
                    
                    var dropdownBtn = document.getElementById('fieldSummaryDropdownBtn');
                    var dropdown = document.getElementById('fieldSummaryDropdown');
                    var checkboxes = dropdown.querySelectorAll('.field-checkbox');
                    var selectAllBtn = document.getElementById('selectAllBtn');
                    var clearAllBtn = document.getElementById('clearAllBtn');

                    // Toggle dropdown
                    dropdownBtn.addEventListener('click', function(e) {{
                        e.stopPropagation();
                        dropdown.classList.toggle('show');
                    }});

                    // Close dropdown when clicking outside
                    document.addEventListener('click', function(e) {{
                        if (!dropdown.contains(e.target) && e.target !== dropdownBtn) {{
                            dropdown.classList.remove('show');
                        }}
                    }});

                    // Checkbox logic
                    function updateFieldSections() {{
                        var checked = Array.from(checkboxes).filter(cb => cb.checked).map(cb => cb.value);
                        var sectionFilter = document.getElementById('sectionTypeFilter').value;
                        
                        // Find all field sections that have data-column attributes (individual field sections)
                        document.querySelectorAll('.field-section[data-column]').forEach(function(section) {{
                            var col = section.getAttribute('data-column');
                            var isVisible = checked.includes(col);
                            
                            // Apply section type filter
                            if (sectionFilter !== 'all') {{
                                var isInCriticalSection = section.closest('.critical-section') !== null;
                                var isInOtherSection = section.closest('.other-section') !== null;
                                
                                if (sectionFilter === 'critical' && !isInCriticalSection) {{
                                    isVisible = false;
                                }} else if (sectionFilter === 'other' && !isInOtherSection) {{
                                    isVisible = false;
                                }}
                            }}
                            
                            section.style.display = isVisible ? '' : 'none';
                        }});
                        
                        // Also handle the critical and other section containers
                        var criticalSection = document.querySelector('.critical-section');
                        var otherSection = document.querySelector('.other-section');
                        
                        if (criticalSection) {{
                            var criticalFields = criticalSection.querySelectorAll('.field-section[data-column]');
                            var hasVisibleCriticalFields = Array.from(criticalFields).some(function(section) {{
                                return section.style.display !== 'none';
                            }});
                            criticalSection.style.display = hasVisibleCriticalFields ? '' : 'none';
                        }}
                        
                        if (otherSection) {{
                            var otherFields = otherSection.querySelectorAll('.field-section[data-column]');
                            var hasVisibleOtherFields = Array.from(otherFields).some(function(section) {{
                                return section.style.display !== 'none';
                            }});
                            otherSection.style.display = hasVisibleOtherFields ? '' : 'none';
                        }}
                    }}
                    checkboxes.forEach(cb => cb.addEventListener('change', updateFieldSections));

                    // Section type filter
                    var sectionTypeFilter = document.getElementById('sectionTypeFilter');
                    if (sectionTypeFilter) {{
                        sectionTypeFilter.addEventListener('change', updateFieldSections);
                    }}

                    // Select All
                    selectAllBtn.addEventListener('click', function(e) {{
                        e.preventDefault();
                        checkboxes.forEach(cb => cb.checked = true);
                        updateFieldSections();
                    }});
                    // Clear All
                    clearAllBtn.addEventListener('click', function(e) {{
                        e.preventDefault();
                        checkboxes.forEach(cb => cb.checked = false);
                        updateFieldSections();
                    }});
                }});
            </script>
        </body>
        </html>
        """

        return html_template

    def _get_bar_color(self, rate):
        """Get color for bar chart based on success rate"""
        if rate >= 90:
            return '#28a745'
        elif rate >= 50:
            return '#ffc107'
        else:
            return '#dc3545'

    def _get_enhanced_bar_color(self, rate):
        """Get enhanced color for bar chart based on success rate"""
        if rate >= 90:
            return '#27ae60'
        elif rate >= 70:
            return '#f39c12'
        elif rate >= 50:
            return '#3498db'
        else:
            return '#dc3545'

    def _get_enhanced_border_color(self, rate):
        """Get enhanced border color for bar chart based on success rate"""
        if rate >= 90:
            return '#27ae60'
        elif rate >= 70:
            return '#f39c12'
        elif rate >= 50:
            return '#3498db'
        else:
            return '#dc3545'

    def _darken_color(self, hex_color, amount=0.2):
        """Darken a hex color by a given amount."""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join([c * 2 for c in hex_color])
        if len(hex_color) != 6:
            return '#000000' # Return black for invalid hex
        try:
            r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
            r = max(0, min(255, int(r * (1 - amount))))
            g = max(0, min(255, int(g * (1 - amount))))
            b = max(0, min(255, int(b * (1 - amount))))
            return '#{:02x}{:02x}{:02x}'.format(r, g, b)
        except ValueError:
            return '#000000' # Return black for invalid hex

    def _generate_field_summary_html(self):
        """Generate HTML for field summary tab with enhanced statistics and quality scores, sorted by critical data elements"""
        if not hasattr(self, 'df') or self.df is None:
            return "<p>No data available for field summary.</p>"

        def classify_data_type(col_data):
            """Classify data type with user-friendly names"""
            if pd.api.types.is_datetime64_any_dtype(col_data):
                return "Date/Time"
            elif pd.api.types.is_bool_dtype(col_data):
                return "Boolean"
            elif pd.api.types.is_numeric_dtype(col_data):
                if 'int' in str(col_data.dtype).lower():
                    return "Integer"
                elif 'float' in str(col_data.dtype).lower():
                    return "Decimal"
                else:
                    return "Numeric"
            elif pd.api.types.is_object_dtype(col_data):
                # Enhanced detection for object types
                sample = col_data.dropna().head(100)
                if len(sample) > 0:
                    # Check for boolean-like values first
                    bool_count = sum(1 for x in sample if isinstance(x, bool))
                    if bool_count == len(sample):
                        return "Boolean"
                    
                    # Check for string representations of booleans
                    bool_string_count = sum(1 for x in sample if str(x).lower() in ['true', 'false', '1', '0'])
                    if bool_string_count == len(sample):
                        return "Boolean"
                    
                    # Check if it's actually string data
                    if all(isinstance(x, str) for x in sample):
                        return "Text/String"
                return "Text/String"  # Default for object types
            else:
                # Convert any remaining pandas-specific names to user-friendly
                dtype_str = str(col_data.dtype).lower()
                if 'category' in dtype_str:
                    return "Category"
                else:
                    return dtype_str.title()

        def calculate_quality_scores(col_data):
            """Calculate quality scores for a column"""
            total_count = len(col_data)
            null_count = col_data.isnull().sum()
            distinct_count = col_data.nunique()
            
            # Completeness Score
            completeness = ((total_count - null_count) / total_count) * 100 if total_count > 0 else 0
            
            # Uniqueness Score
            uniqueness = (distinct_count / total_count) * 100 if total_count > 0 else 0
            
            # Consistency Score (basic implementation)
            consistency = 100.0  # Start with perfect score
            if pd.api.types.is_object_dtype(col_data):
                # Check for mixed data types in object columns
                sample = col_data.dropna().head(100)
                if len(sample) > 0:
                    type_counts = {}
                    for item in sample:
                        item_type = type(item).__name__
                        type_counts[item_type] = type_counts.get(item_type, 0) + 1
                    
                    # If more than 80% are the same type, consider it consistent
                    most_common_type_count = max(type_counts.values()) if type_counts else 0
                    consistency = (most_common_type_count / len(sample)) * 100
            
            return {
                'completeness': round(completeness, 1),
                'uniqueness': round(uniqueness, 1),
                'consistency': round(consistency, 1)
            }

        def is_critical_data_element(col_data):
            """Determine if a column is a critical data element based on data quality metrics"""
            total_count = len(col_data)
            null_count = col_data.isnull().sum()
            
            # Calculate completeness percentage
            completeness = ((total_count - null_count) / total_count) * 100 if total_count > 0 else 0
            
            # Calculate data density (how much actual data vs nulls)
            data_density = (total_count - null_count) / total_count if total_count > 0 else 0
            
            # A column is considered critical if:
            # 1. It has high completeness (>= 80%)
            # 2. It has good data density (>= 70%)
            # 3. It's not mostly empty
            is_critical = (completeness >= 80 and data_density >= 0.7 and null_count < total_count * 0.5)
            
            return is_critical

        def get_score_color(score):
            """Get color class for score display"""
            if score >= 90:
                return "score-excellent"
            elif score >= 70:
                return "score-good"
            elif score >= 50:
                return "score-warning"
            else:
                return "score-poor"

        # Separate columns into critical and other
        critical_columns = []
        other_columns = []
        
        for column in self.df.columns:
            col_data = self.df[column]
            if is_critical_data_element(col_data):
                critical_columns.append(column)
            else:
                other_columns.append(column)

        # Sort critical columns by completeness (highest first)
        critical_columns.sort(key=lambda col: calculate_quality_scores(self.df[col])['completeness'], reverse=True)
        
        # Sort other columns by completeness (highest first)
        other_columns.sort(key=lambda col: calculate_quality_scores(self.df[col])['completeness'], reverse=True)

        html_parts = []
        
        # Generate HTML for critical data elements section
        if critical_columns:
            html_parts.append(f"""
            <div class="field-section critical-section" style="border-left: 4px solid #28a745; background: linear-gradient(135deg, rgba(40, 167, 69, 0.05), rgba(40, 167, 69, 0.02));">
                <h3 style="color: #28a745; margin-bottom: 15px;">🔑 Critical Data Elements ({len(critical_columns)} columns)</h3>
                <p style="color: #666; font-style: italic; margin-bottom: 20px;">These columns contain high-quality, well-populated data with minimal missing values.</p>
            """)
            
            for column in critical_columns:
                col_data = self.df[column]
                data_type = classify_data_type(col_data)
                quality_scores = calculate_quality_scores(col_data)
                
                html_parts.append(f"""
                <div class="field-section" data-column="{column}" style="margin-left: 20px; border-left: 3px solid #28a745;">
                    <h3>Column: {column}</h3>
                    <div class="field-stats">
                        <div class="field-stats-header">Field Statistics</div>
                        <div class="stats-grid">
                            <div class="stat-row data-type-row">
                                <span class="stat-label">Data Type</span>
                                <span class="stat-value data-type-value">{data_type}</span>
                            </div>
                            <div class="stat-row">
                                <span class="stat-label">Total Count</span>
                                <span class="stat-value">{len(col_data)}</span>
                            </div>
                            <div class="stat-row">
                                <span class="stat-label">Distinct Count</span>
                                <span class="stat-value">{col_data.nunique()}</span>
                            </div>
                            <div class="stat-row">
                                <span class="stat-label">Null Count</span>
                                <span class="stat-value">{col_data.isnull().sum()}</span>
                            </div>
                """)

                # Add descriptive statistics for numeric columns
                if pd.api.types.is_numeric_dtype(col_data):
                    # Calculate mode (most frequent value)
                    mode_values = col_data.mode()
                    most_frequent = mode_values.iloc[0] if len(mode_values) > 0 else "N/A"
                    mode_count = (col_data == most_frequent).sum() if most_frequent != "N/A" else 0
                    
                    html_parts.append(f"""
                        <div class="stat-row">
                            <span class="stat-label">Mean</span>
                            <span class="stat-value">{col_data.mean():.2f}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Median</span>
                            <span class="stat-value">{col_data.median():.2f}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Most Frequent</span>
                            <span class="stat-value">{most_frequent}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Highest Frequency</span>
                            <span class="stat-value">{mode_count}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Min</span>
                            <span class="stat-value">{col_data.min()}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Max</span>
                            <span class="stat-value">{col_data.max()}</span>
                        </div>
                    """)

                # Add date-specific statistics for datetime columns
                elif pd.api.types.is_datetime64_any_dtype(col_data):
                    # Convert to datetime if needed and handle nulls
                    date_data = pd.to_datetime(col_data, errors='coerce')
                    non_null_dates = date_data.dropna()
                    
                    if len(non_null_dates) > 0:
                        oldest_date = non_null_dates.min()
                        newest_date = non_null_dates.max()
                        median_date = non_null_dates.median()
                        
                        # Calculate most frequent date
                        mode_dates = non_null_dates.mode()
                        most_frequent_date = mode_dates.iloc[0] if len(mode_dates) > 0 else None
                        mode_date_count = (non_null_dates == most_frequent_date).sum() if most_frequent_date is not None else 0
                        
                        html_parts.append(f"""
                            <div class="stat-row">
                                <span class="stat-label">Oldest Date</span>
                                <span class="stat-value">{oldest_date.strftime('%Y-%m-%d')}</span>
                            </div>
                            <div class="stat-row">
                                <span class="stat-label">Most Recent</span>
                                <span class="stat-value">{newest_date.strftime('%Y-%m-%d')}</span>
                            </div>
                            <div class="stat-row">
                                <span class="stat-label">Median Date</span>
                                <span class="stat-value">{median_date.strftime('%Y-%m-%d')}</span>
                            </div>
                            <div class="stat-row">
                                <span class="stat-label">Most Frequent</span>
                                <span class="stat-value">{most_frequent_date.strftime('%Y-%m-%d') if most_frequent_date else 'N/A'}</span>
                            </div>
                            <div class="stat-row">
                                <span class="stat-label">Highest Frequency</span>
                                <span class="stat-value">{mode_date_count}</span>
                            </div>
                            <div class="stat-row">
                                <span class="stat-label">Date Range</span>
                                <span class="stat-value">{(newest_date - oldest_date).days} days</span>
                            </div>
                        """)
                    else:
                        html_parts.append(f"""
                            <div class="stat-row">
                                <span class="stat-label">Date Info</span>
                                <span class="stat-value">No valid dates</span>
                            </div>
                        """)

                # Add sample values for non-numeric, non-datetime columns
                elif not pd.api.types.is_numeric_dtype(col_data) and not pd.api.types.is_datetime64_any_dtype(col_data):
                    # Calculate most frequent value for categorical/string columns
                    mode_values = col_data.mode()
                    most_frequent = mode_values.iloc[0] if len(mode_values) > 0 else "N/A"
                    mode_count = (col_data == most_frequent).sum() if most_frequent != "N/A" else 0
                    
                    sample_values = col_data.dropna().head(5).tolist()
                    html_parts.append(f"""
                        <div class="stat-row">
                            <span class="stat-label">Most Frequent</span>
                            <span class="stat-value">{most_frequent}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Highest Frequency</span>
                            <span class="stat-value">{mode_count}</span>
                        </div>
                        <div class="stat-row sample-values">
                            <span class="stat-label">Sample Values</span>
                            <span class="stat-value">{', '.join(map(str, sample_values))}</span>
                        </div>
                    """)

                # Close the stats grid and field-stats container
                html_parts.append("""
                        </div>
                    </div>
                """)

                # Add quality scores in separate container
                html_parts.append(f"""
                    <div class="quality-scores">
                        <h4>Quality Scores</h4>
                        <div class="score-grid">
                            <div class="score-item">
                                <span class="score-label">Completeness</span>
                                <span class="score-value {get_score_color(quality_scores['completeness'])}">{quality_scores['completeness']}%</span>
                            </div>
                            <div class="score-item">
                                <span class="score-label">Uniqueness</span>
                                <span class="score-value {get_score_color(quality_scores['uniqueness'])}">{quality_scores['uniqueness']}%</span>
                            </div>
                            <div class="score-item">
                                <span class="score-label">Consistency</span>
                                <span class="score-value {get_score_color(quality_scores['consistency'])}">{quality_scores['consistency']}%</span>
                            </div>
                        </div>
                    </div>
                </div>
                """)
            
            html_parts.append("</div>")  # Close critical-section

        # Generate HTML for other fields section
        if other_columns:
            html_parts.append(f"""
            <div class="field-section other-section" style="border-left: 4px solid #6c757d; background: linear-gradient(135deg, rgba(108, 117, 125, 0.05), rgba(108, 117, 125, 0.02)); margin-top: 30px;">
                <h3 style="color: #6c757d; margin-bottom: 15px;">📊 Other Fields ({len(other_columns)} columns)</h3>
                <p style="color: #666; font-style: italic; margin-bottom: 20px;">These columns may have more missing data or lower data quality metrics.</p>
            """)
            
            for column in other_columns:
                col_data = self.df[column]
                data_type = classify_data_type(col_data)
                quality_scores = calculate_quality_scores(col_data)
                
                html_parts.append(f"""
                <div class="field-section" data-column="{column}" style="margin-left: 20px; border-left: 3px solid #6c757d;">
                    <h3>Column: {column}</h3>
                    <div class="field-stats">
                        <div class="field-stats-header">Field Statistics</div>
                        <div class="stats-grid">
                            <div class="stat-row data-type-row">
                                <span class="stat-label">Data Type</span>
                                <span class="stat-value data-type-value">{data_type}</span>
                            </div>
                            <div class="stat-row">
                                <span class="stat-label">Total Count</span>
                                <span class="stat-value">{len(col_data)}</span>
                            </div>
                            <div class="stat-row">
                                <span class="stat-label">Distinct Count</span>
                                <span class="stat-value">{col_data.nunique()}</span>
                            </div>
                            <div class="stat-row">
                                <span class="stat-label">Null Count</span>
                                <span class="stat-value">{col_data.isnull().sum()}</span>
                            </div>
                """)

                # Add descriptive statistics for numeric columns
                if pd.api.types.is_numeric_dtype(col_data):
                    # Calculate mode (most frequent value)
                    mode_values = col_data.mode()
                    most_frequent = mode_values.iloc[0] if len(mode_values) > 0 else "N/A"
                    mode_count = (col_data == most_frequent).sum() if most_frequent != "N/A" else 0
                    
                    html_parts.append(f"""
                        <div class="stat-row">
                            <span class="stat-label">Mean</span>
                            <span class="stat-value">{col_data.mean():.2f}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Median</span>
                            <span class="stat-value">{col_data.median():.2f}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Most Frequent</span>
                            <span class="stat-value">{most_frequent}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Highest Frequency</span>
                            <span class="stat-value">{mode_count}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Min</span>
                            <span class="stat-value">{col_data.min()}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Max</span>
                            <span class="stat-value">{col_data.max()}</span>
                        </div>
                    """)

                # Add date-specific statistics for datetime columns
                elif pd.api.types.is_datetime64_any_dtype(col_data):
                    # Convert to datetime if needed and handle nulls
                    date_data = pd.to_datetime(col_data, errors='coerce')
                    non_null_dates = date_data.dropna()
                    
                    if len(non_null_dates) > 0:
                        oldest_date = non_null_dates.min()
                        newest_date = non_null_dates.max()
                        median_date = non_null_dates.median()
                        
                        # Calculate most frequent date
                        mode_dates = non_null_dates.mode()
                        most_frequent_date = mode_dates.iloc[0] if len(mode_dates) > 0 else None
                        mode_date_count = (non_null_dates == most_frequent_date).sum() if most_frequent_date is not None else 0
                        
                        html_parts.append(f"""
                            <div class="stat-row">
                                <span class="stat-label">Oldest Date</span>
                                <span class="stat-value">{oldest_date.strftime('%Y-%m-%d')}</span>
                            </div>
                            <div class="stat-row">
                                <span class="stat-label">Most Recent</span>
                                <span class="stat-value">{newest_date.strftime('%Y-%m-%d')}</span>
                            </div>
                            <div class="stat-row">
                                <span class="stat-label">Median Date</span>
                                <span class="stat-value">{median_date.strftime('%Y-%m-%d')}</span>
                            </div>
                            <div class="stat-row">
                                <span class="stat-label">Most Frequent</span>
                                <span class="stat-value">{most_frequent_date.strftime('%Y-%m-%d') if most_frequent_date else 'N/A'}</span>
                            </div>
                            <div class="stat-row">
                                <span class="stat-label">Highest Frequency</span>
                                <span class="stat-value">{mode_date_count}</span>
                            </div>
                            <div class="stat-row">
                                <span class="stat-label">Date Range</span>
                                <span class="stat-value">{(newest_date - oldest_date).days} days</span>
                            </div>
                        """)
                    else:
                        html_parts.append(f"""
                            <div class="stat-row">
                                <span class="stat-label">Date Info</span>
                                <span class="stat-value">No valid dates</span>
                            </div>
                        """)

                # Add sample values for non-numeric, non-datetime columns
                elif not pd.api.types.is_numeric_dtype(col_data) and not pd.api.types.is_datetime64_any_dtype(col_data):
                    # Calculate most frequent value for categorical/string columns
                    mode_values = col_data.mode()
                    most_frequent = mode_values.iloc[0] if len(mode_values) > 0 else "N/A"
                    mode_count = (col_data == most_frequent).sum() if most_frequent != "N/A" else 0
                    
                    sample_values = col_data.dropna().head(5).tolist()
                    html_parts.append(f"""
                        <div class="stat-row">
                            <span class="stat-label">Most Frequent</span>
                            <span class="stat-value">{most_frequent}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Highest Frequency</span>
                            <span class="stat-value">{mode_count}</span>
                        </div>
                        <div class="stat-row sample-values">
                            <span class="stat-label">Sample Values</span>
                            <span class="stat-value">{', '.join(map(str, sample_values))}</span>
                        </div>
                    """)

                # Close the stats grid and field-stats container
                html_parts.append("""
                        </div>
                    </div>
                """)

                # Add quality scores in separate container
                html_parts.append(f"""
                    <div class="quality-scores">
                        <h4>Quality Scores</h4>
                        <div class="score-grid">
                            <div class="score-item">
                                <span class="score-label">Completeness</span>
                                <span class="score-value {get_score_color(quality_scores['completeness'])}">{quality_scores['completeness']}%</span>
                            </div>
                            <div class="score-item">
                                <span class="score-label">Uniqueness</span>
                                <span class="score-value {get_score_color(quality_scores['uniqueness'])}">{quality_scores['uniqueness']}%</span>
                            </div>
                            <div class="score-item">
                                <span class="score-label">Consistency</span>
                                <span class="score-value {get_score_color(quality_scores['consistency'])}">{quality_scores['consistency']}%</span>
                            </div>
                        </div>
                    </div>
                </div>
                """)
            
            html_parts.append("</div>")  # Close other-section

        return ''.join(html_parts)

    def _generate_data_table_html(self):
        """Generate HTML for data table tab with sorting and filtering"""
        if not hasattr(self, 'df') or self.df is None:
            return "<p>No data available for data table.</p>"

        # Create table headers
        headers = list(self.df.columns)
        
        # Add validation result columns
        validation_columns = []
        if hasattr(self, 'results') and self.results:
            for result in self.results:
                column = result.get('column', '')
                rule = result['rule']
                validation_columns.append(f'{column}_{rule}')
        
        # Create header HTML
        header_html = '<tr>'
        # Data columns
        for i, col in enumerate(headers):
            header_html += f'<th class="sortable" onclick="sortTable(document.getElementById(\'dataTable\'), {i}, \'text\')">{col}</th>'
        # Validation columns
        for i, val_col in enumerate(validation_columns):
            col_idx = len(headers) + i
            # Keep the original column name format for easier matching
            header_html += f'<th class="sortable" onclick="sortTable(document.getElementById(\'dataTable\'), {col_idx}, \'text\')">{val_col}</th>'
        header_html += '</tr>'

        # Create table rows with validation data (limit to first 100 rows for performance)
        rows_html = ""
        for idx, row in self.df.head(100).iterrows():
            # Create data cells
            cells = ''.join([f'<td>{str(val) if pd.notna(val) else ""}</td>' for val in row])
            
            # Add validation result cells
            validation_cells = ""
            if hasattr(self, 'results') and self.results:
                for result in self.results:
                    column = result.get('column', '')
                    rule = result['rule']
                    success_rate = result['success_rate']
                    
                    # Check if this specific row's value passes the validation
                    # Find the column with case-insensitive matching
                    matching_column = None
                    for col in row.index:
                        if col.lower() == column.lower():
                            matching_column = col
                            break
                    
                    row_value = row[matching_column] if matching_column else None
                    cell_status = "N/A"
                    
                    # Only process validation if the column exists in the table data
                    if matching_column:
                        if rule == 'not null':
                            is_not_null = pd.notna(row_value)
                            cell_status = "Passed" if is_not_null else "Failed"
                        elif rule == 'unique':
                            # Check if this specific value is unique in the column
                            value_count = (self.df[matching_column] == row_value).sum()
                            cell_status = "Passed" if value_count == 1 else "Failed"
                        elif rule == 'in allowed set':
                            allowed_values = result.get('details', {}).get('allowed_values', [])
                            cell_status = "Passed" if row_value in allowed_values else "Failed"
                        elif rule == 'matches regex':
                            import re
                            pattern = result.get('details', {}).get('pattern', '')
                            try:
                                matches = bool(re.match(pattern, str(row_value)))
                                cell_status = "Passed" if matches else "Failed"
                            except Exception as e:
                                cell_status = "N/A"
                        elif rule.startswith('in range'):
                            # Extract min and max from rule name like "in range 18-60"
                            try:
                                range_part = rule.replace('in range ', '')
                                if '-' in range_part:
                                    min_val, max_val = map(float, range_part.split('-'))
                                    in_range = min_val <= row_value <= max_val
                                    cell_status = "Passed" if in_range else "Failed"
                                else:
                                    cell_status = "N/A"
                            except Exception as e:
                                cell_status = "N/A"
                        elif rule.startswith('in date range'):
                            # Extract min and max dates from rule name
                            try:
                                date_range_part = rule.replace('in date range ', '')
                                if '-' in date_range_part:
                                    min_date_str, max_date_str = date_range_part.split('-')
                                    min_date = pd.to_datetime(min_date_str)
                                    max_date = pd.to_datetime(max_date_str)
                                    row_date = pd.to_datetime(row_value)
                                    in_range = min_date <= row_date <= max_date
                                    cell_status = "Passed" if in_range else "Failed"
                                else:
                                    cell_status = "N/A"
                            except Exception as e:
                                cell_status = "N/A"
                        else:
                            cell_status = "Passed" if success_rate >= 90 else "Failed"
                    else:
                        # Column doesn't exist in table data, show N/A
                        cell_status = "N/A"
                    
                    # Color code the status
                    status_color = "#28a745" if cell_status == "Passed" else "#dc3545" if cell_status == "Failed" else "#6c757d"
                    validation_cells += f'<td style="color: {status_color}; font-weight: bold;">{cell_status}</td>'
            
            rows_html += f'<tr>{cells}{validation_cells}</tr>'

        # Get validation rules for filtering and create column-rule mapping
        validation_rules = []
        column_rule_mapping = {}
        
        if hasattr(self, 'results') and self.results:
            for result in self.results:
                rule = result['rule']
                column = result.get('column', '')
                if rule not in validation_rules:
                    validation_rules.append(rule)
                
                if column not in column_rule_mapping:
                    column_rule_mapping[column] = []
                if rule not in column_rule_mapping[column]:
                    column_rule_mapping[column].append(rule)
        
        # Generate column options HTML
        column_options = ''.join([f'<option value="{i+1}">{col}</option>' for i, col in enumerate(headers)])
        
        # Generate validation rule options HTML
        rule_options = ''.join([f'<option value="{rule}">{rule.replace("_", " ").title()}</option>' for rule in validation_rules])

        return f"""
        <script>
        const columnRuleMapping = {str(column_rule_mapping)};
        const dataColumnCount = {len(headers)};
        </script>
        <div class="data-table-controls">
            <div class="filter-row">
            <label>Filter by column:</label>
                <select id="filterColumn" onchange="updateValidationRules(); setTimeout(filterTable, 100);">
                <option value="0">All Columns</option>
                    {column_options}
            </select>
            </div>
            <div class="filter-row">
                <label>Filter by validation rule:</label>
                <select id="filterRule" onchange="filterTable()">
                    <option value="">All Rules</option>
                    {rule_options}
                </select>
            </div>
            <div class="filter-row">
                <label>Filter by status:</label>
                <select id="filterStatus" onchange="filterTable()">
                    <option value="">All Status</option>
                    <option value="passed">Passed</option>
                    <option value="failed">Failed</option>
                </select>
            </div>
            <div class="filter-row">
            <label>Filter value:</label>
            <input type="text" id="filterInput" placeholder="Enter filter value..." onkeyup="filterTable()">
            </div>
        </div>
        <div class="data-table-container">
            <table id="dataTable">
                <thead>{header_html}</thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        <p><em>Showing first 100 rows. Use filters to find specific data.</em></p>
        """
