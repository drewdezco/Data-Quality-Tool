"""
Simple Data Quality Test with Inline DataFrame and JSON Expectations
"""

import pandas as pd
import json
from data_quality_checker import DataQualityChecker

# Create a simple sample DataFrame
data = {
    'employee_id': ['EMP001', 'EMP002', 'EMP003', 'EMP004', 'EMP005', 'EMP006'],
    'name': ['John Doe', 'Jane Smith', 'Bob Johnson', None, 'Alice Brown', 'Charlie Wilson'],
    'email': ['john@company.com', 'jane@company.com', 'invalid-email', 'alice@company.com', 'bob@company.com', 'charlie@company.com'],
    'department': ['IT', 'HR', 'IT', 'Finance', 'IT', 'Marketing'],
    'salary': [75000, 85000, 95000, 70000, 80000, 90000],
    'age': [28, 32, 45, 29, 35, 41]
}

my_df = pd.DataFrame(data)

# Define expectations as inline JSON
expectations_json = {
    "expect_column_values_to_not_be_null": [
        {"column": "employee_id"},
        {"column": "name"},
        {"column": "email"}
    ],
    "expect_column_values_to_be_unique": [
        {"column": "employee_id"},
        {"column": "email"}
    ],
    "expect_column_values_to_be_in_set": [
        {"column": "department", "allowed_values": ["IT", "HR", "Finance", "Marketing", "Sales"]}
    ],
    "expect_column_values_to_match_regex": [
        {"column": "email", "pattern": "^[^@]+@[^@]+\\.[^@]+$"}
    ],
    "expect_column_values_to_be_in_range": [
        {"column": "salary", "min_val": 50000, "max_val": 150000},
        {"column": "age", "min_val": 18, "max_val": 65}
    ]
}


def quality_check(df, expectations_json, title, dataset_name):
    checker = DataQualityChecker(df, dataset_name=dataset_name)
    checker.run_rules_from_json(expectations_json)
    results = checker.get_results()
    data_docs = checker.generate_data_docs(
        title=title,
        dataset_name=dataset_name
    )
    comprehensive_results = checker.get_comprehensive_results(title=title)
    
    # Save to CSV for historical tracking
    csv_filename = checker.save_comprehensive_results_to_csv(title=title, csv_filename="employee_data_quality_history.csv")
    
    return data_docs, results, comprehensive_results

data_docs, results, comprehensive_results = quality_check(my_df, expectations_json, "Simple Data Quality Report", "Employee Sample Data")

with open("test_data_quality.html", "w", encoding="utf-8") as f:
    f.write(data_docs)
