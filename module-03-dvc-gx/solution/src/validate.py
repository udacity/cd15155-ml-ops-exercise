import json
import os
import sys

import great_expectations as gx
import pandas as pd
import yaml


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def build_suite(context, val):
    # TODO Create an Expectation suite
    # An Expectation suite contains a group of Expectations
    # applied to the same dataset
    # Read more here: https://docs.greatexpectations.io/docs/core/define_expectations/organize_expectation_suites?procedure=sample_code
    suite = gx.ExpectationSuite(name="sales_suite")

    # TODO Add the suite to the Data Context
    suite = context.suites.add(suite)

    # TODO Create an Expectation that ensures all expected columns are present after preprocessing.
    # This catches regressions where a preprocessing step accidentally drops
    # or renames a column that downstream stages depend on.
    # Expectation Gallery: https://greatexpectations.io/expectations/
    # Hint: https://greatexpectations.io/expectations/expect_table_columns_to_match_set/
    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchSet(column_set=val["expected_columns"])
    )

    # TODO Create an Expectation that ensures key columns do not contain null values.
    # Expectation Gallery: https://greatexpectations.io/expectations/
    for col in val["non_null_columns"]:
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=col))

    # TODO Create an Expectation that checks numeric columns stay within specific ranges.
    # Expectation Gallery: https://greatexpectations.io/expectations/
    for col, bounds in val["value_ranges"].items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=col,
                min_value=bounds["min"],
                max_value=bounds["max"],
            )
        )

    return suite


def run_validation(df, params):
    val = params["validation"]
    report_path = params["paths"]["report"]

    # TODO Initialize the data context to manage configuration and metadata
    # Use ephemeral data context for in-memory storage
    # Or file data context to store information as YAML files
    # Read more here: https://docs.greatexpectations.io/docs/core/set_up_a_gx_environment/create_a_data_context
    context = gx.get_context(mode="ephemeral")

    # TODO Connect GX to your data source
    # Create a dataframe data source
    # Read more here: https://docs.greatexpectations.io/docs/core/connect_to_data/dataframes/
    data_source = context.data_sources.add_pandas("pandas_source")

    # TODO Add the training dataframe to the data source
    data_asset = data_source.add_dataframe_asset("processed_train")

    # TODO Add a batch definition to describe how the data
    # within a data asset should be retrieved for validation
    # The batch definition should cover the whole dataframe
    batch_definition = data_asset.add_batch_definition_whole_dataframe("full_batch")

    # TODO Create the Expectation suite by calling build_suite
    suite = build_suite(context, val)

    # TODO Create a validation definition that links the batch definition to the Expectation suite
    # Read more: https://docs.greatexpectations.io/docs/core/run_validations/create_a_validation_definition
    validation_def = gx.ValidationDefinition(
        name="sales_validation",
        data=batch_definition,
        suite=suite,
    )
    # TODO Add the validation definition to the Data Context
    validation_definition = context.validation_definitions.add(validation_def)

    # TODO Run the validation
    # Read more: https://docs.greatexpectations.io/docs/core/run_validations/run_a_validation_definition?procedure=sample_code
    result = validation_definition.run(batch_parameters={"dataframe": df})

    # TODO Save the validation result as JSON
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(result.to_json_dict(), f, indent=2)

    print(f"Validation report saved to {report_path}")
    return result


if __name__ == "__main__":
    params = load_params()
    df = pd.read_csv(params["paths"]["processed_train"])
    print(f"Validating {len(df)} rows from training set...")

    result = run_validation(df, params)

    # TODO Make sure that the script exits with an error code if the validation is unsuccessful
    if not result.success:
        print("Validation FAILED. Inspect the report for details.")
        sys.exit(1)

    print("Validation passed.")
