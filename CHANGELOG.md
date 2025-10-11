# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

- Initial public release of stable-cart package.
- Implemented LessGreedyHybridRegressor model with stability-enhancing techniques.
- Removed GreedyCARTExact in favor of using sklearn's DecisionTreeRegressor for comparisons.
- Added evaluation utilities: prediction stability and accuracy metrics.
- Set up unit, integration, and end-to-end tests with pytest.