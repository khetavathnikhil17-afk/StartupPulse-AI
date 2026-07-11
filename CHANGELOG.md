# Changelog

All notable changes to StartupPulse AI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2026-07-11

### Added
- Docker support with multi-stage builds
- Docker Compose configuration
- GitHub Actions CI/CD pipeline
- Unit test suite with pytest
- CHANGELOG.md
- CONTRIBUTING.md
- DOCKER.md documentation

### Changed
- Improved code quality across all modules
- Added comprehensive docstrings to all functions
- Refactored loading stages in dashboard for better maintainability
- Updated configuration with additional constants
- Improved error handling and logging

### Fixed
- Removed unused imports across codebase
- Fixed hardcoded values in evaluate.py
- Fixed inconsistent logging patterns
- Fixed .gitignore contradictions
- Fixed security vulnerability with HTML injection

### Removed
- Empty directories (notebooks/, outputs/)
- Dead code in download_dataset.py

## [1.2.1] - 2026-07-10

### Added
- Professional branding assets (logo, favicon, GitHub banner)
- Comprehensive README with 14 badges
- 11 documentation files
- Installation, Usage, Architecture, Model, SHAP, Dashboard, Configuration, Troubleshooting, FAQ, Roadmap guides

### Changed
- Expanded Model Performance section
- Updated dashboard sidebar with logo and version badge
- Improved CSS styling for dark theme

### Fixed
- Repository audit and cleanup
- Version sync between pyproject.toml and dashboard
- Image paths in README

## [1.2.0] - 2026-07-09

### Added
- SHAP explainability integration
- Token-level explanations
- Waterfall and bar summary plots
- Interactive HTML visualizations

### Changed
- Upgraded to DeBERTa-v3-base model
- Improved prediction accuracy

## [1.1.0] - 2026-07-08

### Added
- Streamlit dashboard
- Dark theme with premium styling
- Loading animations
- Session state management

### Changed
- Improved UI/UX design
- Added example reviews

## [1.0.0] - 2026-07-07

### Added
- Initial release
- DeBERTa-v3 sentiment classifier
- 3-class sentiment analysis
- Basic evaluation metrics
