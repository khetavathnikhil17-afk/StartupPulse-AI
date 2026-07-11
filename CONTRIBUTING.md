# Contributing to StartupPulse AI

Thank you for your interest in contributing to StartupPulse AI! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)

## Code of Conduct

We are committed to providing a welcoming and inclusive experience for everyone. Please be respectful and considerate in all interactions.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a branch for your changes
4. Make your changes
5. Run tests
6. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.10+
- pip
- Git
- Docker (optional)

### Installation

```bash
# Clone your fork
git clone https://github.com/your-username/StartupPulse-AI.git
cd StartupPulse-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov flake8 black isort
```

### Docker Development

```bash
# Build and run with Docker
docker-compose up --build

# Or run directly
docker build -t startuppulse-ai .
docker run -p 8501:8501 startuppulse-ai
```

## How to Contribute

### Types of Contributions

1. **Bug Fixes**: Fix issues in existing code
2. **Features**: Add new functionality
3. **Documentation**: Improve or add documentation
4. **Tests**: Add or improve test coverage
5. **Refactoring**: Improve code quality without changing behavior

### Contribution Workflow

1. **Find or Create an Issue**
   - Check existing issues for something to work on
   - Create a new issue if you find a bug or have a feature idea

2. **Set Up Your Development Environment**
   - Fork and clone the repository
   - Create a feature branch: `git checkout -b feature/your-feature-name`

3. **Make Your Changes**
   - Write code following our coding standards
   - Add tests for new functionality
   - Update documentation as needed

4. **Test Your Changes**
   - Run the test suite: `pytest tests/ -v`
   - Run linting: `flake8 src/ dashboard/`
   - Test manually with the dashboard

5. **Submit a Pull Request**
   - Push your changes to your fork
   - Create a pull request with a clear description
   - Link any related issues

## Coding Standards

### Python Style Guide

- Follow PEP 8
- Use Black for code formatting
- Use isort for import sorting
- Maximum line length: 100 characters

### Code Quality

- Write docstrings for all public functions and classes
- Use type hints where appropriate
- Keep functions focused and small
- Avoid global variables when possible
- Use meaningful variable and function names

### Example

```python
"""
Module for sentiment prediction.

This module provides a singleton-based sentiment predictor.
"""
from typing import Dict
from src.config.config import SENTIMENT_MAPPING


def predict_sentiment(text: str) -> Dict[str, any]:
    """
    Predict sentiment for a given text.
    
    Args:
        text: Input text to classify
        
    Returns:
        Dictionary containing label, confidence, and probabilities
    """
    # Implementation here
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test
pytest tests/test_core.py::TestPrediction -v
```

### Writing Tests

- Create test files in the `tests/` directory
- Name test files `test_*.py`
- Use descriptive test names
- Test both success and error cases
- Mock external dependencies when appropriate

## Documentation

### Code Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Include type hints
- Provide examples where helpful

### Project Documentation

- Update README.md for major changes
- Update relevant documentation files in `docs/`
- Add entries to CHANGELOG.md

## Pull Request Process

### Before Submitting

1. Ensure all tests pass
2. Run linting and formatting
3. Update documentation
4. Add changelog entry

### PR Description

Include:
- Summary of changes
- Related issues
- Testing performed
- Screenshots (if applicable)

### Review Process

1. PR will be reviewed by maintainers
2. Address any feedback
3. Once approved, PR will be merged

## Reporting Bugs

### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior.

**Expected behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment**
- OS: [e.g., Windows 11]
- Python version: [e.g., 3.10]
- Browser: [e.g., Chrome 120]
```

## Suggesting Features

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
What you want to happen.

**Describe alternatives you've considered**
Other solutions you've thought about.

**Additional context**
Any other context or screenshots.
```

## Questions?

Feel free to open an issue with the label "question" if you have any questions about contributing.
