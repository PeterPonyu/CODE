# Contributing to CODE

Thank you for your interest in contributing to CODE! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/CODE.git
   cd CODE
   ```

3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

## Development Process

### Setting Up Your Development Environment

1. Install the package in editable mode with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

2. Install pre-commit hooks (optional but recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

### Making Changes

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following the code style guidelines

3. Write or update tests for your changes

4. Run tests to ensure everything works:
   ```bash
   pytest tests/
   ```

5. Format your code:
   ```bash
   black src/CODE/
   ```

6. Check for common issues:
   ```bash
   flake8 src/CODE/
   ```

### Submitting Changes

1. Commit your changes with a clear commit message:
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```

2. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

3. Open a Pull Request on GitHub

## Code Style Guidelines

- Follow PEP 8 style guide for Python code
- Use meaningful variable and function names
- Write docstrings for all public functions, classes, and modules
- Keep functions focused and relatively small
- Add type hints where appropriate

### Docstring Format

Use Google-style docstrings:

```python
def example_function(arg1: int, arg2: str) -> bool:
    """
    Brief description of the function.
    
    More detailed description if needed.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When something goes wrong
    """
    pass
```

## Testing

- Write unit tests for new functionality
- Ensure existing tests pass
- Aim for good test coverage
- Test edge cases and error conditions

## Reporting Issues

When reporting issues, please include:

- A clear, descriptive title
- Detailed description of the issue
- Steps to reproduce the problem
- Expected behavior
- Actual behavior
- System information (OS, Python version, etc.)
- Relevant code snippets or error messages

## Feature Requests

We welcome feature requests! Please:

- Check if the feature has already been requested
- Clearly describe the feature and its use case
- Explain why it would be useful to the community

## Questions?

If you have questions, feel free to:

- Open an issue on GitHub
- Start a discussion in the GitHub Discussions tab

## License

By contributing to CODE, you agree that your contributions will be licensed under the MIT License.

## Acknowledgments

Thank you for helping improve CODE! Your contributions make this project better for everyone.
