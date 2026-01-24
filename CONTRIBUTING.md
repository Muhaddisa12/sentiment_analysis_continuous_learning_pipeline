# Contributing to Self-Training Sentiment Analysis System

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## 🎯 Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help maintain code quality and documentation
- Follow the project's coding standards

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- MySQL database (for full functionality)
- Understanding of machine learning concepts

### Development Setup

1. **Fork the repository** and clone your fork:
```bash
git clone https://github.com/your-username/Setiment_analysis.git
cd Setiment_analysis
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up your development environment**:
   - Copy `config.py` and update with your database credentials
   - Ensure MySQL is running and database is set up
   - Run initial training: `python -c "from ml.trainer import train_pipeline; train_pipeline(force_retrain=True)"`

## 📝 Development Workflow

### Branch Naming

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions/updates

### Making Changes

1. **Create a feature branch**:
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes**:
   - Write clean, documented code
   - Follow existing code style
   - Add comments for complex logic
   - Update relevant documentation

3. **Test your changes**:
   - Ensure existing functionality still works
   - Test edge cases
   - Check for linting errors

4. **Commit your changes**:
```bash
git add .
git commit -m "feat: add your feature description"
```

### Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

Examples:
```
feat: add new drift detection method using KS test
fix: resolve model loading error on Windows
docs: update API documentation
refactor: improve model lifecycle management
```

### Pull Request Process

1. **Update your branch**:
```bash
git fetch origin
git rebase origin/main
```

2. **Push your branch**:
```bash
git push origin feature/your-feature-name
```

3. **Create a Pull Request** on GitHub:
   - Provide a clear title and description
   - Reference any related issues
   - Include screenshots for UI changes
   - Ensure all CI checks pass

4. **Respond to feedback**:
   - Address review comments
   - Update your PR as needed
   - Keep your branch up to date

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_drift.py

# Run with coverage
pytest --cov=ml --cov=utils
```

### Writing Tests

- Place tests in the `tests/` directory
- Follow naming convention: `test_*.py`
- Write unit tests for new functions
- Include integration tests for complex workflows
- Aim for >80% code coverage

## 📐 Code Style

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 100 characters (soft limit)
- **Indentation**: 4 spaces
- **Imports**: Grouped (stdlib, third-party, local)
- **Docstrings**: Google style

### Example

```python
"""
Module docstring explaining the purpose.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from ml.models import MODELS
from config import DRIFT_THRESHOLD


def my_function(param1, param2):
    """
    Function description.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
    """
    # Implementation
    pass
```

### Linting

We use:
- **Black** for code formatting
- **flake8** for linting
- **mypy** for type checking (optional)

Run before committing:
```bash
black .
flake8 .
```

## 📚 Documentation Standards

### Code Documentation

- **Module docstrings**: Explain purpose and usage
- **Function docstrings**: Google style with Args, Returns, Raises
- **Inline comments**: Explain "why", not "what"
- **Type hints**: Use where helpful (Python 3.8+)

### Documentation Files

- Update README.md for user-facing changes
- Update ARCHITECTURE.md for structural changes
- Update API.md for API changes
- Add examples for new features

## 🐛 Reporting Bugs

### Before Reporting

1. Check existing issues
2. Verify the bug on the latest version
3. Gather relevant information

### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce:
1. ...
2. ...

**Expected behavior**
What you expected to happen.

**Environment**
- OS: [e.g., Windows 10]
- Python version: [e.g., 3.9]
- Package versions: [from pip freeze]

**Additional context**
Screenshots, logs, etc.
```

## 💡 Feature Requests

### Before Requesting

1. Check if the feature already exists
2. Consider if it fits the project scope
3. Think about implementation approach

### Feature Request Template

```markdown
**Feature Description**
Clear description of the feature.

**Use Case**
Why is this feature needed?

**Proposed Solution**
How should it work?

**Alternatives Considered**
Other approaches you've thought about.

**Additional Context**
Any other relevant information.
```

## 🔬 Research Contributions

This project emphasizes research-grade code. When contributing:

1. **Document research rationale**: Explain why your approach is chosen
2. **Cite relevant papers**: If implementing from research
3. **Include experiments**: Show results if adding new methods
4. **Maintain reproducibility**: Ensure experiments can be reproduced

## 📋 Review Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guidelines
- [ ] Tests pass and coverage is maintained
- [ ] Documentation is updated
- [ ] Commit messages follow conventions
- [ ] No hardcoded credentials or paths
- [ ] Error handling is appropriate
- [ ] Logging is added where needed
- [ ] Code is commented appropriately

## 🎓 Learning Resources

- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [MLOps Best Practices](https://ml-ops.org/)
- [Drift Detection Papers](https://arxiv.org/search/?query=concept+drift+detection)
- [Flask Documentation](https://flask.palletsprojects.com/)

## 📞 Getting Help

- **Issues**: Open a GitHub issue
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers (if listed)

## 🙏 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Credited in release notes
- Acknowledged in relevant documentation

Thank you for contributing to this project! 🚀
