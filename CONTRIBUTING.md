# Contributing to RL for Data Scientists

Thank you for contributing! Here's how to help:

## Types of Contributions

1. **Bug fixes**: typos, broken code, incorrect outputs
2. **Domain adaptations**: applying Chapter 18's pipeline to a new domain (e.g., clinical notes, legal text, financial analysis)
3. **Exercise solutions**: adding solution notebooks (`*_solutions.ipynb`) for exercises
4. **New exercises**: adding exercises aligned with chapter content

## Process

1. Fork the repository
2. Create a branch: `git checkout -b fix/description` or `feat/description`
3. Make changes and test locally
4. Submit a pull request with a clear description

## Notebook Standards

- Each notebook must run top-to-bottom without errors
- First cell: setup/imports with clear instructions
- Include expected output in markdown cells
- GPU-intensive cells should have CPU fallback or a clear skip instruction
- All code comments explain *why*, not just *what*

## Testing

```bash
pip install nbmake pytest
pytest --nbmake notebooks/chapter_XX_*/your_notebook.ipynb
```
