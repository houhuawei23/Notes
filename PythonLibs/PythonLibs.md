# Python Libs

## Misc

### [Pylint](https://pylint.pycqa.org/en/latest/index.html): Static Code Analyser

Additional tools in pylint

- [pyreverse](https://pylint.readthedocs.io/en/latest/additional_tools/pyreverse/index.html)(standalone tool that generates package and class diagrams.)
- [symilar](https://pylint.readthedocs.io/en/latest/additional_tools/symilar/index.html)(duplicate code finder that is also integrated in pylint)

#### pyreverse

```bash
pyreverse [options] <packages>

# eg.
# generate classes.dot and packages.dot of marko lib
pyreverse marko 
01_Misc dot -Tsvg classes.dot > classes.svg
01_Misc dot -Tsvg packages.dot > packages.svg
```

