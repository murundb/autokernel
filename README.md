# Read Me

## List

```
conda env create -f environment.yml
```

Create a file called `.env` and add Anthropic key as `ANTHROPIC_API_KEY="your-key"`.


## VSCode Debugging Setup

For debugging the project in VSCode with your conda environment:

1. Create a launch configuration that specifies the Python interpreter from your conda environment directly:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: main.py (threedai env)",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/main.py",
            "console": "integratedTerminal",
            "justMyCode": false,
            "python": "${command:python.interpreterPath}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            }
        }
    ]
}
```

2. Save this as `.vscode/launch.json` in your project directory.

3. Then use the VSCode Python extension to select your conda environment:
   - Press Ctrl+Shift+P (or Cmd+Shift+P on macOS)
   - Type "Python: Select Interpreter"
   - Choose the desired conda environment from the list
