# Read Me

## Running

To run the project initially:

1. Create a new file in the root directory called `.env` and add the following variables:
```
ANTHROPIC_API_KEY=
LLM_PROVIDER=
OPENAI_API_KEY=
AUTOKERNEL_BACKEND=
```

2. Create the conda environment:
```
conda env create -f environment.yml
```

3. Activate the conda environment:
```
conda activate <your-environment-name>
```

4. Run the project:
```
python main.py
```

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
