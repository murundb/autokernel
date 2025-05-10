# Project README

## Getting Started

Follow these steps to set up and run the project:

### 1. Environment Variables

Create a `.env` file in the root directory and add the following variables:

```
ANTHROPIC_API_KEY=
LLM_PROVIDER=           # Options: anthropic, openai
OPENAI_API_KEY=
AUTOKERNEL_BACKEND=     # Options: cuda, opencl
```

### 2. Set Up the Conda Environment

Install the required dependencies by creating a conda environment:

```
conda env create -f environment.yml
```

### 3. Activate the Environment

Activate your newly created conda environment:

```
conda activate <your-environment-name>
```

### 4. Run the Project

Start the application with:

```
python main.py
```

## Debugging in VSCode

To debug the project using Visual Studio Code and your conda environment:

1. Create a launch configuration specifying the Python interpreter from your conda environment:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: main.py (conda env)",
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

2. Save this configuration as `.vscode/launch.json` in your project directory.

3. Select your conda environment in VSCode:
   - Open the command palette (Ctrl+Shift+P or Cmd+Shift+P on macOS)
   - Type "Python: Select Interpreter"
   - Choose the appropriate conda environment from the list

---

For additional information, please refer to the project documentation or open an issue if you encounter any problems.
