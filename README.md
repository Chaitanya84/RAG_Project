### Setting up a Virtual Environment

To create and activate a virtual environment for this project, follow these steps:

1. **Create the virtual environment**:
    ```bash
    python3 -m venv venv
    ```

2. **Activate the virtual environment**:
    - On Linux/macOS:
      ```bash
      source venv/bin/activate
      ```
    - On Windows:
      ```bash
      .\venv\Scripts\activate
      ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

Remember to activate the virtual environment each time you work on the project.
### Installing requirement
To install the required dependencies for this project, ensure you are in the virtual environment and run:

```bash
pip install -r requirements.txt
```

This will install all the necessary packages listed in the `requirements.txt` file.


### Deactivating the Virtual Environment

To deactivate the virtual environment, simply run:

```bash
deactivate
```

This will return you to the system's default Python environment.