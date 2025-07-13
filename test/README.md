# AudioMuse-AI Developer Tests

These integration tests are for developer purposes to verify the functionality of the AudioMuse-AI API endpoints. They are not included in the production Docker container and should be run from a local development machine.

## Prerequisites

- Python 3.8+
- `pip` and `venv`
- Jellyfin installed with 40+ song albums
- Have already run the analysis of 40+ albums with AudioMuse-AI (the analysis test jur run 1 album for test, it is not enough for other test)

## Setup Instructions

You can choose to clone the entire repository or just the `test` directory.

### Clone the  Repository

This is the simplest method.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/NeptuneHub/AudioMuse-AI.git
    ```

2.  **Navigate to the test directory:**
    ```bash
    cd AudioMuse-AI/test
    ```

### Environment Setup (from within the `test` directory)

Once you are inside the `test` directory, follow these steps:

1.  **Create and activate a Python virtual environment:**

    *   **On macOS/Linux:**
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        ```
    *   **On Windows:**
        ```bash
        python -m venv .venv
        .\.venv\Scripts\activate
        ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure the API endpoint:**
    Open the `test.py` file and update the `BASE_URL` to point to your running AudioMuse-AI instance.
    ```python
    # test/test.py
    BASE_URL = 'http://YOUR_AUDIOMUSE_IP:8000'
    ```

## Running the Tests

To run all tests, execute the following command from your terminal (while in the `test` directory with the virtual environment activated):

```bash
python test.py
```