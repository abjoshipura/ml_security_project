# Group 5: RAG-based Unlearning

**Creator:** Akshara Joshipura

## Setup Instructions

### Windows
**Important:** This project requires **Python < 3.13 and >= 3.10**. The dependency comes from `llm_guard`, which relies on an older version of `sentencepiece` that does not support Python 3.13. I discovered this compatibility issue on 11/29/2025 while testing the project on a friend's Windows machine.

To avoid installation issues, please install **Python 3.12.4**:

https://www.python.org/ftp/python/3.12.4/python-3.12.4-amd64.exe

Thank you for your understanding.

After installing Python, run the setup script:

1. Copy the template.env file into .env:
```bash
copy template.env .env
```

2. Fill the newly created .env with the API Keys (see below for details)

3. Run the setup script:
```bash
setup.bat
```

4. Activate the virtual environment:
```bash
.venv\Scripts\activate.bat
```

5. Run the script(s) of your choice!

### Mac/Linux
1. Copy the template.env file into .env:
```bash
cp template.env .env
```

2. Fill the newly created .env with the API Keys (see below for details)

3. Make the script executable and run it:
```bash
chmod +x setup.sh
./setup.sh
```

4. Activate the virtual environment:
```bash
source .venv/bin/activate
```

5. Run the script(s) of your choice!

## API Keys Required

The setup script will prompt you to add two API keys to the `.env` file:

1. **OpenAI API Key** (for GPT-4o)
   - Get your key here: https://platform.openai.com/api-keys

2. **Google API Key** (for Gemini 2.5 Flash)
   - Get your key here: https://aistudio.google.com/app/apikey
