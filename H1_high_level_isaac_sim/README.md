# Part 1: LLM and VLM integration with Whisper transcription

## 🏗️  Codebase structure

```shell
.
├── main.py
├── src/
│   ├── mistral_ai/
│   │   ├── prompts/
│   │   │   ├── recipe_prompt.py # for LLM
│   │   │   └── vision_prompt.py # for VLM
│   │   ├── scripts/
│   │   │   ├── llm_script.txt # LLM reponses history
│   │   │   ├── vlm_script.json # JSON extract from VLM history
│   │   │   └── vlm_script.txt # VLM reponses history
│   │   ├── mistral.py # api and structure
│   │   ├── llm.py
│   │   └── vlm.py
│   ├── transcribe/
│   │   ├── sst.py 
│   │   ├── tts.py
│   │   └── transcription.txt
│   └── utils.py
├── main.py
├── run.sh
└── requirements.txt
```

## 🔧 Installation guide

1. **Clone the repository**:

2. **Set up a virtual environment**:
   ```bash
   # on macos
   brew install uv
   # on Ubuntu
   sudo apt-get install uv
   # on Arch Linux
   sudo pacman -S uv

   uv sync
   ```

3. **Install dependencies**:
   ```bash
   uv pip install -r requirements.txt
   ```

4. **Set environment variables in shell config**:
   ```bash
   export MISTRAL_API_KEY="your_mistral_api_key_here"
   ```

   ## 🎯 Operation guide
1. **Run the application**:
   ```bash
   ./run.sh
   ```
#TODO
