# Airwindows Plugins Expert
Chatbot providing information about Airwindows plugins. The chatbot 
is created with LangChain framework as a RAG QA Bot. The information 
for generation is taken from [Airwindopedia](https://www.airwindows.com/wp-content/uploads/Airwindopedia.txt)

## Prerequisites
At the moment Ollama with gemma3:1b is used for text generation. Please
make sure that Ollama is installed in your environment and gemma3:1b
is pulled.

## To run locally
1. [Install Ollama](https://ollama.com/download)
2. Run gemma3:1b 
```bash
   ollama run gemma3:1b
```
3. Clone this repo and cd to its directory
4. Start virtual environment
```bash
   python3 -m venv venv
   source venv/bin/activate
 ```
5. Install dependencies
```bash
   pip install -r requirements.txt
```
6. Start application
```shell
   python main.py
```
7. Open [http://localhost:7860](http://localhost:7860) in your browser
8. Ask questions