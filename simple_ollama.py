from ollama import chat
from ollama import ChatResponse
from colorama import Fore, Style

def query_ollama(prompt: str) -> ChatResponse:
    response: ChatResponse = chat(
        model='nemotron-3-nano:30b',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response

if __name__ == '__main__':
    print(f'{Fore.GREEN}Welcome to the OLLAMA CLI!{Style.RESET_ALL}')
    print(f'{Fore.GREEN}Type "exit", "quit", "bye", or "q" to exit.{Style.RESET_ALL}')
    print(f'{Fore.YELLOW}If this is the first time you are using OLLAMA, please wait for the model to load. Larger models may take a few minutes to load.{Style.RESET_ALL}')
    while True:
        prompt = input(f'{Fore.YELLOW}>> {Style.RESET_ALL}')
        if prompt.lower() in ['exit', 'quit', 'bye', 'q']:
            break
        try:
            response = query_ollama(prompt)
            print(f'{Fore.GREEN}{response.message.content}{Style.RESET_ALL}')
        except Exception as e:
            print(f'{Fore.RED}Error: {e}{Style.RESET_ALL}')