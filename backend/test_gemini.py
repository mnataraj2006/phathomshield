import os, sys
from dotenv import load_dotenv
load_dotenv('.env')
key = os.environ.get('GEMINI_API_KEY', 'NOT_FOUND')
print(f'API Key: {key[:12]}...{key[-4:]}')

try:
    from google import genai
    from google.genai import types
    print('google-genai imported OK')
    client = genai.Client(api_key=key)
    resp = client.models.generate_content(
        model='gemini-2.0-flash',
        contents='Say: CONNECTED',
        config=types.GenerateContentConfig(max_output_tokens=10)
    )
    print(f'Gemini says: {resp.text}')
except Exception as e:
    print(f'ERROR type : {type(e).__name__}')
    print(f'ERROR detail: {e}')
