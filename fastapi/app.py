from fastapi import Response, FastAPI
import requests

app = FastAPI()

@app.get('/embed')
async def embedding_text():
    res = requests.post("http://localhost:11434/api/embeddings", 
                        json={
                            "model": "all-minilm",
                            "input": "Why is the sky blue?"
                        })
    return Response(content=res.text, media_type="application/json")