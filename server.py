from fastapi import FastAPI
import json

app = FastAPI()

@app.get("/regime")
def get_regime():
    with open("../output/regime.json") as f:
        return json.load(f)
