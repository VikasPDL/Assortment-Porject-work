from fastapi import FastAPI

app = FastAPI(title="Assortment Project")


@app.get("/")
def read_root():
    return {"message": "Hello, world!"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}
