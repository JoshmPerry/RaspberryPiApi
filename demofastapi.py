from fastapi import FastAPI
import uvicorn

app = FastAPI()

myglobalVars={
    'hi':'car'
}

def obsurefunc(myString):
    return myString*2


@app.get("/")
async def read_root():
    return {"message":"Welcome to my api!"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}


@app.get("/recognizeDigit")
async def read_item(vector: str):
    returnvalue=[]
    temp = vector.split(',')
    for i in temp:
        returnvalue.append(float(i))
    return {"recievedItem": returnvalue}

@app.get("/localization")
async def read_item():
    return {"computed": obsurefunc(myglobalVars['hi'])}


if __name__ == '__main__':
    uvicorn.run("demofastapi:app", host='0.0.0.0', reload=True, port=8000)
