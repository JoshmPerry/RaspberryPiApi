from fastapi import FastAPI
import uvicorn
from controllers.RoutesHandler import RoutesHandler

app = FastAPI()

routeHandler = RoutesHandler()
routeHandler.register_routes(app)

if __name__ == '__main__':
    uvicorn.run("App:app", host='0.0.0.0', reload=True, port=8000)