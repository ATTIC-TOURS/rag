from fastapi import FastAPI, Query, Path, Body, Header, Cookie, status, HTTPException
from enum import Enum
from typing import Annotated, Literal
from pydantic import BaseModel, Field, HttpUrl


class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


class FilterParams(BaseModel):
    model_config = {"extra": "forbid"}

    limit: int = Field(100, gt=0, le=100)
    offset: int = Field(0, ge=0)
    order_by: Literal["created_at", "updated_at"] = "created_at"
    tags: list[str] = []


app = FastAPI()


# POST, GET, PUT, DELETE (http methods) (operations - in OpenAPI)
@app.get("/")  # path operation decorator
async def root() -> dict[str, str]:
    return {"message": "Hello World"}


@app.get("/kenji")
async def kenji() -> list[int]:
    return [
        1,
        2,
    ]


@app.get("/items/")
async def read_items(filter_query: Annotated[FilterParams, Query()]) -> FilterParams:
    return filter_query


@app.get("/items/{item_id}")
async def read_item(
    item_id: Annotated[int, Path(title="The ID of the item", ge=1, le=100)],
) -> dict[str, int]:
    return {"item_id": item_id}


@app.get("/models/{model_name}")
async def get_model(
    model_name: Annotated[ModelName, Path(title="The Enum Model")],
) -> dict[str, str | ModelName]:
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}

    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}
    return {"model_name": model_name, "message": "Have some residuals"}


@app.get("/files/{file_path:path}")
async def read_file(
    file_path: Annotated[str, Path(title="file path")],
) -> dict[str, str]:
    return {"file_path": file_path}


class Image(BaseModel):
    url: HttpUrl
    name: str


class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None
    tags: set[str] = set()
    image: Image | None = None


class User(BaseModel):
    username: str
    full_name: str | None = None


class UpdateItemResponse(BaseModel):
    item_id: int
    item: Item


# @app.put("/items/{item_id}", response_model=UpdateItemResponse)
# async def update_item(
#     item_id: Annotated[int, Path()], item: Annotated[Item, Body()]
# ) -> UpdateItemResponse:
#     results = {"item_id": item_id, "item": item}
#     return UpdateItemResponse(**results)


@app.put(
    "/items/{item_id}",
    response_model=UpdateItemResponse,
    status_code=status.HTTP_201_CREATED,
    summary="update the item",
    description="put the newer one to specified item id",
    tags=['items'],
    deprecated=True
)
async def update_item(
    item_id: Annotated[int, Path()],
    item: Annotated[
        Item,
        Body(
            openapi_examples={
                "normal": {
                    "summary": "A normal example",
                    "description": "A **normal** item works correctly.",
                    "value": {
                        "name": "Foo",
                        "description": "A very nice Item",
                        "price": 35.4,
                        "tax": 3.2,
                        "tags": ["fsad"],
                        "image": "",
                    },
                },
                "converted": {
                    "summary": "An example with converted data",
                    "description": "FastAPI can convert price `strings` to actual `numbers` automatically",
                    "value": {
                        "name": "Bar",
                        "price": "35.4",
                    },
                },
                "invalid": {
                    "summary": "Invalid data is rejected with an error",
                    "value": {
                        "name": "Baz",
                        "price": "thirty five point four",
                    },
                },
            },
        ),
    ],
    user_agent: Annotated[str | None, Header()] = None,
    user_profie: Annotated[str | None, Cookie()] = None,
) -> UpdateItemResponse:
    results = {"item_id": item_id, "item": item}
    return UpdateItemResponse(**results)
