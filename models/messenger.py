from pydantic import BaseModel, Field


class Message(BaseModel):
    text: str = Field(examples=["ano po ang mga requiremest para sa tourist visa po?"])

    model_config = {
        "json_schema_extra": {
            "examples": [{"text": "ano po ang mga requirements para sa tourist po?"}]
        }
    }
