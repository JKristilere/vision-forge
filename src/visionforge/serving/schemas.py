from pydantic import BaseModel


class Prediction(BaseModel):
    class_index: int
    label: str
    confidence: float


class PredictResponse(BaseModel):
    prediction: Prediction
    top_k: list[Prediction]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    architecture: str
    num_parameters: int
    device: str
    checkpoint_path: str
    checkpoint_epoch: int | None
    checkpoint_val_loss: float | None
    class_names: list[str]
    image_size: int
    mean: list[float]
    std: list[float]
    top_k: int
