import json
import time

import numpy as np
import pandas as pd
from fastapi import FastAPI, Response, status
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from ecommerce_recommender.config.api_logging_middleware import (
    LoggingMiddleware,
)
from ecommerce_recommender.config.logging_config import setup_api_logger
from ecommerce_recommender.prometheus.metrics import (
    AVG_CONFIDENCE,
    PREDICTION_DURATION,
    PREDICTIONS_TOTAL,
)
from ecommerce_recommender.recommendation import recommend_top_n
from ecommerce_recommender.training import (
    MLFLOW_TRACKING_URI,
)
from ml_prep_kit import (
    ModelPredictor,
)

LOGGER = setup_api_logger()

MODEL_NAME = "ecommerce_recommender_pytorch_mlp"
MODEL_ALIAS = "production"

def _load_predictor() -> object:
    predictor = None
    
    try:
        predictor = ModelPredictor(tracking_uri=MLFLOW_TRACKING_URI)
        predictor.find_model(
            registered_model_name=MODEL_NAME,
            model_alias=MODEL_ALIAS,
        )
        predictor.load_preprocessor()
        predictor.load_pytorch_model()

    except Exception as e:
        LOGGER.error(f"Erro ao carregar o modelo: {e}")

    return predictor


predictor = _load_predictor()

tags_metadata = [
    {
        "name": "aisles",
        "description": (
            "Endpoints relacionados à obtenção de"
            " informações sobre os corredores dos produtos."
        )
    },
    {
        "name": "departments",
        "description": (
            "Endpoints relacionados à obtenção de"
            " informações sobre os departamentos dos produtos."
        )
    },
    {
        "name": "recomendações",
        "description": (
            "Endpoints relacionados à geração de"
            " recomendações para os usuários."
        )
    },
]

app = FastAPI(openapi_tags=tags_metadata)
app.add_middleware(LoggingMiddleware)
Instrumentator().instrument(app).expose(
    app, 
    endpoint="/metrics", 
    summary="Métricas de desempenho da API para Prometheus"
)

@app.middleware("http")
async def not_loaded_model_middleware(request, call_next):
    if predictor is None:
        return Response(
            content=json.dumps({
                "status": "erro", 
                "mensagem": "Falha na carga do modelo"
            }),
            media_type="application/json",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    response = await call_next(request)
    return response

class RecommendationRequest(BaseModel):
    user_id: int
    product_id: int
    
    candidate_from_history: bool
    candidate_from_cooccurrence: bool
    candidate_from_favorite_category: bool
    candidate_from_similar_users: bool
    candidate_was_previously_purchased: bool
    candidate_is_new_product_for_user: bool
    is_favorite_department: bool
    is_favorite_aisle: bool

    cooccurrence_score: float
    category_popularity_score: float
    similar_user_score: float
    purchase_count: int
    reorder_rate: int
    avg_cart_position: float
    first_order_number: float
    last_order_number: float
    orders_since_last_purchase: float
    purchase_frequency: int
    user_total_orders: int
    user_total_items: int
    user_unique_products: int
    user_reorder_rate: int
    user_avg_cart_position: float
    user_avg_days_between_orders: float
    user_avg_order_hour: float
    user_avg_basket_size: int
    product_total_orders: int
    product_unique_users: int
    product_reorder_rate: float
    product_avg_cart_position: float
    user_department_purchase_count: int
    user_department_purchase_rate: int
    user_aisle_purchase_count: int
    user_aisle_purchase_rate: int

    aisle: str
    department: str

    def to_dict(self) -> dict[str, object]:
        return {
            "user_id":
                self.user_id,
            "product_id":
                self.product_id,
            "candidate_from_history":
                self.candidate_from_history,
            "candidate_from_cooccurrence":
                self.candidate_from_cooccurrence,
            "candidate_from_favorite_category":
                self.candidate_from_favorite_category,
            "candidate_from_similar_users":
                self.candidate_from_similar_users,
            "candidate_was_previously_purchased":
                self.candidate_was_previously_purchased,
            "candidate_is_new_product_for_user":
                self.candidate_is_new_product_for_user,
            "is_favorite_department":
                self.is_favorite_department,
            "is_favorite_aisle":
                self.is_favorite_aisle,
            "cooccurrence_score":
                self.cooccurrence_score,
            "category_popularity_score":
                self.category_popularity_score,
            "similar_user_score":
                self.similar_user_score,
            "purchase_count":
                self.purchase_count,
            "reorder_rate":
                self.reorder_rate,
            "avg_cart_position":
                self.avg_cart_position,
            "first_order_number":
                self.first_order_number,
            "last_order_number":
                self.last_order_number,
            "orders_since_last_purchase":
                self.orders_since_last_purchase,
            "purchase_frequency":
                self.purchase_frequency,
            "user_total_orders":
                self.user_total_orders,
            "user_total_items":
                self.user_total_items,
            "user_unique_products":
                self.user_unique_products,
            "user_reorder_rate":
                self.user_reorder_rate,
            "user_avg_cart_position":
                self.user_avg_cart_position,
            "user_avg_days_between_orders":
                self.user_avg_days_between_orders,
            "user_avg_order_hour":
                self.user_avg_order_hour,
            "user_avg_basket_size":
                self.user_avg_basket_size,
            "product_total_orders":
                self.product_total_orders,
            "product_unique_users":
                self.product_unique_users,
            "product_reorder_rate":
                self.product_reorder_rate,
            "product_avg_cart_position":
                self.product_avg_cart_position,
            "user_department_purchase_count":
                self.user_department_purchase_count,
            "user_department_purchase_rate":
                self.user_department_purchase_rate,
            "user_aisle_purchase_count":
                self.user_aisle_purchase_count,
            "user_aisle_purchase_rate":
                self.user_aisle_purchase_rate,
            "aisle":
                self.aisle,
            "department":
                self.department,
        }

class UserRecommendation(BaseModel):
    user_id: int
    product_id: int
    recommendation_score: float

class RecommendationResponse(BaseModel):
    recommendations: list[UserRecommendation]

class ResponseError(BaseModel):
    status: str
    mensagem: str


@app.get("/health", summary="Verificar saúde da API")
def read_health() -> dict[str, str] | ResponseError:
    return {"status": "sucesso"}

@app.get(
    "/aisles", 
    tags=["aisles"], 
    summary="Obter lista de corredores"
)
def read_aisles() -> list[str] | ResponseError:
    return predictor.get_categorical_column_values("aisle")

@app.get(
    "/departments", 
    tags=["departments"], 
    summary="Obter lista de departamentos"
)
def read_departments() -> list[str] | ResponseError:
    return predictor.get_categorical_column_values("department")

@app.post(
    "/recomendacoes", 
    tags=["recomendações"], 
    summary="Gerar recomendações para os usuários"
)
def read_recommendations(
    prediction_request: list[RecommendationRequest],
    response: Response,
) -> RecommendationResponse | ResponseError:
    if len(prediction_request) == 0:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return { 
            "status": "erro", 
            "mensagem": (
                "A lista de requisições de recomendação"
                " não pode estar vazia." 
            )
        }

    data_dict = [req.to_dict() for req in prediction_request]
    data = pd.DataFrame(
        [list(d.values()) for d in data_dict], 
        columns=list(data_dict[0].keys())
    )

    start_time = time.perf_counter()
    recommendations, proba = recommend_top_n(
        predictor=predictor,
        candidates=data,
        top_n=1,
    )
    duration = time.perf_counter() - start_time

    response = [
        UserRecommendation(**rec) 
            for rec 
            in recommendations.to_dict(orient='records')
        ]

    LOGGER.info(
        "predicao_gerada",
        extra={
            "duracao": duration / len(prediction_request),
            "confianca": np.mean(proba),
            "dados_entrada": data_dict,
            "recomendacoes": response,
        },
    )

    PREDICTION_DURATION.observe(duration / len(prediction_request))
    PREDICTIONS_TOTAL.inc(len(prediction_request))
    AVG_CONFIDENCE.observe(np.mean(proba))

    return RecommendationResponse(recommendations=response)
