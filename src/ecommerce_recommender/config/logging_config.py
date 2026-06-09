import logging
import sys
from datetime import UTC, datetime

from pythonjsonlogger import json


class JsonFormatter(json.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record["timestamp"] = datetime.now(UTC).isoformat()
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["servico"] = "ecommerce_recommender_api"

        if not log_record.get("mensagem"):
            log_record["mensagem"] = record.getMessage()

def setup_api_logger():
    logger = logging.getLogger("ecommerce_recommender_api")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = JsonFormatter()
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)
    return logger
