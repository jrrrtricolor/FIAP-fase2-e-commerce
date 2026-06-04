"""Executa o fluxo principal de treino do Tech Challenge Fase 2."""

import logging
import sys

import pandas as pd
from colorama import Fore, Style, init

from ecommerce_recommender import torch_training, training

LOGGER = logging.getLogger("ecommerce_recommender.main")


def main() -> None:
    """Executa os treinamentos e exibe uma tabela final.

    Exemplo:
        poetry run python -m ecommerce_recommender.main
    """
    init(autoreset=True)
    configure_console_logging()

    # Executar modelos clássicos usados como baseline comparativo.
    classic_report = training.run_training(configurar_logs=False)

    # Executar modelo neural PyTorch.
    torch_result = torch_training.run_training(configurar_logs=False)

    # Consolidar todos os resultados em uma tabela única.
    final_report = pd.concat(
        [classic_report, pd.DataFrame([torch_result])],
        ignore_index=True,
    )

    # Exibir resultados finais no terminal.
    show_training_report(final_report)


def configure_console_logging() -> None:
    """Configura logs simples para leitura no terminal.

    Exemplo:
        configure_console_logging()
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(levelname)s | %(name)s | %(message)s")
    )

    logging.getLogger().handlers.clear()
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)


def show_training_report(report: pd.DataFrame) -> None:
    """Exibe a tabela final de resultados dos modelos.

    Exemplo:
        show_training_report(report)
    """
    metric = training.BEST_MODEL_METRIC
    ordered_report = report.sort_values(metric, ascending=False).reset_index(
        drop=True,
    )

    # Marcar o melhor modelo de acordo com a métrica definida.
    ordered_report["melhor_modelo"] = ""
    ordered_report.loc[0, "melhor_modelo"] = "SIM"

    # Selecionar as colunas principais para leitura no terminal.
    display_columns = [
        "melhor_modelo",
        "model_name",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "train_loss",
        "run_id",
    ]
    display_columns = [
        column for column in display_columns if column in ordered_report
    ]

    output = ordered_report[display_columns].rename(
        columns={
            "melhor_modelo": "Melhor",
            "model_name": "Modelo",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "roc_auc": "ROC AUC",
            "train_loss": "Train loss",
            "run_id": "Run ID",
        }
    )
    output = format_report_for_terminal(output)

    best_model = ordered_report.iloc[0]

    print()
    print(color_text("Resumo dos modelos treinados", Fore.CYAN))
    print(
        color_text(
            f"Critério para melhor modelo: {metric}",
            Fore.YELLOW,
        )
    )
    print(build_table(output))
    print(
        color_text(
            "\nMelhor modelo: "
            f"{best_model['model_name']} "
            f"({metric}={best_model[metric]:.4f})",
            Fore.GREEN,
        )
    )

    LOGGER.info("Tabela final gerada com sucesso.")


def format_report_for_terminal(report: pd.DataFrame) -> pd.DataFrame:
    """Formata resultados para exibição no terminal.

    Exemplo:
        output = format_report_for_terminal(report)
    """
    report = report.copy()
    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC AUC",
        "Train loss",
    ]

    # Formatar métricas numéricas com quatro casas decimais.
    for column in metric_columns:
        if column in report:
            report[column] = report[column].apply(format_metric_value)

    # Encurtar o ID da execução para manter a tabela legível.
    if "Run ID" in report:
        report["Run ID"] = report["Run ID"].astype(str).str[:8]

    # Substituir valores ausentes por traço.
    return report.fillna("-")


def build_table(report: pd.DataFrame) -> str:
    """Monta uma tabela simples para exibição no terminal.

    Exemplo:
        text = build_table(report)
    """
    headers = list(report.columns)
    rows = report.astype(str).values.tolist()
    widths = calculate_column_widths(headers, rows)

    separator = build_separator(widths)
    lines = [
        separator,
        color_text(build_row(headers, widths), Fore.CYAN),
        separator,
    ]

    for row in rows:
        line = build_row(row, widths)
        if row[0] == "SIM":
            line = color_text(line, Fore.GREEN)
        lines.append(line)

    lines.append(separator)
    return "\n".join(lines)


def calculate_column_widths(
    headers: list[str],
    rows: list[list[str]],
) -> list[int]:
    """Calcula a largura de cada coluna da tabela.

    Exemplo:
        widths = calculate_column_widths(headers, rows)
    """
    widths = []

    for index, header in enumerate(headers):
        values = [row[index] for row in rows]
        widths.append(max(len(header), *(len(value) for value in values)))

    return widths


def build_separator(widths: list[int]) -> str:
    """Monta a linha separadora da tabela.

    Exemplo:
        separator = build_separator([6, 10])
    """
    parts = ["-" * (width + 2) for width in widths]
    return "+" + "+".join(parts) + "+"


def build_row(values: list[str], widths: list[int]) -> str:
    """Monta uma linha da tabela.

    Exemplo:
        row = build_row(["SIM", "modelo"], [6, 10])
    """
    cells = []

    for value, width in zip(values, widths, strict=True):
        cells.append(f" {value:<{width}} ")

    return "|" + "|".join(cells) + "|"


def color_text(text: str, color: str) -> str:
    """Aplica cor em um texto do terminal.

    Exemplo:
        text = color_text("sucesso", Fore.GREEN)
    """
    return f"{color}{text}{Style.RESET_ALL}"


def format_metric_value(value: float) -> str:
    """Formata uma métrica numérica para a tabela final.

    Exemplo:
        value = format_metric_value(0.91234)
    """
    if pd.isna(value):
        return "-"

    return f"{value:.4f}"


if __name__ == "__main__":
    main()
