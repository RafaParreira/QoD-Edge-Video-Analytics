
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Diretórios do projeto
RAW_DIR = Path("results/raw")
PROCESSED_DIR = Path("results/processed")
FIGURES_DIR = Path("results/figures")

# Deadlines utilizados no cálculo do DMR
DEADLINES_MS = [100, 150, 200]


def calculate_metrics(data, dataset_name):
    """
    Calcula estatísticas descritivas das métricas
    de tempo HTTP e inferência YOLO.
    """

    metrics = []

    columns = {
        "request_time_ms": "HTTP response time",
        "inference_time_ms": "YOLO inference time",
    }

    for column, description in columns.items():

        values = data[column].dropna()

        metrics.append({
            "dataset": dataset_name,
            "metric": description,
            "count": len(values),
            "mean_ms": values.mean(),
            "std_ms": values.std(),
            "min_ms": values.min(),
            "p50_ms": values.quantile(0.50),
            "p95_ms": values.quantile(0.95),
            "p99_ms": values.quantile(0.99),
            "max_ms": values.max(),
        })

    return metrics


def calculate_dmr(data, dataset_name):
    """
    Calcula o Deadline Miss Ratio (DMR).

    Uma requisição é considerada uma violação quando:
    - O tempo de resposta ultrapassa o deadline;
    - A requisição não obteve sucesso;
    - O tempo de resposta é inválido ou inexistente.

    O DMR é expresso em porcentagem.
    """

    results = []

    total_requests = len(data)

    for deadline in DEADLINES_MS:

        misses = (
            (data["request_time_ms"] > deadline)
            | (data["status"] != "success")
            | (data["request_time_ms"].isna())
        )

        deadline_misses = int(misses.sum())

        dmr = (
            deadline_misses / total_requests * 100
            if total_requests > 0
            else 0
        )

        results.append({
            "dataset": dataset_name,
            "deadline_ms": deadline,
            "total_requests": total_requests,
            "deadline_misses": deadline_misses,
            "deadline_compliant": total_requests - deadline_misses,
            "dmr_percent": round(dmr, 3),
        })

    return results


def main():

    parser = argparse.ArgumentParser(
        description="Análise dos resultados Baseline."
    )

    parser.add_argument(
        "csv_file",
        nargs="?",
        help=(
            "Caminho do arquivo CSV. "
            "Se omitido, utiliza o Baseline mais recente."
        )
    )

    parser.add_argument(
        "--keep-first-frame",
        action="store_true",
        help="Inclui o primeiro frame na análise principal."
    )

    args = parser.parse_args()

    # Cria os diretórios de saída, caso não existam
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Localiza o arquivo CSV
    if args.csv_file:

        csv_path = Path(args.csv_file)

    else:

        files = sorted(
            RAW_DIR.glob("baseline_*.csv"),
            key=lambda path: path.stat().st_mtime
        )

        if not files:
            raise FileNotFoundError(
                "Nenhum CSV Baseline encontrado em results/raw/."
            )

        csv_path = files[-1]

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {csv_path}"
        )

    print(f"\nAnalisando: {csv_path}")

    # Carrega os dados
    data = pd.read_csv(csv_path)

    # Verifica as colunas obrigatórias
    required_columns = [
        "frame_id",
        "request_time_ms",
        "inference_time_ms",
        "num_detections",
        "status",
    ]

    missing = set(required_columns) - set(data.columns)

    if missing:
        raise ValueError(
            f"Colunas ausentes no CSV: {missing}"
        )

    # Converte as métricas para valores numéricos
    for column in [
        "frame_id",
        "request_time_ms",
        "inference_time_ms",
        "num_detections",
    ]:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # Separa as requisições bem-sucedidas
    successful = data[
        data["status"] == "success"
    ].copy()

    # Identifica o primeiro frame do conjunto original
    first_frame_id = data["frame_id"].min()

    # Separa os dados do regime estável
    if args.keep_first_frame:

        steady_state = successful.copy()
        dmr_steady_state = data.copy()

    else:

        steady_state = successful[
            successful["frame_id"] != first_frame_id
        ].copy()

        dmr_steady_state = data[
            data["frame_id"] != first_frame_id
        ].copy()

    # Calcula as métricas estatísticas
    metrics = []

    metrics.extend(
        calculate_metrics(successful, "all_frames")
    )

    metrics.extend(
        calculate_metrics(steady_state, "steady_state")
    )

    summary = pd.DataFrame(metrics)

    # Salva o resumo estatístico
    summary_path = (
        PROCESSED_DIR / f"{csv_path.stem}_summary.csv"
    )

    summary.to_csv(summary_path, index=False)

    # Calcula o DMR para todos os frames e regime estável
    dmr_results = []

    dmr_results.extend(
        calculate_dmr(data, "all_frames")
    )

    dmr_results.extend(
        calculate_dmr(dmr_steady_state, "steady_state")
    )

    dmr_summary = pd.DataFrame(dmr_results)

    # Salva o relatório de DMR
    dmr_path = (
        PROCESSED_DIR / f"{csv_path.stem}_dmr.csv"
    )

    dmr_summary.to_csv(dmr_path, index=False)

    # Calcula indicadores de execução
    total_requests = len(data)
    successful_requests = len(successful)
    failed_requests = total_requests - successful_requests

    # Exibe o resumo experimental
    print("\n===== RESUMO EXPERIMENTAL =====")

    print(f"Total de requisições: {total_requests}")
    print(f"Sucessos: {successful_requests}")
    print(f"Falhas: {failed_requests}")

    if total_requests > 0:

        success_rate = (
            successful_requests / total_requests
        ) * 100

        print(f"Taxa de sucesso: {success_rate:.2f}%")

    print(f"Primeiro frame: {first_frame_id}")
    print(f"Frames na análise principal: {len(steady_state)}")

    # Exibe as métricas estatísticas
    print("\n===== MÉTRICAS =====")

    print(
        summary[
            [
                "dataset",
                "metric",
                "mean_ms",
                "p50_ms",
                "p95_ms",
                "p99_ms",
                "max_ms",
            ]
        ].to_string(index=False)
    )

    # Os gráficos utilizam o regime estável,
    # excluindo o primeiro frame, salvo opção contrária.

    # Gráfico 1: evolução do tempo HTTP
    plt.figure(figsize=(12, 5))

    plt.plot(
        steady_state["frame_id"],
        steady_state["request_time_ms"],
        linewidth=0.7,
    )

    plt.xlabel("Frame")
    plt.ylabel("HTTP response time (ms)")
    plt.title("Baseline - HTTP response time per frame")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / f"{csv_path.stem}_timeseries.png",
        dpi=300,
    )

    plt.close()

    # Gráfico 2: distribuição do tempo HTTP
    plt.figure(figsize=(9, 5))

    plt.hist(
        steady_state["request_time_ms"].dropna(),
        bins=40,
        edgecolor="black",
    )

    plt.xlabel("HTTP response time (ms)")
    plt.ylabel("Frequency")
    plt.title("Baseline - HTTP response time distribution")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / f"{csv_path.stem}_histogram.png",
        dpi=300,
    )

    plt.close()

    # Gráfico 3: comparação entre HTTP e inferência YOLO
    plt.figure(figsize=(8, 5))

    plt.boxplot(
        [
            steady_state["request_time_ms"].dropna(),
            steady_state["inference_time_ms"].dropna(),
        ],
        tick_labels=[
            "HTTP response",
            "YOLO inference",
        ],
    )

    plt.ylabel("Time (ms)")
    plt.title("Baseline - Response and inference time")
    plt.grid(True, axis="y")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / f"{csv_path.stem}_boxplot.png",
        dpi=300,
    )

    plt.close()

    # Exibe os resultados do DMR
    print("\n===== ANÁLISE DE DMR =====")

    print(
        dmr_summary[
            [
                "dataset",
                "deadline_ms",
                "total_requests",
                "deadline_misses",
                "deadline_compliant",
                "dmr_percent",
            ]
        ].to_string(index=False)
    )

    # Exibe os arquivos gerados
    print("\n===== ARQUIVOS GERADOS =====")

    print(f"Resumo estatístico: {summary_path}")
    print(f"Relatório DMR: {dmr_path}")
    print(f"Gráficos: {FIGURES_DIR}/")

    print("\nAnálise concluída!")


if __name__ == "__main__":
    main()
