

import gradio as gr
from gradio.components import Markdown, Textbox, Button
import pandas as pd
import os

from tools import MLFunctionSelector


text_data = "Analysis Report:"
report_file = "data/reports/analysis-"

ml_function_selector = MLFunctionSelector()


# anomaly_detection, assoc_rule_mining, k_means_clustering, kde, pca, gan, lda, nmf, self_organizing_maps, word_embeddings, vae
def ml_function(selected_function=""):
    global ml_function_selector
    result = ml_function_selector.select_function(selected_function)
    print(result)
    yield result
    return result



def load_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading CSV: {str(e)}")
        return None

def analyze_dataset(df):
    global text_data
    if df is None:
        return
    text_data = "Analysis Report:"
    text_data += f"\n\nDataset Information:\n{df.info()}"
    text_data += f"\n\nSummary Statistics:\n{df.describe()}"
    text_data += f"\n\nColumn Data Types:\n{df.dtypes}"
    text_data += f"\n\nMissing Values:\n{df.isnull().sum()}"
    text_data += f"\n\nUnique Values:"
    for column in df.columns:
        text_data += f"\n{column}: {len(df[column].unique())} unique values"
        text_data += f"\nValue Counts for {column}: {df[column].value_counts()}"
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_columns) > 1:
        text_data += f"\n\nCorrelation between Numeric Columns:\n{df[numeric_columns].corr()}"


def save_file(data, filename=report_file):
    with open(filename, 'w') as f:
        f.write(str(data))

def load_dataset(csv_file) -> str:
    global text_data, report_file
    df = load_csv(csv_file)
    analyze_dataset(df)
    csv_name = report_file + csv_file.split("/")[3] + ".txt"
    save_file(text_data, csv_name)
    yield text_data

def get_datasets() -> list:
    return list(list_files("./data/datasets"))

def list_files(path_dir) -> list:
    file_paths = []
    for root, dirs, files in os.walk(path_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

def upload_files(files):
    file_paths = [file.name for file in files]
    return file_paths


with gr.Blocks(title="Data Viewer", analytics_enabled=False) as chatbot:
    with gr.Tab("Dataset Analysis"):
        gr.Markdown("# Data Viewer")
        gr.Markdown("Welcome to Data Viewer!")
        with gr.Row():
            with gr.Column(scale=1):
                dataset_dropdown = gr.Dropdown(get_datasets(), label="Dataset Directory:")
        with gr.Row():
            with gr.Column(scale=1):
                analysis_output = gr.Textbox(lines=10, label="Dataset Analysis Report:")
        with gr.Row():
            with gr.Column(scale=1):
                btn_analyze = gr.Button(value="Run & Save Analysis")
            with gr.Column(scale=1):
                btn_ml = gr.Button(value="Run & Save ML Report")
            btn_analyze.click(fn=load_dataset, inputs=dataset_dropdown, outputs=analysis_output)
            btn_ml.click(fn=load_dataset, inputs=dataset_dropdown)
    with gr.Tab("Anomaly Detection"):
        gr.Markdown("# Anomaly Detection")
        gr.Markdown("Welcome to Anomaly Detection!")
        with gr.Row():
            with gr.Column(scale=1):
                ml_input = gr.Textbox(value="anomaly_detection",  interactive=False)
                analysis_output = gr.Textbox(lines=10, label="Dataset Analysis Report:")
        with gr.Row():
            with gr.Column(scale=1):
                btn_ml = gr.Button(value="Run & Save ML Report")
            btn_ml.click(fn=ml_function, inputs=ml_input, outputs=analysis_output)
    with gr.Tab("Associative Rule Mining"):
        gr.Markdown("# Associative Rule Mining")
        gr.Markdown("Welcome to Associative Rule Mining!")
        with gr.Row():
            with gr.Column(scale=1):
                ml_input = gr.Textbox(value="assoc_rule_mining",  interactive=False)
                analysis_output = gr.Textbox(lines=10, label="Dataset Analysis Report:")
        with gr.Row():
            with gr.Column(scale=1):
                btn_ml = gr.Button(value="Run & Save ML Report")
            btn_ml.click(fn=ml_function, inputs=ml_input, outputs=analysis_output)
    with gr.Tab("Clustering"):
        gr.Markdown("# Clustering")
        gr.Markdown("Welcome to Clustering!")
        with gr.Row():
            with gr.Column(scale=1):
                ml_input = gr.Textbox(value="k_means_clustering",  interactive=False)
                analysis_output = gr.Textbox(lines=10, label="Dataset Analysis Report:")
        with gr.Row():
            with gr.Column(scale=1):
                btn_ml = gr.Button(value="Run & Save ML Report")
            btn_ml.click(fn=ml_function, inputs=ml_input, outputs=analysis_output)
    with gr.Tab("Density Estimation"):
        gr.Markdown("# Density Estimation")
        gr.Markdown("Welcome to Density Estimation!")
        with gr.Row():
            with gr.Column(scale=1):
                ml_input = gr.Textbox(value="kde",  interactive=False)
                analysis_output = gr.Textbox(lines=10, label="Dataset Analysis Report:")
        with gr.Row():
            with gr.Column(scale=1):
                btn_ml = gr.Button(value="Run & Save ML Report")
            btn_ml.click(fn=ml_function, inputs=ml_input, outputs=analysis_output)
    with gr.Tab("Dimensionality Reduction"):
        gr.Markdown("# Dimensionality Reduction")
        gr.Markdown("Welcome to Dimensionality Reduction!")
        with gr.Row():
            with gr.Column(scale=1):
                ml_input = gr.Textbox(value="pca",  interactive=False)
                analysis_output = gr.Textbox(lines=10, label="Dataset Analysis Report:")
        with gr.Row():
            with gr.Column(scale=1):
                btn_ml = gr.Button(value="Run & Save ML Report")
            btn_ml.click(fn=ml_function, inputs=ml_input, outputs=analysis_output)
    with gr.Tab("GAN"):
        gr.Markdown("# GAN")
        gr.Markdown("Welcome to GAN!")
        with gr.Row():
            with gr.Column(scale=1):
                ml_input = gr.Textbox(value="gan",  interactive=False)
                analysis_output = gr.Textbox(lines=10, label="Dataset Analysis Report:")
        with gr.Row():
            with gr.Column(scale=1):
                btn_ml = gr.Button(value="Run & Save ML Report")
            btn_ml.click(fn=ml_function, inputs=ml_input, outputs=analysis_output)
    with gr.Tab("LDA"):
        gr.Markdown("# LDA")
        gr.Markdown("Welcome to LDA!")
        with gr.Row():
            with gr.Column(scale=1):
                ml_input = gr.Textbox(value="lda",  interactive=False)
                analysis_output = gr.Textbox(lines=10, label="Dataset Analysis Report:")
        with gr.Row():
            with gr.Column(scale=1):
                btn_ml = gr.Button(value="Run & Save ML Report")
            btn_ml.click(fn=ml_function, inputs=ml_input, outputs=analysis_output)
    with gr.Tab("NMF"):
        gr.Markdown("# NMF")
        gr.Markdown("Welcome to NMF!")
        with gr.Row():
            with gr.Column(scale=1):
                ml_input = gr.Textbox(value="nmf",  interactive=False)
                analysis_output = gr.Textbox(lines=10, label="Dataset Analysis Report:")
        with gr.Row():
            with gr.Column(scale=1):
                btn_ml = gr.Button(value="Run & Save ML Report")
            btn_ml.click(fn=ml_function, inputs=ml_input, outputs=analysis_output)
    with gr.Tab("RNN"):
        gr.Markdown("# RNN")
        gr.Markdown("Welcome to RNN!")
        with gr.Row():
            with gr.Column(scale=1):
                ml_input = gr.Textbox(value="rnn",  interactive=False)
                analysis_output = gr.Textbox(lines=10, label="Dataset Analysis Report:")
        with gr.Row():
            with gr.Column(scale=1):
                btn_ml = gr.Button(value="Run & Save ML Report")
            btn_ml.click(fn=ml_function, inputs=ml_input, outputs=analysis_output)
    with gr.Tab("Self Organizing Maps"):
        gr.Markdown("# Self Organizing Maps")
        with gr.Row():
            with gr.Column(scale=1):
                ml_input = gr.Textbox(value="self_organizing_maps",  interactive=False)
                analysis_output = gr.Textbox(lines=10, label="Dataset Analysis Report:")
        gr.Markdown("Welcome to Self Organizing Maps!")
        with gr.Row():
            with gr.Column(scale=1):
                btn_ml = gr.Button(value="Run & Save ML Report")
            btn_ml.click(fn=ml_function, inputs=ml_input, outputs=analysis_output)
    with gr.Tab("VAE"):
        gr.Markdown("# VAE")
        gr.Markdown("Welcome to VAE!")
        with gr.Row():
            with gr.Column(scale=1):
                ml_input = gr.Textbox(value="vae",  interactive=False)
                analysis_output = gr.Textbox(lines=10, label="Dataset Analysis Report:")
        with gr.Row():
            with gr.Column(scale=1):
                btn_ml = gr.Button(value="Run & Save ML Report")
            btn_ml.click(fn=ml_function, inputs=ml_input, outputs=analysis_output)
    with gr.Tab("Embeddings"):
        gr.Markdown("# Embeddings")
        gr.Markdown("Welcome to Embeddings!")
        with gr.Row():
            with gr.Column(scale=1):
                ml_input = gr.Textbox(value="word_embeddings",  interactive=False)
                analysis_output = gr.Textbox(lines=10, label="Dataset Analysis Report:")
        with gr.Row():
            with gr.Column(scale=1):
                btn_ml = gr.Button(value="Run & Save ML Report")
            btn_ml.click(fn=ml_function, inputs=ml_input, outputs=analysis_output)




chatbot.queue().launch(server_name="0.0.0.0", server_port=7864, show_api=False)