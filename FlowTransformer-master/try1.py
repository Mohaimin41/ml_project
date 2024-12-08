#  FlowTransformer 2023 by liamdm / liam@riftcs.com

import os

import pandas as pd

from framework.dataset_specification import NamedDatasetSpecifications
from framework.enumerations import EvaluationDatasetSampling
from framework.flow_transformer import FlowTransformer
from framework.flow_transformer_parameters import FlowTransformerParameters
from framework.framework_component import FunctionalComponent
from implementations.classification_heads import *
from implementations.input_encodings import *
from implementations.pre_processings import StandardPreProcessing
from implementations.transformers.basic_transformers import BasicTransformer
from implementations.transformers.named_transformers import *
from framework.model_input_specification import ModelInputSpecification

encodings = [
    RecordLevelEmbed(64),
    RecordLevelEmbed(64, project=True)
]

classification_heads = [
    FeaturewiseEmbedding(project=False),
    LastTokenClassificationHead()
]

transformers: List[FunctionalComponent] = [
    BasicTransformer(2, 128, n_heads=2),
    GPTSmallTransformer()
]

flow_file_path = r"dataset/"

datasets = [
    ("CSE_CIC_IDS", os.path.join(flow_file_path, "NF_CSECICID2018/data/NF-CSE-CIC-IDS2018-v2.csv"), NamedDatasetSpecifications.cse_cic_ids_2018, 0.01, EvaluationDatasetSampling.RandomRows)
]

print("Imported and defined")
pre_processing = StandardPreProcessing(n_categorical_levels=32)

# Define the transformer
ft = FlowTransformer(pre_processing=pre_processing,
                     input_encoding=encodings[0],
                     sequential_model=transformers[0],
                     classification_head=classification_heads[0],
                     params=FlowTransformerParameters(window_size=8, mlp_layer_sizes=[128], mlp_dropout=0.1))

print("Defined flowtransformer, starting load")

# Load the specific dataset
dataset_name, dataset_path, dataset_specification, eval_percent, eval_method = datasets[0]
ft.load_dataset(dataset_name, dataset_path, dataset_specification, evaluation_dataset_sampling=eval_method, evaluation_percent=eval_percent)

print("Loaded dataset, building model")

# Build the transformer model
m = ft.build_model()
m.summary()

# Compile the model
m.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'], jit_compile=True)

# Get the evaluation results
eval_results: pd.DataFrame
(train_results, eval_results, final_epoch) = ft.evaluate(m, batch_size=128, epochs=5, steps_per_epoch=64, early_stopping_patience=5)


print(eval_results)