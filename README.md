### Network Flow Classification & Anomaly Detection

This is our term project for undergraduate senior year Machine Learning Lab in CSE, BUET. We chose to work with a well known **Network Intrusion Detection Database**, the **CSE-CIC-IDS2018**. We used the paper [Towards a Standard Feature Set for Network Intrusion Detection System Datasets](https://doi.org/10.1007/s11036-021-01843-0) as the cleaned up, modified version of the dataset. The dataset had roughly 18 million entries. Our target was to use **transformer based models to perform multi-class classification of network flows**.

#### Papers
For this we tried to follow up on two papers -
- [FlowTransformer: A Transformer Framework for Flow-based Network Intrusion Detection Systems (2024)](https://doi.org/10.1016/j.eswa.2023.122564): Focuses on general encoder/decoder transformers and specialized models such as GPT, Bert for binary classification of network flows. This is the paper which we worked on.
- [A Novel Multi-Stage Approach for Hierarchical Intrusion Detection (2023)](https://doi.org/10.1109/TNSM.2023.3259474): This is the state of the art work on this dataset. Using a 2 stage approach where first stage uses One Class Support Vector Machine to do binary classification in Attacks vs Benign flow and the second stage uses Random Forest models to classify the Attacks, it achieves high accuracy and a good F1-Score.

#### Models
We tested different ML models such as **Random Forests**, **K-Nearest Neighbours**, **Transformer Based Models** etc. Of the Transformer based models, we chose to use -
- **Encoder based models: BERT and variants** such as CodeBERT, DistillBERT, RoBERTa, Electra, SpanBERT
- **Decoder based models: GPT 2**

#### Method & Result
We had to downsample the data as the whole dataset was too large to train/test on our available setup. We kept the original ratio while downsampling. **First** we successfully **replicated the FlowTransformer paper**'s high accuracy and F1-score. We also **tried out Random Forests** as these simple models are most effective in network flow classification. We found out that **feature selection** beforehand can improve Random Forest accuracy and F1-score even more. 

Then we tried the different BERT variants. Here **CodeBERT and SpanBERT** did **comparatively better**, but achieved a lower F1-score compared to the state of the art(SOTA) multi-stage approach. We switched to decoder based model and chose **GPT 2**. Here we obtained **high enough accuracy and F1-score, comparable to the SOTA**. Our findings indicate transformer based models coupled with feature selection, can perform multiclass network flow classification with good accuracy and F1-score. The only drawback would be a large training time, but it can be mitigated with moderate computing power. More about our findings can be found [**here**](https://github.com/Mohaimin41/ml_project/blob/main/1905058%201905041%20ML%20Project%20Final%20Presentation.pdf). This repository also contains the notebooks along with the results of all model runs. 

#### Acknowledgements
We are thankful to authors of both the papers and the datasets for their vast contributions in the field of Network Intrusion Detection. Their papers and codebases guided us on this project. We are also grateful to our project supervisor [**Dr. Muhammad Masroor Ali**](https://cse.buet.ac.bd/faculty/faculty_detail/mmasroorali) for his motivation and direction. 
