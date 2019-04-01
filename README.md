# Recommendation System
This repository provides examples and best practices for building recommendation systems, provided as Jupyter notebooks. The examples detail our learnings on five key tasks: 
- [Prepare Data](notebooks/01_prepare_data/README.md): Preparing and loading data for each recommender algorithm
- [Model](notebooks/02_model/README.md): Building models using various classical and deep learning recommender algorithms such as Alternating Least Squares ([ALS](https://spark.apache.org/docs/latest/api/python/_modules/pyspark/ml/recommendation.html#ALS)) or eXtreme Deep Factorization Machines ([xDeepFM](https://arxiv.org/abs/1803.05170)).
- [Evaluate](notebooks/03_evaluate/README.md): Evaluating algorithms with offline metrics
- [Model Select and Optimize](notebooks/04_model_select_and_optimize): Tuning and optimizing hyperparameters for recommender models
- [Operationalize](notebooks/05_operationalize/README.md): Operationalizing models in a production environment on Azure

Several utilities are provided in [reco_utils](reco_utils) to support common tasks such as loading datasets in the format expected by different algorithms, evaluating model outputs, and splitting training/test data. Implementations of several state-of-the-art algorithms are provided for self-study and customization in your own applications.

## Algorithms

The table below lists recommender algorithms available in the repository at the moment.

| Algorithm | Environment | Type | Description | 
| --- | --- | --- | --- |
| [Smart Adaptive Recommendations (SAR)<sup>*</sup>](notebooks/00_quick_start/sar_movielens.ipynb) | Python CPU | Collaborative Filtering | Similarity-based algorithm for implicit feedback dataset |
| [Surprise/Singular Value Decomposition (SVD)](notebooks/02_model/surprise_svd_deep_dive.ipynb) | Python CPU | Collaborative Filtering | Matrix factorization algorithm for predicting explicit rating feedback in datasets that are not very large | 
| [Neural Collaborative Filtering (NCF)](notebooks/00_quick_start/ncf_movielens.ipynb) |  Python CPU / Python GPU | Collaborative Filtering | Deep learning algorithm with enhanced performance for implicit feedback | 
| [Restricted Boltzmann Machines (RBM)](notebooks/00_quick_start/rbm_movielens.ipynb) |  Python CPU / Python GPU | Collaborative Filtering | Neural network based algorithm for learning the underlying probability distribution for explicit or implicit feedback | 
| [FastAI Embedding Dot Bias (FAST)](notebooks/00_quick_start/fastai_movielens.ipynb)  |  Python CPU / Python GPU | Collaborative Filtering | General purpose algorithm with embeddings and biases for users and items |
| [Alternating Least Squares (ALS)](notebooks/00_quick_start/als_movielens.ipynb) | PySpark | Collaborative Filtering | Matrix factorization algorithm for explicit or implicit feedback in large datasets, optimized by Spark MLLib for scalability and distributed computing capability | 
| [Vowpal Wabbit Family (VW)<sup>*</sup>](notebooks/02_model/vowpal_wabbit_deep_dive.ipynb) | Python CPU (train online) | Collaborative, Content-Based Filtering | Fast online learning algorithms, great for scenarios where user features / context are constantly changing |
| [LightGBM/Gradient Boosting Tree<sup>*</sup>](notebooks/00_quick_start/lightgbm_tinycriteo.ipynb) | Python CPU | Content-Based Filtering | Gradient Boosting Tree algorithm for fast training and low memory usage in content-based problems |
| [Deep Knowledge-Aware Network (DKN)<sup>*</sup>](notebooks/00_quick_start/dkn_synthetic.ipynb) |  Python CPU / Python GPU | Content-Based Filtering | Deep learning algorithm incorporating a knowledge graph and article embeddings to provide powerful news or article recommendations | 
| [Extreme Deep Factorization Machine (xDeepFM)<sup>*</sup>](notebooks/00_quick_start/xdeepfm_synthetic.ipynb) | Python CPU / Python GPU | Hybrid | Deep learning based algorithm for implicit and explicit feedback with user/item features | 
| [Wide and Deep](notebooks/00_quick_start/wide_deep_movielens.ipynb) | Python CPU / Python GPU | Hybrid | Deep learning algorithm that can memorize feature interactions and generalize user features |

**Preliminary Comparison**

Provided a [comparison notebook](notebooks/03_evaluate/comparison.ipynb) to illustrate how different algorithms could be evaluated and compared. In this notebook, data (MovieLens 1M) is randomly split into training/test sets at a 75/25 ratio. A recommendation model is trained using each of the collaborative filtering algorithms below. We utilize empirical parameter values reported in literature [here](http://mymedialite.net/examples/datasets.html). For ranking metrics we use k = 10 (top 10 recommended items). 

| Algo | MAP | nDCG@k | Precision@k | Recall@k | RMSE | MAE | R<sup>2</sup> | Explained Variance | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| [ALS](notebooks/00_quick_start/als_movielens.ipynb) | 0.002020 | 0.024313 | 0.030677 | 0.009649 | 0.860502 | 0.680608 | 0.406014 | 0.411603 | 
| [SVD](notebooks/02_model/surprise_svd_deep_dive.ipynb) | 0.010915 | 0.102398 | 0.092996 | 0.025362 | 0.888991 | 0.696781 | 0.364178 | 0.364178 | 
| [FastAI](notebooks/00_quick_start/fastai_movielens.ipynb) | 0.023022 |0.168714 |0.154761 |0.050153 |0.887224 |0.705609 |0.371552 |0.374281 |

