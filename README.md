# Recommendation System

Recommendation System basically are data filtering tools that make use of algorithms and data to recommend the most relevant items to a particular user.

## Types of recommendations
There are basically three important types of recommendation engines:

### Collaborative filtering:
This filtering method is usually based on collecting and analyzing information on user’s behaviors, their activities or preferences and predicting what they will like based on the similarity with other users. A key advantage of the collaborative filtering approach is that it does not rely on machine analyzable content and thus it is capable of accurately recommending complex items such as movies without requiring an “understanding” of the item itself. Collaborative filtering is based on the assumption that people who agreed in the past will agree in the future, and that they will like similar kinds of items as they liked in the past. For example, if a person A likes item 1, 2, 3 and B like 2,3,4 then they have similar interests and A should like item 4 and B should like item 1.

Further, there are several types of collaborative filtering algorithms:

1. **User-User Collaborative Filtering**: Here, we try to search for lookalike customers and offer products based on what his/her lookalike has chosen. This algorithm is very effective but takes a lot of time and resources. This type of filtering requires computing every customer pair information which takes time. So, for big base platforms, this algorithm is hard to put in place.
   
   Well, UB-CF uses that logic and recommends items by finding similar users to the active user (to whom we are trying to recommend a movie). A specific application of this is the user-based Nearest Neighbor algorithm. This algorithm needs two tasks:
    - Find the K-nearest neighbors (KNN) to the user a, using a similarity function w to measure the distance between each pair of users:
    - Predict the rating that user a will give to all items the k neighbors have consumed but a has not. We Look for the item j with the best predicted rating. In other words, we are creating a User-Item Matrix, predicting the ratings on items the active user has not see, based on the other similar users. This technique is memory-based.


PROS:
Easy to implement.
Context independent.
Compared to other techniques, such as content-based, it is more accurate.
CONS:
Sparsity: The percentage of people who rate items is really low.
Scalability: The more K neighbors we consider (under a certain threshold), the better my classification should be. Nevertheless, the more users there are in the system, the greater the cost of finding the nearest K neighbors will be.
Cold-start: New users will have no to little information about them to be compared with other users.
New item: Just like the last point, new items will lack of ratings to create a solid ranking (More of this on ‘How to sort and rank items’).
   
2. Item-Item Collaborative Filtering: It is very similar to the previous algorithm, but instead of finding a customer look alike, we try finding item look alike. Once we have item look alike matrix, we can easily recommend alike items to a customer who has purchased any item from the store. This algorithm requires far fewer resources than user-user collaborative filtering. Hence, for a new customer, the algorithm takes far lesser time than user-user collaborate as we don’t need all similarity scores between customers. Amazon uses this approach in its recommendation engine to show related products which boost sales.

### Content-based filtering:
These filtering methods are based on the description of an item and a profile of the user’s preferred choices. In a content-based recommendation system, keywords are used to describe the items; besides, a user profile is built to state the type of item this user likes. In other words, the algorithms try to recommend products which are similar to the ones that a user has liked in the past. The idea of content-based filtering is that if you like an item you will also like a ‘similar’ item. For example, when we are recommending the same kind of item like a movie or song recommendation. This approach has its roots in information retrieval and information filtering research.

A major issue with content-based filtering is whether the system is able to learn user preferences from users actions about one content source and replicate them across other different content types. When the system is limited to recommending the content of the same type as the user is already using, the value from the recommendation system is significantly less when other content types from other services can be recommended. For example, recommending news articles based on browsing of news is useful, but wouldn’t it be much more useful when music, videos from different services can be recommended based on the news browsing.

### Hybrid Recommendation systems:
Recent research shows that combining collaborative and content-based recommendation can be more effective. Hybrid approaches can be implemented by making content-based and collaborative-based predictions separately and then combining them. Further, by adding content-based capabilities to a collaborative-based approach and vice versa; or by unifying the approaches into one model.

---
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

