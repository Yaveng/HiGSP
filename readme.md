Hi there👋.

This is the official implementation of  **Hierarchical Graph Signal Processing for Collaborative Filtering**, HiGSP for short, which is accepted by The Web Conference 2024.

We hope this code helps you well. If you use this code in your work, please cite our paper.



#### How to run this code

##### Step 1: Check the compatibility of your running environment. 

Generally, different running environments will still have a chance to cause different experimental results though all random processes are fixed in the code. Our running environment is 

```
- CPU: Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz
- GPU: Tesla T4
- Memory: 251.5G
- Operating System: Ubuntu 16.04.7 LTS (GNU/Linux 4.15.0-142-generic x86_64)
- CUDA Version: 10.0
- Python packages:
	- numpy: 1.19.5
	- pandas: 1.1.5
	- python: 3.6.13
	- pytorch: 1.9.0
	- scikit-learn: 0.22.0 
	- scipy: 1.5.2 
	- sparsesvd: 0.2.2 
```



##### Step 2: Prepare the datasets. 

Please put your datasets under the directory `dataset/XXX/`, where ```xxx``` is the name of the dataset, e.g., ```ml100k```. If the directory doesn't exist, please create it first. Please note that if you use your own datasets, check their format so as to make sure that it matches the input format of `HiGSP`. 



##### Step 3: Run the code.

* For ```ml100k``` dataset, please use the following code:

    ```python
    python main.py --dataset ml100k --alpha1 0.08 --alpha2 0.73 --order1 2 --order2 12 --pri_factor 80  --n_clusters 25 
    ```
    
* For ```ml1m``` dataset, please use the following code:

    ```python
    python main.py --dataset ml1m --alpha1 0.3 --alpha2 0.9 --order1 8 --order2 8 --pri_factor 256 --n_clusters 6 
    ```

* For ```Beauty``` dataset, please use the following code:

    ```python
python main.py --dataset beauty --alpha1 0.9 --alpha2 0.1 --order1 8 --order2 8 --pri_factor 370 --n_clusters 3
    ```

* For ```LastFM``` dataset, please use the following code:

    ```python
python main.py --dataset lastfm --alpha1 0.1 --alpha2 0.9 --order1 4 --order2 8 --pri_factor 85 --n_clusters 10
    ```

* For ```BX``` dataset, please use the following code:

    ```python
python main.py --dataset bx --alpha1 0.3 --alpha2 0.1 --order1 8 --order2 10 --pri_factor 206 --n_clusters 5
    ```

* For ```Netflix``` dataset, please use the following code:

    ```python
    python main.py --dataset netflix --alpha1 0.1 --alpha2 0.7 --order1 8 --order2 8 --pri_factor 256 --n_clusters 5
    ```



#### More information for reproducibility

Due to copyright issues, the raw datasets and the code of baselines cannot be included in our repository. Instead, we provide online access links of these datasets and baselines in our code repository to help improve the reproducibility of experimental results in our paper.

##### 1 Dataset Link

We provide the open access links of six commonly used public datasets in the table below:

| Dataset | Open Access Link                                             |
| :------ | :----------------------------------------------------------- |
| ML100K  | https://grouplens.org/datasets/movielens/                    |
| ML1M    | https://grouplens.org/datasets/movielens/                    |
| Beauty  | https://nijianmo.github.io/amazon/index.html                 |
| LastFM  | http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html  |
| BX      | https://grouplens.org/datasets/book-crossing/                |
| Netflix | https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data |



##### 2 Baseline Link

We provide the open access links of all baselines in the table below:

| Dataset  | Open Access Link                                             |
| :------- | :----------------------------------------------------------- |
| LR-GCCF  | https://github.com/newlei/LR-GCCF                            |
| LCFN     | https://github.com/Wenhui-Yu/LCFN                            |
| DGCF     | https://github.com/xiangwang1223/disentangled_graph_collaborative_filtering |
| LightGCN | https://github.com/gusye1234/LightGCN-PyTorch                |
| IMP-GCN  | https://github.com/liufancs/IMP_GCN                          |
| SimpleX  | https://github.com/reczoo/RecZoo/tree/main/matching/cf/SimpleX |
| UltraGCN | https://github.com/ouomarabdessamade/UltraGCN                |
| GF-CF    | https://github.com/yshenaw/GF_CF                             |
| PGSP     | https://github.com/jhliu0807/PGSP                            |



##### 3 Hyper-parameter Configuration

We provide the details for tuning the hyper-parameters of various baselines in the following table:

| Model    | Hyper-parameters Configuration                               |
| :------- | :----------------------------------------------------------- |
| LR-GCCF  | (1) embedding dimension: from 32 to 256; (2) learning rate: from 0.001 to 0.03. |
| LCFN     | (1) embedding dimension: from 32 to 256; (2) layer number: from 1 to 5; (3) user_freq: from 100 to 400; (4) item_freq: from 100 to 400; (5) learning rate: from 0.001 to 0.03. |
| DGCF     | (1) embedding dimension: from 32 to 256; (2) layer number: from 1 to 5; (3) n_factors: [2, 4, 8]; (4) learning rate: from 0.001 to 0.03. |
| LightGCN | (1) embedding dimension: from 32 to 256; (2) layer number: from 1 to 5; (3) keep_prob: from 0.1 to 1.0; (4) learning rate: from 0.001 to 0.03. |
| IMP-GCN  | (1) embedding dimension: from 32 to 256; (2) layer number: from 1 to 5; (3) adj_type: [pre, plain, norm, mean]; (4) regs: from 0.0001 to 0.01; (5) learning rate: from 0.001 to 0.03. |
| SimpleX  | (1) embedding dimension: from 32 to 256; (2) num_negs: [500, 1000, 1500, 2000]; (3) aggregator: [mean, user_attention, self-attention]; (4) enable_bias_str: [True, False]; (5) margin: from 0.3 to 1.0; (6) learning rate: from 0.001 to 0.03. |
| UltraGCN | (1) embedding dimension: from 32 to 256; (2) ii_neighbor_num: [5, 10, 15, 20, 30]; (3) lambda: 0.001 to 1.0 (4) learning rate: from 0.001 to 0.03. |
| GF-CF    | (1) primary components: from 32 to 1024, (2) weight coefficient: from 0.0 to 2.0. |
| PGSP     | (1) P0: [True, False], (2) P1: [True, False], (3) weight coefficient $\phi$: from 0.0 to 1.0; (4) primary components: from 32 to 1024. |

