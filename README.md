# GL-Coarsener
A python implementation of GL-Coarsener method, as proposed in [GL-Coarsener: A Graph representation learning framework to construct coarse grid hierarchy for AMG solvers, arXiv:2011.09994](https://arxiv.org/abs/2011.09994).

If you find our code or paper useful in your research, please consider citing:
```
@misc{namazi2020glcoarsener,
      title={GL-Coarsener: A Graph representation learning framework to construct coarse grid hierarchy for AMG solvers}, 
      author={Reza Namazi and Arsham Zolanvari and Mahdi Sani and Seyed Amir Ali Ghafourian Ghahramani},
      year={2020},
      eprint={2011.09994},
      archivePrefix={arXiv},
      primaryClass={math.NA}
}
```

# Installation
```bash
pip install gl-coarsener
```

# Usage
### Import the package
```python
from glcoarsener import Coarsener
```
### Input
You should construct a _Coarsener_ object with the adjacency matrix that you want to coarsen. The input must be of type _scipy sparse CSR matrix_.
```python
from scipy import sparse
adjacency_matrix = sparse.random(1000, 1000, format='csr')
coarsener = Coarsener(adjacency_matrix)
```
### Applying the method
```python
prolongation_operator = coarsener.apply(
    dimensions=100,
    walk_length=20,
    num_walks=10,
    p=0.1,
    q=1,
    number_of_clusters=adjacency_matrix.shape[0] // 5,
    clustering_method='kmeans',
    workers=1
)
```
### Getting the coarse (smaller) adjacency matrix
```python
restriction_operator = prolongation_operator.transpose()
coarse_matrix = restriction_operator.dot(adjacency_matrix).dot(prolongation_operator)
print(coarse_matrix)
```
# Contributing
Any contribution is **greatly appreciated**. If you think you can improve this work, please open a pull request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/Improvement`)
3. Commit your Changes (`git commit -m 'Add some Improvement'`)
4. Push to the Branch (`git push origin feature/Improvement`)
5. Open a Pull Request

# Contact
If you have any questions, please do not hesitate to contact me:
- Personal Webpage: https://rezanmz.com/
- Email: rezanmz@ymail.com