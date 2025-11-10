
# Attention Meets Post-hoc Interpretability: A Mathematical Perspective

Code for the paper [*Attention Meets Post-hoc Interpretability: A Mathematical Perspective*](https://proceedings.mlr.press/v235/lopardo24a.html), ICML 2024. 

## Getting Started

To install the necessary dependencies, run the following command:

    pip install -r requirements.txt

## Code Structure
-   `multi_head_trainer.py`: This script is responsible for training the multi-head classifier. The classifier is defined in the `models/multi_head.py` file and its structure is detailed in Section 2 of the paper. 
-   `params.py`: This file contains all the parameters required for the model and the experiments. It serves as a centralized location for managing experiment configurations.


## Notebooks

The repository includes several Jupyter notebooks for generating the figures in the paper:

-   `attention_meets_xai.ipynb`: generates Figure 1.
-   `attention_heads.ipynb`: generates Figure 3.
-   `lime_meets_attention.ipynb`: generates Figure 4.
-   `gradient_meets_attention.ipynb`: generates Figure 5.

The generated figures can be found in the  `results/paper`  directory.

## Quantitative experiments

-   `quant_gradient.py`  and  `quant_lime.py`: These scripts contain the code for large-scale quantitative experiments for the Gradient and LIME sections, respectively.

## Citation

If you use this code or find our work helpful, please cite our paper:

```bibtex
@inproceedings{
	lopardo2024attention,
	title={Attention Meets Post-hoc Interpretability: A Mathematical Perspective},
	author={Gianluigi Lopardo and Frederic Precioso and Damien Garreau},
	booktitle={Forty-first International Conference on Machine Learning},
	year={2024},
	url={https://openreview.net/forum?id=wnkC5T11Z9}
}
```
