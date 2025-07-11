# Inverse Stochastic Microstructure Design

[Paper](https://www.sciencedirect.com/science/article/abs/pii/S1359645424002301)

## Description

This repository contains research code associated with 
*"Inverse Stochastic Microstructure Design"*. The proposed framework enables identifying
a conditional density over 2-point spatial correlations given a set of target thermal
conductivities.

If you find this code useful, interesting, and are open to collaboration, please reach out!
Alternatively, if you have any questions regarding the contents of this repository or portions of the work, feel free
to as well at: [agenerale3@gatech.edu](agenerale3@gatech.edu). Please consider citing this work (expand for BibTeX).

<details>
<summary>
Generale, A. P., Robertson, A. E., Kelly, C., & Kalidindi, S. R. (2024). Inverse stochastic microstructure design. Acta Materialia, 271, 119877. https://doi.org/10.1016/j.actamat.2024.119877
</summary>

```bibtex
@article{generale_inverse_2024,
	title = {Inverse stochastic microstructure design},
	journal = {Acta Materialia},
	volume = {271},
	pages = {119877},
	year = {2024},
	issn = {1359-6454},
	doi = {https://doi.org/10.1016/j.actamat.2024.119877},
	url = {https://www.sciencedirect.com/science/article/pii/S1359645424002301},
	author = {Adam P. Generale and Andreas E. Robertson and Conlain Kelly and Surya R. Kalidindi},
	keywords = {Inverse design, Generative modeling, Uncertainty quantification, Bayesian inference, Computational materials design, Microstructure},
	}
```
</details>

## Examples
The framework is briefly displayed below. A statistical representation of microstructure, namely, 2-point spatial correlations are first subjected to a orthogonal transformation through Principal Component Analysis (PCA), and subsequently through a nonlinear embedding with a Variational Autoencoder (VAE) for the construction of an information dense hierarchical latent space. This latent space enables the probabilistic inversion of the structure-property linkage.

![My Image](images/framework_pictoral.png)

In the PC latent space (left), the framework is capable of identifying statistical microstructure representations giving rise to target property sets under uncertainty of the forward model. The unseen microstructure representation (red) corresponding to the target set of orthotropic thermal conductivities is recovered by the posterior (blue), with high specificity relative to the complete ensemble (black).

The posterior similarly, recovers the unseen microstructure representation (red) in 2-point spatial correlation space (right), shown in 1D, rather than the complete set in 3D for visual clarity.

![My Image](images/github_img.png)


## Contents
This section provides a brief description of the contents of this repository.

1. *Models*: Contains code for instantiating the Gaussian mixture model (GMM), sparse variational multi-output
 Gaussian process (SV-MOGP), flow-based generative model, and variational auto-encoder (VAE) used in this work.
 
2. *Data*: Contains PC scores (computed from 2-point spatial correlations) of initial microstructure
 dataset, alongside corresponding location in the latent space of the VAE and property set.
 
3. *Checkpoints*: *mogp_likelihood_state.pth, mogp_model_state.pth, vae.pth* Model state dictionaries for the SV-MOGP forward model and VAE.

4. *main.py*: Main executable for training and post-processing results from the flow-based generative model.

## Execute
Inference of the conditional microstructure distributions provided above can be replicated as
```
python main.py --micro 0
```
where the *micro* flag can be swept from 0-2 for the three current test cases.

For postprocessing of results from the trained model, plotting results over the space of 2-point spatial correlations requires *pca_autocorr_1024.pkl*. Due to its prohibitive size, it has been left out from these associated files. If needed for your own use case, please feel free to reach out.
