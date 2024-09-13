Given the page limitations, it was challenging to provide a detailed explanation of the computation of our margin metric $\mathcal{M}_{\alpha}$. We actually followed the methodology of reference [19] to derive it. Below is a summary of our approach:

Since computing the exact distance of a point to the decision boundary in a deep network is computationally infeasible, the authors of [19] are approximating this distance using a first-order Taylor approximation, i.e., 

For a latent sample $\boldsymbol{x}^l$, with $\boldsymbol{x}^0 = \boldsymbol{x}$ for the input layer, the approximate margin distance for $p=2$ from the representation $\boldsymbol{x}^l$ to the decision boundary between class pair $(i, j)$ is:


##  $$d_{f,(i, j)}\left(\boldsymbol{x}^l\right) = \frac{f_i\left(\boldsymbol{x}^l\right) - f_j\left(\boldsymbol{x}^l\right)}{\left\|\nabla_{\boldsymbol{x}^l} f_i\left(\boldsymbol{x}^l\right) - \nabla_{\boldsymbol{x}^l} f_j\left(\boldsymbol{x}^l\right)\right\|_2}$$

where $f_i\left(\boldsymbol{x}^l\right)$ represents the logit for class $i$ given $\boldsymbol{x}^l$ as defined in the paper. The sign of this distance indicates whether the sample is on the "correct" (positive) or "wrong" (negative) side of the decision boundary. While this distance can be computed for all $(i, j)$ pairs, the authors assume $i$ corresponds to the ground truth label and $j$ is either the second-highest or highest class in case of misclassification. 

With this approximation, our training data $\boldsymbol{x}$ induces a distribution of margin distances (latent margins) at each layer $l$. These distributions can then be resumed through a vector of descriptive statistics $\mu$. For these statistics, we chose the five key statistics proposed by [19]: the first quartile, the median, the third quartile, and the upper and lower fences measuring variability outside the upper and lower quartiles. 

*N.B : We also experimented what happened with more extensive sets of statistics, and our results did not show significant improvements compared to the 5-statistics case.*

While our method is indeed largely inspired by the approach described in [19], it is important to clarify that we redefined its goal. Instead of focusing on **predicting the exact generalization gap using latent margins for image classification** our objective is rather to **use latent margins of training samples to derive a metric enabling to assess the robustness against post-processing in forensic applications**. This is a new point of view not explored in forensics that could benefit the forensic community greatly. 

Please note that our initial choice to sum the statistics raised to a specific $\alpha$ was an intuitive approach that successfully demonstrated the link between margin and generalization gap. We however believe there are more effective ways to create margin metrics that can better capture the generalization ability of splicing detectors. This work is a first step in encouraging the search of such metrics.

**Important Information: The results of our latent margin computations are available at [this link](https://drive.google.com/drive/folders/1q0Y0vEVPaqH23j5zIrv6jut3uH5PWnV4?usp=sharing). You need to download them and put them in a folder named  ```margin_data``` to reproduce our results in the playground notebook).**
