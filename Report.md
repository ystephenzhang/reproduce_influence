---
title: Report

---

## Report
### Theoretical Framework
The figure to reproduce plots predicted difference in loss (represented by influence function as $-\frac{1}{n}I_{loss}(z,z_{test})$) against actual difference in loss (represented by difference of loss after retraining as in $L(z_{test}, \theta/\{z\}) - L(z_{test}, \theta)$), of 500 samples with the highest influence $|I_{loss}(z,z_{test})|$. The two quantities' high correlation should show influence function's stability in reflecting the effect of removing one training sample.

The specific figure to reproduce is the middle one below, corresponding to a LiSSA-based approximation method.

![Screenshot 2025-01-15 at 09.59.49](https://hackmd.io/_uploads/rkZj85VPJx.png)

There are two aspects to reproducing this figure: the actual difference and the predicted the difference. The former can be efficiently done with parralleled re-training, while the latter incorporates key acceleration methods. **The following is a report of a PyTorch based attempt.**
### Implementation
*Predicted Loss.* This value i.e. values of the influence function is represented as $\mathcal{I}_{\text{up,loss}}(z, z_{\text{test}}) = - \nabla_{\theta} L(z_{\text{test}}, \hat{\theta})^\top H_{\hat{\theta}}^{-1} \nabla_{\theta} L(z, \hat{\theta})$. To accelerate, the original work proposed the following pipeline:
1. Calculate $\nabla_{\theta} L(z_{\text{test}}, \hat{\theta})$, which can be done in O\(p\).
2. Use a LiSSA-based approximation method to calculate $s_{test} := \nabla_{\theta} L(z_{\text{test}}, \hat{\theta})^\top H_{\hat{\theta}}^{-1}$ without explicitly storing $H_{\hat{\theta}}$. 
3. Calculate $\nabla_{\theta} L(z_i, \hat{\theta})$ for the target training sample $z_i$.
4. Calculate the final result through dot product $-s_{test}\cdot \nabla_{\theta} L(z_i, \hat{\theta})$

For step 1, `influence::hvp::calculate_grad_L` provided a batched implementation using `torch::autograd`. 
For step 2, I tried two implementations to compare their speed, as well as another slow but exact implementation as the ground truth in `influence::hvp::inverse_hvp`. 
* **Implementation A** explicitly calculate the Hessian using `torch::autograd::functional::hessian` to act as ground truth for correction. A eigenvalue analysis of the Hessian is also implemented to study convergence issues.
* **Implementation B** uses `torch::autograd::functional::hvp` to calculate the Hessian-vector production at each step of the iteration. This turned out to be much faster than the original implementation, which is based on an older version of Tensorflow.
* **Implementation C** uses a similar logic to the original work.

Step 3 and 4 are integrated in `main::calculate_influence`.

*Actual Loss.* This is done by simply removing samples and retraining. To efficiently complete retraining of the 500 models, `scripts::train::calculate_retrained_loss` provided a distributed data parallel implementation. Also, to avoid repeated retraining, all retrained models' parameters will be saved locally for direct loading.
### Results and Reflection
*Explosion in Magnitude.* The first problem I encounter is that when calculating the inverse Hessian product through iteration, magnitudes of the product explodes long before it is supposed to converge. 
![Screenshot 2025-01-15 at 18.01.55](https://hackmd.io/_uploads/Bk7iDWHwyg.png)
 This is solved by adding `1e-4 * product` to the Hessian-vector product at each iteration so as to avoid problems caused by inversing, and by performing scaling to the product at each step.
 
*Random Results.* Although the Hessian vector product is converging, results are seemingly random, as shown in the below figures with $top-30$ and $top-200$ influence examples, respectively.
![figure1](https://hackmd.io/_uploads/HysTK7BPkl.png)
![figure2](https://hackmd.io/_uploads/SybRFXHvkg.png)
After studying eigenvalues of the final Hessian matrix, I identified the problem as parameters of the logistic model being sub-optimal: most Hessian estimations are far from positive definite.

Moreover, I deviced a parallel pipeline in `influence::hvp::actual_H_with_data_parallel` to calculate the actual Hessian of all 6000 losses w.r.t. model parameters, and it is still not positive semi definite. The oracle Hessian's eigenvalues are counted as follows:

| $\lambda$<0 | $\lambda$=0 | 0<$\lambda$<1 | $\lambda$>1    |
| -------- | -------- | -------- | --- |
| 587     | 0     | 7253     |  10  |

*Next Steps.* 
* Given the simple nature of a logistic model on MNIST classification, I'll continue on trying to produce a roughly optimal model.
* In the meantime I'll adapt the loss function to $$
\begin{aligned}
\tilde{L}(z, \theta) &= L(z, \hat{\theta}) + 
\nabla L(z, \hat{\theta})^\top (\theta - \hat{\theta})\\
&+ \frac{1}{2} (\theta - \hat{\theta})^\top (H_{\hat{\theta}} + \lambda I)(\theta - \hat{\theta})
\end{aligned}
$$
    According to the original work as a solution to non-convex situations.
