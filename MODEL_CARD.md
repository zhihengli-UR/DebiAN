# Model Card

## Model Details

DebiAN is proposed to discover and mitigate unknown biases for image classification tasks.

DebiAN contains two networks---a discoverer and a classifier. The discoverer tries to identify unknown biases from the classifier, and the classifier tries to mitigate the biases identified by the discoverer.

Here, the bias is defined as the violation of the Equal Opportunity fairness criterion.

### Model Date

July 2022


### Documents

- [DebiAN Paper](https://arxiv.org/abs/2207.10077)


## Model Use

The model (DebiAN) is intended to identify and mitigate biases violates the Equal Opportunity fairness criterion.

In other words, DebiAN is not intended to (or cannot) capture other biases, e.g., biases against historically disadvantaged groups.


## Datasets

DebiAN is trained or evaluated on the following datasets:

- Multi-Color MNIST
- Colored MNIST
- CelebA
- bFFHQ
- Transects
- BAR
- Places
- LSUN

## Limitations
Here are some limitations that DebiAN has not fully resolved.

First, the work only assumes that the bias attribute is binary or continuously valued from 0 to 1 (i.e., two bias attribute groups). Future works can focus on extending DebiAN to discover and mitigate unknown biases with more than two groups.

Second, DebiAN can only discover the biases caused by spurious correlation rather than lack of coverage. For example, suppose a face image dataset only contains long-hair female images and does not contain any short-hair female images. In that case, DebiAN cannot discover the hair length bias attribute because the discoverer does not have samples to categorize female images into two groups in terms of the hair length bias attribute.

Finally, in terms of interpreting the discovered biases, DebiAN’s approach, using the saliency maps on real-world images, is not as easy as [interpreting biases from synthesized counterfactual images](https://arxiv.org/abs/2104.14556). Future works can further explore better interpreting the discovered unknown biases on real-world images.

## Potential Negative Social Impact

One potential negative social impact is that DebiAN’s discovered biases could be used as a way to choose real-world images as the adversarial images to attack visual models in some safety-critical domains, e.g., self-driving cars. Therefore, we encourage the defender to use DebiAN to mitigate the biases as the defense strategy.

Since our bias discovery approach relies on the fairness criterion based on equations, e.g., equal true positive rates among two groups, our method cannot identify the biases that a fairness criterion cannot capture, e.g., discrimination against the historically disadvantaged group. Therefore, we include this model card to clarify that our method’s intended use case is discovering and mitigating biases that violate the equal opportunity fairness criterion, and the model’s out-of-scope use case is identifying or mitigating other biases that cannot be captured by the equation of a fairness criterion, e.g., discrimination against the historically disadvantaged group.

## Feedback

If you have any questions, please contact Zhiheng Li (Homepage: [https://zhiheng.li](https://zhiheng.li)).
