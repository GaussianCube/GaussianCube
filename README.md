# GaussianCube: Structuring Gaussian Splatting for 3D Generative Modeling using Optimal Transport

By [Bowen Zhang](http://home.ustc.edu.cn/~zhangbowen), [Yiji Cheng](https://www.linkedin.com/in/yiji-cheng-a8b922213/?originalSubdomain=cn), [Jiaolong Yang](https://jlyang.org/), [Chunyu Wang](https://www.chunyuwang.org/), [Feng Zhao](https://en.auto.ustc.edu.cn/2021/0616/c26828a513169/page.htm), [Yansong Tang](https://andytang15.github.io/), [Dong Chen](http://www.dongchen.pro/), [Baining Guo](https://www.microsoft.com/en-us/research/people/bainguo/).

[Paper]() | [Project Page](https://gaussiancube.github.io/) | [Code](https://github.com/GaussianCube/GaussianCube)

## Abstract

> In this work, we introduce GaussianCube, a novel representation crafted for 3D generative modeling. Recent advances in Gaussian Splatting have demonstrated considerable improvements in expressiveness and efficiency. However, its unstructured nature poses a significant challenge to its deployment in generative applications. To fully unleash the potential of Gaussian Splatting in generative modeling, we present GaussianCube, a powerful and efficient spatial-structured representation for 3D generation. We propose to perform densification-constrained fitting to obtain fix-length Gaussians for each 3D asset, and then systematically arrange the Gaussians into the structured voxel grid via Optimal Transport. Furthermore, we perform generative modeling on the proposed GaussianCube using 3D diffusion models. The spatial coherence of GaussianCube allows us to utilize a standard 3D U-Net as our diffusion network without elaborate design. Extensive experiments conducted on ShapeNet and challenging large-vocabulary OmniObject3D show the superiority to prior works, underscoring its potential as a versatile 3D representation.

## Todo

- [ ] Release the inference code.
- [ ] Release the training code.
