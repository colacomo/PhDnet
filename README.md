# PhDnet (IF 2024)
### 📖[**Paper**](https://www.sciencedirect.com/science/article/abs/pii/S1566253524000551) | 🖼️[**PDF**](/figs/main.png)

PyTorch codes for "[PhDnet: A novel physic-aware dehazing network for remote sensing images](https://www.sciencedirect.com/science/article/abs/pii/S1566253524000551)", **Information Fusion (IF)**, 2024.

- Authors:  [Ziyang LiHe](Ziyang_Lihe@whu.edu.cn), [Jiang He](https://jianghe96.github.io/), [Qiangqiang Yuan*](http://qqyuan.users.sgg.whu.edu.cn/), [Xianyu Jin](jin_xy@whu.edu.cn), [Yi Xiao](https://xy-boy.github.io/), and [Liangpei Zhang](http://www.lmars.whu.edu.cn/prof_web/zhangliangpei/rs/index.html)<br>
- Wuhan University,


## Abstract
> Remote sensing haze removal is a popular computational imaging technique that directly obtains clear remote sensing data from hazy remote sensing images. Apart from prior-based methods, deep-learning-based methods have performed well in the past years for their powerful non-linear mapping from hazy to haze-free domains. Most dehazing networks are established on atmospheric scattering models. However, these models cannot be easily embedded into deep learning to guide feature extraction. In this study, we introduce a haze extraction model that combines residual learning with the atmospheric scattering model and build a novel physic-aware dehazing network based on this model to achieve effective haze removal for remote sensing images with physical interpretability. By combining multi-scale-gating convolution with a haze extraction unit, PhDnet meets the need for remote sensing image haze removal. We verify the advantage of PhDnet by applying this model on multiple synthetic hazy datasets and real hazy images. We also implement comprehensive studies about the effectiveness of its haze extraction unit and other components.
## Network  
 ![image](/figs/main.png)
 


## 🎁 Dataset
Please download the following remote sensing benchmarks:
[RSHaze+](https://zenodo.org/records/13837162) 
### Train
```
python train.py
```

## Contact
If you have any questions or suggestions, feel free to contact me.  
Email: Ziyang_Lihe@whu.edu.cn

## Citation
If you find our work helpful in your research, please consider citing it. We appreciate your support！😊

```
@article{LIHE2024102277,
title = {PhDnet: A novel physic-aware dehazing network for remote sensing images},
journal = {Information Fusion},
volume = {106},
pages = {102277},
year = {2024},
issn = {1566-2535},
doi = {https://doi.org/10.1016/j.inffus.2024.102277},
url = {https://www.sciencedirect.com/science/article/pii/S1566253524000551},
author = {Ziyang Lihe and Jiang He and Qiangqiang Yuan and Xianyu Jin and Yi Xiao and Liangpei Zhang},
keywords = {Haze removal, Physics-aware network, Deep learning, Remote sensing imaging},
abstract = {Remote sensing haze removal is a popular computational imaging technique that directly obtains clear remote sensing data from hazy remote sensing images. Apart from prior-based methods, deep-learning-based methods have performed well in the past years for their powerful non-linear mapping from hazy to haze-free domains. Most dehazing networks are established on atmospheric scattering models. However, these models cannot be easily embedded into deep learning to guide feature extraction. In this study, we introduce a haze extraction model that combines residual learning with the atmospheric scattering model and build a novel physic-aware dehazing network based on this model to achieve effective haze removal for remote sensing images with physical interpretability. By combining multi-scale-gating convolution with a haze extraction unit, PhDnet meets the need for remote sensing image haze removal. We verify the advantage of PhDnet by applying this model on multiple synthetic hazy datasets and real hazy images. We also implement comprehensive studies about the effectiveness of its haze extraction unit and other components. The source code and pre-trained models are available at https://github.com/colacomo/PhDnet.}
}
```
