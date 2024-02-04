<div align="center">

# Weakly Supervised Video Emotion Detection and Prediction via Cross-Modal Temporal Erasing Network [CVPR2023]


<i>Zhicheng Zhang, Lijuan Wang, and Jufeng Yang</i>

<a href=" "><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
[![Conference](https://img.shields.io/badge/CVPR-2023-green)](https://cvpr2023.thecvf.com/)
[![License](https://img.shields.io/badge/license-Apache%202-blue)](./LICENSE)

</div>

This is the official implementation of our **CVPR 2023** paper.
## News
- [ ] Adding comments
- [ ] reconstruct code

## Publication

**Weakly Supervised Video Emotion Detection and Prediction via Cross-Modal Temporal Erasing Network**<br>
<i>IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2023</i>.
</br>
[[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Weakly_Supervised_Video_Emotion_Detection_and_Prediction_via_Cross-Modal_Temporal_CVPR_2023_paper.pdf) [[PDF]](./assests/cvpr23_WECL.pdf) [[Video]](https://www.youtube.com/watch?v=ebD_xNQLuCY) [[Demo]](https://zzcheng.top/projects/VER) </br>



## Abstract

<img src="./assests/motivation_video3-1.png" width="50%" align="right">
Automatically predicting the emotions of user-generated videos (UGVs) receives increasing interest recently. However, existing methods mainly focus on a few key visual frames, which may limit their capacity to encode the context that depicts the intended emotions. To tackle that, in this paper, we propose a cross-modal temporal erasing network that locates not only keyframes but also context and audio-related information in a weakly-supervised manner. In specific, we first leverage the intra- and inter-modal relationship among different segments to accurately select keyframes. Then, we iteratively erase keyframes to encourage the model to concentrate on the contexts that include complementary information. Extensive experiments on three challenging benchmark datasets demonstrate that the proposed method performs favorably against the state-of-the-art approaches.


## Running

You can easily train and evaluate the model by running the script below.

You can adjust more details such as epoch, batch size, etc. Please refer to [`opts.py`](./opts.py).

~~~~
$ bash run.sh
~~~~

The used datasets are provided in [Ekman-6](https://github.com/kittenish/Frame-Transformer-Network), [VideoEmotion-8](https://drive.google.com/drive/folders/0B5peJ1MHnIWGd3pFbzMyTG5BSGs?resourcekey=0-hZ1jo5t1hIauRpYhYIvWYA), and [CAER](https://caer-dataset.github.io/).



## References

We referenced the repo of [VAANet](https://github.com/maysonma/VAANet) for the code.

## Citation
If you find this repo useful in your project or research, please consider citing the relevant publication.

**Bibtex Citation**
````
@InProceedings{Zhang_2023_CVPR,
    author    = {Zhang, Zhicheng and Wang, Lijuan and Yang, Jufeng},
    title     = {Weakly Supervised Video Emotion Detection and Prediction via Cross-Modal Temporal Erasing Network},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {18888-18897}
}
````
