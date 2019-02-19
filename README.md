# LATTADV: [Relation Extraction](https://en.wikipedia.org/wiki/Relationship_extraction) With Leveled Adversarial Neural Networks and MDL Database Expansion.

_Contributed by [Diyah Puspitaningrum](http://diyahpuspitaningrum.net/)_



These recent years, state-of-the-art relation extraction aims to extract relations from plain text with the help of deep learning method to classify ontology of a relation that a given document point to. In this project we demonstrate the use of database expansion through Minimum Description Length (MDL) based semantic identification based on entity pair as a preprocessing step to improve classifier performance on relation extraction task. For classifier, since a relation is viewed as semantic composition of sentence embedding, the false labelling problem can be addressed via leveled sentence-level attention over multiple instances and leveled adversarial training. The selective attention aims to dynamically reduce weights of noisy instances while adversarial training aims to improve robustness of a classifier through regularization of loss function's gradient direction given small perturbations to the classifier system. As a final result, given input data _X_ and consider the word embedding of all the words in _X_ is _V_, the relation is extracted by leveled mechanism aims hopefully beneficial to improve performance of the standard adversarial network. While our proposed model, _LATTADV_ is evaluated on a popular benchmark dataset in relation extraction, i.e. New York Times dataset, the same as used by [Riedel et. al. (2010)](https://www.researchgate.net/publication/220698997_Modeling_Relations_and_Their_Mentions_without_Labeled_Text), the LATTADV model has shown significant improvement as compared with other PCNN based methods on relation extraction. We have shown that the MDL data expansion and the leveled adversarial strategy both can be two supporting strategies to improve the PCNN performance.

We inspired from the paper "Adversarial Training for Relation Extraction" [Wu et al., 2017](http://www.aclweb.org/anthology/D17-1187) and the paper "Neural Relation Extraction with Selective Attention over Instances" [Lin et al.,2016](http://www.aclweb.org/anthology/P16-1200) and a good understanding about MDL principle from the paper "Compression Picks Item Sets That Matter" [van Leeuwen, et al., 2006](https://link.springer.com/content/pdf/10.1007/11871637_59.pdf). Our contribution are three-fold: (i). Database expansion with the use of semantic understanding of entity identification is a new approach in relation extraction preprocessing technique. (ii). The leveled classifier with adversarial training, LATTADV, is an independent classifier which means that the model can also be applied on any classification tasks. (iii). Tested on state-of-the-art methods (PCNN ATT [Lin et al.,2016](http://www.aclweb.org/anthology/P16-1200), PCNN ADV, PCNN MAX-ADV [Wu et al., 2017](http://www.aclweb.org/anthology/D17-1187)), the proposed LATTADV on MDL data expansion can be significantly improve the deep learning (PCNN) performance.

# Evaluation Results:

### Performances comparison between LATTADV-ATT and other baselines:

![](./images/Table.png)

 
Precision/Recall curve of our method (LATTADV) with database expansion compared to state-of-the-art methods:

![](./images/pr_curve_jac_k5.png)



# Data
We use the same dataset(NYT10, New York Times Annotated Corpus ([LDC Data LDC2008T19](https://catalog.ldc.upenn.edu/LDC2008T19)) as in [Lin et al.,2016] (https://github.com/thunlp/OpenNRE/tree/old_version)) and Riedel et. al. ([2010](https://github.com/diyahpus/RiedelNYT0506)) and we expand it. We provide the dataset in origin_data/ directory.

To run our code, the dataset should be put in the folder origin_data/ using the following format, containing four files:
- train.txt: training file.
- test.txt: testing file.
- relation2id.txt: all relations and corresponding ids.
- vec.txt: the pre-train word embedding file

# Codes
The source codes are in the current main directory. `network.py` contains the whole neural network's definition. This model is deveoped from work by  [Tianyu Gao, Xu Han, Lumin Tang, Yankai Lin, Zhiyuan Liu](https://github.com/thunlp/OpenNRE/tree/old_version).

# Requirements
- Python (>=2.7)
- TensorFlow (>=1.4.1), GPU CUDA(>=8.0)
- scikit-learn (>=0.18)
- Matplotlib (>=2.0.0)



## Quick Start

### Process Data

```bash
python3 gen_data.py
```
The processed data will be stored in `./data`

### Train Model
```
python3 train.py --model_name att_adv_att_adv_att
```

### Test Model see available model `./model`
```bash
python test.py --model_name att_adv_att_adv_att
```

All checkpoints are stored in `./checkpoint`. Best checkpoint in `./test_result`.

### Plot
```bash
python draw_plot.py att_adv_att_adv_att
```

The plot will be saved as `./test_result/pr_curve.png`.


# Request for additional results (optional)
We provide .ipynb files as log for our experiments as well as best checkpoint for drawing charts. Contact: diyahpuspitaningrum@gmail.com .




## Reference

1. **Compression Picks Item Sets That Matter.** _Matthijs van Leeuwen, Jilles Vreeken, and Arno Siebes._ PKDD2006. [paper](https://link.springer.com/content/pdf/10.1007/11871637_59.pdf)

2. **Widened Learning of Bayesian Network Classifiers.** _Oliver R. Sampson and Michael R. Berthold._ IDA2016. [paper](https://www.researchgate.net/publication/309195760_Widened_Learning_of_Bayesian_Network_Classifiers)

3. **Neural Relation Extraction with Selective Attention over Instances.** _Yankai Lin, Shiqi Shen, Zhiyuan Liu, Huanbo Luan, Maosong Sun._ ACL2016. [paper](http://www.aclweb.org/anthology/P16-1200)

4. **Adversarial Training for Relation Extraction.** _Yi Wu, David Bamman, Stuart Russell._ EMNLP2017. [paper](http://www.aclweb.org/anthology/D17-1187)

5. **A Soft-label Method for Noise-tolerant Distantly Supervised Relation Extraction.** _Tianyu Liu, Kexiang Wang, Baobao Chang, Zhifang Sui._ EMNLP2017. [paper](http://aclweb.org/anthology/D17-1189)

6. **Reinforcement Learning for Relation Classification from Noisy Data.** _Jun Feng, Minlie Huang, Li Zhao, Yang Yang, Xiaoyan Zhu._ AAAI2018. [paper](https://tianjun.me/static/essay_resources/RelationExtraction/Paper/AAAI2018Denoising.pdf)

7. **Modeling relations and their mentions without labeled text.** _Sebastian Riedel, Limin Yao, and Andrew McCallum._ ECMLPKDD'10(III). [paper](https://www.researchgate.net/publication/220698997_Modeling_Relations_and_Their_Mentions_without_Labeled_Text)

