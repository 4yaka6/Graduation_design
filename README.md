# 毕业设计（Graduation Design）
该项目仓库未包含使用的数据集，如果需要下载使用可以参考以下链接：  
（The project repository does not include the datasets used. If you need to download them, you can refer to the following links:）  
AffectNet： https://www.kaggle.com/datasets/mstjebashazida/affectnet  
iemocap：https://sail.usc.edu/iemocap/  
MELD: https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz  
      OR:https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz  
MediaEval 2018实际是DEAM的子集完整的下载地址在：  
https://cvml.unige.ch/databases/DEAM/

**摘    要**   
现有的数字音乐平台的推荐方法通常依赖播放历史、收藏行为和协同过滤结果，虽然能够在一定程度上提升推荐效率，但难以反映用户在特定时刻的即时情绪需求。针对这一问题，本文围绕“基于情绪识别的智能音乐推荐系统设计”开展研究，以用户当前情绪为核心驱动因素，构建了一个集多模态情绪识别、音乐情感建模与桌面推荐交互于一体的原型系统。  
在用户情绪识别部分，系统允许用户通过文本输入与图像输入表达当前情绪。文本模态以IEMOCAP与MELD数据集中的句子文本为训练对象，基于RoBERTa构建分类与VA回归联合学习模型；图像模态以AffectNet数据集数据集为基础，采用ResNet18完成六类情绪识别，并通过离散情绪到Valence-Arousal连续情绪空间的映射得到坐标表示。音乐情感建模部分以音频声学特征为核心，在MediaEval2018数据集上提取MFCC、GTF、Chroma特征分别再建立Arousal与 Valence回归分支，为曲库中每首歌曲生成统一的二维情绪表示。  
在系统中，本文实现了多模态输入、情绪推理、曲库检索与播放器控制等模块，并通过欧氏距离完成用户情绪与歌曲情绪之间的近邻匹配。实验记录表明，音乐 Arousal 分支的 R² 约为0.59，Valence分支的R²约为0.35，证明基于连续情绪空间的建模方式具备一定的可行性。本文工作的价值在于将多模态情绪理解与音乐推荐流程打通，为实时情绪感知推荐系统的工程实现提供了可复用的代码框架与实验路径。  

关键词: 情绪识别；音乐推荐系统；多模态融合；双向长短时记忆网络  

**ABSTRACT**  
Existing recommendation methods in digital music platforms typically rely on listening history, user preference behaviors, and collaborative filtering. Although these approaches improve recommendation efficiency to some extent, they fail to reflect users’ instantaneous emotional needs in specific contexts. To address this limitation, this paper investigates the design of an intelligent music recommendation system based on emotion recognition, taking the user’s current emotional state as the core driving factor. A prototype system is developed that integrates multimodal emotion recognition, music emotion modeling, and desktop-based interactive recommendation.  
For user emotion recognition, the system supports both textual and visual inputs. In the text modality, sentence data from the IEMOCAP and MELD datasets are used to train a RoBERTa-based joint learning model for emotion classification and valence–arousal (VA) regression. In the image modality, the AffectNet dataset is employed, and a ResNet18 model is adopted for six-class emotion recognition. The predicted discrete emotion categories are further mapped into the continuous VA space to obtain coordinate representations.For music emotion modeling, acoustic features are taken as the primary basis. Using the MediaEval 2018 dataset, MFCC, Gammatone Frequency Cepstral Coefficients (GTF), and Chroma features are extracted, and separate regression branches are constructed for arousal and valence prediction, enabling a unified two-dimensional emotional representation for each song.  
The system implements modules for multimodal input processing, emotion inference, music library retrieval, and playback control. Nearest-neighbor matching between user emotions and song emotions is performed based on Euclidean distance in the continuous emotion space. Experimental results show that the R² score of the arousal branch reaches approximately 0.59, while that of the valence branch is about 0.35, demonstrating the feasibility of modeling music emotion in a continuous affective space. This work bridges multimodal emotion understanding with the music recommendation pipeline and provides a reusable code framework and experimental methodology for real-time emotion-aware recommendation systems.  

Key words: Emotion recognition; music recommendation system; Multimodal fusion；BiLSTM  