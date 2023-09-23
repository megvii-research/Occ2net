# Occ2net [ICCV 2023]
#### Occ2Net: Robust Image Matching Based on 3D Occupancy Estimation for Occluded Regions
  
#### Occ2Net: 一种基于3D 占据估计的有效且稳健的带有遮挡区域图像匹配方法

#### 论文链接：https://arxiv.org/abs/2308.16160

Image matching is a fundamental and critical task in various visual applications, such as Simultaneous Localization and Mapping (SLAM) and image retrieval, which require accurate pose estimation. However, most existing methods ignore the occlusion relations between objects caused by camera motion and scene structure. In this paper, we propose Occ2Net, a novel image matching method that models occlusion relations using 3D occupancy and infers matching points in occluded regions. Thanks to the inductive bias encoded in the Occupancy Estimation (OE) module, it greatly simplifies bootstrapping of a multi-view consistent 3D representation that can then integrate information from multiple views. Together with an Occlusion-Aware (OA) module, it incorporates attention layers and rotation alignment to enable matching between occluded and visible points. We evaluate our method on both real-world and simulated datasets and demonstrate its superior performance over state-of-the-art methods on several metrics, especially in occlusion scenarios.
<img width="568" alt="图片" src="https://github.com/megvii-research/Occ2net/assets/106311350/681a9bb8-593c-4f25-84f1-e62dad238051">

图像匹配是各种视觉应用中的基础和关键任务，如同时定位和映射（SLAM）和图像检索，这些任务都需要精确的姿态估计。然而，大多数现存的方法忽视了由相机运动和场景结构引起的对象间的遮挡关系。在本文中，我们提出了一种新颖的图像匹配方法Occ2Net，该方法使用3D占位图模型来描述遮挡关系，并推断出被遮挡区域的匹配点。借助占位估计（OE）模块编码的归纳偏差，它大大简化了构建一个多视图一致的3D表示的过程，该表示能够整合多视图信息。再结合遮挡感知（OA）模块，通过引入注意力层和旋转对齐，实现了被遮挡点和可见点的匹配。我们在真实世界和模拟数据集上评估了我们的方法，结果显示其在多项指标上，尤其是在遮挡场景下，优于当前最先进的方法。


<img width="1100" alt="image" src="https://github.com/megvii-research/Occ2net/assets/106311350/b0a2f03b-67bf-48fc-813e-d4b2f37d5070">





