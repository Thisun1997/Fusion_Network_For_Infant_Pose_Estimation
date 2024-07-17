<h1>Introduction</h1>

This codebase is related to fusion networks developed for privacy preserved 2D infant pose estimation. The following fusion techniques are available in the implementation:
- Addition
- Concatenation
- Iterative Attentional Feature Fusion (iAFF)

<h1>How to get started?</h1>

You can either train a fusion model from random weight initialization or fine-tune a model initialized with the weights of a fusion network fully trained using the SLP dataset. The steps are as follows:
- Download the [datasets](#datasets). 
- To train a fusion model from the beginning using the SMaL dataset:
    1. Navigate to [notebook](infant_pose_estimation/model_train_and_test.ipynb), which can be easily run on Colab.
    2. Setup fusion model options under "setup options". Here, the ``fuse_stage`` and ``fuse_type`` can be configured along with other required options, which are explained in the notebook itself. Possible options for ``fuse_stage`` and ``fuse_type`` are:
        - fuse_stage - stage at which fusion operation takes place ( either 2 or 3 )
        - fuse_type - which fusion type is used ( either add, concat, or iAFF)
    3. Execute the notebook.
    4. The trained model will be available in the output folder.
- To fine tune a model initialized with the weights of a fusion network fully trained using the SLP dataset:
    1. Locate the root folder of the codebase.
    2. Install the dependencies available in [requirements.txt](requirements.txt).
    3. Execute the following command to train a new model using the SLP dataset:
    `python main.py --modelConf config/HRposeFuseNetNewUnweighted_v2.conf --mod_src depth PM --fuse_stage n --fuse_type 'fuse_type' --ds_fd '/path/to/SLP_full_dataset'`.
    Possible options for ``fuse_stage`` and ``fuse_type`` are the same as above.
    4. The trained model will be available in the output folder.
    5. Now fine tuning can be done by initializing the fuison model with the above trained model. For this navigate to [notebook](infant_pose_estimation/model_train_and_test.ipynb), which can be easily run on Colab. Under "setup options", provide the path to model_best.pth to ``opts.bestpath_file`` and set ``opts.fine_tune`` to True.
    6. Execute the notebook, and the trained model will be available in the output folder.

<a id="datasets"></a><h1>Datasets</h1>

- [SLP dataset](https://web.northeastern.edu/ostadabbas/2019/06/27/multimodal-in-bed-pose-estimation/)
- [SMaL dataset](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/C8HGRU)

<h1>References</h1>

- *Sun, Ke, et al.* **‘Deep High-Resolution Representation Learning for Human Pose Estimation’**. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2019, pp. 5686–5696, https://doi.org10.1109/CVPR.2019.00584.
- *Dai, Yimian, et al.* **‘Attentional Feature Fusion’**. 2021 IEEE Winter Conference on Applications of Computer Vision (WACV), IEEE, 2021, pp. 3559–3568, https://doi.org10.1109/WACV48630.2021.00360. [[codebase](https://github.com/YimianDai/open-aff)]
- *Liu, Shuangjun, Xiaofei Huang, et al.* **‘Simultaneously-Collected Multimodal Lying Pose Dataset: Enabling In-Bed Human Pose Monitoring’**. IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 1, Jan. 2023, pp. 1106–1118, https://doi.org10.1109/TPAMI.2022.3155712. [[codebase](https://github.com/ostadabbas/SLP-Dataset-and-Code)]
- *Dayarathna, Thisun, et al.* **‘Privacy-Preserving in-Bed Pose Monitoring: A Fusion and Reconstruction Study’**. Expert Systems with Applications, vol. 213, Mar. 2023, p. 119139, https://doi.org10.1016/j.eswa.2022.119139.
- *Kyrollos, Daniel G., et al.* **‘Under the Cover Infant Pose Estimation Using Multimodal Data’**. IEEE Transactions on Instrumentation and Measurement, vol. 72, 2023, pp. 1–12, https://doi.org10.1109/TIM.2023.3244220. [[codebase](https://github.com/DanielKyr/SMaL)]
