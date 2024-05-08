<h1>Introduction</h1>

This codebase is related to fusion networks dveloped for privacy preserved infant pose estimation. The following fusion techniques are avaialble in the implenetation.
- Addition
- Concatenation
- Iterative Attentional Feature Fusion

<h1>How to get started?</h1>

The code supports 2D human pose estimation of infants. You can either train a fusion model from randon weight initialization or fine tune a model initialised with weights of a fuion network fully trained using SLP dataset. The steps are as follows:
- Download the [datasets](#datasets). 
- To train a fusion model from the begining using SMaL dataset
    - Navigate to [notebook](infant_pose_estimation/model_train_and_test.ipynb), which can be easily run on Colab.
    - Setup fusion model options under "setup options". Here the ``fuse_stage`` and ``fuse_type`` can be configured along with other required options which are explained in the notebook itself. Possible options for ``fuse_stage`` and ``fuse_type`` are:
        - fuse_stage - stage at which fusion operation takes place ( either 2 or 3 )
        - fuse_type - which fusion type is used ( either add, concat or iAFF)
    - Execute the notebook.
    - The trained model will be availbe in the output folder.
- To fine tune a model initialised with weights of a fuion network fully trained using SLP dataset
    - Locate at the root folder of the codebase.
    - Intsall the dependencies available in [requitements.txt](requirements.txt).
    - Execute the following command to train a new model usinf SLP dataset:</n>
    `python main.py --modelConf config/HRposeFuseNetNewUnweighted_v2.conf --mod_src depth PM --fuse_stage n --fuse_type 'fuse_type' --ds_fd '/path/to/SLP_full_dataset'`.</n>
    Possible options for ``fuse_stage`` and ``fuse_type`` are same as above.
    - The trained model will be avaialble in output folder.
    - Now fine tuning can be done by initialising the fuison model with above trained model. For this navigate to [notebook](infant_pose_estimation/model_train_and_test.ipynb), which can be easily run on Colab. under "setup options", provide the path to model_best.pth to ``opts.bestpath_file`` and set ``opts.fine_tune`` to True.
    - Execute the notebook and the trained model will be availbe in the output folder.

<a id="datasets"></a><h1>Datasets</h1>

- [SLP dataset](https://web.northeastern.edu/ostadabbas/2019/06/27/multimodal-in-bed-pose-estimation/)
- [SMaL dataset](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/C8HGRU)

<h1>References</h1>

- *Sun, Ke, et al.* **‘Deep High-Resolution Representation Learning for Human Pose Estimation’**. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2019, pp. 5686–5696, https://doi.org10.1109/CVPR.2019.00584.
- *Dai, Yimian, et al.* **‘Attentional Feature Fusion’**. 2021 IEEE Winter Conference on Applications of Computer Vision (WACV), IEEE, 2021, pp. 3559–3568, https://doi.org10.1109/WACV48630.2021.00360. [[codebase](https://github.com/YimianDai/open-aff)]
- *Liu, Shuangjun, Xiaofei Huang, et al.* **‘Simultaneously-Collected Multimodal Lying Pose Dataset: Enabling In-Bed Human Pose Monitoring’**. IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 1, Jan. 2023, pp. 1106–1118, https://doi.org10.1109/TPAMI.2022.3155712. [[codebase](https://github.com/ostadabbas/SLP-Dataset-and-Code)]
- *Dayarathna, Thisun, et al.* **‘Privacy-Preserving in-Bed Pose Monitoring: A Fusion and Reconstruction Study’**. Expert Systems with Applications, vol. 213, Mar. 2023, p. 119139, https://doi.org10.1016/j.eswa.2022.119139.
- *Kyrollos, Daniel G., et al.* **‘Under the Cover Infant Pose Estimation Using Multimodal Data’**. IEEE Transactions on Instrumentation and Measurement, vol. 72, 2023, pp. 1–12, https://doi.org10.1109/TIM.2023.3244220. [[codebase](https://github.com/DanielKyr/SMaL)]