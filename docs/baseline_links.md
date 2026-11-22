I went through your GATE-IR experiment matrix and searched for the GitHub implementations, organized by the same stages as your proposed experiments. I’ve included official author repositories where I could verify them, marked community implementations separately, and flagged methods for which I could not verify a public repository. The list follows the baselines in your survey and experiment tables.&#x20;

Pasted markdown(20261108-203034).md



Checked: 9 November 2026.

### Repository status

- Official — repository linked by the authors or identified as their implementation.
- Community / port — a usable implementation, but not verified as the authors’ release.
- Not verified — I could not find a reliable public GitHub implementation. This does not prove none exists.

## 1. Stage A — All-in-one restoration and adaptive routing

These are important competitors to your explicit weather gate: they attempt to handle multiple degradations without requiring a separate hard-routing decision.

| Baseline         | GitHub repository                                                                                          | Status / notes                                              |
| ---------------- | ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| AirNet           | [XLearning-SCU/2022-CVPR-AirNet](https://github.com/XLearning-SCU/2022-CVPR-AirNet?utm_source=chatgpt.com) | Official                                                    |
| TransWeather     | [jeya-maria-jose/TransWeather](https://github.com/jeya-maria-jose/TransWeather?utm_source=chatgpt.com)     | Official; check completeness of training/evaluation scripts |
| WeatherDiffusion | [IGITUGraz/WeatherDiffusion](https://github.com/IGITUGraz/WeatherDiffusion?utm_source=chatgpt.com)         | Official                                                    |
| WGWS-Net         | [zhuyr97/WGWS-Net](https://github.com/zhuyr97/WGWS-Net?utm_source=chatgpt.com)                             | Official                                                    |
| PromptIR         | [va1shn9v/PromptIR](https://github.com/va1shn9v/PromptIR?utm_source=chatgpt.com)                           | Official                                                    |
| DA-CLIP          | [Laid0513/daclip](https://github.com/Laid0513/daclip?utm_source=chatgpt.com)                               | Official                                                    |
| Histoformer      | [sunshangquan/Histoformer](https://github.com/sunshangquan/Histoformer?utm_source=chatgpt.com)             | Official                                                    |
| AdaIR            | [c-yn/AdaIR](https://github.com/c-yn/AdaIR?utm_source=chatgpt.com)                                         | Official                                                    |
| InstructIR       | [mv-lab/InstructIR](https://github.com/mv-lab/InstructIR?utm_source=chatgpt.com)                           | Official                                                    |
| DiffUIR          | [iSEE-Laboratory/DiffUIR](https://github.com/iSEE-Laboratory/DiffUIR?utm_source=chatgpt.com)               | Official                                                    |
| MWFormer         | [taco-group/MWFormer](https://github.com/taco-group/MWFormer?utm_source=chatgpt.com)                       | Official; repository has been described as work in progress |
| WM-MoE           | No verified public repository found                                                                        | Do not substitute an unrelated weather-forecasting MoE      |
| M2Restore        | [yz-wang/M2Restore](https://github.com/yz-wang/M2Restore?utm_source=chatgpt.com)                           | Author repository; check implementation completeness        |
| M2IR             | [Im34v/M2IR](https://github.com/Im34v/M2IR?utm_source=chatgpt.com)                                         | Public repository                                           |
| UniRestorer      | [mrluin/unirestorer](https://github.com/mrluin/unirestorer?utm_source=chatgpt.com)                         | Official                                                    |
| MoCE-IR          | [eduardzamfir/MoCE-IR](https://github.com/eduardzamfir/MoCE-IR?utm_source=chatgpt.com)                     | Official                                                    |
| ClusIR           | No verified public repository found                                                                        | —                                                           |

Repository checks:&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+14



### Gate-classifier baselines

| Baseline                                          | Repository                                                                                   | Notes                                                            |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| CLIP zero-shot classifier                         | [openai/CLIP](https://github.com/openai/CLIP?utm_source=chatgpt.com)                         | General CLIP implementation                                      |
| OpenCLIP alternative                              | [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip?utm_source=chatgpt.com) | Useful for experimenting with alternative CLIP checkpoints       |
| DA-CLIP                                           | [Laid0513/daclip](https://github.com/Laid0513/daclip?utm_source=chatgpt.com)                 | Degradation-aware CLIP                                           |
| Small CNN classifier                              | Your own implementation                                                                      | Train on the same weather labels as the proposed MLP             |
| Handcrafted-feature MLP                           | Your own implementation                                                                      | Proposed gate                                                    |
| No-gate / oracle-gate / always-both / soft-router | Your own implementation                                                                      | These are experimental controls, not standalone published models |

ADSM: I could not verify a reliable public implementation. The same applies to a dedicated implementation of the exact WM-MoE weather-restoration router described in your survey.

## 2. Stage B1 — Fog and dehazing baselines

### Deep dehazing methods

| Baseline          | GitHub repository                                                                                      | Status                           |
| ----------------- | ------------------------------------------------------------------------------------------------------ | -------------------------------- |
| AOD-Net           | [weberwcwei/AODnet-by-pytorch](https://github.com/weberwcwei/AODnet-by-pytorch?utm_source=chatgpt.com) | Community PyTorch implementation |
| GridDehazeNet     | [proteus1991/GridDehazeNet](https://github.com/proteus1991/GridDehazeNet?utm_source=chatgpt.com)       | Official                         |
| FFA-Net           | [lfliu/FFA-Net-dehaze](https://github.com/lfliu/FFA-Net-dehaze?utm_source=chatgpt.com)                 | Official                         |
| MSBDN             | No verified official repository found                                                                  | —                                |
| Dehamer           | [Li-Chongyi/Dehamer](https://github.com/Li-Chongyi/Dehamer?utm_source=chatgpt.com)                     | Official                         |
| DehazeFormer      | [IDKiro/DehazeFormer](https://github.com/IDKiro/DehazeFormer?utm_source=chatgpt.com)                   | Official                         |
| C2PNet            | [YuZheng9/C2PNet](https://github.com/YuZheng9/C2PNet?utm_source=chatgpt.com)                           | Official                         |
| DEA-Net           | [cecret3350/DEA-Net](https://github.com/cecret3350/DEA-Net?utm_source=chatgpt.com)                     | Official                         |
| RIDCP             | [RQ-Wu/RIDCP](https://github.com/RQ-Wu/RIDCP?utm_source=chatgpt.com)                                   | Official                         |
| DCMPNet           | [zhoushen1/DCMPNet](https://github.com/zhoushen1/DCMPNet?utm_source=chatgpt.com)                       | Official                         |
| MB-TaylorFormerV2 | [FVL2020/MB-TaylorFormerV2](https://github.com/FVL2020/MB-TaylorFormerV2?utm_source=chatgpt.com)       | Official                         |

Sources:&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+9



### Classical fog-enhancement baselines

These are usually better implemented as controlled preprocessing operations than treated as separate deep-learning repositories.

| Baseline                          | Suggested implementation                                                                   |
| --------------------------------- | ------------------------------------------------------------------------------------------ |
| CLAHE                             | [opencv/opencv](https://github.com/opencv/opencv?utm_source=chatgpt.com)                   |
| Adaptive gamma / gamma correction | OpenCV or custom NumPy/PyTorch operation                                                   |
| AGCWD / AGCCPF                    | Custom implementation                                                                      |
| DCP — Dark Channel Prior          | Custom implementation; be cautious applying RGB dark-channel assumptions to thermal images |
| SSR / MSR / MSRCR Retinex         | Custom implementation                                                                      |
| Guided filter                     | [opencv/opencv_contrib](https://github.com/opencv/opencv_contrib?utm_source=chatgpt.com)   |

For your proposed fog-stage swap, the most useful deep baseline repositories are DEA-Net, C2PNet and FFA-Net. Your matrix also specifically calls for CLAHE and AGCWD as lightweight alternatives.&#x20;

Pasted markdown(20261108-203034).md

Pasted markdown(20261108-203034).md



## 3. Stage B2 — Rain-removal / deraining baselines

### Deraining methods

| Baseline    | GitHub repository                                                                              | Status / notes                                     |
| ----------- | ---------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| DerainNet   | No verified official repository found                                                          | —                                                  |
| DDN         | [biubiubiu/derain-toolbox](https://github.com/biubiubiu/derain-toolbox?utm_source=chatgpt.com) | Community toolbox implementation                   |
| RESCAN      | [XiaLiPKU/RESCAN](https://github.com/XiaLiPKU/RESCAN?utm_source=chatgpt.com)                   | Official                                           |
| PReNet      | [biubiubiu/derain-toolbox](https://github.com/biubiubiu/derain-toolbox?utm_source=chatgpt.com) | Community toolbox implementation                   |
| MSPFN       | [kuijiang94/mspfn](https://github.com/kuijiang94/mspfn?utm_source=chatgpt.com)                 | Author implementation                              |
| RCDNet      | [hongwang01/RCDNet](https://github.com/hongwang01/RCDNet?utm_source=chatgpt.com)               | Official                                           |
| MPRNet      | [swz30/MPRNet](https://github.com/swz30/MPRNet?utm_source=chatgpt.com)                         | Official                                           |
| SPDNet      | [Joyies/SPDNet](https://github.com/Joyies/SPDNet?utm_source=chatgpt.com)                       | Official                                           |
| Uformer     | [ZhendongWang6/Uformer](https://github.com/ZhendongWang6/Uformer?utm_source=chatgpt.com)       | Official                                           |
| Restormer   | [swz30/Restormer](https://github.com/swz30/Restormer?utm_source=chatgpt.com)                   | Official                                           |
| DRSformer   | [cschenxiang/DRSformer](https://github.com/cschenxiang/DRSformer?utm_source=chatgpt.com)       | Official                                           |
| NeRD-Rain   | [cschenxiang/NeRD-Rain](https://github.com/cschenxiang/NeRD-Rain?utm_source=chatgpt.com)       | Official                                           |
| DRT         | [YC-Liang/DRT](https://github.com/YC-Liang/DRT?utm_source=chatgpt.com)                         | Lightweight deraining baseline                     |
| LIR         | [dongqi-fan/lir](https://github.com/dongqi-fan/lir?utm_source=chatgpt.com)                     | Lightweight implementation                         |
| SFNet       | No verified official repository found                                                          | —                                                  |
| RainMamba   | [TonyHongtaoWu/RainMamba](https://github.com/TonyHongtaoWu/RainMamba?utm_source=chatgpt.com)   | Check task scope; this is a video-deraining method |
| DeRainMamba | No verified public repository found                                                            | —                                                  |

Sources:&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+12



Useful toolbox: [biubiubiu/derain-toolbox](https://github.com/biubiubiu/derain-toolbox?utm_source=chatgpt.com) bundles several classical deraining methods, including DDN, RESCAN, PReNet, RCDNet and SPDNet. Another option is [XiaoXiao-Woo/derain](https://github.com/XiaoXiao-Woo/derain?utm_source=chatgpt.com), which includes implementations of several deraining and restoration methods.

For your paper's main rain-stage comparison, I would prioritize:

1. Your LSRB.
2. DRT — lightweight competitor.
3. DRSformer — heavier, stronger competitor.
4. Restormer — general restoration baseline.

That matches the essential comparison in your experiment matrix.&#x20;

Pasted markdown(20261108-203034).md

Pasted markdown(20261108-203034).md



## 4. General image-restoration baselines

These are useful as strong generic preprocessing methods, even if they are not specialized for fog or rain.

| Baseline            | GitHub repository                                                                                  |
| ------------------- | -------------------------------------------------------------------------------------------------- |
| Restormer           | [swz30/Restormer](https://github.com/swz30/Restormer?utm_source=chatgpt.com)                       |
| Uformer             | [ZhendongWang6/Uformer](https://github.com/ZhendongWang6/Uformer?utm_source=chatgpt.com)           |
| NAFNet              | [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet?utm_source=chatgpt.com)         |
| SwinIR              | [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR?utm_source=chatgpt.com)               |
| MambaIR / MambaIRv2 | [csguoh/MambaIR](https://github.com/csguoh/MambaIR?utm_source=chatgpt.com)                         |
| WeatherDiffusion    | [IGITUGraz/WeatherDiffusion](https://github.com/IGITUGraz/WeatherDiffusion?utm_source=chatgpt.com) |
| DiffUIR             | [iSEE-Laboratory/DiffUIR](https://github.com/iSEE-Laboratory/DiffUIR?utm_source=chatgpt.com)       |

For SwinIR and other restoration architectures, [XPixelGroup/BasicSR](https://github.com/XPixelGroup/BasicSR?utm_source=chatgpt.com) is also a useful supporting framework.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+3



## 5. Task-driven / joint restoration + object detection

These are particularly important because they challenge the assumption that improving image quality necessarily improves detection.

| Baseline                                                 | GitHub repository                                                                                  | Status / notes                                                                             |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| IA-YOLO                                                  | [wenyyu/Image-Adaptive-YOLO](https://github.com/wenyyu/Image-Adaptive-YOLO?utm_source=chatgpt.com) | Official; older TensorFlow-based implementation                                            |
| DSNet                                                    | No verified official repository for the cited work                                                 | Avoid similarly named unrelated projects                                                   |
| TogetherNet                                              | [yz-wang/TogetherNet](https://github.com/yz-wang/TogetherNet?utm_source=chatgpt.com)               | Official                                                                                   |
| DENet                                                    | [NIvykk/DENet](https://github.com/NIvykk/DENet?utm_source=chatgpt.com)                             | Official                                                                                   |
| BAD-Net / Detection-Friendly Dehazing                    | No verified official repository found                                                              | —                                                                                          |
| FriendNet                                                | [fanyihua0309/FriendNet](https://github.com/fanyihua0309/FriendNet?utm_source=chatgpt.com)         | Public testing code; training code was not available in the repository information I found |
| D-YOLO                                                   | No verified official repository found                                                              | —                                                                                          |
| RDMNet                                                   | [xfwang23/RDMNet](https://github.com/xfwang23/RDMNet?utm_source=chatgpt.com)                       | Public implementation                                                                      |
| WARLearn                                                 | [ShubhamAgarwal12/WARLearn](https://github.com/ShubhamAgarwal12/WARLearn?utm_source=chatgpt.com)   | Public implementation; based on YOLOv3                                                     |
| ERUP-YOLO                                                | No verified public repository found                                                                | —                                                                                          |
| UniDet-D                                                 | No verified public repository found                                                                | —                                                                                          |
| DTRDNet                                                  | No verified official repository found                                                              | —                                                                                          |
| Aloufi et al. (2024), weather classification + detection | No verified author repository found                                                                | Paper is available                                                                         |

Sources:&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+6



Important: I would not silently replace DTRDNet or BAD-Net with another model. Your survey identifies them as close prior work; if you cannot obtain their code, explain the limitation and use the closest reproducible alternatives, especially IA-YOLO, TogetherNet and RDMNet.

## 6. Stage C — Object-detection architectures

### YOLO family

| Baseline | GitHub repository                                                                            |
| -------- | -------------------------------------------------------------------------------------------- |
| YOLOv5   | [ultralytics/yolov5](https://github.com/ultralytics/yolov5?utm_source=chatgpt.com)           |
| YOLOv7   | [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7?utm_source=chatgpt.com)             |
| YOLOv8   | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics?utm_source=chatgpt.com) |
| YOLOv9   | [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9?utm_source=chatgpt.com)             |
| YOLOv10  | [THU-MIG/yolov10](https://github.com/THU-MIG/yolov10?utm_source=chatgpt.com)                 |
| YOLO11   | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics?utm_source=chatgpt.com) |
| YOLOv12  | [sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12?utm_source=chatgpt.com)     |
| YOLOv13  | [iMoonLab/yolov13](https://github.com/iMoonLab/yolov13?utm_source=chatgpt.com)               |

### DETR and other real-time detectors

| Baseline        | GitHub repository                                                                                                |
| --------------- | ---------------------------------------------------------------------------------------------------------------- |
| RT-DETR         | [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR?utm_source=chatgpt.com)                                   |
| RT-DETRv2       | [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR?utm_source=chatgpt.com)                                   |
| RT-DETRv3       | [clxia12/RT-DETRv3](https://github.com/clxia12/RT-DETRv3?utm_source=chatgpt.com)                                 |
| D-FINE          | [Peterande/D-FINE](https://github.com/Peterande/D-FINE?utm_source=chatgpt.com)                                   |
| DEIM            | [Intellindust-AI-Lab/DEIM](https://github.com/Intellindust-AI-Lab/DEIM?utm_source=chatgpt.com)                   |
| RF-DETR         | [roboflow/rf-detr](https://github.com/roboflow/rf-detr?utm_source=chatgpt.com)                                   |
| RTMDet          | [open-mmlab/mmyolo](https://github.com/open-mmlab/mmyolo?utm_source=chatgpt.com)                                 |
| Faster R-CNN    | [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection?utm_source=chatgpt.com)                       |
| DETR            | [facebookresearch/detr](https://github.com/facebookresearch/detr?utm_source=chatgpt.com)                         |
| Deformable DETR | [fundamentalvision/Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR?utm_source=chatgpt.com) |
| DINO            | [IDEA-Research/DINO](https://github.com/IDEA-Research/DINO?utm_source=chatgpt.com)                               |
| Co-DETR         | [Sense-X/Co-DETR](https://github.com/Sense-X/Co-DETR?utm_source=chatgpt.com)                                     |

Sources:&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+11



### Thermal-specific detectors

| Baseline                           | GitHub repository                                                                                                | Status / notes                                          |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| MS-YOLO                            | [JialiZhang1016/YOLO_thermal_small](https://github.com/JialiZhang1016/YOLO_thermal_small?utm_source=chatgpt.com) | Public implementation; check training-code completeness |
| CRT-YOLO                           | No verified public repository found                                                                              | —                                                       |
| High-altitude IR pruning + NMS     | No verified public repository found                                                                              | —                                                       |
| Shyam & Yoo thermal SR + detection | No verified public repository found                                                                              | —                                                       |

The MS-YOLO repository corresponds to the infrared detector using MobileNetV4 and SlideLoss.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

JialiZhang1016/YOLO_thermal_small · GitHub+1



### Small-object detection, necks and losses

| Baseline   | GitHub repository                                                                                                | Notes                                                        |
| ---------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| BiFPN      | [google/automl — EfficientDet](https://github.com/google/automl/tree/master/efficientdet?utm_source=chatgpt.com) | Reference implementation                                     |
| PAFPN      | [open-mmlab/mmyolo](https://github.com/open-mmlab/mmyolo?utm_source=chatgpt.com)                                 | Framework implementation                                     |
| P2 head    | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics?utm_source=chatgpt.com)                     | Usually implemented as a detector configuration/model change |
| QueryDet   | [Small-Object-Detection/QueryDet](https://github.com/Small-Object-Detection/QueryDet?utm_source=chatgpt.com)     | Official                                                     |
| SAHI       | [obss/sahi](https://github.com/obss/sahi?utm_source=chatgpt.com)                                                 | Official slicing-aided inference                             |
| NWD        | [jwwangchn/NWD](https://github.com/jwwangchn/NWD?utm_source=chatgpt.com)                                         | Official                                                     |
| Focal-EIoU | No single canonical repository verified                                                                          | Implement in the detector loss                               |
| SlideLoss  | [JialiZhang1016/YOLO_thermal_small](https://github.com/JialiZhang1016/YOLO_thermal_small?utm_source=chatgpt.com) | Included with MS-YOLO                                        |

Sources:&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+3



## 7. RGB–thermal fusion baselines

These use paired RGB and thermal inputs, so your survey correctly treats them as extra-modality upper bounds, not directly equivalent competitors to an IR-only GATE-IR system.

| Baseline     | GitHub repository                                                                                                    | Status                |
| ------------ | -------------------------------------------------------------------------------------------------------------------- | --------------------- |
| CFT          | [DocF/multispectral-object-detection](https://github.com/DocF/multispectral-object-detection?utm_source=chatgpt.com) | Public implementation |
| ProbEn       | [guanning03/ProbEn](https://github.com/guanning03/ProbEn?utm_source=chatgpt.com)                                     | Public implementation |
| ICAFusion    | [chanchanchan97/ICAFusion](https://github.com/chanchanchan97/ICAFusion?utm_source=chatgpt.com)                       | Official              |
| CSAA         | No verified repository found                                                                                         | —                     |
| RSDet        | No verified repository found                                                                                         | —                     |
| CAFF-DINO    | No verified repository found                                                                                         | —                     |
| Fusion-Mamba | No verified repository found                                                                                         | —                     |
| WaveMamba    | No verified repository found                                                                                         | —                     |

Sources:&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+2



## 8. Cross-modal translation baselines

These are relevant to the proposed CycleGAN → pseudo-RGB → teacher pathway.

| Baseline | GitHub repository                                                                                                                | Status                |
| -------- | -------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| CycleGAN | [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix?utm_source=chatgpt.com)           | Official              |
| CUT      | [taesungp/contrastive-unpaired-translation](https://github.com/taesungp/contrastive-unpaired-translation?utm_source=chatgpt.com) | Official              |
| UNIT     | [mingyuliutw/UNIT](https://github.com/mingyuliutw/UNIT?utm_source=chatgpt.com)                                                   | Public implementation |
| MUNIT    | [NVlabs/MUNIT](https://github.com/NVlabs/MUNIT?utm_source=chatgpt.com)                                                           | Official              |
| DRIT++   | [HsinYingLee/DRIT](https://github.com/HsinYingLee/DRIT?utm_source=chatgpt.com)                                                   | Public implementation |
| DiffV2IR | No verified public repository found                                                                                              | —                     |
| F-ViTA   | No verified public repository found                                                                                              | —                     |
| CM-Diff  | No verified public repository found                                                                                              | —                     |
| T2V-DDPM | No verified public repository found                                                                                              | —                     |
| DiffTV   | No verified public repository found                                                                                              | —                     |
| InfraGAN | No verified public repository found                                                                                              | —                     |

The classic CycleGAN implementation is available here:&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



## 9. Knowledge-distillation baselines

This is particularly important for your Table 4 experiment.

| Baseline                       | GitHub repository                                                                              | Status / notes                                                        |
| ------------------------------ | ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| FitNets                        | [HikariTJU/LD](https://github.com/HikariTJU/LD?utm_source=chatgpt.com)                         | Includes FitNets feature imitation option                             |
| FGD                            | [yzd-v/FGD](https://github.com/yzd-v/FGD?utm_source=chatgpt.com)                               | Official                                                              |
| MGD                            | [yzd-v/MGD](https://github.com/yzd-v/MGD?utm_source=chatgpt.com)                               | Official                                                              |
| CWD                            | [git.io/Distiller](https://git.io/Distiller?utm_source=chatgpt.com)                            | Linked by the ICCV paper                                              |
| PKD                            | No verified official repository found                                                          | —                                                                     |
| Localization Distillation (LD) | [HikariTJU/LD](https://github.com/HikariTJU/LD?utm_source=chatgpt.com)                         | Official                                                              |
| CrossKD                        | [jbwang1997/CrossKD](https://github.com/jbwang1997/CrossKD?utm_source=chatgpt.com)             | Official                                                              |
| BCKD                           | [TinyTigerPan/BCKD](https://github.com/TinyTigerPan/BCKD?utm_source=chatgpt.com)               | Official                                                              |
| FreqKD                         | [tkeval/freq_decoupled_kd](https://github.com/tkeval/freq_decoupled_kd?utm_source=chatgpt.com) | Public code release; README describes it as an anonymous code release |
| No-distillation                | Your own implementation                                                                        | Control baseline                                                      |
| Logit-only KD                  | Your own implementation                                                                        | Control baseline                                                      |
| Standard feature KD            | Your own implementation                                                                        | Control baseline                                                      |
| RGB pretraining without KD     | Your own training pipeline                                                                     | Control baseline                                                      |

Sources:&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+6



One useful detail: the LD repository also contains FitNets-style feature imitation, so you can reuse it rather than implementing that baseline from scratch.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



## 10. Weather augmentation / robustness baselines

| Baseline                     | GitHub repository                                                                                                  |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| AugMix                       | [google-research/augmix](https://github.com/google-research/augmix?utm_source=chatgpt.com)                         |
| General image augmentation   | [albumentations-team/albumentations](https://github.com/albumentations-team/albumentations?utm_source=chatgpt.com) |
| YOLO weather augmentation    | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics?utm_source=chatgpt.com)                       |
| Domain randomization         | Usually implemented in your own augmentation pipeline                                                              |
| Synthetic weather corruption | Usually custom; use real-weather data for the primary evaluation                                                   |

Your matrix specifically calls for comparing against a detector trained with weather augmentation but without preprocessing. This is an important control, not just another optional experiment.&#x20;

Pasted markdown(20261108-203034).md

Pasted markdown(20261108-203034).md



## What I would actually download first

You do not need to clone every repository above and run every model. Your own experiment matrix recommends independent stage-wise swaps rather than a huge full-factorial experiment.&#x20;

Pasted markdown(20261108-203034).md



Here is the practical shortlist I would start with:

| Priority | Experiment                          | Repositories to start with                                                                                                                                                                                                                                                    |
| -------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1        | All-in-one restoration              | [PromptIR](https://github.com/va1shn9v/PromptIR?utm_source=chatgpt.com), [Histoformer](https://github.com/sunshangquan/Histoformer?utm_source=chatgpt.com), [DA-CLIP](https://github.com/Laid0513/daclip?utm_source=chatgpt.com)                                              |
| 2        | Task-driven restoration + detection | [IA-YOLO](https://github.com/wenyyu/Image-Adaptive-YOLO?utm_source=chatgpt.com), [TogetherNet](https://github.com/yz-wang/TogetherNet?utm_source=chatgpt.com), [RDMNet](https://github.com/xfwang23/RDMNet?utm_source=chatgpt.com)                                            |
| 3        | Fog enhancement                     | [DEA-Net](https://github.com/cecret3350/DEA-Net?utm_source=chatgpt.com), [C2PNet](https://github.com/YuZheng9/C2PNet?utm_source=chatgpt.com), OpenCV CLAHE                                                                                                                    |
| 4        | Rain enhancement                    | [DRT](https://github.com/YC-Liang/DRT?utm_source=chatgpt.com), [DRSformer](https://github.com/cschenxiang/DRSformer?utm_source=chatgpt.com), [Restormer](https://github.com/swz30/Restormer?utm_source=chatgpt.com)                                                           |
| 5        | Detection architectures             | [Ultralytics](https://github.com/ultralytics/ultralytics?utm_source=chatgpt.com), [YOLOv12](https://github.com/sunsmarterjie/yolov12?utm_source=chatgpt.com), [RT-DETR](https://github.com/lyuwenyu/RT-DETR?utm_source=chatgpt.com)                                           |
| 6        | Small-object detection              | [QueryDet](https://github.com/Small-Object-Detection/QueryDet?utm_source=chatgpt.com), [NWD](https://github.com/jwwangchn/NWD?utm_source=chatgpt.com), BiFPN                                                                                                                  |
| 7        | Distillation                        | [FGD](https://github.com/yzd-v/FGD?utm_source=chatgpt.com), [MGD](https://github.com/yzd-v/MGD?utm_source=chatgpt.com), [CrossKD](https://github.com/jbwang1997/CrossKD?utm_source=chatgpt.com), [FreqKD](https://github.com/tkeval/freq_decoupled_kd?utm_source=chatgpt.com) |
| 8        | RGB–thermal upper bound             | [ICAFusion](https://github.com/chanchanchan97/ICAFusion?utm_source=chatgpt.com), [ProbEn](https://github.com/guanning03/ProbEn?utm_source=chatgpt.com)                                                                                                                        |

My recommendation: start with the first five rows, then add the KD experiments. Those establish whether the central GATE-IR claim—weather-aware routing improves detection while avoiding unnecessary restoration compute—is actually competitive. The other repositories are useful extensions, but they should not distract from that central comparison.

One final caveat: most restoration repositories above were designed and trained for RGB images, not single-channel thermal imagery. Their availability does not mean they are immediately valid thermal baselines. You will need to check input-channel handling, preprocessing, training data, and retraining requirements before comparing their results with GATE-IR.