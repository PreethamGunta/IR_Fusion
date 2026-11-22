Engineering Roadmap and Architectural Design for Physics-Informed Multimodal Weather Synthesis
The deployment of automated driving systems (ADS) and advanced driver assistance systems (ADAS) in complex, real-world environments requires robust perception capabilities that function reliably across all weather and lighting conditions. To satisfy the stringent safety requirements outlined in international standards such as ISO 26262 for Functional Safety (FuSa) and ISO 21448 for the Safety of the Intended Functionality (SOTIF), autonomous driving systems must be rigorously validated against hazardous edge-case scenarios1. While visible-light (RGB) sensors provide high-resolution textural and semantic information, they suffer severe degradation in adverse weather conditions such as rain and fog due to atmospheric scattering4. Conversely, thermal infrared (IR) sensors offer resilience to visual obscurants and low-light conditions by detecting heat signatures, though they lack detailed geometric textures and are subject to complex thermodynamic attenuation6.
Because collecting strictly aligned, multimodal paired data under severe and highly specific weather conditions is logistically prohibitive, dangerous, and expensive, synthesizing realistic adverse weather onto clean RGB-IR datasets has emerged as a critical data augmentation and SOTIF validation strategy3. However, modeling environmental degradation across distinct spectra presents a profound engineering challenge: weather phenomena manifest through fundamentally different physical mechanisms in the visible versus the thermal domains.
This technical design document details a comprehensive engineering roadmap and architecture for a state-of-the-art (SOTA) data generation pipeline. The pipeline leverages Physics-Informed Latent Diffusion Models (PID) to synthesize highly realistic rain and fog distortions onto strictly aligned RGB and thermal image pairs. By intertwining deep generative priors with deterministic atmospheric scattering and thermodynamic attenuation models, the proposed architecture ensures that the synthetic outputs are perceptually convincing, physically accurate across both modalities, and perfectly aligned in their semantic geometry.
Architecture and Model Selection
The fundamental challenge in multimodal degradation synthesis is applying domain-specific distortions while preserving the underlying semantic geometry and strictly maintaining the spatial alignment of the source images. The architecture must simultaneously serve as a generative engine for complex artifacts (e.g., rain streaks) and a highly constrained physical renderer.
The landscape of generative modeling for image-to-image translation has evolved rapidly. Historically, Generative Adversarial Networks (GANs), including architectures such as CycleGAN, Pix2Pix, and their derivatives, were the frameworks of choice11. However, in the context of safety-critical dataset generation for autonomous driving, GANs exhibit severe limitations. They are notoriously prone to mode collapse, exhibit training instability, and frequently hallucinate semantically incorrect artifacts11. For instance, unsupervised CycleGAN models often generate unpredictable content during translation, which can corrupt the bounding box annotations required for downstream object detection tasks11.
Recent advancements have established Conditional Latent Diffusion Models (LDMs) and Flow-Matching models as the definitive state-of-the-art for high-fidelity image synthesis6. Diffusion models provide stable, iterative optimization trajectories, superior mode coverage, and the critical ability to strictly condition the output on physical priors. For the specific task of paired multi-modal weather degradation, a Physics-Informed Diffusion (PID) model architecture is the optimal choice6. Standard diffusion models treat infrared images merely as a stylistic variation of RGB data, ignoring fundamental physical laws6. A PID architecture injects physical constraints directly into the diffusion denoising loop, ensuring that the synthesized weather artifacts obey atmospheric scattering laws in the RGB domain and thermodynamic emission laws in the IR domain16.
When processing strictly paired RGB and IR data, architects must carefully determine how the modalities interact within the network. The choice of fusion strategy dictates the network's ability to maintain spatial alignment while allowing for domain-specific physical rendering.

Fusion Strategy
Architecture Description
Feasibility for RGBT Weather Synthesis
Drawbacks / Limitations
Early Fusion
Concatenation of RGB (3-channel) and IR (1-channel) inputs at the pixel level before the initial convolutional layers18.
Poor. Forces the network to learn a joint distribution of fundamentally distinct physical phenomena (reflected photons vs. emitted radiation) at the lowest feature level.
The network struggles to untangle why fog increases luminance in RGB pixels but simply lowers thermal contrast in the corresponding IR pixels.
Late Fusion
Processing modalities completely independently through separate backbones, merging only high-level semantic features prior to the output head18.
Suboptimal. Fails to leverage cross-modal structural cues during the intermediate generative steps.
High risk of spatial misalignment. Synthesized rain streaks in the RGB image might not correspond geometrically to the thermal occlusions in the IR image.
Mid Fusion
Intermediate feature maps are fused (e.g., via addition or concatenation) at various stages of the network18.
Moderate. Allows for some cross-talk between modalities, but simple summation often dilutes the physically specific features of each spectrum.
Can lead to feature entanglement where modality-specific, content-specific, and task-specific information becomes conflated19.
Parallel Cross-Attention
Dual parallel networks operating in a shared latent space, exchanging information explicitly through cross-modal attention mechanisms4.
Optimal. Allows distinct physical rendering per channel while enforcing strict geometric consistency across the latent space.
Higher computational memory footprint during training due to maintaining dual U-Net or DiT backbones.

For this pipeline, the Parallel Networks with Shared Latent Spaces and Cross-Modal Attention strategy is highly recommended20. The architecture utilizes two parallel U-Net backbones operating in a compressed latent space provided by a Variational Autoencoder (VAE)21. The RGB stream processes the visual scattering features of rain and fog, while the IR stream processes the thermodynamic attenuation features. To bridge these streams, Spatial Cross-Attention layers are deployed at each resolution scale within the U-Net. In these blocks, the Query vector is derived from the RGB stream, while the Key and Value vectors are derived from the IR stream (and vice versa symmetrically). This mathematical interaction enforces geometric structural consistency—ensuring that a raindrop occluding a specific coordinate occludes the exact same coordinate in both modalities—while allowing the channel-wise feature representations to remain physically distinct20.
Handling Physics and Modality Differences
Rain and fog interact with electromagnetic radiation differently depending on the wavelength of the sensor. The visible spectrum (400–700 nm) is primarily governed by scattering and reflection, whereas the long-wave infrared spectrum (LWIR, 8–14 m) is governed by thermal emission, absorption, and surface emissivity modifications22. The synthesis pipeline must mathematically decouple and model these physical realities within the respective network branches.
Modeling Visible Light Degradation (RGB Domain)
In the RGB domain, fog and rain are modeled primarily through atmospheric scattering and optical attenuation. Fog consists of suspended water droplets that are comparable in size to visible light wavelengths, leading to Mie scattering23. The visual degradation caused by fog is traditionally modeled by the Koschmieder equation and the atmospheric scattering model26:

In this formulation,  represents the observed degraded RGB image,  is the clear scene radiance (the clean input image), and  is the global atmospheric light. The variable  represents the transmission map, defining the portion of light that successfully reaches the camera sensor without undergoing scattering. According to the Beer-Lambert law, this transmission map decays exponentially with the depth of the scene29:

Here,  is the atmospheric scattering coefficient (which is proportional to the density of the fog) and  is the physical distance or scene depth29. The first term of the Koschmieder equation () dictates the direct attenuation of the original scene, while the second term () represents the airlight, which causes the characteristic white or gray wash-out associated with fog26.
Rain introduces dynamic, high-frequency spatial anomalies. Visually, rain manifests as bright, semi-transparent streaks caused by the refraction and reflection of ambient light by falling water droplets, alongside localized droplet occlusions on the camera lens itself32. The physical model for synthesizing a rainy RGB image expands upon the scattering model:

In this equation,  is a spatial binary mask representing immediate droplet occlusion,  represents the specular highlights and motion blur of the falling rain streaks, and the final term accounts for the "veiling effect" (a light, localized fog) that frequently accompanies heavy precipitation due to aerosolized water particles32.
Modeling Thermal Infrared Degradation (IR Domain)
Thermal cameras do not measure reflected photons; they measure the radiant power emitted by an object, which is governed by Planck's Law and the Stefan-Boltzmann Law24. The total radiant energy  emitted by a surface is proportional to the fourth power of its absolute temperature :

Where  is the surface emissivity of the specific material (ranging from 0 to 1) and  is the Stefan-Boltzmann constant ()24.
When rain or fog is introduced into an environment, the resulting thermal degradation is governed by physical mechanisms that are vastly different from visual degradation:
Temperature Masking and Convective Cooling (Rain): Rain physically cools the surfaces it touches through convective and evaporative cooling. A vehicle's hood or a human pedestrian, which typically appear as bright hotspots in a clean IR image, will gradually dim as rainwater pools on them, altering their apparent kinetic temperature24.
Emissivity Alteration (Rain): Water possesses an exceptionally high emissivity () in the LWIR band. When a naturally dry, low-emissivity surface (such as concrete or metal) becomes wet, its surface emissivity approaches that of water. This results in an "emissivity wash-out" that severely degrades the thermal contrast of the scene, making distinct objects blend into the background24.
Atmospheric Attenuation (Fog/Rain): While LWIR radiation penetrates light fog far better than visible light due to its longer wavelength (which renders the size parameter for Mie scattering much smaller)23, dense fog and heavy rain droplets still absorb and scatter thermal radiation35.
To ensure the diffusion model accurately simulates these thermal changes, the architecture must utilize a Temperature-Emissivity-Texture (TeV) decomposition technique prior to the generative diffusion process6. Instead of generating raw IR pixel intensities, the IR branch of the PID model predicts perturbations in the decoupled TeV space21. The Temperature () is modulated to simulate evaporative cooling, the Emissivity () is modulated to simulate surface wetness, and the Thermal Texture () captures the remaining geometric residuals6.
The physical synthesis formulation for the IR branch is governed by adjusting these decomposed parameters before applying the Beer-Lambert attenuation:

Where  is the effective attenuation coefficient specifically calibrated for the LWIR band22.
Architectural Pipeline Design for Physical Integration
To operationalize these physical models, the data generation pipeline must incorporate a pre-trained monocular depth estimator (e.g., Depth Anything or MiDaS) to generate a dense depth map  directly from the clean RGB image36. During the forward pass of the Conditional Latent Diffusion Model, this depth map, combined with a scalar condition vector representing the target weather severity (e.g., rainfall rate in mm/h, or Meteorological Optical Range visibility in meters), is used to deterministically generate the RGB transmission map  and the IR attenuation map37. The diffusion network is then constrained by these physical maps, bridging the gap between unconstrained data-driven hallucination and rigid physics-based rendering38.
Modality
Primary Physical Phenomenon
Governing Equation
Diffusion Network Constraint
Visible (RGB)
Mie Scattering, Reflection, Refraction
Koschmieder Model, Beer-Lambert Law
Deterministic depth-based transmission map  applied as a spatial attention bias.
Thermal (IR)
Emissivity Shift, Evaporative Cooling, Absorption
Planck's Law, Stefan-Boltzmann Law
TeV Decomposition space modulation, preventing impossible thermal spikes in wetted areas.

Loss Functions and Spatial Alignment
Standard diffusion model training relies almost entirely on minimizing the Mean Squared Error (MSE) between the network's predicted noise and the actual added Gaussian noise during the forward diffusion process (). However, for the specific task of paired multi-modal weather degradation, relying solely on simple noise prediction will result in structural drift. This drift destroys the strict spatial alignment between the RGB and IR pairs, rendering the synthetic dataset useless for downstream multispectral object detection tasks39.
To counteract this, the LDM must be fine-tuned using a composite loss function that simultaneously enforces physical accuracy, perceptual quality, and rigid cross-modal spatial consistency.
Cross-Modal Contrastive Loss (PatchNCE)
The most critical requirement for paired dataset synthesis is that the semantic geometry of the scene remains spatially locked between the RGB and IR output arrays12. If a pedestrian's bounding box is located at pixel coordinates  in the clean RGB image, that pedestrian must remain exactly at  in both the degraded RGB and the degraded IR images, despite the addition of heavy fog or rain.
To guarantee this alignment, the architecture must implement a Patch-wise Contrastive Loss (PatchNCE), a specialized derivation of the InfoNCE objective12. This loss operates on the principle of maximizing the mutual information between corresponding spatial locations across different modalities12.
Given a robust feature extractor (such as a frozen, pre-trained VGG-16 network), spatial feature maps are extracted from the generated degraded RGB image () and the generated degraded IR image (). The algorithm selects a query feature patch  from a specific spatial location in . The corresponding spatially aligned patch in  acts as the positive key . Crucially,  patches sampled from entirely different spatial locations within the same image act as negative keys 12.
The PatchNCE loss is mathematically formulated as:

Where  is a temperature hyperparameter (typically set to 0.07) that scales the dot-product similarity between the patches41. By minimizing this contrastive loss, the network is heavily penalized if the semantic features of the RGB image shift spatially relative to the IR image during the weather synthesis process. It effectively forces the cross-attention layers of the parallel U-Nets to lock the geometry in place.
Physics-Informed Losses
To ensure the diffusion model does not hallucinate physically impossible weather effects—such as fog that ignores scene depth, or thermal reflections that violate the laws of thermodynamics—specific physics-informed loss terms are integrated into the training objective6.
The Physical Reconstruction Loss () ensures that the generated infrared image aligns cohesively with the decomposed physical components. It operates continuously in the diffusion model to penalize non-physical generations:

Where  represents the sub-network responsible for extracting the Temperature, Emissivity, and Texture components, and  reconstructs the image from those components6.
The Physical TeV Space Loss () measures the discrepancy in the high-dimensional physical parameter space between the predicted degraded image  and a physically simulated target  (which is generated via the deterministic rendering equations outlined in Section 2)6. This highlights physically unreasonable aspects of the translated images that standard pixel-wise MSE would miss:

Perceptual and Structural Losses
While the diffusion objective inherently captures high-frequency textural details, the preservation of sharp object boundaries—which is essential for downstream SOTIF validation involving object detectors—requires a perceptual loss (such as LPIPS)15.

Here,  denotes the feature maps extracted at layer  of a pre-trained network. For the IR modality, standard VGG features (which are trained exclusively on visible-light ImageNet data) are often suboptimal. It is highly recommended to use a VGG or ResNet architecture fine-tuned on an expansive thermal dataset (e.g., FLIR or M3FD) to extract thermally relevant perceptual features accurately4.
Total Composite Objective
The final training objective for the multimodal diffusion model integrates the standard latent diffusion denoising loss () with the proposed regularization terms:

Proper balancing of the weight lambdas () is crucial to prevent one loss term from overwhelming the gradient landscape. A recommended starting configuration is setting , , and . Implementing dynamic weight averaging during training based on the magnitude of the gradients ensures stable convergence.
Training Dynamics and Data Pipeline Engineering
The engineering of the data pipeline and the stabilization of the training dynamics are just as critical to the success of the project as the architectural design itself. Paired multimodal datasets pose unique memory bandwidth, datatype, and synchronization challenges when implemented in PyTorch.
Structuring PyTorch DataLoaders for Multimodal Tasks
When dealing with strictly aligned RGB and IR pairs sourced from comprehensive autonomous driving datasets like KAIST, FLIR, LLVIP, or M3FD4, the PyTorch Dataset and DataLoader classes must be engineered to guarantee that every geometric transformation applied to the RGB tensor is identically applied to the corresponding IR tensor43.
The standard torchvision.transforms library (e.g., RandomResizedCrop, RandomAffine, RandomHorizontalFlip) generates a new random state upon each execution43. To ensure perfect spatial alignment during data augmentation, developers must utilize functional transforms (torchvision.transforms.functional). In this paradigm, the random parameters (such as crop coordinates or rotation angles) are generated exactly once per sample pair and applied iteratively to both the RGB and IR tensors.
Furthermore, datatype precision and modality scaling must be handled with extreme care. RGB data is typically represented as 8-bit integers and subsequently scaled to  for ingestion by diffusion models. Thermal data, however, often contains absolute radiometric temperature information encoded in 16-bit TIFF arrays. It is an engineering fallacy to naively normalize 16-bit radiometric IR data using standard 8-bit RGB logic (e.g., simply dividing the array by 255). The IR tensors must be globally standardized using the dataset's specific radiometric mean and standard deviation to preserve the absolute temperature deltas; destroying these deltas cripples the network's ability to perform accurate TeV decomposition45.
Optimal data augmentation strategies for this pipeline include:
Geometric Augmentations: Horizontal flipping and random scaling (applied strictly jointly)43.
Radiometric Augmentations (IR only): Emissivity jittering or thermal contrast scaling to simulate varying ambient temperatures.
Photometric Augmentations (RGB only): ColorJitter (modifying brightness, contrast, and saturation) to simulate varied ambient lighting conditions prior to the application of the synthetic weather43.
Because the model ingests dual modalities and operates via heavy parallel U-Nets, activation memory will quickly exceed standard GPU capacities, leading to data starvation. The PyTorch DataLoader must leverage pin_memory=True, configure num_workers scaled to the CPU core count, and utilize prefetch factors to maintain maximum GPU utilization.
Mitigating Training Instabilities in Diffusion Models
Generative models—especially diffusion models conditioned on high-dimensional physical priors—are prone to specific training instabilities that must be mitigated.
In Latent Diffusion Models, the variance schedule () dictates the rate of noise addition over the diffusion timesteps. Standard linear schedules often fail when the model is conditioned on complex multimodal priors, as the network struggles to balance high-frequency textural noise (rain streaks) with low-frequency global changes (volumetric fog or temperature masking). Implementing a Cosine Noise Schedule, or optimizing the Signal-to-Noise Ratio (SNR) using a zero-terminal SNR strategy, prevents the model from retaining unwanted artifacts from the clean base image by ensuring the final timestep is pure Gaussian noise47.
Furthermore, diffusion models exhibit highly oscillatory trajectories in their weight space during training. Maintaining an Exponential Moving Average (EMA) of the model weights (typically with a decay rate of 0.9999) is mandatory. The EMA weights are exclusively used for final inference, resulting in significantly sharper and more stable synthesized artifacts.
When utilizing Classifier-Free Guidance (CFG) to enforce the intensity of the weather (e.g., conditioning the synthesis on text prompts like "heavy driving rain"), setting the guidance scale too high () frequently causes color saturation in the RGB domain and thermodynamic impossibilities in the IR spectrum. Utilizing Dynamic Thresholding or maintaining a relatively low CFG scale () mitigates this phenomenon48.
Finally, to prevent exploding gradients caused by the complex physical loss terms (), strict gradient clipping (max norm = 1.0) must be employed. Developers should utilize PyTorch's Automatic Mixed Precision (AMP) utilizing the bfloat16 datatype rather than float16. bfloat16 offers a significantly wider dynamic range, which is highly beneficial for preserving the extreme continuous values present in the physical transmission maps and radiometric IR tensors without encountering underflow.
Frameworks, Datasets, and Baselines
Avoiding the reinvention of foundational architectures is crucial for rapid prototyping. The pipeline should be built atop existing, robust open-source foundations, heavily optimizing for the unique constraints of dual-stream inference and adverse weather datasets.
Foundational Datasets
To train and evaluate the multimodal synthesis pipeline, the following datasets provide the necessary strictly aligned RGB and Thermal pairs:

Dataset
Characteristics
Ideal Use Case in Pipeline
FLIR (Teledyne)
Massive collection of aligned thermal and visible images, annotated for ADAS object detection4.
Foundation for training the baseline feature extractors and verifying bounding-box retention post-synthesis.
KAIST Multispectral
Covers diverse urban environments across daytime and nighttime scenarios49.
Evaluating the model's performance on varying ambient lighting conditions prior to weather application.
M3FD
Multi-modal dataset featuring complex scenarios with diverse environmental challenges4.
Fine-tuning the network's ability to handle highly cluttered urban scenes during heavy weather.
LLVIP
Visible-infrared paired dataset specifically optimized for low-light vision4.
Testing the pipeline's capability to synthesize nighttime fog and rain phenomena accurately.

Open-Source Frameworks and Model Baselines
The foundational codebase should be built upon the HuggingFace diffusers library. Specifically, extending the StableDiffusionPipeline or ControlNetPipeline allows for the rapid integration of the custom dual-stream cross-attention mechanisms required for the paired RGBT architecture50.
For the specific implementation of the TeV decomposition and infrared physical losses, engineers should leverage the open-source GitHub repository provided by the authors of the PID (Physics-Informed Diffusion) paper6. This repository provides the complex mathematical primitives required for enforcing thermodynamic laws during diffusion, eliminating the need to write custom PyTorch operators for Planck's Law.
Additionally, reviewing the architectures of established weather restoration models—such as TransWeather32 and video diffusion models like WeatherWeaver53—is highly recommended. While these frameworks are primarily designed for weather removal or visual-only synthesis, their feature extraction mechanisms (e.g., the Weather Complexity Adjustment Factors) provide excellent insights into how deep neural networks effectively learn to represent dynamic rain streaks and volumetric fog.
For testing the physical realism of the generated data to ensure compliance with ISO 21448 SOTIF standards, engineers should utilize evaluation frameworks built into simulators like CARLA or LGSVL. These platforms provide testing APIs to cross-reference the deep-learning generated synthetic atmospheric effects with deterministic virtual ray-tracing, ensuring the generative model has not drifted from physical reality1.
Edge Deployment and Inference Optimization
Diffusion models are notoriously slow at inference due to their reliance on an iterative denoising process (typically requiring 20 to 50 steps through a massive U-Net). This high latency becomes computationally unviable when tasked with generating large-scale synthetic datasets containing hundreds of thousands of high-resolution image pairs48. To achieve high-throughput dataset generation, the trained PyTorch model must be heavily optimized, compiled, and exported using NVIDIA TensorRT58.
FP8 Post-Training Quantization
Modern NVIDIA hardware architectures (such as Hopper and Ada Lovelace) feature specialized Tensor Cores designed for 8-bit floating-point (FP8) precision. FP8 quantization significantly reduces the memory bandwidth required by the dual-stream U-Net, accelerating computation dramatically59.
Utilizing NVIDIA TensorRT-LLM or the TorchAO library, the model weights and network activations can be subjected to Post-Training Quantization (PTQ)60. It is critical to note that many legacy quantization methods utilize "fake quantization," which merely simulates lower precision during inference without actually reducing the model's footprint62. True FP8 quantization via TensorRT realizes a physical reduction in memory. For diffusion models, standardizing on the E4M3 format (which dedicates 1 sign bit, 4 exponent bits, and 3 mantissa bits) is preferred over E5M2. E4M3 offers higher granular precision within a smaller dynamic range, which is necessary to preserve the delicate gradients of the physical transmission maps and atmospheric attenuation vectors60. Implementing true FP8 PTQ yields a  to  speedup over standard FP16 inference with negligible drops in the Fréchet Inception Distance (FID) or downstream Object Detection Mean Average Precision (mAP)62.
Solver Optimization and Pipelined Parallelism
Even with quantization, executing 50 steps of a standard DDPM or DDIM solver is excessively slow. Engineers must implement advanced ODE solvers such as DPM-Solver++ or utilize Latent Consistency Models (LCM) distillation48. LCM techniques can compress the entire denoising trajectory down to merely 2 to 4 steps, accelerating the synthetic data generation pipeline by an order of magnitude without sacrificing the physical accuracy or spatial resolution of the synthesized weather phenomena14.
Furthermore, for ultra-high-resolution synthesis (e.g., generating 2K or 4K autonomous driving panoramas), naive inference often causes GPU out-of-memory errors due to the massive spatial dimensions of the cross-attention tensors. Utilizing techniques such as DistriFusion allows the diffusion generation process to be efficiently sharded across multiple GPUs using displaced patch parallelism. By reusing pre-computed feature maps from previous timesteps, this method completely mitigates the fragmentation artifacts that typically plague naive tensor parallelism in diffusion models57.
Conclusion
Synthesizing physically accurate, multi-modal adverse weather is a highly complex engineering challenge that lies at the precise intersection of generative AI, atmospheric optics, and thermodynamics. Standard generative models that rely purely on data-driven pixel manipulation fail to capture the disparate physical realities of the visible and long-wave infrared spectra, rendering their outputs useless for the safety-critical validation required by ISO 21448 SOTIF standards.
By adopting a Physics-Informed Latent Diffusion Model featuring a parallel dual-stream architecture with cross-modal attention, engineers can successfully decouple the optical scattering of rain and fog from their thermal attenuation and emissivity effects. Crucially, implementing rigorous mathematical constraints—such as PatchNCE for absolute spatial alignment, TeV decomposition for thermodynamic accuracy, and Koschmieder-based atmospheric transmission maps—ensures that the generated artifacts transcend mere visual style transfer. The resulting data pipeline, aggressively optimized via NVIDIA TensorRT, FP8 quantization, and Latent Consistency Models, yields mathematically sound, high-fidelity synthetic datasets at scale. These datasets will critically accelerate the training and validation of robust autonomous perception systems, ensuring they remain reliable and safe in the face of unpredictable real-world weather conditions.
Works cited
Abstract 1. Introduction - arXiv, https://arxiv.org/pdf/2503.02498
The Impact of Operational-Data Fidelity when Assessing Safety, https://arxiv.org/pdf/2608.10025
Dataset Safety in Autonomous Driving: Requirements, Risks ... - arXiv, https://arxiv.org/html/2511.08439v2
CFGPNet: Cross-Attention-Based Fused Gradient Programmed, https://arxiv.org/html/2608.06205v1
VALISENS: A Validated Innovative Multi-Sensor System for ... - arXiv, https://arxiv.org/html/2505.06980v1
PID: Physics-Informed Diffusion Model for Infrared Image Generation, https://arxiv.org/html/2407.09299v1
Evaluating an Adaptive Multispectral Turret System for Autonomous, https://arxiv.org/html/2512.22263v1
FireRGBTNet: A Lightweight Forest Fire Detection Model Based on, https://www.mdpi.com/1999-4907/17/8/955
[2603.04325] Scalable Evaluation of the Realism of Synthetic ... - arXiv, https://arxiv.org/abs/2603.04325
From Virtual Environments to Real-World Trials: Emerging Trends in, https://arxiv.org/html/2603.17714v2
PID: Physics-Informed Diffusion Model for Infrared Image Generation, https://arxiv.org/html/2407.09299v2
Generative adversarial networks enable biomimetic topology fusion, https://pmc.ncbi.nlm.nih.gov/articles/PMC13323779/
Unifying Multimodal Generation as Image-in, Image-out Flow Matching, https://arxiv.org/html/2604.06757v1
STREAMING VIDEO-TO-VIDEO TRANSLATION WITH FEATURE, https://proceedings.iclr.cc/paper_files/paper/2025/file/7280f65ed571b7b28321f2c7cf4c60c8-Paper-Conference.pdf
MDD-VIR: Vis-to-IR Remote Sensing Image Generation Method, https://www.mdpi.com/2072-4292/18/10/1502
PID: Physics-Informed Diffusion Model for Infrared Image Generation, https://www.researchgate.net/publication/392453746_PID_Physics-informed_diffusion_model_for_infrared_image_generation
PhysWave: Physics-Guided Latent Diffusion Models for Controllable, https://arxiv.org/html/2608.29549v1
YOLOv11-RGBT - arXiv, https://arxiv.org/pdf/2506.14696
CROSS-MODAL FEATURE DISENTANGLEMENT WITH, https://openreview.net/pdf/6122c3d7f752fcb222c1eb38f010a319b36468d8.pdf
A Dual-Modal Adaptive Pyramid Transformer Algorithm for UAV, https://pmc.ncbi.nlm.nih.gov/articles/PMC12736609/
[Literature Review] PID: Physics-Informed Diffusion Model for, https://www.themoonlight.io/en/review/pid-physics-informed-diffusion-model-for-infrared-image-generation
Infrared Thermal Imaging: Fundamentals, Research and Applications, https://www.researchgate.net/publication/258489786_Infrared_Thermal_Imaging_Fundamentals_Research_and_Applications
Predicting Atmospheric Attenuation Under Pristine Conditions, https://www.imperial.ac.uk/media/imperial-college/faculty-of-engineering/electrical-and-electronic-engineering/public/optical-and-semiconductor-devices/2016_11_IEEE-Access.pdf
Infrared Thermography (IRT) - TarmacView, https://www.tarmacview.com/glossary/infrared-thermography/
A Methodology to Model the Rain and Fog Effect on the ... - MDPI, https://www.mdpi.com/1424-8220/23/15/6891
Atmospheric and Topographic Correction (ATCOR Theoretical, https://www.rese-apps.com/pdf/atcor_atbd.pdf
Atmospheric Chemistry: Air Pollution and Global Warming, https://www.mcgoodwin.net/pages/atmscuw458.pdf
Conference Book - Copernicus Meetings, https://meetings.copernicus.org/fog2010/fogconference_2010_conference_book.pdf
arXiv:1811.04102v1 [physics.ins-det] 9 Nov 2018, https://arxiv.org/pdf/1811.04102
Unified atmospheric attenuation models for visible and infrared, https://www.researchgate.net/publication/384263036_Unified_atmospheric_attenuation_models_for_visible_and_infrared_wavelengths
Chapter 2 - Atmospheric Radiative Transfer, https://uomustansiriyah.edu.iq/media/lectures/6/6_2021_09_15!11_07_28_PM.pdf
Video-and-Single-Image-Deraining/README.md at dev - GitHub, https://github.com/hongwang01/Video-and-Single-Image-Deraining/blob/dev/README.md
Interaction of electromagnetic radiation with the Earth's atmosphere, https://lpl.arizona.edu/hamilton/sites/lpl.arizona.edu.hamilton/files/courses/ptys551/Chapters/Interaction_of_Electromagnetic_Radiation_with_the_Earth's_Atmosphere.pdf
Radiation (Chapter 5) - Urban Climates, https://www.cambridge.org/core/books/urban-climates/radiation/B01B694873B2937FA362B8225915AAA7
Seeing through fog and rain - September 2008, http://www.securitysa.com/article.aspx?pklarticleid=5369
A Disentanglement Approach for RGBD Semantic Segmentation, https://arxiv.org/html/2505.24361v1
harnessing optical imaging limit through atmospheric scattering media, https://arxiv.org/pdf/2404.15082
Computer Vision and Pattern Recognition Dec 2025 - arXiv, https://www.arxiv.org/list/cs.CV/2025-12?skip=2025&show=2000
Physical Adversarial Clothing Evades Visible-Thermal Detectors via, https://www.researchgate.net/publication/404541877_Physical_Adversarial_Clothing_Evades_Visible-Thermal_Detectors_via_Non-Overlapping_RGB-T_Pattern
arXiv:2109.14910v1 [cs.CV] 30 Sep 2021, https://lmb.informatik.uni-freiburg.de/Publications/2021/ZB21/zolfaghariICCV2021.pdf
Dual-Branch Colorization Network for Unpaired Infrared Images, https://www.mdpi.com/2079-9292/13/18/3784
Physics-based diffusion model for thermal image generation in, https://www.researchgate.net/publication/413721627_Physics-based_diffusion_model_for_thermal_image_generation_in_space_orbital_rendez-vous_scenarios
Beginner's Guide to Data Augmentation with PyTorch - Kaggle, https://www.kaggle.com/discussions/getting-started/576485
Data Augmentations in Pytorch - Scaler Topics, https://www.scaler.com/topics/pytorch/data-augmentations-in-python/
PyTorch Image Augmentation Methods | Sebastian Raschka, PhD, https://sebastianraschka.com/blog/2023/data-augmentation-pytorch.html
Image Augmentation for Computer Vision Tasks Using PyTorch, https://www.comet.com/site/blog/image-augmentation-for-computer-vision-tasks-using-pytorch/
Efficient Quantization Strategies for Latent Diffusion Models - arXiv, https://arxiv.org/abs/2312.05431
The Denoising Engine: Diffusion Models, from Basics to Scale, https://svac-llc.com/posts/33-diffusion-models-a-systems-review.html
Visible–Infrared Image Fusion for Computer Vision - MDPI, https://www.mdpi.com/2076-3417/16/13/6757
List of Large Language Vision Models | by Sung Kim | Dev Genius, https://blog.devgenius.io/list-of-computer-vision-models-with-promptable-interface-b7af24a9a61f
Accelerating Diffusion Models with an Open, Plug-and-Play Offering, https://developer.nvidia.com/blog/accelerating-diffusion-models-with-an-open-plug-and-play-offering/
leonmakise/Awesome-All-in-one-Image-Restoration-Methods - GitHub, https://github.com/leonmakise/Awesome-All-in-one-Image-Restoration-Methods
Controllable Weather Synthesis and Removal with Video Diffusion, https://www.researchgate.net/publication/391369076_Controllable_Weather_Synthesis_and_Removal_with_Video_Diffusion_Models
Controllable Weather Synthesis and Removal with Video Diffusion, https://arxiv.org/html/2505.00704v1
SCaRL- A Synthetic Multi-Modal Dataset for Autonomous Driving, https://arxiv.org/abs/2405.17030
Making Diffusion Models Practical: A Survey on Acceleration and, https://www.preprints.org/frontend/manuscript/3ef2013c1dad47746fcc56736b8a27d4/download_pub
Distributed Parallel Inference for High-Resolution Diffusion Models, https://cvpr.thecvf.com/virtual/2024/poster/30271
Visual Generation (Diffusion Models) [Beta] — TensorRT LLM, https://nvidia.github.io/TensorRT-LLM/features/visual-generation.html
FP8 Quantization — TensorRT-LLM - GitHub Pages, https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/fp8-quantization.html
Optimizing Transformer-Based Diffusion Models for Video, https://developer.nvidia.com/blog/optimizing-transformer-based-diffusion-models-for-video-generation-with-nvidia-tensorrt/
1Introduction - arXiv, https://arxiv.org/html/2507.16099v1
arXiv:2501.17343v1 [cs.CV] 28 Jan 2025, https://arxiv.org/pdf/2501.17343
DistriFusion: Distributed Parallel Inference for High-Resolution, https://github.com/mit-han-lab/distrifuser
