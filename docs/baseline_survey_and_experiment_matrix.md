# GATE-IR: A Publication-Grade SOTA Baseline Survey and Experiment Matrix for a Weather-Gated Thermal Object Detector

## TL;DR
- **The GATE-IR design is publishable but sits in a crowded 2022–2026 field; its novelty must be reframed from "components" to "adaptive routing" and defended against three attack vectors:** (1) all-in-one restoration models (PromptIR, Histoformer, DA-CLIP, AdaIR, WM-MoE) that avoid an explicit gate; (2) task-driven joint restoration-detection (IA-YOLO, TogetherNet, BAD-Net, DTRDNet) that shows restoration-for-quality can *hurt* detection; and (3) the physics objection that LWIR barely needs defogging and that the RGB scattering model is invalid for thermal fog synthesis.
- **Comparing only to YOLOv8 in 2026 will draw a reject;** reviewers will demand YOLOv10/YOLO11/YOLOv12 and DETR-family real-time detectors (RT-DETRv2/v3, D-FINE, DEIM), plus oracle-gate/no-gate bounds and gate-error sensitivity analysis, which is the single most important experiment GATE-IR is currently missing.
- **Adaptive gamma correction as a "fog enhancer" and a ViT neck on YOLOv8 are both likely to read as incremental;** the defensible contribution is the *joint system* — a real-time, thermal-specific, gated pipeline benchmarked on real adverse-weather thermal data (M3FD, DENSE) with a Pareto/edge-deployment story — not any single block.

## Key Findings

1. **The "why gate at all" objection is the central threat.** All-in-one restoration (AiOIR) is now the dominant competing paradigm. The strongest current baselines are PromptIR (NeurIPS 2023), DA-CLIP (ICLR 2024), Histoformer (ECCV 2024), AdaIR, and MoE routers such as WM-MoE (2023) and M2Restore/M2IR (2025–2026). GATE-IR's explicit MLP gate is functionally a hard-routing special case of these soft-routing MoE restorers — reviewers will ask why a hard gate is better than a learned soft router.

2. **A near-identical prior system already exists and must be cited.** DTRDNet (Sensors 2024) already couples a degradation category predictor (a gate) with degradation-specific restoration for detection under adverse weather, and Aloufi et al. (Electronics 2024, 13(15):3063) explicitly couple a weather classification head with object detection. Missing these = desk-reject risk.

3. **Restoration can hurt detection — well documented, and it is both GATE-IR's best friend and worst enemy.** IA-YOLO, D-YOLO, TogetherNet and BAD-Net all argue quality-oriented restoration is misaligned with detection. GATE-IR must show its enhancers improve *detection mAP*, not PSNR, or a reviewer will cite this literature against it.

4. **The physics of thermal weather simulation is a real vulnerability.** LWIR (8–14 µm) penetrates fog far better than visible light, and the standard Koschmieder/atmospheric scattering model I=J·t+A·(1−t) is a visible-band model whose airlight term is physically wrong for thermal (emission, not scattering). Synthetic thermal fog must be justified with radiative-transfer references or reframed as approximate corruption.

5. **Real-time SOTA has moved well past YOLOv8.** The 2025–2026 real-time frontier is YOLOv10/v11/v12, RT-DETRv2/v3, D-FINE and DEIM. Real-time claims require Jetson Orin latency with TensorRT FP16/INT8, not just desktop FPS.

## Details

---
### PART 1 — SOTA METHOD INVENTORY PER COMPONENT

#### Component A — Weather/degradation classification & adaptive routing

**A1. All-in-one / unified restoration (the critical competing paradigm — reviewers WILL ask "why gate?").** These implicitly identify degradation rather than explicitly gating.

| Method | Venue/Year | One-line | Code | Notes for GATE-IR |
|---|---|---|---|---|
| **AirNet** | CVPR 2022 | Contrastive degradation encoder, no explicit degradation label | Yes | Standard AiOIR baseline (must-cite) |
| **TransWeather** | CVPR 2022 | Single transformer enc/dec with learnable weather-type queries | Yes | Must-cite; efficient, weather-general |
| **WeatherDiffusion (Özdenizci & Legenstein)** | TPAMI 2023 | Patch-based denoising diffusion for all-weather; slow | Yes | Must-cite; known to hallucinate/leave shadows |
| **Chen et al. two-stage knowledge learning (WGWS)** | CVPR 2023 | Weather-general + weather-specific features | Yes | Must-cite conceptual sibling of gating |
| **PromptIR** | NeurIPS 2023 | Learnable prompts encode degradation; blind AiOIR | Yes | **Must-cite; strongest generic baseline** |
| **DA-CLIP** | ICLR 2024 | CLIP controller predicts degradation type + clean embedding | Yes | **Must-cite; directly does "gate via VLM"** |
| **Histoformer** | ECCV 2024 | Histogram self-attention for adverse weather | Yes | **Must-cite; current transformer SOTA** |
| **AdaIR** | 2024 | Frequency mining/modulation, adaptive AiOIR | Yes | Must-cite as recent reactive baseline |
| **InstructIR** | ECCV 2024 | Text-instruction-guided restoration | Yes | Nice-to-have |
| **DiffUIR** | CVPR 2024 | Selective hourglass diffusion, shared distribution | Yes | Nice-to-have |
| **MWFormer** | 2024/2025 | Degradation-aware transformer; MWFormer-real beats TransWeather-real on RainDrop/Outdoor-Rain/Snow100K | Yes | Optional strong recent |
| **WM-MoE** | 2023 | Weather-aware mixture-of-experts *router* selects experts w/o labels | — | **Must-cite: the MoE analog of GATE-IR's gate** |
| **M2Restore / M2IR / UniRestorer / ClusIR** | 2025–2026 | MoE/Mamba routers with degradation-aware expert selection | Some | Optional; shows soft-routing frontier |

**Key data point for framing (verified):** On the AllWeather Test1 (rain+fog) set, TransWeather (CVPR 2022) reaches **27.96 dB PSNR / 0.9509 SSIM vs the earlier All-in-One's 24.71 dB / 0.8980** (official repo jeya-maria-jose/TransWeather; the CVPR paper frames this as a **+6.34 dB** Test1 gain, **+4.93 dB** on Snow100K-L, **+3.11 dB** on RainDrop, at **0.14 s** inference vs All-in-One 0.33 s and MPRNet 0.18 s). Yet task-specific specialists still win per condition: on RainDrop, **CCN (CVPR 2021) = 31.34 dB / 0.9286**, UDR-S²Former (ICCV 2023) = 32.64 / 0.9427, and Restormer = 32.18 / 0.9408, all above TransWeather's 30.17 (per DA2Diff, arXiv:2504.05135). This is the exact tension GATE-IR exploits: specialists win per-condition, generalists win on average.

**A2. Weather classification (RGB & thermal).** CNN classifiers, handcrafted-feature classifiers, and CLIP/VLM zero-shot. DA-CLIP is the strongest degradation classifier; ADSM (2024, arXiv:2411.07445) uses CLIP latent prompt generators for degradation type/property/caption. GATE-IR's hand-crafted-feature MLP (thermal variance, entropy, Laplacian variance) is a defensible *lightweight* choice but must be benchmarked against a CLIP zero-shot classifier and a small CNN classifier to prove the efficiency trade-off is worth it.

**A3. Routing / conditional computation / MoE literature.** Adaptive Mixture of Local Experts (Jacobs/Jordan, foundational); dynamic networks and early-exit; degradation-aware routers (DDER in M2Restore, arXiv:2506.07814; UIR-LoRA degradation-aware router, arXiv:2409.20197; MoCE-IR complexity experts; UniRestorer degradation+granularity estimation, ICLR). UniRestorer's key insight is directly relevant: it uses *granularity estimation to be robust to degradation-estimation error* — i.e., to gate misclassification, exactly GATE-IR's vulnerability.

**A4. Bounds.** "No-gate" (always bypass) lower bound and "oracle-gate" (ground-truth weather label) upper bound are mandatory. Add an "always-apply-both-enhancers" baseline and a "soft-router" (learned weights instead of hard gate) baseline.

#### Component B1 — Dehazing/defogging (thermal-aware)

- **Classical:** DCP (TPAMI 2010; SOTS ~16–19 dB — weak, and *DCP is largely invalid for single-channel thermal* since it relies on the RGB dark-channel color prior — say this explicitly); CLAHE; AGCWD/AGCCPF (adaptive gamma variants); Retinex family SSR/MSR/MSRCR; guided filter; non-local prior dehazing.
- **Deep (RGB) SOTA ladder:** AOD-Net (2017) → GridDehazeNet (2019) → FFA-Net (AAAI 2020, ~36 dB SOTS) → MSBDN (2020) → Dehamer (2022) → DehazeFormer (2023) → **C2PNet (CVPR 2023: 42.56 dB / 0.9954 SSIM SOTS-indoor, 36.68 dB / 0.9900 SOTS-outdoor, a 3.94 dB indoor gain over prior SOTA; arXiv:2303.14218)** → DEA-Net (TIP 2024, ~41 dB, efficient) → RIDCP (real-world codebook prior) → DCMPNet (CVPR 2024) → MB-TaylorFormerV2 (TPAMI 2025). C2PNet and DEA-Net are the current must-compare deep dehazers.
- **IR-specific enhancement:** thermal contrast enhancement literature is thin; most "thermal enhancement" is CLAHE/gamma/Retinex-style. This thinness is an opportunity for GATE-IR but also means adaptive gamma is not obviously beaten by a heavy dehazer on thermal — the paper should demonstrate this empirically.

#### Component B2 — Deraining (direct competitor to LSRB)

- **Classical/CNN:** DerainNet, DDN (2017), RESCAN (2018), PReNet (CVPR 2019), MSPFN (2020), RCDNet (2020), MPRNet (CVPR 2021), SPDNet (2021).
- **Transformer/SOTA:** Uformer, Restormer (CVPR 2022), IDT, **DRSformer (CVPR 2023: 32.18 dB PSNR Rain200H vs IDT 32.10, SPDNet CNN 31.28; 48.53 dB SPA-Data vs Restormer 47.98; github.com/cschenxiang/DRSformer)**, **NeRD-Rain (CVPR 2024)**. Mamba: RainMamba, DeRainMamba.
- **Lightweight/edge (the true LSRB competitors):** DRT (lightweight recursive transformer), LIR (lightweight baseline), SFNet, and efficient CNN deraining. GATE-IR must compare LSRB against at least one lightweight (DRT/LIR) and one heavy (DRSformer/Restormer) deraining baseline, reporting params/FLOPs/FPS *and* downstream mAP.
- **IR/thermal-specific deraining:** essentially absent — another gap GATE-IR can claim, but also means synthetic thermal rain must be physically justified.

#### Component B3 — General restoration backbones (strong preprocessing baselines)
Restormer (CVPR 2022), Uformer (CVPR 2022), NAFNet (ECCV 2022), SwinIR (ICCV 2021), MambaIR / MambaIRv2 (2024–2025), diffusion restoration (WeatherDiffusion, DiffUIR). NAFNet and Restormer are the standard "strong generic" preprocessors.

#### Component B4 — "Restoration is not the right objective" (reviewers WILL raise this)

- **Task-driven / joint restoration-detection (must-cite cluster):** **IA-YOLO** (AAAI 2022, differentiable image processing predicted by CNN-PP, trained by detection loss); **DSNet** (Huang et al. 2020, joint visibility enhancement + detection); **TogetherNet** (Pacific Graphics/CGF 2022, arXiv:2209.01373, dynamic enhancement learning, restoration branch train-only); **DENet** (ACCV 2022); **BAD-Net / Detection-Friendly Dehazing** (TIP 2023, two-branch so a weak dehazer can't hurt detection); **FriendNet** (2024, arXiv:2403.04443); **D-YOLO** (2024, arXiv:2403.09233, dual-route hazy+clean features); **RDMNet** (IEEE T-IV 2024); ERUP-YOLO (WACV 2025); WARLearn (WACV 2025, arXiv:2411.14095); UniDet-D (2025, arXiv:2506.12324).
- **Empirical picture:** the recurring finding across IA-YOLO, D-YOLO and BAD-Net is that cascading an off-the-shelf restorer before a detector often *does not help and can hurt*, because restoration removes/hallucinates cues the detector relies on and optimizes PSNR not detection. This is the single most likely reviewer attack on GATE-IR's cascade design. **GATE-IR must include the ablation: gate+enhancer vs. detector-only, and show a mAP gain under matched training.**
- **Robustness-by-training alternatives (need-no-preprocessing baselines):** heavy augmentation, AugMix, domain randomization, synthetic-weather augmentation training. Volk et al. (ITSC 2019) synthetic-rain augmentation; causal-intervention adaptation (Zhang et al., TPAMI 2022/2024, 46(3):1742–1756). GATE-IR must beat "YOLO trained with weather augmentation, no preprocessing."

#### Component C — Thermal/IR detection architectures

- **Real-time SOTA ladder (must-compare — YOLOv8-only is outdated in 2026):** YOLOv5, YOLOv7 (CVPR 2023), YOLOv8, YOLOv9, **YOLOv10 (NeurIPS 2024)**, **YOLO11**, **YOLOv12 (attention-centric, NeurIPS 2025)**, YOLOv13; **RT-DETR (CVPR 2024)**, **RT-DETRv2 (2024, arXiv:2407.17140)**, **RT-DETRv3 (WACV 2025, R18 48.1 AP)**, **D-FINE (2024, arXiv:2410.13842, distribution refinement)**, **DEIM (2025, dense matching)**, DEIMv2/RT-DETRv4 (2025, DINOv3 backbones), RF-DETR (2025, first real-time >60 mAP COCO), RTMDet. **Must-compare set: at minimum YOLOv10/YOLO11/YOLOv12 + one DETR-family (RT-DETRv2 or D-FINE/DEIM).**
- **Small-object necks/losses:** BiFPN, PAFPN, ASFF, transformer necks, P2 heads, QueryDet, SAHI (slicing-aided inference); **NWD (Normalized Gaussian Wasserstein Distance, arXiv:2110.13389; ISPRS 2022 NWD-RKA)** for tiny objects, Focal-EIoU, SlideLoss. GATE-IR's "ViT neck for small/distant people" claim must be ablated against BiFPN and a P2 head, and NWD/SlideLoss should be tested as cheaper alternatives.
- **Thermal-specific detectors (last 3 yrs):** MS-YOLO (MobileNetV4+SlideLoss, edge IR, 2025, arXiv:2509.21696); CRT-YOLO (FLIR 79.6 mAP50/39.6 mAP; LLVIP 95.1/59.7); high-altitude IR pruning+NMS (Sensors 2025, YOLOv8, PMC11943301); Shyam & Yoo lightweight thermal SR+detection (WACV 2024).
- **RGB-T fusion detectors (paired RGB — GATE-IR does NOT use paired RGB, but reviewers ask):** CFT (Cross-Modality Fusion Transformer 2021), ProbEn, CSAA, RSDet, **ICAFusion (2024, FLIR 79.2 mAP50 / 41.4 mAP)**, CAFF-DINO (CVPRW 2024), Fusion-Mamba, WaveMamba (ICCV 2025, LLVIP 66.0 mAP). Position these as *upper bounds that use an extra modality*, not fair competitors.
- **General reference detectors:** Faster R-CNN, DETR, Deformable DETR, DINO, Co-DETR.

**Benchmark anchors (what a competitive number looks like):** LLVIP pedestrian mAP is *saturated* — mAP50 routinely 95–98%, so improvements look marginal and mAP@50-95 (~55–70%) is the discriminating metric. FLIR-aligned: mAP50 ~76–89%, mAP ~37–46%. Report mAP@50-95, not just mAP50, especially on LLVIP.

#### Component D — Cross-modal thermal↔RGB transfer

- **Unpaired translation:** CycleGAN, CUT (ECCV 2020), UNIT/MUNIT, DRIT++; diffusion unpaired translation.
- **Thermal↔visible SOTA:** DiffV2IR (2025, arXiv:2503.19012, vision-language-guided visible→IR), F-ViTA (2025, arXiv:2504.02801, foundation-model-guided, SOTA on 5 datasets, LWIR/MWIR/NIR), CM-Diff (2025, arXiv:2503.09514, bidirectional IR↔visible diffusion), T2V-DDPM (FG 2023), DiffTV (ACM MM 2024, thermal→visible faces), InfraGAN (PRL 2022). Notably Ordun et al. (ICIP 2023) "When Visible-to-Thermal GAN Beats Conditional Diffusion" — GAN sometimes still wins, relevant to CycleGAN defensibility.
- **Critiques / hallucination:** thermal→pseudo-RGB is *known-fragile*: FreqKD (2026, arXiv:2606.11572) shows RGB teachers encode appearance cues (color, illumination, reflectance) with *no correspondence* in thermal, and that naïve cross-modal distillation (uniform feature-matching, cosine alignment, response-level) *fails to beat the baseline* when transferring RGB DINOv2 to IR on KAIST. This directly threatens GATE-IR's CycleGAN→pseudo-RGB→teacher pipeline; the translation can hallucinate textures. **Strong reviewer objection.**
- **Alternatives to translation:** UDA detection (domain-adaptive YOLO, MS-DAYOLO), RGB→thermal adaptation, self-training/pseudo-labeling, tensor-factorization cross-modal IR detection (2023, arXiv:2309.16592). InfraNet (2026, arXiv:2607.03795) shows an IR-only student guided by RGB at *training only* reaching 68.4–68.8 mAP on LLVIP without inference-time RGB — a cleaner design than CycleGAN pseudo-RGB.

#### Component E — Knowledge distillation for detection

- **Detection KD SOTA:** FitNets (feature mimic, foundational); **FGD (Focal & Global Distillation, CVPR 2022)**; **MGD (Masked Generative Distillation, ECCV 2022)**; **CWD (Channel-Wise Distillation, ICCV 2021)**; PKD (NeurIPS 2022); **LD (Localization Distillation, CVPR 2022)**; **CrossKD (CVPR 2024, cross-head, 43.7 mAP on GFL)**; BCKD (ICCV 2023, 43.2 mAP). GATE-IR's "Feature Mimic Loss" is closest to FitNets/FGD; it must be benchmarked against FGD/MGD/CWD/CrossKD.
- **Cross-modal (RGB→IR) KD (the key related cluster):** **FreqKD (2026, frequency-decoupled RGB→IR distillation)**; multi-scale graph cross-attention RGB→IR KD (Nagoya VisLab); CKD (coupled KD for RGBT tracking); tensor-factorization cross-modal IR detection (2023); M²C-EvDet (cross-modal event KD 2026, arXiv:2606.24248). FreqKD is the single most important cross-modal-KD citation and a direct methodological rival.
- **Baselines:** no-distillation, logit-only KD, standard feature KD, cross-modal KD variants, and RGB-pretraining (no distillation).

#### Component F — Datasets, degradation simulation, evaluation

**Thermal detection benchmarks (verified sizes):**
| Dataset | Size | Classes | Res | Real weather? | Typical mAP |
|---|---|---|---|---|---|
| FLIR ADAS v1/v2 | ~10–14k thermal (13k aligned pairs) | person/car/bicycle(+) | 640×512 | Day/night, some | mAP50 ~76–79 |
| FLIR-aligned (Zhang) | **5,142 pairs (4,129 train / 1,013 test)** | person/car/bicycle | 640×512 | limited | mAP ~37–46 |
| LLVIP (Jia et al., arXiv:2108.10831) | **15,488 aligned pairs (30,976 imgs; 12,025 train / 3,463 test), 26 locations** | person | 1280×1024 | low-light | mAP50 95–98 (saturated) |
| KAIST Multispectral | ~95k | pedestrian | 640×512 | day/night | MR-based |
| M3FD (TarDAL, CVPR 2022) | **4,200 aligned pairs, 33,603 objects** | 6 (People/Car/Bus/Motorcycle/Truck/Lamp) | — | **real: Daytime/Overcast/Night/Challenge** | — |
| CAMEL, OSU Thermal, VEDAI-IR, HIT-UAV | varied | varied | varied | mostly no | — |
| LYNRED-MDS (2026), ThermEval-D (2026) | new | VRU | — | mixed | — |

**Real adverse-weather (IR/thermal-relevant):** **SeeingThroughFog/DENSE (Bijelic, CVPR 2020, pp. 11682–11692): 12,000 real-world samples + 1,500 fog-chamber samples over 10,000+ km in northern Europe, with 100k labels across lidar, camera, radar, gated NIR and FIR sensors** — the premier real thermal-in-fog set; RADIATE; Boreas; ACDC (RGB); Foggy Cityscapes (synthetic depth-based); Rain100/1400/RainDrop; RESIDE (SOTS). **M3FD and DENSE are the two must-use real-weather thermal sets.** WiSE-OD (2025, arXiv:2507.18925) provides LLVIP-C and FLIR-C corruption benchmarks (fog/brightness/contrast, severity 1–5, applied to IR).

**Simulating fog/rain on thermal — the defensibility problem (reviewers attack synthetic degradation hard):**
- Standard RGB model: Koschmieder/atmospheric scattering I(x)=J(x)t(x)+A(1−t(x)), t=exp(−βd), MOR≈3/β (Sakaridis Foggy Cityscapes, IJCV 2018; Narasimhan & Nayar, IJCV 2002 48(3):233–254 and TPAMI 2003 25(6):713–724).
- **This model is NOT physically valid for LWIR as-is.** Fog extinction is strongly wavelength-dependent and much weaker in LWIR; per the FLIR/Teledyne tech note, for Cat II fog the LWIR band offers roughly four times the range of MWIR, and visible-band transmission is far below both thermal windows. The Kruse (1962) and Kim (SPIE 2001, 4214:26–37) visibility→extinction models were derived for visible→NIR and break down for fog droplets ≫ wavelength (SPIE "Optical attenuation in fog and clouds": Kruse formula "not appropriate" for fog, attenuation "virtually independent of wavelength" from visible to NIR, and fog "definitively less challenging in the mid-IR (10.6 µm)"). U.S. Army ERADCOM (1978, White Sands): extinction ≈ 1/λ for haze but ~wavelength-independent for fog. MODTRAN-based LWIR-in-fog range modeling: Beier & Gemperlein (SPIE 2004).
- **The airlight term A is physically wrong for thermal:** in visible, A is scattered sunlight/skylight; in LWIR the additive path radiance is *thermal emission* from the (possibly warm) fog, not scattered airlight — so estimating A as the brightest pixel (dark-channel assumption) is unjustified. The correct radiative-transfer form is I = J·τ + L_path where L_path is emitted atmospheric radiance (MODTRAN-computed).
- **There is essentially no peer-reviewed work doing true radiometric fog synthesis on LWIR** — GATE-IR should (a) use real thermal-weather data (M3FD, DENSE) as primary and synthetic only as augmentation, explicitly framed as approximate ImageNet-C-style corruption (as WiSE-OD does), and (b) cite the radiative-transfer limitation openly. This converts a fatal objection into a stated limitation and a defensible novelty opening.

**Metrics:** detection — mAP@50, mAP@50-95, AP-small/medium/large, and **MR^-2 (log-average miss rate)** for pedestrian benchmarks (KAIST/LLVIP convention). Restoration — PSNR/SSIM/LPIPS, but **flag explicitly that these are misleading for a detection-driven task** (higher PSNR ≠ higher mAP; cite the task-driven cluster). Classification — accuracy, per-class F1, confusion matrix for the gate. Efficiency — params, GFLOPs, latency, FPS, throughput, peak memory, energy. **Real-time hardware convention:** reviewers expect desktop GPU (RTX-class) AND edge (Jetson Orin Nano/NX/AGX) latency with TensorRT FP16/INT8; ">30 FPS" is the common real-time threshold, but state the device (e.g., RT-DETRv4 latency is reported on a single NVIDIA T4 under TensorRT FP16). Statistical rigor: multi-seed (≥3) runs with mean±std, and note that detection papers frequently omit CIs — providing them strengthens the paper.

---
### PART 2 — THE EXPERIMENT MATRIX

#### Table 1 — MAIN COMPARISON (full systems). Fix the detector compute budget where possible; columns = mAP@50-95 per condition, FPS (device), params.

| # | System | Clear | Fog | Rain | FPS(GPU) | FPS(Orin) | Params | Priority | Reviewer objection if omitted |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Detector-only, no preprocessing (YOLOv8s) | ✓ | ✓ | ✓ | — | — | — | must | baseline sanity |
| 2 | Detector-only trained w/ weather augmentation | ✓ | ✓ | ✓ | — | — | — | must | "robustness-by-training beats your gate" |
| 3 | All-in-one restoration (PromptIR/Histoformer)+detector | ✓ | ✓ | ✓ | — | — | — | must | "why gate vs AiOIR" |
| 4 | Task-driven joint (IA-YOLO / D-YOLO) | ✓ | ✓ | ✓ | — | — | — | must | canonical rival |
| 5 | DTRDNet / TogetherNet (degradation-aware joint) | ✓ | ✓ | ✓ | — | — | — | must | nearest prior art |
| 6 | Per-weather specialist restoration + oracle routing | ✓ | ✓ | ✓ | — | — | — | must | upper bound |
| 7 | SOTA real-time detector (YOLOv12 / RT-DETRv2) + weather aug | ✓ | ✓ | ✓ | — | — | — | must | "you compared to outdated YOLOv8" |
| 8 | **GATE-IR (full)** | ✓ | ✓ | ✓ | — | — | — | must | — |
| 9 | RGB-T fusion (ICAFusion) [uses paired RGB] | ✓ | ✓ | ✓ | — | — | — | strengthens | extra-modality upper bound |

#### Table 2 — PER-STAGE BASELINE-SWAP (hold rest of pipeline fixed = GATE-IR)

- **Stage A gate swap:** {hand-crafted MLP (proposed), small CNN classifier, CLIP zero-shot, DA-CLIP, no-gate, oracle-gate, soft-router}. Metric: gate acc + downstream mAP + gate latency. Priority: must.
- **Stage B-fog swap:** {adaptive gamma (proposed), CLAHE, AGCWD, MSRCR, DCP, FFA-Net, DEA-Net, C2PNet}. Metric: mAP-fog, FPS. Essential cells: gamma vs CLAHE vs one deep dehazer (DEA-Net). Skippable: multiple heavy dehazers (diminishing info) — justify by cost.
- **Stage B-rain swap:** {LSRB (proposed), PReNet, MPRNet, DRSformer, Restormer, DRT (lightweight)}. Essential: LSRB vs DRT (lightweight rival) vs DRSformer (heavy SOTA). Metric: mAP-rain, params/FLOPs/FPS.
- **Stage C detector swap:** {YOLOv8s (proposed base), YOLOv8s+ViT neck (proposed), YOLOv8s+BiFPN, YOLOv8s+P2 head, YOLOv10s, YOLO11s, YOLOv12s, RT-DETRv2/D-FINE}. Metric: mAP, AP-small, FPS. Priority: must.
- **Stage C neck/loss ablation:** {baseline neck, ViT neck, BiFPN, +NWD loss, +SlideLoss}. Isolates the ViT-neck contribution — mandatory to justify neck novelty.
- **Prioritization:** the full cross-product (~7×8×6×8 cells) is infeasible and unnecessary. Fix all-but-one-stage at the GATE-IR default and sweep each stage independently (additive design). Only run the full-factorial for the {gate × enhancer} pair, because gate errors interact with enhancer choice — that interaction is the paper's core claim.

#### Table 3 — GATING-SPECIFIC (the paper's core, currently missing)
- Gate accuracy + confusion matrix (must).
- Oracle-gate upper bound vs no-gate lower bound vs GATE-IR (must).
- Always-apply-both-enhancers baseline (must) — isolates value of routing vs value of enhancers.
- **Gate-error sensitivity:** artificially inject X% misclassification and plot mAP degradation; report per-error-type cost (fog→rain routing vs rain→fog). This is the key vulnerability reviewers probe (must-have).
- Gate latency and % of total pipeline latency (must).
- GATE-IR vs implicit AiOIR/soft-router (Table 1 row 3) head-to-head on accuracy AND compute (must).

#### Table 4 — DISTILLATION
- Teacher choice: {YOLOv8-L RGB, YOLOv8-X, RT-DETR-L} (strengthens).
- With/without CycleGAN pseudo-RGB (must — tests the fragile translation step).
- Distillation loss variants: {Feature Mimic (proposed), FGD, MGD, CWD, CrossKD, FreqKD} (must-compare FreqKD, the cross-modal rival).
- vs direct RGB pretraining (no distillation) and vs no-distillation baseline (must).
- vs RGB-T fusion (ICAFusion, upper bound using paired data) (strengthens).
- **Hypothesis to test honestly:** per FreqKD, naïve RGB→IR feature mimic may *fail to beat baseline*; GATE-IR must show its Feature Mimic Loss actually helps, or drop the claim.

#### Table 5 — EFFICIENCY / REAL-TIME
- Per-stage latency breakdown (gate / enhancer / detector; distillation is train-only).
- Throughput on RTX-class GPU + Jetson Orin Nano/NX/AGX.
- With/without TensorRT; FP16 and INT8 quantization.
- Accuracy-latency Pareto frontier plot vs YOLOv10/v12, RT-DETRv2, AiOIR+detector (must — this is where a gated pipeline should win: skip enhancement on clear frames).
- Peak memory and (if possible) energy per inference.

#### Table 6 — GENERALIZATION / ROBUSTNESS
- Cross-dataset transfer (train FLIR-aligned → test M3FD / LLVIP), reporting the known FLIR-ADAS-vs-FLIR-aligned distribution gap.
- Real vs synthetic weather gap: train on synthetic fog, test on real fog (DENSE/M3FD) — directly addresses the physics objection (must).
- Unseen degradations: snow, low-light, sensor noise, thermal non-uniformity/fixed-pattern noise, dead pixels — tests gate behavior on out-of-taxonomy inputs (must; gate has no "snow" class → what does it do?).
- Severity sweeps (WiSE-OD-style severity 1–5) with mAP-vs-severity curves.
- Failure-mode analysis with qualitative examples (gate misfires, enhancer hallucination, CycleGAN artifacts).

---
### PART 3 — REVIEWER-PROOFING (candid)

**1. Novelty: the "naive combination" risk is real.** As described (weather classifier + off-the-shelf gamma/derain + off-the-shelf YOLO + off-the-shelf CycleGAN + off-the-shelf KD), GATE-IR reads as engineering integration. DTRDNet (2024), Aloufi et al. (2024), WM-MoE (2023) and the whole task-driven cluster already occupy this space. **Reposition around ONE defensible thesis and prove it:** e.g., "hard gating + condition-specific lightweight enhancement achieves the accuracy of all-in-one restoration at a fraction of the edge latency by skipping computation on clear frames." Substantiating experiments: Table 3 (gating value + error robustness), Table 5 (Pareto/edge win), Table 6 (real-vs-synthetic). If GATE-IR cannot beat a soft-router / AiOIR on the Pareto frontier, the contribution collapses.

**2. Adaptive gamma as a fog enhancer — likely seen as too weak in 2026.** Gamma correction addresses global contrast/brightness, not scattering; for thermal, where fog barely scatters LWIR, gamma may actually be *appropriate* (there is little haze veil to remove) — but the paper must *argue this from the physics* (LWIR fog ≈ contrast loss + emissive path radiance, not a scattering veil) rather than presenting gamma as a generic dehazer. Stronger yet still real-time alternatives: CLAHE, AGCWD, a tiny learned curve/DIP module (à la IA-YOLO's differentiable filters), or a 1–2-layer detail-enhancement conv. Add CLAHE and a tiny learned-curve baseline to Table 2 to show gamma is a deliberate efficient choice, not a lazy one.

**3. ViT neck on YOLOv8 — incremental unless isolated.** Transformer necks are well-trodden. The neck is only a contribution if Table 2's neck ablation shows a clear AP-small gain attributable to the ViT neck *over* BiFPN/P2/attention necks at matched FLOPs, ideally with the small-object loss (NWD) controlled. If the gain is <1 AP or comes with a large FLOP cost, drop the "contribution" framing and present it as a design detail.

**4. Thermal-detection pitfalls to preempt:** (a) FLIR ADAS annotation issues and the aligned-vs-unaligned split — state exactly which version and split; the aligned set (5,142 pairs) does not represent the full FLIR-ADAS distribution (cross-dataset tests confirm FLIR-aligned under-covers FLIR-ADAS). (b) LLVIP saturation — report mAP@50-95, not mAP50, or gains look marginal. (c) Inconsistent number reporting across papers (mAP vs mAP50, different splits, IR-only vs fused) — build your own controlled re-runs rather than copying table numbers. (d) Dataset leakage from video frames (adjacent frames in train/test) — use scene-disjoint splits. (e) Report per-class AP (person/car/bicycle) since class imbalance dominates FLIR.

**5. Must-cite closest prior work (omission → desk-reject):** DTRDNet (Sensors 2024), Aloufi et al. weather-classification+detection (Electronics 2024, 13(15):3063), TogetherNet (PG/CGF 2022), BAD-Net/Detection-Friendly Dehazing (TIP 2023), IA-YOLO (AAAI 2022), D-YOLO (2024), RDMNet (T-IV 2024), DENet (ACCV 2022), WM-MoE (2023), AW-MoE (2026, weather-aware routing for 3D detection), WARLearn (WACV 2025), ERUP-YOLO (WACV 2025), UniDet-D (2025), causal-intervention adaptation (TPAMI 2022/2024); thermal-in-weather: Bhadoriya et al. (Sensors 2022, 22(12):4567), WiSE-OD (2025), M3FD (CVPR 2022), Shyam & Yoo (WACV 2024); cross-modal KD: FreqKD (2026); AiOIR: PromptIR, DA-CLIP, Histoformer, TransWeather, WeatherDiffusion, AdaIR.

## Recommendations

**Stage 1 — De-risk novelty before running large experiments.** Reframe the paper around the single claim "gated, thermal-specific, real-time adverse-weather detection that matches all-in-one restoration accuracy at edge-deployable latency." Run Table 3 (gating value + oracle/no-gate bounds + gate-error sensitivity) and one Table 1 comparison (GATE-IR vs AiOIR+detector vs weather-augmented YOLOv12) first. **Go/no-go threshold:** if GATE-IR does not dominate the accuracy-latency Pareto frontier against a soft-router and AiOIR, redesign before investing further.

**Stage 2 — Establish SOTA credibility.** Add YOLOv10/YOLO11/YOLOv12 and one DETR-family detector (RT-DETRv2 or D-FINE) to Stage-C swaps. Use M3FD and DENSE (real weather) as primary; use synthetic fog/rain only as augmentation, explicitly framed as approximate corruption with the LWIR physics caveat. Report mAP@50-95 and MR^-2, multi-seed mean±std.

**Stage 3 — Validate the fragile pieces.** Run Table 4 with FreqKD as the cross-modal-KD rival and a no-CycleGAN ablation; if the CycleGAN pseudo-RGB path doesn't beat direct RGB-pretraining or a FreqKD-style train-only scheme, cut it — it is the most attackable component. Run Table 2's fog/rain/neck swaps to prove gamma, LSRB and the ViT neck are deliberate efficient choices.

**Stage 4 — Edge story.** Table 5 on Jetson Orin with TensorRT FP16/INT8 and the Pareto plot; Table 6 real-vs-synthetic and unseen-degradation (especially: what the gate does with snow/noise it was never trained on).

**Benchmarks that change the recommendation:** if all-in-one restorers (PromptIR/Histoformer) match GATE-IR within noise on real thermal weather at comparable latency, pivot the contribution to the gate-error-robustness and edge-efficiency angle. If the ViT neck gains <1 AP-small over BiFPN, demote it. If CycleGAN distillation ≤ baseline, remove it and adopt an InfraNet/FreqKD-style train-only RGB-guided scheme.

## Caveats
- Several cited 2025–2026 works are arXiv preprints (e.g., AW-MoE, UniDet-D, FreqKD, RT-DETRv4, WiSE-OD, InfraNet); verify final venue, exact arXiv IDs and reported numbers before submission — preprint IDs and figures can change.
- Reported mAP numbers are drawn from heterogeneous papers with different splits, detectors, and IR-only-vs-fused settings; treat them as ballpark context, not directly comparable, and re-run controlled baselines.
- The "LWIR penetrates fog" evidence base leans partly on vendor (FLIR/Teledyne) and free-space-optics literature; anchor the physics claim in peer-reviewed sources (Narasimhan & Nayar IJCV 2002 / TPAMI 2003; Beier & Gemperlein MODTRAN SPIE 2004; the SPIE fog-attenuation literature) for a CV audience.
- No peer-reviewed radiometric LWIR fog-synthesis model was found; this is both a defensible novelty opening and a limitation that must be stated openly.
- This survey maps the competitive landscape and experiment design; it does not itself establish that GATE-IR achieves SOTA — that depends on the empirical results of the matrix above.