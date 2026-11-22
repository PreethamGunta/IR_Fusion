# Graph-Structured Knowledge Distillation for Infrared Object Detection

## Experimental research plan and implementation specification

**Project:** BTP --- Infrared Object Detection\
**Document type:** Research experiment plan\
**Status:** Proposal for implementation and ablation; not a claim of
validated performance\
**Primary deployment constraint:** The deployed detector receives IR
only. RGB and all teachers are training-time components.

------------------------------------------------------------------------

## 1. Executive recommendation

Build the work around an **IR-only detector trained with ground-truth
supervision**, then test whether structured teacher guidance improves
it. The most promising graph-KD contribution is not simply matching
teacher and student logits element by element. It is to represent
detections as relational graphs, pass contextual information over those
graphs, and align teacher/student structures even when they produce
different numbers of candidate detections.

Recommended order:

1.  **Baseline:** train an IR-only YOLO detector with the project's
    verified FLIR setup.
2.  **Teacher A:** train a paired RGB-IR detector, then distil its fused
    prediction graph into an IR-only student.
3.  **Teacher B:** train independent RGB and IR teachers; use
    reliability-aware graph consensus to guide the IR student.
4.  **Teacher C (optional):** combine the independent teachers and
    paired RGB-IR teacher through a common student graph.
5.  Compare graph matching methods: Hungarian assignment, Sinkhorn
    optimal transport (OT), fused Gromov-Wasserstein (FGW), and partial
    OT.
6.  Add graph condensation and ground-truth-anchored matching only after
    the basic pipeline is stable.

**Important alignment with the current project design:** Project Brain
v7.0 recommends an IR-main detector with a training-only RGB auxiliary
pathway as the primary design, and keeps explicit multimodal-to-thermal
KD (including DecomKD-style decomposition) as a controlled alternative.
It explicitly cautions against the older default chain of multimodal
teacher → copied IR teacher → student, and says company-labelled IR
boxes should directly supervise the IR detector. This document therefore
treats graph-KD as an experimental extension, not a replacement for that
simpler baseline. The existing proposal-stage document also describes a
two-stage teacher/student pipeline; the experiments below should be
compared against the latest v7.0 design before being adopted as the
project default.

------------------------------------------------------------------------

## 2. Research questions and hypotheses

### RQ1 --- Does relational matching transfer more useful information than ordinary prediction KD?

**Hypothesis:** A graph loss that preserves object-object geometry and
semantic relations will improve localization and/or recall over standard
logit KD, particularly when teacher and student candidate counts differ.

### RQ2 --- Does paired RGB-IR teacher guidance improve an IR-only student?

**Hypothesis:** A quality-aware RGB-IR teacher can provide useful
privileged information, but its benefit depends on pair alignment,
sensor compatibility, and whether the RGB evidence is relevant to the IR
task.

### RQ3 --- Is a set of independent modality teachers more reliable than a fused teacher?

**Hypothesis:** RGB and IR teachers have complementary failure modes.
Reliability-aware consensus should outperform naïve averaging,
especially when RGB is degraded or poorly aligned.

### RQ4 --- Which graph correspondence method is best?

Compare: - Hungarian assignment: strong, simple one-to-one baseline. -
Sinkhorn OT: soft correspondence with entropy regularization. - FGW:
combines node-feature similarity and graph-structure similarity. -
Partial OT / dustbin: allows unmatched detections and different object
counts.

### RQ5 --- Can graph condensation preserve useful teacher knowledge at lower compute?

**Hypothesis:** Selecting a compact, informative subgraph can retain
most of the KD benefit while reducing graph construction and matching
costs.

### RQ6 --- Does GT-anchored matching reduce false correspondences?

**Hypothesis:** Using ground-truth objects as shared anchors can
stabilize matching when teachers and students produce different
candidate sets, especially for crowded scenes and small objects.

------------------------------------------------------------------------

## 3. Data strategy and compatibility gates

### 3.1 Primary dataset: FLIR ADAS

Use the exact FLIR release already available to the project as the first
benchmark. Do not assume all FLIR releases have identical annotations,
pair identifiers, splits, or class definitions.

Before training, create a dataset audit report covering:

-   Release/version, file counts, annotation format, and official
    train/validation/test split.
-   Number of RGB and IR images and the number of true paired frames.
-   Pair identifiers, timestamps, synchronization, and registration
    quality.
-   Thermal representation: native high-bit-depth/radiometric data,
    pre-AGC values, 8-bit AGC grayscale, or palette rendering.
-   RGB/IR resolution, aspect ratio, field of view, and object-scale
    distribution.
-   Class names, class frequencies, ignored/crowd labels, and annotation
    coverage.
-   Whether the chosen FLIR subset is the aligned paired release or an
    unpaired/partially paired release.

**Do not pair images by filename order or approximate timestamp alone.**
Use explicit pair metadata and manually inspect a sample of pairs. If
registration is imperfect, use object-level matching rather than
assuming pixelwise correspondence.

### 3.2 Secondary datasets

  -----------------------------------------------------------------------------
  Dataset           Main value        Use in this project  Caveat
  ----------------- ----------------- -------------------- --------------------
  M3FD              Paired            External paired-data Reconcile its
                    visible-thermal   test; cross-dataset  six-class ontology
                    road scenes,      generalization       with FLIR before
                    multiple object                        joint training
                    categories                             

  LLVIP             Paired visible/IR Pedestrian-focused   Narrower class
                    low-light         transfer test        scope; verify
                    pedestrian scenes                      sensor/data
                                                           characteristics

  KAIST             Paired            Established          Pedestrian-centric
  Multispectral     visible-thermal   pedestrian           and sequence leakage
                    pedestrian        cross-modal          must be controlled
                    benchmark         benchmark            

  DroneVehicle      Paired aerial     Test transfer to     Different viewpoint
                    RGB-IR vehicle    aerial vehicle       and domain; not a
                    scenes            detection if         direct FLIR
                                      relevant             substitute
  -----------------------------------------------------------------------------

Use FLIR for the main development cycle. Treat
M3FD/LLVIP/KAIST/DroneVehicle as external tests or separate transfer
studies, not as interchangeable samples in one pooled dataset.

### 3.3 Splitting and leakage control

-   Split by complete sequence, recording session, camera, or site
    before frame extraction.
-   Keep RGB-IR pairs and all derived crops/augmentations in the same
    partition.
-   Reserve validation data for architecture, thresholds, and KD-weight
    selection.
-   Keep a locked test set for final reporting.
-   Where feasible, add a held-out camera/site split.
-   Avoid random frame-level splits across adjacent video frames.

### 3.4 Class ontology

The FLIR release and the chosen repository may expose different class
subsets (for example, a reduced person/bicycle/car setup versus a larger
ontology). Select one canonical mapping before training. Report the
mapping and ignored classes. Never compare models trained/evaluated with
different class ontologies as if the results were directly comparable.

------------------------------------------------------------------------

## 4. Model choices and initialization

### 4.1 Detector family

YOLO12 is a candidate family because the current project is already
considering YOLO-style detectors. Treat model size and input resolution
as experimental variables, not fixed decisions.

Suggested starting configuration:

  -----------------------------------------------------------------------
  Role                    Initial candidate       Alternatives
  ----------------------- ----------------------- -----------------------
  RGB unimodal teacher    YOLO12l                 YOLO12m if compute is
                                                  limited

  IR unimodal teacher     YOLO12l                 YOLO12m

  RGB-IR teacher          Two-stream YOLO12m/l    Established RGB-T
                                                  detector as an external
                                                  architecture baseline

  IR student              YOLO12n                 YOLO12s as a second
                                                  capacity point

  Optional intermediate   YOLO12s/m               Only if needed for
  student                                         capacity scaling
  -----------------------------------------------------------------------

A YOLO12n student is a useful compression target, but do not assume it
is the best production model. Benchmark accuracy, latency, memory, and
export support on the actual target hardware. Keep YOLOv8s or another
stable one-stage detector as a reproducibility baseline if the YOLO12
implementation or training behavior proves unstable.

### 4.2 IR input adaptation

The detector must accept the project's actual IR representation. If the
model's first convolution expects three channels and the input is
single-channel, compare:

1.  **Mean-kernel adaptation:** average the three RGB input-channel
    kernels into one IR channel.
2.  **Replicated grayscale:** replicate the IR channel to three channels
    and retain the RGB stem.
3.  **IR-specific initialization:** use a checkpoint trained on a
    compatible thermal dataset, if its data, class, and sensor
    provenance are verified.
4.  **Random stem / pretrained remainder:** initialize the stem
    separately while retaining compatible pretrained weights elsewhere.

Run these as a controlled initialization ablation. Do not assume RGB
pretraining is always best for thermal imagery.

### 4.3 Pretrained weights by role

  -----------------------------------------------------------------------
  Model             Recommended       Fine-tuning       Checks
                    starting weights  procedure         
  ----------------- ----------------- ----------------- -----------------
  RGB teacher       COCO-pretrained   Fine-tune on RGB  Confirm
                    RGB detector      FLIR training     class-head
                    weights           images and the    remapping and RGB
                                      canonical project preprocessing
                                      classes           

  IR teacher        Same COCO         Fine-tune on IR   Validate thermal
                    detector weights  FLIR training     representation
                    with one-channel  images            and normalization
                    stem adaptation;                    
                    compare                             
                    IR-specific                         
                    checkpoint if                       
                    available                           

  RGB-IR teacher    Initialize RGB    Train both        Fusion modules
                    branch from RGB   branches and      are usually new
                    COCO weights;     fusion/head on    and need
                    initialize IR     paired RGB-IR     training; check
                    branch from       data              pair registration
                    adapted COCO or                     
                    compatible                          
                    thermal weights                     

  IR student        COCO-pretrained   Train with GT     Keep student
                    weights with IR   plus the selected architecture
                    stem adaptation;  KD objective      identical across
                    optionally                          KD baselines
                    initialize from                     
                    the IR teacher                      
  -----------------------------------------------------------------------

**Do not treat a public FLIR checkpoint as automatically safe.** Verify
its training split, class mapping, annotation version, and whether it
has seen the evaluation images. A checkpoint trained on the target test
split invalidates the comparison.

### 4.4 RGB-IR teacher architecture

Build a two-stream detector:

``` text
RGB image ──> RGB backbone ──> RGB P3/P4/P5 ──┐
                                               ├─> quality-aware fusion ─> neck ─> detection head
IR image  ──> IR backbone  ──> IR P3/P4/P5  ──┘
```

Implementation recommendations:

-   Separate modality-specific backbones and normalization.
-   Extract multi-scale P3/P4/P5 features.
-   Start with feature concatenation followed by a 1×1 projection and a
    standard feature pyramid neck.
-   Then test gated fusion or cross-attention at selected scales.
-   Use a learned quality gate to reduce reliance on RGB when it is
    blurred, saturated, misregistered, or otherwise unreliable.
-   Apply the same valid boxes/classes to both modalities.
-   Keep a direct IR prediction path and its detection loss active
    throughout training.

A simple fusion baseline is essential. Do not start with a large
collection of attention modules; add complexity only through ablations.

### 4.5 Established RGB-T architectures to reproduce or compare

The following are useful external baselines/reference implementations.
Their availability, exact checkpoint provenance, dependencies, and
dataset split must be checked before use:

-   **MMTOD:** https://github.com/tdchaitanya/MMTOD
-   **CSFNet RGB-T Object Detection:**
    https://github.com/Danial-Qashqai/RGBT-ObjectDetection
-   **CDFNet:** https://github.com/WenCongWu/CDFNet
-   **LCAFNet:** https://github.com/WenCongWu/LCAFNet
-   **TarDAL:** https://github.com/chandra-mizzou/TarDAL

These are not interchangeable architectures. Some use Faster R-CNN or
other detector families, and TarDAL is primarily a visible-infrared
fusion framework. Use them as reference implementations or external
baselines rather than assuming they can be dropped into a YOLO12
training loop unchanged.

------------------------------------------------------------------------

## 5. Graph representation

### 5.1 What is a node?

For object detection, the most direct graph is a **prediction-instance
graph**. Each node corresponds to one candidate detection before final
NMS.

Node features can include:

-   Class probability/logit vector.
-   Objectness or confidence.
-   Normalized bounding box `(cx, cy, w, h)`.
-   Scale level (P3/P4/P5).
-   ROI-pooled or proposal-aligned feature embedding.
-   Optional localization uncertainty and quality estimates.
-   Optional GT-association indicator during training.

Project all node embeddings to a common dimension, e.g. 128 or 256,
before graph matching.

### 5.2 What is an edge?

Create sparse edges using a combination of:

-   Spatial proximity between box centers.
-   Box overlap (IoU).
-   Relative geometry: center offset, width/height ratios, relative
    scale.
-   Semantic similarity between node embeddings or class distributions.
-   Optional same-region / same-GT association.

Start with a sparse union of top-k nearest neighbors and overlap edges
(for example, k=5). Normalize geometric features by image dimensions.
Avoid dense all-pairs graphs unless candidate counts are very small.

### 5.3 Candidate generation

-   Use pre-NMS detector outputs rather than only final NMS detections.
-   Keep a permissive confidence threshold and a capped top-K candidate
    set per image.
-   Preserve candidates matched to GT objects, including low-confidence
    candidates, during training graph construction.
-   Apply NMS only in the ordinary evaluation pipeline.
-   Detach teacher outputs; no KD gradient should update teacher
    parameters.

The candidate budget, confidence threshold, and graph size must be held
constant across graph-matching ablations.

------------------------------------------------------------------------

## 6. Core graph-KD objective

Let the frozen teacher produce graph (G_T=(V_T,E_T)) and the student
produce (G_S=(V_S,E_S)). Let (C\_{ij}) be the cost of matching teacher
node (i) to student node (j).

A practical node matching cost is:

\[ C\_{ij} = `\alpha`{=tex}*c D*{`\mathrm{cls}`{=tex}}(p_i^T,p_j^S)
+`\alpha`{=tex}*b D*{`\mathrm{box}`{=tex}}(b_i^T,b_j^S)
+`\alpha`{=tex}*f D*{`\mathrm{feat}`{=tex}}(z_i^T,z_j^S)
+`\alpha`{=tex}*o D*{`\mathrm{obj}`{=tex}}(o_i^T,o_j^S). \]

Use class-distribution divergence, normalized box distance / GIoU cost,
projected feature distance, and objectness difference. Normalize the
terms so one component does not dominate only because of scale.

Let (P) be the soft correspondence matrix. A general objective is:

\[ L\_{`\mathrm{total}`{=tex}} = L\_{`\mathrm{det}`{=tex}}\^{GT}
+`\lambda`{=tex}*{`\mathrm{node}`{=tex}}L*{`\mathrm{node}`{=tex}}
+`\lambda`{=tex}*{`\mathrm{edge}`{=tex}}L*{`\mathrm{edge}`{=tex}}
+`\lambda`{=tex}*{`\mathrm{struct}`{=tex}}L*{`\mathrm{struct}`{=tex}}
+`\lambda`{=tex}*{`\mathrm{pred}`{=tex}}L*{`\mathrm{pred}`{=tex}}. \]

Where:

-   (L\_{`\mathrm{det}`{=tex}}\^{GT}): the detector's ordinary
    supervised loss. **Always retain it.**
-   (L\_{`\mathrm{node}`{=tex}}): matched teacher/student class,
    objectness, box, and embedding transfer.
-   (L\_{`\mathrm{edge}`{=tex}}): relation consistency between matched
    node pairs.
-   (L\_{`\mathrm{struct}`{=tex}}): graph-structure alignment, e.g. FGW.
-   (L\_{`\mathrm{pred}`{=tex}}): conventional dense/anchor/query
    prediction KD, if compatible with the detector.

Start with a small number of terms. Add edge and structure losses
separately so their marginal value can be measured.

------------------------------------------------------------------------

## 7. Graph matching methods to implement

### 7.1 Hungarian matching --- required baseline

-   Construct the pairwise cost matrix (C).
-   Solve a one-to-one assignment.
-   Ignore unmatched nodes initially, then add explicit unmatched
    handling.
-   Use this as the simplest graph correspondence baseline.

### 7.2 Sinkhorn optimal transport --- first soft matching method

-   Convert costs to a kernel,
    e.g. (K=`\exp`{=tex}(-C/`\epsilon`{=tex})).
-   Apply Sinkhorn normalization to obtain a soft transport matrix (P).
-   Use balanced marginals only when teacher/student node mass should be
    fully matched.
-   Add a dustbin/unmatched mass for detections with no counterpart.

### 7.3 Fused Gromov-Wasserstein --- principal research method

FGW combines node-feature cost with relational structure cost.
Conceptually:

\[ L\_{`\mathrm{FGW}`{=tex}}(P)=
(1-`\eta`{=tex})`\langle `{=tex}C,P`\rangle`{=tex}
+`\eta`{=tex}`\sum`{=tex}*{i,k,j,l} \|A^T\_{ik}-A^S*{jl}\|\^2
P\_{ij}P\_{kl}, \]

where (A^T,A^S) encode teacher/student edge relations and (`\eta`{=tex})
balances feature and structure matching.

Implementation sequence:

1.  Implement Hungarian and verify it on synthetic graphs.
2.  Implement Sinkhorn OT and test transport marginals.
3.  Add FGW using a tested OT library or a small verified
    implementation.
4.  Use entropic regularization and sparse graph relations.
5.  Track convergence, transport entropy, runtime, and memory.
6.  Compare FGW against feature-only OT; otherwise any gain cannot be
    attributed to graph structure.

FGW can be computationally expensive. Keep node counts capped and graphs
sparse. Do not begin with multi-marginal GW.

### 7.4 Partial OT / dustbin

Object detectors often disagree on the number of candidate detections.
Balanced OT can force a poor match for every node. Add a dustbin node or
partial transport so unmatched teacher/student detections can remain
unmatched. Compare:

-   Balanced OT.
-   OT with dustbin.
-   Partial OT with a controlled transported mass.

This is particularly important for missed objects, false positives, and
different teacher/student recall.

------------------------------------------------------------------------

## 8. Proposed experiment families

## Experiment 0 --- Reproducible baselines

**Purpose:** Establish the accuracy/latency reference and verify the
data pipeline.

Train: 1. IR-only YOLO12n student. 2. IR-only YOLO12s student. 3.
RGB-only YOLO12l teacher. 4. IR-only YOLO12l teacher.

Use the same data split, class mapping, image size, augmentations, and
evaluation script where applicable.

**Record:** AP50:95, AP50, AP75, AP-small/medium/large, per-class AP,
recall, precision, ECE, parameters, latency, memory, and training time.

**Pass condition:** Stable training across at least three seeds and
verified pair/split integrity. Do not start graph KD until these
baselines are reproducible.

## Experiment A --- Paired RGB-IR teacher → IR student using prediction graphs

**Priority:** Highest.

### A1. Teacher architecture

Use the two-stream RGB-IR detector from Section 4.4. Train it on
verified paired FLIR data with shared GT annotations. Use a direct IR
detection loss plus fused-path detection loss:

\[ L_T = L\_{`\mathrm{det}`{=tex}}(IR) + `\lambda`{=tex}*f
L*{`\mathrm{det}`{=tex}}(RGB,IR). \]

Keep the IR branch directly supervised. Freeze the converged teacher for
student training.

### A2. Student

Use YOLO12n with a single-channel IR input. It receives only IR during
training and inference. Train with GT loss plus teacher graph KD.

### A3. Graph distillation

For each paired training sample:

1.  Run the frozen teacher on RGB+IR and collect pre-NMS candidates,
    logits, boxes, and ROI embeddings from its fused prediction path.
2.  Run the IR student on IR and collect its pre-NMS candidates and
    corresponding embeddings.
3.  Build teacher and student prediction graphs.
4.  Compute matching with Hungarian, Sinkhorn, and FGW in separate runs.
5.  Transfer class distributions, objectness, box geometry, and selected
    embeddings for matched nodes.
6.  Transfer pairwise relation structure for matched node pairs.
7.  Keep GT detection loss active for every student batch.

### A4. Loss ablations

  Variant   Objective
  --------- ---------------------------------------
  A0        IR student + GT only
  A1        GT + conventional prediction/logit KD
  A2        GT + graph node KD
  A3        GT + node KD + edge relation KD
  A4        GT + node/edge KD + Sinkhorn matching
  A5        GT + node/edge KD + FGW
  A6        A5 + partial OT / dustbin
  A7        A5 + reliability weighting

**Main question:** Does graph structure add value beyond ordinary
prediction KD?

### A5. Essential controls

-   Same student initialization and training schedule for all variants.
-   Same teacher checkpoint and candidate budget.
-   Same number of training steps.
-   Report graph matching runtime separately.
-   Run an IR-only teacher KD control to distinguish cross-modal value
    from generic teacher-capacity value.

## Experiment B --- Independent RGB teacher + IR teacher → IR student

**Priority:** High.

### B1. Teachers

Train separate YOLO12l RGB and IR teachers on their corresponding FLIR
modalities. Freeze both after validation-based model selection.

### B2. Reliability estimates

For each teacher prediction, estimate reliability using observable
signals:

-   Calibrated confidence.
-   Predictive entropy of class distribution.
-   Localization uncertainty (if modelled or approximated).
-   Agreement with the other teacher after object-level matching.
-   Modality quality descriptors (blur, saturation, contrast,
    registration confidence).

Do not equate raw confidence with calibrated correctness. Calibrate on
validation data, and validate reliability against actual
localization/classification correctness.

### B3. Consensus graph

1.  Build separate RGB and IR prediction graphs.
2.  Align their object candidates using Hungarian or OT with geometry
    and class costs.
3.  For matched detections, compute reliability-weighted class/box
    guidance.
4.  Retain a modality-specific prediction when the other modality is
    unreliable.
5.  Downweight or suppress KD when teachers strongly disagree and
    neither is clearly reliable.
6.  Construct a consensus graph and match it to the student graph.

### B4. Consensus cases

-   **Agreement:** both teachers predict compatible class and box;
    provide stronger consensus supervision.
-   **One reliable teacher:** use the reliable teacher's soft target
    with a calibrated weight.
-   **Disagreement:** reduce KD weight; preserve GT supervision rather
    than forcing an arbitrary average.
-   **Unmatched prediction:** allow partial OT/dustbin rather than
    forcing a false pair.

### B5. Ablations

-   IR teacher only.
-   RGB teacher only.
-   Naïve average of teacher predictions.
-   Confidence-weighted average.
-   Calibrated reliability-weighted consensus.
-   Graph consensus with Hungarian.
-   Graph consensus with Sinkhorn/FGW/partial OT.

**Main question:** Does modality-aware reliability improve over the best
single teacher and naïve ensemble?

## Experiment C --- Three teachers → one IR student

**Priority:** Exploratory; only after A and B work.

Teachers: 1. RGB-only teacher. 2. IR-only teacher. 3. Paired RGB-IR
teacher.

Build one graph per teacher and align each to a common student graph.
Use pairwise transports (P\_{RGB,S}), (P\_{IR,S}), and (P\_{RGBIR,S}).
Add a consistency penalty so the transports do not provide contradictory
guidance for the same student node.

Do not begin with a fully joint multi-marginal OT solver. Start with
factorized pairwise matching and reliability-weighted losses. Compare: -
Best individual teacher. - Equal-weight three-teacher KD. -
Reliability-weighted three-teacher KD. - Pairwise graph alignment plus
transport-consistency loss.

**Main question:** Does the third teacher add information beyond the
best two-teacher setup, enough to justify training and compute cost?

## Experiment D --- Ground-truth-anchored heterogeneous graph

**Priority:** Medium; potentially useful where candidate sets are
unstable.

Create a graph with two node types: - GT object nodes (shared
anchors). - Teacher/student prediction nodes.

Associate predictions with GT objects using IoU/class criteria. Distil
each teacher/student prediction through its common GT anchor, and
optionally transfer object-object relations between GT anchors.

This reduces dependence on direct teacher/student candidate
correspondence. It also allows analysis of missed objects and false
positives.

Compare: 1. Prediction-only graph matching. 2. GT-anchored node
matching. 3. GT-anchored + object-object relation KD.

**Caution:** GT anchors are available only during training. The method
must not require GT at inference.

## Experiment E --- Teacher graph condensation

**Priority:** Later optimization experiment.

Compress a large teacher graph to a smaller graph before matching.
Candidate selection strategies:

-   Top confidence.
-   GT-associated candidates.
-   Small-object / rare-class prioritization.
-   Graph centrality.
-   Uncertainty plus importance.
-   Learned condensation module.

Compare full graph vs fixed top-K vs each condensation strategy at
matched compute budgets. Report retained-node coverage, KD performance,
matching runtime, and memory.

Do not call a graph "condensed" merely because top-K filtering is used;
distinguish heuristic node selection from learned graph condensation.

## Experiment F --- Decomposed cross-modal student representation

**Priority:** Controlled alternative if direct multimodal KD harms the
IR student.

Multimodal and thermal-only representations are not identical. A single
forced feature-matching objective can transfer modality-specific
nuisance information. Test a student with:

-   IR-preserving feature path.
-   Cross-modal compensation path.
-   Integration module.
-   Instance-level feature KD on selected objects.
-   Uncertainty-weighted prediction KD.

This follows the motivation of DecomKD-style approaches. Compare it
against direct graph KD; do not combine all modules at once.

------------------------------------------------------------------------

## 9. Training schedule and implementation order

### 9.1 Staged schedule

Use this as an initial schedule to tune, not as a proven optimum:

1.  **Baseline stage:** train unimodal IR and RGB detectors.
2.  **Multimodal teacher stage:** train paired RGB-IR teacher.
3.  **Graph validation stage:** save teacher/student candidate outputs
    and test graph construction/matching offline.
4.  **Student warm-up:** train student with GT loss only for an initial
    portion of training.
5.  **KD ramp-up:** gradually increase KD weight to avoid early noisy
    teacher gradients.
6.  **Joint GT + KD stage:** train with selected graph loss and GT
    detection loss.
7.  **Target-domain adaptation:** fine-tune the IR student on labelled
    target-domain IR data with GT supervision. Add a separate IR teacher
    only if it demonstrates value over direct supervised fine-tuning.
8.  **Robustness stage:** test sensor-calibrated degradation
    augmentation and adversarial robustness as separate extensions.
9.  **Deployment stage:** remove all teachers, graph builders, OT
    solvers, and training-only branches; export the IR detector only.

### 9.2 Suggested software structure

``` text
IR-Graph-KD/
  configs/
    dataset_flir.yaml
    rgb_teacher.yaml
    ir_teacher.yaml
    rgbir_teacher.yaml
    ir_student.yaml
    graph_kd.yaml
  data/
    paired_dataset.py
    pair_alignment_audit.py
    class_mapping.py
    sequence_split.py
  models/
    rgb_teacher.py
    ir_teacher.py
    rgbir_teacher.py
    ir_student.py
    fusion/
      concat_fusion.py
      gated_fusion.py
      cross_attention_fusion.py
  graph/
    candidate_extractor.py
    graph_builder.py
    node_features.py
    edge_features.py
    graph_condensation.py
  matching/
    hungarian.py
    sinkhorn.py
    fused_gw.py
    partial_ot.py
  distillation/
    prediction_kd.py
    node_kd.py
    edge_kd.py
    graph_kd.py
    reliability.py
    gt_anchored_kd.py
    decomposed_kd.py
  training/
    train_unimodal.py
    train_rgbir_teacher.py
    train_student.py
    evaluate.py
    export_ir.py
  experiments/
    exp0_baselines.py
    expA_rgbir_to_ir.py
    expB_multiteacher.py
    expC_three_teacher.py
    expD_gt_anchored.py
    expE_condensation.py
    expF_decomposed.py
```

### 9.3 Engineering requirements

-   Cache teacher candidate outputs and embeddings to avoid repeated
    teacher inference when feasible.
-   Store raw pre-NMS predictions, not only rendered detections.
-   Save pair IDs and split IDs with every cached sample.
-   Use deterministic candidate caps and graph construction for
    reproducibility.
-   Unit-test box normalization, IoU, edge construction, transport
    marginals, and unmatched-node behavior.
-   Verify that teacher parameters receive no gradients.
-   Log each loss term independently.
-   Track NaNs, transport convergence, entropy, and graph size.
-   Keep the student inference graph independent of all teacher and
    graph-KD modules.

------------------------------------------------------------------------

## 10. Evaluation protocol

### 10.1 Detection performance

Report: - COCO-style AP50:95. - AP50 and AP75. - AP-small, AP-medium,
AP-large. - Per-class AP. - Precision, recall, and critical-class miss
rate.

### 10.2 Reliability and robustness

Report: - Expected Calibration Error (ECE) and Brier score. - Clean vs
degraded AP. - AP retention ratio: (AP\_{degraded}/AP\_{clean}). -
Worst-condition recall. - OOD detection metrics if OOD monitoring is
included. - Attack success rate and clean/attacked AP for adversarial
experiments.

Degradation tests must be calibrated to the actual target sensor once
its band, representation, and operating envelope are known. Do not label
generic image corruptions as physically realistic thermal degradation
without justification.

### 10.3 Efficiency

Report: - Parameters and model size. - FLOPs if measured consistently. -
P50/P95/P99 inference latency. - Sustained FPS and peak memory. -
Training time. - Graph construction and matching time per image. - Peak
memory for graph matching. - Student-only deployment latency (exclude
teachers and graph KD).

### 10.4 Statistical protocol

-   At least three seeds for key comparisons.
-   Same split, initialization policy, schedule, and training budget
    across ablations.
-   Report mean and standard deviation.
-   Use validation only for model/loss selection.
-   Final locked test is evaluated after the design is frozen.

### 10.5 Data efficiency

Train selected variants with 10%, 25%, 50%, and 100% of labelled IR
training data. Plot AP against annotation budget. This tests whether
privileged RGB/teacher guidance reduces label requirements rather than
merely improving full-data AP.

------------------------------------------------------------------------

## 11. Minimum ablation matrix

  -----------------------------------------------------------------------
  Question                            Required comparison
  ----------------------------------- -----------------------------------
  Does any teacher help?              GT-only student vs best teacher KD

  Does RGB guidance help?             IR teacher vs RGB teacher vs RGB-IR
                                      teacher

  Does graph structure help?          Conventional KD vs node KD vs
                                      node+edge KD

  Which matcher helps?                Hungarian vs Sinkhorn vs FGW vs
                                      partial OT

  Does reliability help?              Equal weights vs calibrated
                                      reliability weights

  Does GT anchoring help?             Prediction-only vs GT-anchored
                                      graph

  Does condensation help?             Full graph vs top-K vs
                                      importance/learned selection

  Is the student too small?           YOLO12n vs YOLO12s

  Is the gain robust?                 Clean vs sensor-calibrated degraded
                                      test

  Is the gain worth deployment cost?  AP vs latency/memory/parameters
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 12. Risks and decision rules

  ------------------------------------------------------------------------
  Risk                    Why it matters          Response
  ----------------------- ----------------------- ------------------------
  RGB/IR misregistration  Incorrect spatial       Audit pairs; use
                          matching creates        object-level matching or
                          harmful supervision     exclude bad pairs

  Spectral mismatch       A public thermal        Check band, bit depth,
                          dataset may not match   optics, AGC, and thermal
                          the target camera       representation

  Teacher overconfidence  Bad pseudo-targets can  Calibrate;
                          dominate GT             reliability-weight; keep
                                                  GT loss

  Candidate-count         Balanced matching       Dustbin / partial OT
  mismatch                forces false            
                          correspondences         

  FGW compute cost        Matching can dominate   Sparse graphs, top-K
                          training                candidates, cached
                                                  outputs

  Class ontology mismatch Invalid class transfer  Canonical mapping and
                                                  explicit ignored-class
                                                  policy

  Data leakage            Inflated evaluation     Sequence/session-level
                                                  splitting

  Student capacity too    Cannot absorb teacher   Compare nano and small
  small                   guidance                students

  Too many simultaneous   Cannot identify what    Add one component at a
  modules                 caused a gain           time

  Public checkpoint       Invalid benchmark       Audit checkpoint
  leakage                 claims                  provenance and training
                                                  split
  ------------------------------------------------------------------------

### Go/no-go rules

-   If paired data fail alignment/compatibility checks, do not train the
    RGB-IR teacher; continue with IR-only supervised training and an IR
    teacher baseline.
-   If conventional KD does not improve over GT-only, do not add FGW
    complexity yet.
-   If node KD helps but edge KD does not, retain node KD and drop the
    edge term.
-   If FGW does not outperform feature-only OT at comparable compute, do
    not claim structural matching is the source of improvement.
-   If the RGB-IR teacher is not better than the IR teacher on the
    relevant validation split, prioritize the IR teacher.
-   If student gains vanish on the locked target-domain test, do not
    present public-dataset gains as target-domain success.

------------------------------------------------------------------------

## 13. Recommended first implementation sprint

**Sprint 1 --- Data and baselines** 1. Confirm FLIR release, class
mapping, pair list, sensor representation, and sequence split. 2. Train
IR YOLO12n and YOLO12s baselines. 3. Train RGB and IR YOLO12l teachers.
4. Produce per-class and object-size diagnostics.

**Sprint 2 --- Paired teacher** 1. Verify paired RGB-IR data. 2.
Implement two-stream teacher with simple concatenation fusion. 3. Train
and evaluate fused teacher and direct IR path. 4. Save pre-NMS
predictions and P3/P4/P5 / ROI embeddings.

**Sprint 3 --- Graph machinery** 1. Implement prediction graph builder.
2. Test Hungarian matching. 3. Add Sinkhorn OT. 4. Add FGW and partial
OT. 5. Unit-test synthetic cases with unequal node counts and known
correspondences.

**Sprint 4 --- Student distillation** 1. Train conventional KD baseline.
2. Train graph node KD. 3. Add edge relation KD. 4. Compare matchers
under identical candidate budgets. 5. Run at least three seeds for
promising variants.

**Sprint 5 --- Multi-teacher** 1. Train/validate independent RGB and IR
teachers. 2. Calibrate their confidences. 3. Implement reliability-aware
consensus. 4. Compare against the best single teacher.

Do not begin three-teacher joint transport or learned graph condensation
until these sprints produce stable results.

------------------------------------------------------------------------

## 14. Expected research contribution

A defensible contribution could be:

> **Reliability-Aware Graph Distillation for Cross-Modal Infrared Object
> Detection:** a training-only framework that constructs sparse
> object-prediction graphs from one or more RGB/IR teachers, uses
> reliability-aware node and relation transfer, and aligns
> teacher/student graphs with partial fused optimal transport to train a
> compact IR-only detector.

The contribution is credible only if experiments establish all of the
following:

1.  Graph relations improve over conventional prediction KD.
2.  The chosen matching method improves over Hungarian or feature-only
    OT.
3.  Reliability-aware multi-teacher fusion improves over the best single
    teacher.
4.  Gains persist on a held-out, leakage-controlled IR test set.
5.  The deployed student remains IR-only and meets the target
    latency/memory budget.

Do not claim that graph matching, OT, GNN message passing, multimodal
fusion, or KD are individually new. Novelty must come from the specific
formulation, the way these components are combined for the task, and
empirical evidence against strong baselines.

------------------------------------------------------------------------

## 15. Selected references and implementation resources

### Project-source references

The following project documents should be read together with this
experiment plan:

-   `IR_Object_Detection_Project_Brain_v7(1).pdf` --- latest project
    architecture decision and evidence review.
-   `IR-Object-Detection-proposal.txt` --- earlier proposal-stage
    pipeline and robustness/privacy extensions.
-   `Graph Knowledge Distillation Frameworks.txt` --- graph
    construction, message passing, matching, condensation, and
    literature index.

### Detection and graph distillation

-   **DETRDistill (ICCV 2023):**
    https://openaccess.thecvf.com/content/ICCV2023/html/Chang_DETRDistill_A_Universal_Knowledge_Distillation_Framework_for_DETR-families_ICCV_2023_paper.html
-   **Label-Guided Self-Distillation for Object Detection (AAAI 2022):**
    https://doi.org/10.1609/aaai.v36i3.20240
-   **Gradient-Guided Knowledge Distillation for Object Detectors (WACV
    2024):**
    https://openaccess.thecvf.com/content/WACV2024/html/Lan_Gradient-Guided_Knowledge_Distillation_for_Object_Detectors_WACV_2024_paper.html
-   **GKD implementation:** https://github.com/lanqz7766/GKD

### RGB-thermal detection references / repositories

-   **MMTOD:** https://github.com/tdchaitanya/MMTOD
-   **CSFNet:** https://github.com/Danial-Qashqai/RGBT-ObjectDetection
-   **CDFNet:** https://github.com/WenCongWu/CDFNet
-   **LCAFNet:** https://github.com/WenCongWu/LCAFNet
-   **TarDAL:** https://github.com/chandra-mizzou/TarDAL

### Dataset entry points

-   **FLIR ADAS:** https://www.flir.com/oem/adas/adas-dataset-form/
-   **M3FD / TarDAL repository:**
    https://github.com/chandra-mizzou/TarDAL
-   **LLVIP:** https://github.com/bupt-ai-cz/LLVIP
-   **DroneVehicle:** https://github.com/VisDrone/DroneVehicle

### Implementation note

Repository links are starting points, not a guarantee that the current
default branch, pretrained weights, or environment remain compatible.
Pin commit hashes, record checkpoint URLs and hashes, and document the
dataset split used by every checkpoint.

------------------------------------------------------------------------

## 16. Final recommended experiment order

1.  **E0:** IR-only baselines and unimodal teachers.
2.  **A:** Paired RGB-IR teacher → IR student, first conventional KD
    then graph KD.
3.  **B:** Independent RGB + IR teachers → reliability-aware graph
    consensus → IR student.
4.  **Matcher study:** Hungarian vs Sinkhorn vs FGW vs partial OT.
5.  **D:** GT-anchored graph KD.
6.  **F:** Decomposed cross-modal student as an alternative if direct
    transfer conflicts.
7.  **C:** Three-teacher graph distillation.
8.  **E:** Graph condensation as a compute/scale optimization.

**Bottom line:** First prove that a clean IR-only baseline is strong and
that paired RGB-IR data are valid. Then test whether graph-structured
matching adds measurable value over ordinary KD. The most important
scientific comparison is not "complex graph KD versus no KD"; it is
**graph KD versus a well-tuned conventional KD baseline under the same
teacher, student, data, and compute budget**.

