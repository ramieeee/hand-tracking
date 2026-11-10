# Improving ASL fingerspelling across hand angles

Research date: 2026-11-10. Scope: this repository's webcam ASL manual alphabet recognizer, especially A, R, and W. This is a code audit, a synthetic sensitivity experiment, and a literature review. No real-signer accuracy measurement or trained replacement model is claimed.

## Recommendation

Hand angle is a plausible cause, and the implementation has specific weaknesses that explain it. Start with a labeled replay benchmark, correct the image aspect-ratio distortion, and separate recording controls from alphabet input. Then compare the existing rules against a small trained classifier using 3D hand-relative features **plus retained orientation**. Add a temporal model for fluent spelling after the held-letter task works. Use RGB hand crops if landmark failures remain concentrated in crossed fingers and hidden thumbs.

Do not make all rotations equivalent: orientation distinguishes some letters. Also, 3D coordinates estimated from one RGB camera cannot reveal a completely hidden thumb reliably.

## What the current code actually does

These are repository observations, separate from literature findings:

| Observation | Code location | Consequence / hypothesis |
| --- | --- | --- |
| ASL uses fixed rules, not a trained ASL classifier. | `gestures/asl_alphabet.py`: `classify_asl_letter`, `_fist_letter` | Detector confidence settings do not tune letter decision boundaries. |
| Nearly every measurement uses `:2`; depth is discarded. | `hand_features` | Out-of-plane rotation changes apparent bend, spacing, and thumb reach. |
| Distances use raw normalized x/y, without frame dimensions. | `main.py`: construction of `frame_hands`; `hand_features` | x and y have different pixel scales on nonsquare images. At 1920×1080, normalized x displacements are 0.5625 times normalized y displacements for equal pixel lengths. |
| Scale is projected index-to-pinky MCP distance. | `hand_features` | Turning the palm edge-on shrinks the denominator and amplifies noise. |
| Local palm axes are normalized separately, not orthogonalized. | `hand_features`: `distal`, `across` | Local projections remain coupled under oblique views. |
| R requires two fingers to be classified as extended, then a reversal of projected tip order and a small gap. | `classify_asl_letter`: `two`, `crossed` | Overlap or foreshortening can prevent reaching the R branch even when crossing is detected. |
| W requires three extended fingers, a curled pinky, two minimum gaps, and upright orientation. | W branch | Any one gate can reject a valid-looking W. |
| A depends on all four curled states and a thumb-position cutoff. | `_fist_letter` | Hidden thumb position can confuse A/S/T/M/N; angle is only one possible cause. |
| The S key's One Euro smoothing affects drawing and control geometry, while ASL receives raw landmarks. | `main.py`: `points`, `frame_hands`, `speller.update` | Toggling S does not smooth ASL classifier inputs. |
| F overlaps OK; `Closed_Fist` can overlap closed letters. | `main.py`: control holds; `config.py` | F can finish a recording after 1 second; fist-like letters can cancel after 2 seconds. A canceled session is discarded. |
| Motion paths divide each absolute point by that frame's palm width. | `_scaled_point` | A changing denominator can produce apparent J/Z/repeat movement without equivalent physical movement. |
| Ground truth, image dimensions, world landmarks, and model identity are not saved in the session schema. | `common/recording.py`; frame construction in `main.py` | Existing unlabelled landmark recordings cannot establish accuracy or fully reproduce extraction. |

Google documents image landmarks with x normalized by width and y by height, and separate world landmarks in meters. The current Gesture Recognizer already returns world landmarks; a second hand detector is not required to collect them. These are model estimates, not depth-sensor measurements. [Google Gesture Recognizer](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer), [Google Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker).

## Reproducible local evidence

Run from the repository root:

```sh
uv run --with numpy python research/asl-fingerspelling/angle_probe.py > research/asl-fingerspelling/angle-probe-results.csv
uv run --with numpy python -m unittest discover -s tests -p test_asl_alphabet.py
```

The eight existing ASL tests passed. The probe uses their 24 static fixtures; J/Z need motion and are excluded. It rotates the fixtures around the wrist and passes their projected XY to the unchanged classifier. It separately simulates width/height normalization of isotropic geometry.

| Transformation | Predictions agreeing with original fixture labels |
| --- | --- |
| Original fixture geometry | 24/24 |
| +30° yaw | 21/24 |
| +45° yaw | 20/24 |
| +60° yaw | 15/24; R becomes K |
| +45° pitch | 19/24 |
| +60° pitch | 6/24; A and W become unknown |
| +60° roll | 11/24; W becomes unknown |
| Separate x/y normalization for a 16:9 image | 15/24; R becomes K |

**These counts are sensitivity checks, not recognition accuracy.** Fixtures are flat synthetic skeletons designed for the rules; they do not simulate a real hand, perspective, occlusion, or landmark estimation. Large rotations may change a letter's intended orientation, so disagreement is not automatically an error. The aspect-ratio experiment isolates a coordinate-metric problem. The angle experiments demonstrate fragility but cannot determine how often it explains the user's failures. Full output: [angle-probe-results.csv](angle-probe-results.csv); implementation: [angle_probe.py](angle_probe.py).

## What research supports

| Primary source | Finding relevant here | Practical implication and limitation |
| --- | --- | --- |
| [Uboweja et al., 2023: On-device Real-time Custom Hand Gesture Recognition](https://arxiv.org/abs/2309.10858) | Describes a pretrained single-hand embedding and custom gesture training for on-device use. | A custom static alphabet model is a practical baseline. It does not by itself solve J/Z or continuous spelling. [Model Maker API](https://developers.google.com/edge/api/mediapipe/python/mediapipe_model_maker/gesture_recognizer/GestureRecognizer) provides training, evaluation, and export. |
| [Shi et al., 2019: Fingerspelling Recognition in the Wild With Iterative Visual Attention](https://arxiv.org/abs/1908.10546) and the [authors' dataset page](https://home.ttic.edu/~klivescu/ChicagoFSWild.htm) | Natural fingerspelling is a visual sequence problem. ChicagoFSWild contains 7,304 sequences from 160 signers; ChicagoFSWild+ has 55,232 from 260. | Use natural sequences for later evaluation. Respect supplied train/dev/test partitions; extract compatible landmarks before applying this app. These are not isolated-letter webcam datasets. |
| [Georg et al., 2024: FSboard](https://arxiv.org/abs/2407.15806) | Smartphone ASL data from 147 Deaf signers; over 250 hours and 3 million characters. The paper reports 11.1% CER with MediaPipe Holistic inputs and ByT5-Small on unique test phrases and signers. | Especially relevant to camera-based text entry and landmark sequences. Its CER belongs to that model/data/protocol; this app has not reproduced it. |
| [Cha et al., 2026: OpenFS](https://arxiv.org/abs/2602.22949) | Addresses signing-hand ambiguity, sequence alignment, and unseen words using pose sequences. | Relevant when progressing past right-hand-first selection and held letters. [Official implementation](https://github.com/AIRC-KETI/OpenFS) has preprocessing, training/evaluation scripts, and checkpoint download instructions; its stated license is non-commercial research only. Its tested environment is Ubuntu/CUDA/NVIDIA A40, so local Mac latency and portability require evaluation. |
| [Papadimitriou et al., ICCV workshop 2025: Seeing in 2D, Thinking in 3D](https://openaccess.thecvf.com/content/ICCV2025W/CV4A11y/html/Papadimitriou_Seeing_in_2D_Thinking_in_3D_3D_Hand_Mesh-Guided_Feature_ICCVW_2025_paper.html) | Distills hand-mesh features into an RGB recognizer, with temporal modeling and CTC; the teacher is not used at inference. | Supports evaluating learned structure and image appearance together. This is a larger research option, not a ready replacement. The CVF indexed abstract was available; direct page retrieval returned 403 during this review, so detailed results were not independently inspected. |

Use [Bill Vicars' ASL alphabet teaching reference](https://www.lifeprint.com/asl101/fingerspelling/abc.htm) to check elicitation, with review by a fluent ASL signer. It is not an accuracy benchmark. Do not copy its alphabet artwork into this project; the page restricts republication.

## Concrete implementation plan

### 1. Establish a replay benchmark and remove collection interference

- Add a dedicated alphabet capture mode with keyboard start/finish/cancel, so F and closed letters cannot trigger control gestures.
- Prompt an intended letter or word and save that human-supplied ground truth separately from predictions. Start with A/R/W and confusions A/S/T/M/N, R/U/V/K, W/B/F. Retain all 24 static letters as regression coverage.
- Save timestamp, actual frame width/height, mirrored-input flag, image landmarks, available world landmarks, handedness and score, model version/hash, method ID, signer/session/clip IDs, and approximate view condition. Do not assume the API supplies reliable per-joint confidence. Preserve null hands and tracking dropouts.
- Optionally save short consented RGB clips/crops for inspecting occlusion and rerunning extraction. The existing recordings cannot supply RGB retrospectively.
- First reproduce the user's failures across separate capture sessions. Then recruit several signers, including held-out people, before making general accuracy claims.

### 2. Fix and compare coordinate representations

For image-plane geometry use `(x * width, y * height)`, or `(x * width / height, y)`. Keep original normalized coordinates for display/storage compatibility. Old recordings lack actual dimensions: only replay with an explicitly recorded assumption, not the current config silently applied retroactively.

For a 3D candidate use world landmarks, wrist translation removal, and a palm scale from multiple stable bone lengths. Construct an orthonormal palm frame, for example:

```text
u = normalize(P5 - P17)
v0 = P9 - P0
v = normalize(v0 - dot(v0, u) * u)
n = cross(u, v)
q_i = [dot(Pi-P0,u), dot(Pi-P0,v), dot(Pi-P0,n)] / palm_scale
```

Reject degenerate frames. Verify handedness/mirror conventions using known left/right captures before canonicalizing hands. Evaluate actual PIP/DIP joint angles, normalized fingertip distances, thumb relationships, and crossing/depth relationships. **Keep palm orientation and image direction as additional features** so K/P and G/Q remain distinguishable. Global camera orientation alone is still imperfect without body/forearm context. These feature choices are proposed experiments, not a literature guarantee of accuracy.

### 3. Train a small held-letter classifier

Compare a regularized MLP or tree-based classifier on engineered features against the rules, then optionally MediaPipe's custom embedding classifier. Include a learned `none/transition` class with real nonletters and transitions. Use class probabilities, a validation-selected acceptance threshold and top-two margin; display uncertainty instead of forcing a letter. Calibrate scores on held-out validation data before treating them as confidence.

Collect repeated clips across comfortable yaw/pitch/roll, camera distance, lighting/background, and both hands. Approximate view bins such as front, ±15°, ±30°, and ±45° are proposed test conditions, not promised supported angles. Keep meaningful letter orientation intact. Add modest landmark noise, scale/translation, and plausible rotations during training; synthetic rotations cannot create real occlusions. Review augmented examples rather than assigning the same label to arbitrary rotations.

### 4. Improve temporal decoding, then add appearance if necessary

Smooth probabilities over a time window and use hysteresis, stable hand identity, and an explicit unknown state. Use actual timestamps so variable FPS does not change the smoothing duration. Repair motion features with wrist-relative fingertip paths plus a separate wrist trajectory, using a stable segment scale; retain wrist rotation features for J and avoid removing meaningful movement.

For continuous spelling, compare a small TCN/GRU with CTC against the held-letter decoder; OpenFS offers a further alignment alternative. Repeated letters require correct blank/repeat handling, and J/Z need labeled motion clips. Evaluate streaming delay explicitly; an offline model may use future frames. If R or closed-thumb letters remain poor when landmarks are visibly wrong, compare an RGB crop classifier or RGB-plus-landmark model. A depth camera is an optional later experiment, with extra hardware and a new capture setup.

## Evaluation and decision gates

Split by signer and session **before** frame sampling and augmentation. Never put neighboring frames of one clip into train and test. For personalization, hold out a different session and label the result as user-specific; for generalization, hold out whole signers. Select thresholds/hyperparameters on validation only.

Report per-letter precision/recall and macro-F1, the full confusion matrix, missing-hand rate, and results by view condition and hand. Count dropped/unrecognized examples in end-to-end results. For rejection, report both accepted-prediction accuracy and coverage. For words, report CER `(substitutions + deletions + insertions) / reference characters`, exact-string accuracy, and repeated-letter/J/Z failures. Also measure FPS, prediction stability, and median/p95 commit latency on this Mac.

Run ablations in order: current rules → aspect-corrected rules → 3D/orientation-aware features → trained static classifier → temporal model → optional RGB fusion. Retune each changed feature representation on training/validation data; current 2D thresholds should not be copied into a 3D metric unchanged. Promote a candidate only after improvement on held-out A/R/W clips without material regressions across the remaining letters, acceptable unknown coverage, and interactive latency. Set numerical targets after measuring the real baseline.

The first implementation batch should therefore be **capture/labels + replay evaluation + aspect correction + nonconflicting alphabet controls**. This provides the evidence needed to choose between a small landmark model and a larger appearance/sequence model. Recognition runtime is unchanged by this research commit.
