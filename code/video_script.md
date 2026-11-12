# Three-minute video demonstration - recording guide

Show the presenter's face throughout narration (picture in picture is fine). If this is team work, share the spoken sections and make sure every member appears. Record and submit a real video of at most three minutes. Add actual student names and IDs on the opening screen.

| Time | Narration | Screen |
|---|---|---|
| 0:00-0:25 | "We chose Option 1: image classification. Our system predicts a host-disease class from one RGB plant photograph. We use macro-F1 because each class should count equally, even when some diseases have fewer images." | Show notebook title, team names/IDs, and a few representative training images. |
| 0:25-0:55 | "We use nine required hosts from verified PlantSeg v7, giving 43 classes and 4,349 images. We keep all images in the supplied train, validation and test splits. Our CNN starts from ImageNet-pretrained ResNet-18; we compare a frozen-head baseline with augmentation and class-weighted loss as separate changes." | Show the archive checksum, manifest summary, class distribution, and experiment table. |
| 0:55-1:40 | "Here is a held-out test example, clearly identified as such. I load the validation-selected checkpoint and class mapping, then apply the same deterministic preprocessing as evaluation. It predicts apple black rot with a maximum softmax score of 0.93. This score is not a calibrated disease probability." | Open the notebook demo cell or run `python code/inference.py --image "plantseg/plantseg/images/test/apple_black_rot_144.jpg"`. Show the test image, command, and printed prediction. Keep face visible. |
| 1:40-2:25 | "We selected class-weighted ResNet-18 using validation macro-F1 before final testing. On the test set, it achieved macro-F1 0.632, accuracy 0.669, and balanced accuracy 0.644. The basic CNN's macro-F1 was 0.631, so the gain is small. The most frequent error was tomato early blight predicted as tomato septoria leaf spot, in ten images." | Show validation comparison, frozen plan, final test table, and that confusion-matrix cell. |
| 2:25-3:00 | "One inspected close-up labelled tomato early blight was predicted as potato early blight; little of the host plant is visible. The supplied splits also contain repeated photographs, so the test score may be optimistic for new photos. Thank you." | Show the inspected incorrect example, then the report and notebook outputs. End before 3:00. |

The narrated values come from `result/final_test_metrics.json` and the saved inference output. Do not describe this test image as an unseen external photograph.
