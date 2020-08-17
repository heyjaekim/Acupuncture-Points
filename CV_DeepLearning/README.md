# DL Application to predict Acupuncture Points 
## Overview
### 1. Image(Input) &rarr; 2. ConvNet (Feature Extraction) &rarr; 3. Coordinate Regression (Output)
- Backbone : ResNet ( advancements will be made  )
- **Colored Mnist Dataset** designed and used to check out basic performance

- Toy Model
![20200731 발표자료_최종 (15)](https://user-images.githubusercontent.com/63584973/89733880-7703b200-da93-11ea-8d2b-1b78dadedfcf.png)

---
### TODO LIST
- [x] Learning Rate Schduler
- [x] Train / Validation Splitter
- [x] Checkpoint Saver/Loader
- [x] Quick Test Data Checking 
- [x] Train Monitoring Utils (loss, time, pixel distance, real distance)
- [ ] TB Utils (almost done)
- [ ] Parser (ongoing)
- [ ] Model Cfgs (ongoing)
- [x] DataLoader (Hand Dataset)
- [ ] Dataset Augmentation (Rotate, Flip, Crop, Add noise, erase, Cutmix) (need to check)
- [ ] XAI
- [ ] Hyperparameter Optimization 
- [ ] Execution Speed 
- [ ] New Data 
- [ ] Graphics 
