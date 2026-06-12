# Optical Character Recognition (OCR) using Python

**Author:** Vansh Rao

A machine learning project that performs OCR on handwritten/printed characters using image preprocessing techniques and classical ML classifiers.

## Dataset
- `samples1000/` — Combined images (5 letters per image)
- `samples_split/` — Individual character images (after splitting)

## Approach

### Image Preprocessing
- Splitting combined images into individual character images
- Grayscale conversion
- Morphological operations: erosion, dilation, opening
- Filters: Gaussian blur, Median filter, Bilateral filter
- Thresholding: Binary, Adaptive, Otsu
- Denoising, Histogram Equalization, Resize, Deskew
- Final pipeline: Resize → Grayscale → Histogram Eq → Dilate → Denoise → Median Filter

### Feature Extraction
- HOG (Histogram of Oriented Gradients) features

### Models
- Logistic Regression
- Support Vector Machine (SVM) — best performer
- Random Forest

### Hyperparameter Tuning
- GridSearchCV on SVM (C, degree, coef0)
- Best: `kernel='poly', C=10, degree=2, coef0=0.0`

### Additional Approach
- CNN-based classifier also explored
