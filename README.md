# AiryNet-Glassify
A Machine Learning project that utilizes Convolutional Network to analyze facial images and generate recommendations for eyewear products.

Created in 24 hours for **[TechX](https://xacademy.cc/index-en.html) Capstone Project 2021** (Intro to Machine Learning).

<br/>


## Project Structure

```
  ├ Webpage: Main                              => Clicking Button: Start
  ├ Webpage: Webcam                            => Clicking Button: Capture
  ├ Image: 120 * 120 * 3 (Array)
  ├ Input: 1 * 3 * 120 * 120 (Tensor)          => Apply AiryNet
  ├ Output: Probability of being first class (Small-frame)
  | ├ Decision 1
  | | ├ Webpage: Big-frame
  | ├ Decision 2
  | | ├ Webpage: Half-Half
  | ├ Decision 1
  └ └ └ Webpage: Small-frame
```

<br />

## AiryNet Architecture

```
1. Image: 3(C) * 120(H) * 120(W)
2. Convolution with 3 * 3 kernel: 32 * 18 * 118
3. Normalization & Relu
4. Convolution with 5 * 5 kernel: 64 * 114 * 114
5. Normalization & Relu
6. Pool with 2 * 2 max kernel + 2 stride: 64 * 57 * 57
7. Convolution with 3 * 3 kernel: 128 * 55 * 55
8. Normalization * Relu
9. Pool with 2 * 2 max kernel + 2 strideL 128 * 27 * 27
10. Convolution with 3 * 3 kernel: 64 * 25 * 25
11. Normalization & Relu
12. Convolution with 3 * 3 kernel: 32 * 23 * 23
13. Normalization & Relu
14. Pool with 2 * 2 max kernel + 2 stride: 32 * 11 * 11
15. Flatten
16. FC: 3872 to 2 fully conncted neurons
17. Output: Probability of being first class (Small-frame)
```

