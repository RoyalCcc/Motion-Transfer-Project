# CSE691 Image and Processing Motion TransferProject
NN, CycelGan, Pix2Pix and Everybody Dance Now for motion transfer.


## Download Pre-trained Models
For pre-trained models of Openpose and PerceptualSimilarity:
[Dropbox link](https://www.dropbox.com/sh/ghrackr5yem5lpy/AADa743FXzhLsS9vrhvg3y8ma?dl=0)

1. For Openpose:
Please put pose_model.pth into Baseline-NN/Openpose/network/ for NN and Openpose_Gen_Skeleton/lib/network/weight/
pose_model.pth is used to extract the keypoints of the body.
NN would choose the image which has the mininmum L2 distance between training images and current skeleton.

2. For PerceptualSimilarity：
Please put alex.pth, squeeze.pth, vgg.pth into ./models/weights/v0.1/ for each methods.
They are used to calculate the metrics between synthisezed images and grount-truths.


