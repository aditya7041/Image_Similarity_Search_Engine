# Image_Similarity_Search_Engine

Description: 
This is an image simlarity search engine trained on 2000 chairs and sofa and is used as encoder to find the visually similar images. 
Please look at result_demo.doc for high level architecture, and the results on couple of test cases.

Main components: 
1) Faster RCNN: Used to get the objects from the images with the bounding box. We have used the open source library written in Caffee for our purpose, so not including in this repo. 
2) Web crawler : Used to get the images and corresponding URL from different external sites like Amazon, overstock etc. based on the class detected by faster-RCNN (File : WebCrawler.py)
3) Image ranker: This is the core brain of the search engine which is written in tensorflow and is trained on 2000 images of chair and sofa. This is used to find the score of images based on the simlarity i.e. cosine similarity of fully connected layer of the image.
