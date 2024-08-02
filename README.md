
Image Forgery detection using Machine learning(Transfer learning)

Problem satetement:

The proliferation of digital image editing tools has exacerbated the issue of image 
forgery, raising concerns about the trustworthiness of visual content. Detecting these forged 
images manually is time-consuming and prone to errors, necessitating automated solutions. 
Thus, the problem at hand is to develop a robust image forgery detection system that can 
accurately identify various types of manipulations, such as copy-move, splicing, and
retouching, in digital images.

What does our model do:

The proposed model for image forgery detection leverages advanced deep learning 
techniques, specifically employing transfer learning with the ResNet-50 model within a 
sequential architecture. This approach capitalizes on the powerful feature extraction 
capabilities of ResNet-50, pre-trained on the ImageNet dataset, to automatically identify 
and analyze subtle cues indicative of image manipulations. By fine-tuning the ResNet-50 
model on a forgery detection task, the system can effectively learn to distinguish between 
authentic and manipulated images without the need for extensive labeled data, making it 
both efficient and robust.
 To enhance usability and accessibility, the system integrates with Streamlit, a userfriendly web application framework. The Streamlit interface provides a seamless platform 
for users to interact with the forgery detection model. Users can upload images directly 
through the interface, and the model processes them in real-time, generating classification 
results and visualizations. This immediate feedback empowers users to quickly assess the 
authenticity of images and detect potential forgeries.
 In practice, the proposed system follows a streamlined workflow: images are 
preprocessed for consistency, fed into the trained sequential model for inference, and the 
results are displayed dynamically through the Streamlit app. The system's architecture 
ensures scalability, allowing it to handle diverse types of image manipulations and 
accommodate varying levels of complexity in forgery detection tasks


Steps involved in buliding this model:

1.Dataset Acquisition and Preprocessing

2.Image Denoising Techniques

3.Visualization and Graph Plotting

4.Model Evaluation and Performance Analysis

5.Confusion Matrix Generation

6.Transfer Learning with ResNet-50

7.Transfer Learning Model Evaluation

Dataset :

https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset