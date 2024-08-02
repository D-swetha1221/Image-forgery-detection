import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load your trained image recognition model
def load_model():
    model = tf.keras.models.load_model('model_casia_run2.h5')  # Replace with your model path
    return model

model = load_model()

# Define the classes or labels for your model predictions
classes = ['class_1', 'class_2', 'class_3']  # Replace with your actual class names

def predict(image):
    try:
        # Convert uploaded image to numpy array
        img = np.array(image)

        # Resize image to match model's expected input size
        img = np.array(Image.fromarray(img).resize((224, 224)))

        # Normalize pixel values (assuming image is in RGB format)
        img = img.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]

        # Add batch dimension (model expects input shape [batch_size, height, width, channels])
        img = np.expand_dims(img, axis=0)

        # Verify input shape
        print("Input image shape:", img.shape)

        # Make prediction
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]

        return classes[predicted_class], confidence
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, 0.0  # Return default values in case of error


def calculate_ratio(a, b):
    # Function to calculate the ratio a/b
    if b != 0:
        return a / b
    else:
        return None  # Return None if denominator is zero


# Example usage that might cause a TypeError
numerator = 10
denominator = 0

result = calculate_ratio(numerator, denominator)
print(result)  # This will print None

# Handling the NoneType error
if result is not None:
    print(f"The result of {numerator}/{denominator} is: {result}")
else:
    print("Cannot calculate ratio. Denominator is zero or invalid.")


# Streamlit app
def main():
    st.title('Image Recognition App')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        if st.button('Predict'):
            with st.spinner('Predicting...'):
                prediction, confidence = predict(image)
                st.success(f'Prediction: {prediction}, Confidence: {confidence:.2f}')

if __name__ == '__main__':
    main()
