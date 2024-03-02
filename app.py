import streamlit as st
from api.model import PredictionPipeline

def main():
    st.title("Potato Leaf classification")
    st.write("Upload images")

    file = st.file_uploader("Upload file", type=['jpg', 'png', 'jpeg'])

    if file:
        # Display the uploaded image
        st.image(file, caption='Uploaded Image.', use_column_width="auto",width=100)

        # Resize the image to [224, 224, 3]
        prediction_output=PredictionPipeline(file)
        output=prediction_output.predict()
        st.write(f"Prediction: {output}")

    else:
        st.text("You have not uploaded an image yet")    


if __name__=="__main__":
    main()

    