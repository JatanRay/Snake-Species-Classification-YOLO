from ultralytics import YOLO
import streamlit as st
from PIL import Image
import cv2


# Load trained YOLO model
model = YOLO("best.pt")


# App title
st.title("🐍 Snake Species Detection")
st.write("Upload a snake image to identify its species.")


# Upload image
uploaded_file = st.file_uploader(
    "Choose a snake image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(uploaded_file)

    # Show uploaded image
    st.image(image, caption="Uploaded Image")


    # Detect button
    if st.button("Detect Snake"):

        # YOLO prediction
        results = model(image, conf=0.3)

        result = results[0]

        # Convert image to OpenCV format
        result_image = result.orig_img.copy()


        # Draw detection boxes
        for box in result.boxes:

            # Box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Confidence
            confidence = float(box.conf[0])

            # Class ID
            class_id = int(box.cls[0])

            # Species name
            name = model.names[class_id]

            # Original name + confidence
            label = f"{name} {confidence:.2f}"


            # Draw bounding box
            cv2.rectangle(
                result_image,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )


            # Font settings
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2


            # Check text size
            (text_width, text_height), _ = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness
            )


            # Automatically make text smaller
            # if it is wider than the snake box
            while text_width > (x2 - x1 - 10) and font_scale > 0.2:

                font_scale -= 0.05

                (text_width, text_height), _ = cv2.getTextSize(
                    label,
                    font,
                    font_scale,
                    thickness
                )


            # Label background inside the box
            cv2.rectangle(
                result_image,
                (x1, y1),
                (x1 + text_width + 8, y1 + text_height + 10),
                (255, 0, 0),
                -1
            )


            # Write complete label
            cv2.putText(
                result_image,
                label,
                (x1 + 4, y1 + text_height + 4),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )


        # Convert BGR to RGB
        result_image = cv2.cvtColor(
            result_image,
            cv2.COLOR_BGR2RGB
        )


        # Show final result
        st.image(
            result_image,
            caption="Detection Result"
        )


        st.success("Detection completed!")