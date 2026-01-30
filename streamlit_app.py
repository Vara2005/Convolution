import streamlit as st
import numpy as np
from PIL import Image

from convolution import apply_convolution

st.set_page_config(page_title="Image Kernel Convolution Engine", layout="wide")

st.title("Image Kernel Convolution Engine")

st.write(
    "Upload an image and apply predefined or custom convolution kernels "
    "to see the transformed output."
)

# -----------------------------------
# Image Upload
# -----------------------------------
uploaded_file = st.file_uploader(
    "Upload an image (JPG, PNG, JPEG)",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------------
# Predefined Kernels
# -----------------------------------
PREDEFINED_KERNELS = {
    "Sharpen": np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ]),
    "Blur": (1 / 9) * np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]),
    "Edge Detection": np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ]),
    "Emboss": np.array([
        [-2, -1, 0],
        [-1,  1, 1],
        [0,   1, 2]
    ])
}

# -----------------------------------
# Kernel Selection
# -----------------------------------
kernel_mode = st.radio(
    "Kernel Mode",
    ["Predefined Kernel", "Custom Kernel"]
)

kernel_to_apply = None

if kernel_mode == "Predefined Kernel":
    kernel_name = st.selectbox(
        "Select a predefined kernel",
        list(PREDEFINED_KERNELS.keys())
    )
    kernel_to_apply = PREDEFINED_KERNELS[kernel_name]

else:
    st.subheader("Custom Kernel Input (3×3 or 5×5)")
    st.write("Enter values row-wise, separated by spaces.")
    st.write("Example:")
    st.code("0 -1 0\n-1 5 -1\n0 -1 0")

    custom_kernel_text = st.text_area(
        "Custom Kernel Matrix",
        height=120
    )

    if custom_kernel_text.strip():
        try:
            rows = custom_kernel_text.strip().split("\n")
            kernel = np.array(
                [[float(value) for value in row.split()] for row in rows]
            )

            if kernel.shape in [(3, 3), (5, 5)]:
                kernel_to_apply = kernel
            else:
                st.error("❌ Kernel must be either 3×3 or 5×5.")

        except ValueError:
            st.error("❌ Invalid input. Please enter numeric values only.")

# -----------------------------------
# Apply Convolution & Display Results
# -----------------------------------
if uploaded_file is not None and kernel_to_apply is not None:
    image = Image.open(uploaded_file).convert("L")
    image_array = np.array(image)

    output_array = apply_convolution(image_array, kernel_to_apply)
    output_array = np.clip(output_array, 0, 255)
    output_image = Image.fromarray(output_array.astype(np.uint8))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original Image")
        st.image(image, width="stretch")

    with col2:
        st.subheader("Kernel Matrix")
        st.table(kernel_to_apply)

    with col3:
        st.subheader("Transformed Image")
        st.image(output_image, width="stretch")
    st.success("✅ Image transformed successfully!")
    
