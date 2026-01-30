from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np
import io

from convolution import apply_convolution

app = FastAPI(title="Image Kernel Convolution API")
KERNELS = {
    "sharpen": np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ]),
    "blur": (1/9) * np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]),
    "edge": np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ]),
    "emboss": np.array([
        [-2, -1, 0],
        [-1,  1, 1],
        [ 0,  1, 2]
    ])
}
@app.get("/kernels/list")
def list_kernels():
    return {"available_kernels": list(KERNELS.keys())}
@app.post("/transform/convolution")
async def transform_image(
    kernel_name: str,
    file: UploadFile = File(...)
):
    # Validate kernel
    if kernel_name not in KERNELS:
        raise HTTPException(status_code=400, detail="Invalid kernel name")

    # Read uploaded image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image_array = np.array(image)

    # Apply convolution
    kernel = KERNELS[kernel_name]
    output_array = apply_convolution(image_array, kernel)
    output_array = np.clip(output_array, 0, 255)

    # Convert back to image
    output_image = Image.fromarray(output_array.astype(np.uint8))
    output_image.save("output.jpg")

    return {
        "message": "Image transformed successfully",
        "kernel_used": kernel_name
    }

