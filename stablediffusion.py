import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Initialize the pipeline
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)  
pipe = pipe.to("cuda")

# Function to create an image grid
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

# Streamlit app
def main():
    st.title("Hydro Power Plant Image Generation")

    # Input text prompt
    prompt = st.text_area("Enter your construction steps:", height=200)

    # Number of rows and columns for the image grid
    num_cols = st.slider("Number of Columns", min_value=1, max_value=10, value=5)
    num_rows = st.slider("Number of Rows", min_value=1, max_value=10, value=5)

    if st.button("Generate Images"):
        # Split the prompt into equal parts based on the number of rows
        prompt_list = [prompt[i:i + len(prompt) // num_rows] for i in range(0, len(prompt), len(prompt) // num_rows)]

        # Generate images for each part of the prompt
        all_images = []
        for part in prompt_list:
            images = pipe(part).images
            all_images.extend(images)

        # Display the image grid using Streamlit
        grid = image_grid(all_images, rows=num_rows, cols=num_cols)
        st.image(grid, caption="Generated Images", use_column_width=True)

if __name__ == "__main__":
    main()
