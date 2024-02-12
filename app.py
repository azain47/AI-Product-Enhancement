# for ui
import streamlit as st
from typing import Optional
# for app
from ultralytics import YOLO
from diffusers import DPMSolverMultistepScheduler
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionInpaintPipeline
import torch

from PIL import Image
OUTPUT_IMAGE_KEY = "output_img"
LOADED_IMAGE_KEY = "loaded_image"

def get_image(key: str) -> Optional[Image.Image]:
    if key in st.session_state:
        return st.session_state[key]
    return None


def set_image(key: str, img: Image.Image):
    st.session_state[key] = img

valid_objects = ['shoe', 'sneaker', 'bottle', 'cup', 'sandal', 'perfume', 'toy', 
                'sunglasses', 'car', 'water bottle', 'chair', 'office chair',
                'can', 'cap', 'hat', 'couch', 'wristwatch', 'watch', 'glass', 'bag', 'handbag',
                'baggage', 'suitcase', 'headphones', 'jar', 'vase']

@st.cache_resource()
def load_models():

    scheduler = DPMSolverMultistepScheduler.from_pretrained("stabilityai/stable-diffusion-2-1",
                                                subfolder="scheduler", use_karras_sigmas=True,
                                                algorithm_type="sde-dpmsolver++")

    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained('stabilityai/stable-diffusion-2-1',
                    torch_dtype=torch.float16, use_safetensors=True, variant="fp16", scheduler = scheduler)

    # img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained('stabilityai/stable-diffusion-2-1',
    #                 torch_dtype=torch.float16, use_safetensors=True, variant="fp16", scheduler = scheduler)
    # img2img_pipe.unet = torch.compile(img2img_pipe.unet, mode="reduce-overhead", fullgraph=True)
    # img2img_pipe.to("cuda")
    # inpaint_pipe.to("cuda")

    inpaint_pipe.enable_model_cpu_offload()
    # img2img_pipe.enable_model_cpu_offload()

    segmentation_model = YOLO('yolov8x-seg.pt')
    return inpaint_pipe, segmentation_model

def image_uploading():
    image = st.file_uploader("Image", ["jpg", "png"])
    if image:
        image = Image.open(image)
        print(f"loaded input image of size ({image.width}, {image.height})")
        return image

    return get_image(LOADED_IMAGE_KEY)

def segmentation(image,segmentation_model):
    with(st.spinner("Segmentation in progress...")):
        
        segmentation_results = segmentation_model.predict(image,retina_masks = True, max_det = 3)

        mask = None
        inverted_mask = None
        masked_image = None

        for r in segmentation_results:
            # print(r)
            # print(r.boxes.conf)
            # print(r.masks.data.cpu().numpy()*255)
            boxes = r.boxes
            masks = []
            # predicted classes and confidences are in boxes class, hence enumerating
            for i, box_idx in enumerate(boxes.cls.cpu().numpy()):
                
                # check the index with the names dict of model, compare with valid objects and 
                # threshold confidence. if object is valid, add the mask data and the name to masks array.
                # print(r.names[box_idx])
                # print(boxes.conf.cpu().numpy()[i])
                if(r.names[box_idx] in valid_objects and boxes.conf.cpu().numpy()[i] > 0.60):
                    masks.append((r.masks.data[i].cpu().numpy(),r.names[box_idx]))
            
            if(len(masks) == 0):
                st.markdown('# No matching object found, please choose another photo, or click from a different angle.')
            else:
                st.write("The object(s) that are found in the image:")
                objs = {}
                for i, obj in enumerate(masks):
                    objs[obj[1]] = i
                    st.write(obj[1])
                idx = st.selectbox(label = 'Select an Object :', options = objs.keys(), index = None, key = "obj-selector")
                
                if idx:
                    mask = Image.fromarray(masks[objs[idx]][0]*255)
                    inverted_mask = Image.fromarray(255 - masks[objs[idx]][0]*255)
                    maxsize = max(image.size[0],image.size[1])
                    
                    # resizing algo
                    if(maxsize<1280):
                        size = image.size
                    else:
                        if(image.size[0] > image.size[1]):
                            ratio = image.size[0] / image.size[1]
                            size = (int((1280/image.size[0]) * image.size[0]), int((1280/image.size[0]) * image.size[0]/ratio))
                        else:
                            ratio = image.size[1] / image.size[0]
                            size = (int((1280/image.size[1]) * image.size[1]/ratio), int((1280/image.size[1]) * image.size[1])) 

                    # image = image.resize(size).convert('RGB')
                    name = masks[objs[idx]][1]
                    mask = mask.resize(size).convert('RGB')
                    inverted_mask = inverted_mask.resize(size).convert('RGB')
                    return size,inverted_mask,mask, name

def main():
    
    st.set_page_config(layout="wide")
    st.title("AI Product Enhancer and Filter")
    uploaded_img, generated_mask, output_img = st.columns(3)
    inpaint_pipe, segmentation_model = load_models()
    segment_key = "segmented-image"
    output_key = "output-img"
    image = None
    mask = None
    background = None
    obj_name = None

    with uploaded_img:
        st.markdown("## Upload Image")       
        # img_path = Image.open('./2.jpg')
        image = image_uploading()
        if image:
            st.image(image)
        
    with generated_mask:
        st.markdown("## Mask ")
         
        if image:
            # image = st.session_state[uploader_key]
            size, inverted_mask, mask, obj_name = segmentation(image, segmentation_model)
            st.image(mask)
            st.session_state[segment_key] = mask

    with output_img:
        st.markdown("## Output Image")

        if(image and mask):
            
            prompt_fg = f'High quality product photoshoot of {obj_name}, professional lighting, 8k, uhd'
            prompt_bg = f'high quality {obj_name} photoshoot, realistic lighting, realism, finely detailed, ultra quality, 8k uhd'
            # neg_prompt = f'({obj_name}) in background,low quality, cartoon, casual, unprofessional'
            neg_prompt = "(worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), cropped, text, jpeg artifacts, signature, watermark, username, sketch, cartoon, drawing, anime, duplicate, blurry, semi-realistic, out of frame, ugly, deformed, EasyNegative"
            if(st.checkbox("Custom prompt", key = "custom-prompt")):
                final_prompt = prompt_bg + st.text_input('Enter your prompt:') 
            else:
                final_prompt = prompt_bg
            
            # foreground = img2img_pipe(prompt = prompt_fg,negative_prompt=neg_prompt, image=img_path,
            #             num_inference_steps=100, strength=0.2, seed=seed, guidance_scale=4).images[0]
            # foreground.resize(size).show()
            width = st.slider(
                    "Width",
                    min_value=64,
                    max_value=1600,
                    step=8,
                    value=size[0],
                )

            height = st.slider(
                    "Height",
                    min_value=64,
                    max_value=1600,
                    step=8,
                    value=size[1],     
                )
            strength = st.slider("Strength for background image guiding :", 0.0,1.0,0.35,step=0.01, key = "strength")
            steps = st.slider("Number of steps for generation. Low steps = low quality. High steps =/= high quality", 5, 300, 50, step = 1, key = "steps")
            guidance = st.slider("Guidance scale, higher number is more creative, lower number means it'll stick to prompt more.", 1.0 , 13.0, 6.0, 0.3, key = "guidance-scale")
            
            if st.button("Generate image", key = "generate-btn"):
                seed = torch.Generator("cuda").seed()
                with st.spinner("Generating image..."):
                    background = inpaint_pipe(prompt = final_prompt, negative_prompt=neg_prompt, image = image,
                        mask_image=inverted_mask, num_inference_steps=steps, strength=1.0-strength, seed=seed ,guidance_scale=guidance,
                        height = height, width = width).images[0]
                    
                    if background:
                        output = (background.resize(size))
                        st.image(output)
                        from io import BytesIO
                        buf = BytesIO()
                        import datetime
                        output.save(buf, format = "PNG")
                        img_bytes = buf.getvalue()
                        st.download_button("Download Output",img_bytes,file_name="output"+f"{datetime.datetime.now()}.png")
                        st.session_state[output_key] = output
        # background.resize(size).save('op1.jpg')

if __name__ == "__main__":
    main()