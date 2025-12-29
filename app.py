#reconstructing streamlit app 
import torch, uuid
import json, time, os, sys, time, requests
import pandas as pd
from diffusers import StableDiffusionPipeline
from PIL import Image

#cpu server
import streamlit as st
import json, time, os, sys
import pandas as pd

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def text_to_image(text):
  image = pipe(text).images[0]
  return image

  #save image locally
  image_path = uuid.uuid4() +  ".png"
  image.save(image_path)
  return image_path


#streamlit UI
user_input = st.text_input("enter your prompt text: ")

if st.button("process"):
  if user_input:
    
    img_path = text_to_image(user_input)
    st.success("messge sent! waiting for reponse.......")
   
    if image_path:
      st.image(img_path, caption = "Generated image", use_column_width=True)
    else:
      st.error("no image generated within the timeout period...")
  
  else:
      st.warning("please enter a prompt before sending")
          
