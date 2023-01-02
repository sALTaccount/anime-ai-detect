import gradio as gr
from transformers import pipeline

detection_pipeline = pipeline("image-classification", "saltacc/anime-ai-detect")


def detect(img):
    output = detection_pipeline(img, top_k=2)
    final = {}
    for d in output:
        final[d["label"]] = d["score"]
    return final


iface = gr.Interface(fn=detect, inputs=gr.Image(type="pil"), outputs=gr.Label(label="result"))
iface.launch()
