from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("Lightricks/LTX-Video")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt).images[0]

image.save("astronaut_in_jungle.png")