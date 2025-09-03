from PIL import Image

im = Image.open("./images/patch.webp")

im = im.convert("RGBA")

im.save("./images/patch.png")