from PIL import Image, ImageChops

# Open the two images (ensure they have the same dimensions and mode)
img1 = Image.open("datasets/A4paper_3_y0_x0.bmp")
img2 = Image.open("datasets/A4paper_3_y0_x1024.bmp")

# Optional: convert to 'RGB' mode to avoid issues with alpha channels
# img1 = img1.convert('RGB')
# img2 = img2.convert('RGB')

# Calculate the difference
diff = ImageChops.difference(img1, img2)

# Display the difference image
diff.save("output.png")

# Check if images are identical
# If getbbox() returns None, the images are identical (the difference image is all black/transparent)
if diff.getbbox() is None:
    print("Images are identical!")
else:
    print("Images are different. Bounding box of differences:", diff.getbbox())
