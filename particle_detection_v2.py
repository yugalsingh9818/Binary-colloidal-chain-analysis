import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load cropped chain image
img = cv2.imread(
    r"C:\Users\yugal\OneDrive\Desktop\Binary chain (45).tif"
)

if img is None:
    print("Image not found!")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur to reduce noise
gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect circles
circles = cv2.HoughCircles(
    gray_blur,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=15,
    param1=50,
    param2=15,
    minRadius=3,
    maxRadius=40
)

output = img.copy()

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")

    print("Particles detected:", len(circles))

    for i, (x, y, r) in enumerate(circles):

        print(f"Particle {i+1}: Radius = {r}")

        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 2, (0, 0, 255), 3)

else:
    print("No particles detected")

# Show result
plt.figure(figsize=(8,8))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Detected Particles")
plt.axis("off")
plt.show()