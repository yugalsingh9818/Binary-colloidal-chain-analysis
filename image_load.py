import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread(
    r"C:\Users\yugal\OneDrive\Desktop\Binary chain (45).tif"
)

if img is None:
    print("Image not found!")
    exit()

print("Image loaded successfully")
print("Shape:", img.shape)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold
_, thresh = cv2.threshold(
    gray,
    180,
    255,
    cv2.THRESH_BINARY
)

print("White pixels:", np.sum(thresh == 255))
print("Black pixels:", np.sum(thresh == 0))

# Find contours
contours, _ = cv2.findContours(
    255 - thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print("Number of detected objects:", len(contours))

# Draw contours
output = img.copy()

for i, cnt in enumerate(contours):

    area = cv2.contourArea(cnt)

    if area > 5:
        (x, y), radius = cv2.minEnclosingCircle(cnt)

        print(
            f"Object {i+1}: "
            f"Area = {area:.2f}, "
            f"Radius = {radius:.2f}"
        )

        cv2.circle(
            output,
            (int(x), int(y)),
            int(radius),
            (0, 255, 0),
            2
        )

# Display results
plt.figure(figsize=(18,6))

plt.subplot(1,3,1)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(thresh, cmap="gray")
plt.title("Threshold")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Detected Objects")
plt.axis("off")

plt.tight_layout()
plt.show()