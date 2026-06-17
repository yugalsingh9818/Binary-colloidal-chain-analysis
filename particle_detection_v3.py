import cv2
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# LOAD IMAGE
# =====================================

img = cv2.imread(
    r"C:\Users\yugal\OneDrive\Desktop\Binary chain (45).tif"
)

if img is None:
    print("Image not found!")
    exit()

print("Image loaded successfully")
print("Shape:", img.shape)

# =====================================
# GRAYSCALE + BLUR
# =====================================

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray_blur = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)

# =====================================
# DETECT CIRCLES
# =====================================

circles = cv2.HoughCircles(
    gray_blur,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=18,
    param1=60,
    param2=18,
    minRadius=4,
    maxRadius=40
)

output = img.copy()

small_count = 0
large_count = 0

# New dataset
particle_data = []

# =====================================
# ANALYZE PARTICLES
# =====================================

if circles is not None:

    circles = np.round(
        circles[0, :]
    ).astype("int")

    print("\nParticles detected:",
          len(circles))

    print("-" * 40)

    for i, (x, y, r) in enumerate(circles):

        # Classification

        if r < 14:
            particle_type = "Small"
            label = "S"
            small_count += 1

        else:
            particle_type = "Large"
            label = "L"
            large_count += 1

        # Save data
        particle_data.append(
            [x, y, r, label]
        )

        print(
            f"Particle {i+1}: "
            f"X={x}, "
            f"Y={y}, "
            f"Radius={r}, "
            f"Type={particle_type}"
        )

        # Draw circle
        cv2.circle(
            output,
            (x, y),
            r,
            (0, 255, 0),
            2
        )

        # Draw center
        cv2.circle(
            output,
            (x, y),
            2,
            (0, 0, 255),
            3
        )

        # Draw label
        cv2.putText(
            output,
            label,
            (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

else:
    print("No particles detected")

# =====================================
# SUMMARY
# =====================================

print("\n" + "=" * 40)
print("SUMMARY")
print("=" * 40)

print("Small particles =", small_count)
print("Large particles =", large_count)

# =====================================
# PARTICLE DATASET
# =====================================

print("\nParticle Dataset")
print("=" * 40)

for row in particle_data:
    print(row)

# =====================================
# DISPLAY IMAGE
# =====================================

plt.figure(figsize=(10, 10))

plt.imshow(
    cv2.cvtColor(
        output,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    "Particle Detection and Classification"
)

plt.axis("off")

plt.show()