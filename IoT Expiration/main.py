import cv2
import pytesseract
import re
from datetime import datetime, date

now = date.today()


image = cv2.imread("maruja_bar.jpeg")
if image is None:
    print("Error: Image not found")
    exit()

    # ---- 200% zoom into the center of the image ----
h, w = image.shape[:2]

# Center coordinates
cx, cy = w // 2, h // 2

# Crop size (half width & height → 2x zoom)
crop_w, crop_h = w // 2, h // 2

x1 = cx - crop_w // 2
y1 = cy - crop_h // 2
x2 = cx + crop_w // 2
y2 = cy + crop_h // 2

# Crop center
center_crop = image[y1:y2, x1:x2]

# Resize back to original size
image = cv2.resize(center_crop, (w, h), interpolation=cv2.INTER_CUBIC)
# ----------------------------------------------


cv2.imshow("Maruja", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
text = pytesseract.image_to_string(image)

lines = text.split("\n")

dlc_keywords = [
    "dlc",
    "ed",
    "cad",
    "exp",
    "expiration",
    "d"
]

bad_keywords = [
    "prod",
    "p",
    "l&pd",
]

date_pattern = r"\d{2}/\d{2}/\d{4}"

for line in lines:
    lower_line = line.lower()

    if any(bad in lower_line for bad in bad_keywords):
        continue

    if any(word in lower_line for word in dlc_keywords):
        match = re.search(date_pattern, line)
        if match:
            print("DLC:", match.group())

            raw_date = match.group()
            clean_date = re.sub(r"[^\d/]", "", raw_date)

            dlcDate = datetime.strptime(clean_date, "%d/%m/%Y").date()
            dateRemaining = (dlcDate - now).days

            found = True

            print("Remaining time before expiration:", dateRemaining,"days")
if not found:
    print("I didn't obtain the data")



