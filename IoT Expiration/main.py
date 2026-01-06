import cv2
import pytesseract
import re
from datetime import datetime, date

now = date.today()


image = cv2.imread("dateTest2.png")
if image is None:
    print("Error: Image not found")
    exit()

#cv2.imshow("Maruja", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

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

            print("Remaining time before expiration:", dateRemaining,"days") 


