import cv2
import pytesseract
import re
from datetime import datetime, date
import numpy as np

class ExpirationDateDetector:
    def __init__(self):
        self.now = date.today()
        
        self.dlc_keywords = [
            "dlc", "ed", "cad", "exp", "expiration", "expire",
            "best before", "use by", "sell by", "à consommer avant",
            "date limite"
        ]
        
        self.bad_keywords = [
            "prod", "production", "fabrication", "lot", "batch"
        ]
        
        # Multiple date patterns
        self.date_patterns = [
            r"\b(\d{2})[/\-\.](\d{2})[/\-\.](\d{4})\b",  # DD/MM/YYYY or DD-MM-YYYY
            r"\b(\d{2})[/\-\.](\d{2})[/\-\.](\d{2})\b",   # DD/MM/YY
            r"\b(\d{4})[/\-\.](\d{2})[/\-\.](\d{2})\b",   # YYYY/MM/DD
        ]
    
    def preprocess_image(self, image, method="adaptive"):
        """Apply preprocessing to improve OCR accuracy"""
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if method == "adaptive":
            # Adaptive thresholding - works well with varying lighting
            processed = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        elif method == "otsu":
            # Otsu's thresholding - automatic threshold calculation
            _, processed = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif method == "contrast":
            # Increase contrast
            processed = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        else:
            processed = gray
        
        # Denoise
        processed = cv2.fastNlMeansDenoising(processed, None, 10, 7, 21)
        
        # Optional: slight dilation to make text bolder
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.dilate(processed, kernel, iterations=1)
        
        return processed
    
    def extract_text_multiple_methods(self, image):
        """Try multiple preprocessing methods and OCR configs"""
        results = []
        
        methods = ["adaptive", "otsu", "contrast", None]
        
        for method in methods:
            if method:
                processed = self.preprocess_image(image, method)
            else:
                processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Try different PSM modes (Page Segmentation Modes)
            psm_modes = [3, 6, 11]  # 3=auto, 6=single block, 11=sparse text
            
            for psm in psm_modes:
                custom_config = f'--oem 3 --psm {psm}'
                text = pytesseract.image_to_string(processed, config=custom_config)
                results.append({
                    'text': text,
                    'method': method,
                    'psm': psm,
                    'processed': processed
                })
        
        return results
    
    def parse_date(self, date_str):
        """Try to parse date with multiple formats"""
        # Clean the string
        date_str = date_str.strip()
        
        formats = [
            "%d/%m/%Y",
            "%d-%m-%Y",
            "%d.%m.%Y",
            "%d/%m/%y",
            "%d-%m-%y",
            "%Y/%m/%d",
            "%Y-%m-%d"
        ]
        
        for fmt in formats:
            try:
                parsed = datetime.strptime(date_str, fmt).date()
                # Handle 2-digit years
                if parsed.year < 100:
                    parsed = parsed.replace(year=parsed.year + 2000)
                return parsed
            except ValueError:
                continue
        
        return None
    
    def extract_dates_from_text(self, text):
        """Extract all potential dates from text"""
        dates_found = []
        
        for pattern in self.date_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                date_str = match.group()
                # Normalize separators
                date_str = re.sub(r'[/\-\.]', '/', date_str)
                dates_found.append(date_str)
        
        return dates_found
    
    def find_expiration_date(self, image_path, show_preprocessing=False):
        """Main method to find expiration date"""
        
        image = cv2.imread("dateTest2.png")
        if image is None:
            print("Error: Image not found")
            return None
        
        print("Analyzing image with multiple OCR methods...")
        
        results = self.extract_text_multiple_methods(image)
        
        best_result = None
        best_confidence = 0
        
        for i, result in enumerate(results):
            text = result['text']
            lines = text.split("\n")
            
            for line in lines:
                lower_line = line.lower()
                
                # Skip production dates
                if any(bad in lower_line for bad in self.bad_keywords):
                    continue
                
                # Check if line contains DLC keywords
                has_keyword = any(word in lower_line for word in self.dlc_keywords)
                
                dates = self.extract_dates_from_text(line)
                
                for date_str in dates:
                    parsed_date = self.parse_date(date_str)
                    
                    if parsed_date:
                        # Calculate confidence score
                        confidence = 0
                        if has_keyword:
                            confidence += 50
                        if parsed_date > self.now:
                            confidence += 30
                        if parsed_date.year >= 2024 and parsed_date.year <= 2030:
                            confidence += 20
                        
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_result = {
                                'date': parsed_date,
                                'raw_text': line,
                                'date_str': date_str,
                                'confidence': confidence,
                                'method': result['method'],
                                'psm': result['psm'],
                                'processed_image': result['processed']
                            }
        
        if best_result:
            days_remaining = (best_result['date'] - self.now).days
            
            print(f"\n✓ Expiration date found: {best_result['date'].strftime('%d/%m/%Y')}")
            print(f"  Raw text: {best_result['raw_text'].strip()}")
            print(f"  Confidence: {best_result['confidence']}%")
            print(f"  Days remaining: {days_remaining}")
            print(f"  Best method: {best_result['method']} (PSM {best_result['psm']})")
            
            if days_remaining < 0:
                print(f"  ⚠️  EXPIRED {abs(days_remaining)} days ago!")
            elif days_remaining <= 3:
                print(f"  ⚠️  URGENT: Expires soon!")
            elif days_remaining <= 7:
                print(f"  ⚡ Use within a week")
            
            if show_preprocessing:
                cv2.imshow("Original", image)
                cv2.imshow("Best Preprocessing", best_result['processed_image'])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            return best_result
        else:
            print("\n✗ No expiration date found")
            return None


# Usage
if __name__ == "__main__":
    detector = ExpirationDateDetector()
    
    # Test with your image
    result = detector.find_expiration_date("dateTest2.png", show_preprocessing=True)
    
    # Test with multiple images if you have them
    # test_images = ["dateTest1.png", "dateTest2.png", "dateTest3.png"]
    # for img in test_images:
    #     print(f"\n{'='*50}")
    #     print(f"Testing: {img}")
    #     print('='*50)
    #     detector.find_expiration_date(img)