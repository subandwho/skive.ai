import re
from PIL import Image
import pytesseract
import fitz  # PyMuPDF

def is_physical_aadhaar(image_path):
    """
    Detect if the given image is of a physical Aadhaar card.
    """
    print(f"Checking if the image at {image_path} is a physical Aadhaar card...")
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    is_physical = "Government of India" in text and "DOB:" in text and "Aadhaar" in text
    print(f"Extracted Text: {text[:100]}...")  # Print first 100 characters of extracted text
    print(f"Is physical Aadhaar: {is_physical}")
    return is_physical

def extract_text_from_pdf(pdf_path):
    """
    Extract text from all pages of a given PDF.
    """
    print(f"Extracting text from PDF at {pdf_path}...")
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        print(f"Processing page {page_num + 1}...")
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        page_text = pytesseract.image_to_string(image)
        text += page_text
        print(f"Extracted Text from Page {page_num + 1}: {page_text[:100]}...")  # Print first 100 characters of extracted text per page
    return text

def validate_kra_information(text):
    """
    Validate the extracted information from the KRA document.
    """
    print("Validating KRA information...")
    valid_poi = ["Driving License", "Passport", "PAN Card", "Aadhaar"]
    valid_poa = ["Aadhaar"]
    is_valid = True
    
    print("Checking Proof of Identity...")
    if "Proof of Identity" in text:
        poi = extract_poi(text)
        print(f"Extracted POI: {poi}")
        if not any(id_type in poi for id_type in valid_poi):
            is_valid = False
            print("Invalid POI detected.")
    
    print("Checking Proof of Address...")
    if "Proof of Address" in text:
        poa = extract_poa(text)
        print(f"Extracted POA: {poa}")
        if "Aadhaar" in poa and not is_digilocker_aadhaar(text):
            is_valid = False
            print("Invalid POA detected: Aadhaar card is not from DigiLocker.")
    
    print(f"Validation result: {is_valid}")
    return is_valid

def extract_poi(text):
    """
    Extract Proof of Identity (POI) details from the text.
    """
    poi_patterns = {
        "Driving License": r"Driving Licence",
        "Passport": r"Passport",
        "PAN Card": r"Permanent Account Number|PAN Card",
        "Aadhaar": r"UID|Aadhaar"
    }
    extracted_poi = []
    for key, pattern in poi_patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            extracted_poi.append(key)
    return extracted_poi

def extract_poa(text):
    """
    Extract Proof of Address (POA) details from the text.
    """
    poa_patterns = {
        "Aadhaar": r"UID|Aadhaar"
    }
    extracted_poa = []
    for key, pattern in poa_patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            extracted_poa.append(key)
    return extracted_poa

def is_digilocker_aadhaar(text):
    """
    Detect if the Aadhaar card mentioned in the text is from DigiLocker.
    """
    digilocker_indicators = [
        "Issued by DigiLocker",
        "DigiLocker"
    ]
    for indicator in digilocker_indicators:
        if indicator.lower() in text.lower():
            return True
    return False

if __name__ == "__main__":
    # Example usage for detecting physical Aadhaar card
    physical_aadhaar_path = r"C:\Users\USER\Documents\GitHub\skive.ai\Docs\physicalAdhaar.jpg"
    is_physical = is_physical_aadhaar(physical_aadhaar_path)
    print(f"Is physical Aadhaar: {is_physical}")

    # Example usage for extracting and validating KRA information
    kra_pdf_path = r"C:\Users\USER\Documents\GitHub\skive.ai\Docs\BGQPK4512E_12022024032132 (1).pdf"
    kra_text = extract_text_from_pdf(kra_pdf_path)
    is_valid_kra = validate_kra_information(kra_text)
    print(f"Is valid KRA: {is_valid_kra}")
