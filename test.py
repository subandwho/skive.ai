import os
from typing import List, Dict, Any
import pandas as pd
import fitz  # PyMuPDF
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel, pipeline, TrOCRProcessor, VisionEncoderDecoderModel
import re
from langgraph import StateGraph, END, MemorySaver

# Initialize models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

ocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
ocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

ner_extractor = pipeline("ner", model="dslim/bert-base-NER")

reviewer_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

def extract_images_from_pdf(pdf_path: str) -> List[Image.Image]:
    images = []
    pdf_document = fitz.open(pdf_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

def analyze_image(image: Image.Image) -> str:
    inputs = clip_processor(text=["aadhar card", "passport", "driving license", "identity document"], images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    document_types = ["aadhar card", "passport", "driving license", "identity document"]
    return document_types[probs.argmax()]

def perform_ocr(image: Image.Image) -> str:
    pixel_values = ocr_processor(images=image, return_tensors="pt").pixel_values
    generated_ids = ocr_model.generate(pixel_values)
    generated_text = ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def extract_information(text: str, doc_type: str) -> Dict[str, Any]:
    ner_results = ner_extractor(text)
    extracted_info = {
        "name": None,
        "date_of_birth": None,
        "address": None,
        "document_type": doc_type,
        "document_number": None
    }
    
    current_entity = ""
    for item in ner_results:
        if item['entity'].endswith('PER') and not extracted_info['name']:
            current_entity += item['word'].replace('##', '')
            if item['entity'].startswith('I-'):
                extracted_info['name'] = current_entity.strip()
                current_entity = ""
        elif item['entity'].endswith('LOC') and not extracted_info['address']:
            current_entity += item['word'].replace('##', '') + " "
            if item['entity'].startswith('I-'):
                extracted_info['address'] = current_entity.strip()
                current_entity = ""
    
    # Extract date of birth
    dob_pattern = r'\b(?:0[1-9]|[12]\d|3[01])[-/](?:0[1-9]|1[0-2])[-/](?:19|20)\d{2}\b'
    dob_match = re.search(dob_pattern, text)
    if dob_match:
        extracted_info['date_of_birth'] = dob_match.group()
    
    # Extract document number based on document type
    if doc_type == "aadhar card":
        aadhar_pattern = r'\b\d{4}\s\d{4}\s\d{4}\b'
        aadhar_match = re.search(aadhar_pattern, text)
        if aadhar_match:
            extracted_info['document_number'] = aadhar_match.group()
    elif doc_type == "passport":
        passport_pattern = r'[A-Z]{1}[0-9]{7}'
        passport_match = re.search(passport_pattern, text)
        if passport_match:
            extracted_info['document_number'] = passport_match.group()
    elif doc_type == "driving license":
        dl_pattern = r'[A-Z]{2}[0-9]{2} [0-9]{11}'
        dl_match = re.search(dl_pattern, text)
        if dl_match:
            extracted_info['document_number'] = dl_match.group()
    
    return extracted_info

def process_document(state: Dict[str, Any]) -> Dict[str, Any]:
    pdf_path = state['pdf_path']
    images = extract_images_from_pdf(pdf_path)
    extracted_data = []

    for image in images:
        doc_type = analyze_image(image)
        text = perform_ocr(image)
        info = extract_information(text, doc_type)
        extracted_data.append(info)
    
    # Convert extracted data to DataFrame
    df = pd.DataFrame(extracted_data)
    state['extracted_info'] = df
    state['status'] = "success"
    return state

def create_workflow():
    # Define state schema
    state_schema = {
        "pdf_path": str,
        "extracted_info": pd.DataFrame,
        "status": str,
    }

    workflow = LangGraph(input_schema=state_schema, output_schema=state_schema)

    workflow.add_node("process", process_document)

    def check_completion(state: Dict[str, Any]):
        if state['status'] == 'success':
            return END
        else:
            return "process"

    workflow.add_node("check_completion", check_completion)
    workflow.add_edge("process", "check_completion")
    workflow.add_edge("check_completion", "process")
    workflow.set_entry_point("process")

    return workflow.compile()

# Main execution
if __name__ == "__main__":
    pdf_path = "Docs/BGQPK4512E_31012024103628.pdf"  # Replace with the path to your PDF file
    workflow = create_workflow()

    initial_state = {
        'pdf_path': pdf_path,
        'extracted_info': None,
        'status': 'incomplete',
    }
    
    final_state = workflow.invoke(initial_state)
    
    if final_state['status'] == 'success':
        print(f"Document processed successfully.")
        df = final_state['extracted_info']
        print(df)
    else:
        print("Document processing failed.")
