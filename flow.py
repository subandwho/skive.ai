import os
from pdf2image import convert_from_path
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from transformers import pipeline
from transformers import pipeline
import langgraph as lg

def get_pdf_paths(directory):
    pdf_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_paths.append(os.path.join(root, file))
    return pdf_paths

def extract_images_from_pdf(pdf_path):
    return convert_from_path(pdf_path)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def analyze_image(image):
    inputs = clip_processor(text=["aadhar card", "passport", "driving license", "identity document"], images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return probs[0].tolist()

extractor = pipeline("ner", model="dslim/bert-base-NER")
def extract_information(text, doc_type):
    ner_results = extractor(text)
    extracted_info = {
        "name": None,
        "date_of_birth": None,
        "address": None,
        "document_type": doc_type,
        "document_number": None
    }
    
    # (Implement extraction logic based on doc_type)
    # ...

    return extracted_info



reviewer_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

def reviewer_check(extracted_info, required_attributes):
    review_text = f"Extracted information: {extracted_info}"
    result = reviewer_model(review_text)[0]
    is_complete = result['label'] == 'POSITIVE' and result['score'] > 0.9
    
    missing_attributes = [attr for attr in required_attributes if not extracted_info.get(attr)]
    
    return is_complete, missing_attributes


def process_document(pdf_path, required_attributes):
    images = extract_images_from_pdf(pdf_path)
    extracted_info = {}
    
    for image in images:
        probs = analyze_image(image)
        doc_type = ["aadhar card", "passport", "driving license", "identity document"][probs.index(max(probs))]
        
        # Perform OCR (you'll need to implement this)
        text = perform_ocr(image)
        
        info = extract_information(text, doc_type)
        extracted_info.update(info)
        
        is_complete, missing_attributes = reviewer_check(extracted_info, required_attributes)
        
        if is_complete:
            return {"status": "success", "info": extracted_info}
    
    return {"status": "incomplete", "info": extracted_info, "missing": missing_attributes}

def create_workflow(pdf_directory, required_attributes):
    workflow = lg.Workflow()
    
    pdf_paths = get_pdf_paths(pdf_directory)
    
    for pdf_path in pdf_paths:
        workflow.add_task(
            lambda path=pdf_path: process_document(path, required_attributes),
            dependencies=[]
        )
    
    return workflow

# Usage
pdf_directory = "path/to/pdf/directory"
required_attributes = ["name", "date_of_birth", "address", "document_type", "document_number"]

workflow = create_workflow(pdf_directory, required_attributes)
results = workflow.run()

for result in results:
    if result['status'] == 'success':
        print(f"Document processed successfully: {result['info']}")
    else:
        print(f"Incomplete document. Missing: {result['missing']}")
