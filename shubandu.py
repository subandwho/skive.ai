from pdf2image import convert_from_path
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.tools import BaseTool
from langchain.agents import Agent
from langchain.schema import HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
import operator
import pandas as pd
import os

# Initialize the model and tokenizer
model_id = "vikhyatk/moondream2"
revision = "2024-05-20"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, revision=revision)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

# Function to extract images from a PDF
def extract_images_from_pdf(pdf_path):
    return convert_from_path(pdf_path)

# Step 2: Define Tools
class AadhaarTool(BaseTool):
    name = "aadhaar_extraction"
    description = "Extracts address information from Aadhaar card images."

    def _call_(self, image_path: str) -> dict:
        image = Image.open(image_path)
        enc_image = model.encode_image(image)
        result = model.answer_question(enc_image, "Extract address from this Aadhaar card.", tokenizer)
        return {"address": result}

class PanTool(BaseTool):
    name = "pan_extraction"
    description = "Extracts identity information from PAN card images."

    def _call_(self, image_path: str) -> dict:
        image = Image.open(image_path)
        enc_image = model.encode_image(image)
        result = model.answer_question(enc_image, "Extract name and PAN number from this PAN card.", tokenizer)
        return {"identity_info": result}

class VoterIDTool(BaseTool):
    name = "voterid_extraction"
    description = "Extracts identity information from Voter ID images."

    def _call_(self, image_path: str) -> dict:
        image = Image.open(image_path)
        enc_image = model.encode_image(image)
        result = model.answer_question(enc_image, "Extract name and Voter ID from this Voter ID card.", tokenizer)
        return {"identity_info": result}

class AttributeTool(BaseTool):
    name = "attribute_extraction"
    description = "Extracts specified attributes from the document images based on the prompt."

    def _call_(self, image_path: str, attributes: list) -> dict:
        image = Image.open(image_path)
        enc_image = model.encode_image(image)
        results = {}
        for attr in attributes:
            result = model.answer_question(enc_image, f"Extract {attr} from this document.", tokenizer)
            results[attr] = result
        return results

# Step 3: Define Agents
class ProofOfAddressAgent(Agent):
    def _init_(self, tool):
        self.tool = tool

    def _call_(self, state):
        image_path = state["image_path"]
        address_info = self.tool(image_path)
        return {"address_info": address_info}

class ProofOfIdentityAgent(Agent):
    def _init_(self, tools):
        self.tools = tools

    def _call_(self, state):
        image_path = state["image_path"]
        for tool in self.tools:
            identity_info = tool(image_path)
            if identity_info:
                return {"identity_info": identity_info}
        return {"identity_info": None}

class AttributeExtractionAgent(Agent):
    def _init_(self, tool):
        self.tool = tool

    def _call_(self, state):
        image_path = state["image_path"]
        attributes = state["attributes"]
        extracted_attributes = self.tool(image_path, attributes)
        return {"extracted_attributes": extracted_attributes}

class ReviewerAgent(Agent):
    def _call_(self, state):
        address_info = state.get("address_info")
        identity_info = state.get("identity_info")
        extracted_attributes = state.get("extracted_attributes")

        if address_info and identity_info and extracted_attributes:
            data = {**address_info, **identity_info, **extracted_attributes}
            df = pd.DataFrame([data])
            return {"dataframe": df}
        else:
            return {"error": "Missing information"}

# Step 4: Integrate Agents into LangGraph
class AgentState(TypedDict):
    image_path: str
    attributes: list
    address_info: dict
    identity_info: dict
    extracted_attributes: dict

def call_proof_of_address(state):
    agent = ProofOfAddressAgent(AadhaarTool())
    return agent(state)

def call_proof_of_identity(state):
    agent = ProofOfIdentityAgent([PanTool(), VoterIDTool()])
    return agent(state)

def call_attribute_extraction(state):
    agent = AttributeExtractionAgent(AttributeTool())
    return agent(state)

def call_reviewer(state):
    agent = ReviewerAgent()
    return agent(state)
def run():
    workflow = StateGraph(AgentState)
    workflow.add_node("proof_of_address", call_proof_of_address)
    workflow.add_node("proof_of_identity", call_proof_of_identity)
    workflow.add_node("attribute_extraction", call_attribute_extraction)
    workflow.add_node("reviewer", call_reviewer)
    workflow.set_entry_point("proof_of_address")

    workflow.add_edge("proof_of_address", "proof_of_identity")
    workflow.add_edge("proof_of_identity", "attribute_extraction")
    workflow.add_edge("attribute_extraction", "reviewer")
    workflow.add_edge("reviewer", END)

    app = workflow.compile()

    # Step 5: Process Each Page of the Document
    pdf_path = r"Docs\BGQPK4512E_31012024103628.pdf"
    output_folder = "output_images"
    image_paths = extract_images_from_pdf(pdf_path)
    attributes_to_extract = ["DOB", "Issue Date"]

    all_dataframes = []
    for page_number, image in enumerate(image_paths):
        image_path = os.path.join(output_folder, f"page_{page_number + 1}.png")
        image.save(image_path, 'PNG')
        
        inputs = {
            "image_path": image_path,
            "attributes": attributes_to_extract
        }
        output = app.invoke(inputs)
        if "dataframe" in output:
            all_dataframes.append(output["dataframe"])

    # Combine all dataframes
    final_dataframe = pd.concat(all_dataframes, ignore_index=True)
    print(final_dataframe)
run()