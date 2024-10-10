from concurrent.futures import ThreadPoolExecutor
from typing import Literal, TypedDict, Optional
from pathlib import Path
from tempfile import TemporaryDirectory

import filecache
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from docx import Document
import magic
from loguru import logger
import pandas as pd

from openai import OpenAI
from pydantic import BaseModel, Field


class ContentExtractor:

    def __init__(self, _dir: str | Path | None):
        if _dir is None:
            self.dir = None
        else:
            self.dir = Path(_dir)

    PDF, IMAGE, DOCX, TXT, EXCEL, CSV = "PDF", "Image", "DOCX", "TXT", "EXCEL", "CSV"

    def identify_file_type(self, file_path: Path) -> str:
        """
        Identifies the type of file based on its MIME type.
        """
        mime = magic.Magic(mime=True)
        mime_type = mime.from_file(file_path)

        # Determine file type based on the mime type
        if mime_type == 'application/pdf':
            return self.PDF
        elif mime_type in ['image/jpeg', 'image/png', 'image/gif', 'image/tiff']:
            return self.IMAGE
        elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            return self.DOCX
        elif mime_type == 'text/plain':
            return self.TXT
        elif mime_type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                           'application/vnd.ms-excel']:
            return self.EXCEL
        elif mime_type in ['text/csv', 'application/csv']:
            return self.CSV
        else:
            raise Exception(f"Unsupported file type: {file_path.name}")

    def csv_to_text(self, csv_path: Path) -> str:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_path)

        # Convert the DataFrame to a string (plain text) format
        text_output = df.to_string(index=False)

        return text_output

    def convert_image_to_text(self, img_path: Path) -> str:
        """
        Converts an image file to text using OCR.
        """
        image = Image.open(img_path)
        text = pytesseract.image_to_string(image)
        return text

    def convert_xlsx_to_text(self, xlsx_path: Path) -> str:
        # Load the Excel file
        excel_data = pd.ExcelFile(xlsx_path)

        text_output = []

        # Loop through each sheet in the Excel file
        for sheet_name in excel_data.sheet_names:
            text_output.append(f"Sheet: {sheet_name}\n")

            # Read each sheet into a DataFrame
            df = pd.read_excel(excel_data, sheet_name=sheet_name)

            # Convert the DataFrame to text (each row will be converted into a string)
            text_output.append(df.to_string(index=False))
            text_output.append("\n\n")

        # Join the text from all sheets and rows into a single string
        return "\n".join(text_output)

    def convert_pdf_to_text(self, pdf_path: Path) -> str:
        """
        Converts a PDF file to text using OCR for each page.
        """
        pages = convert_from_path(pdf_path, dpi=300)
        extracted_text = ""

        with TemporaryDirectory() as tmp_dir:
            for page_number, page in enumerate(pages):
                image_filename = Path(tmp_dir) / f"page_{page_number + 1}.png"
                page.save(image_filename, "PNG")
                text = pytesseract.image_to_string(Image.open(image_filename))
                extracted_text += f"--- Page {page_number + 1} ---\n{text}\n"

        return extracted_text

    def extract_text_from_docx(self, file_path: Path) -> str:
        """
        Extracts text from a DOCX file.
        """
        doc = Document(str(file_path))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text

    def get_text_from_docs(self, all_files: list[Path]) -> dict:
        """
        Processes a list of files and extracts text from each one based on file type.
        """
        file_content_dict = {}
        for file in all_files:
            logger.info(f"Working on: {file.name}")
            file_type = self.identify_file_type(file)
            if file_type == self.PDF:
                file_content_dict[file.name] = self.convert_pdf_to_text(file)
            elif file_type == self.IMAGE:
                file_content_dict[file.name] = self.convert_image_to_text(file)
            elif file_type == self.DOCX:
                file_content_dict[file.name] = self.extract_text_from_docx(file)
            elif file_type == self.EXCEL:
                file_content_dict[file.name] = self.convert_xlsx_to_text(file)
            elif file_type == self.CSV:
                file_content_dict[file.name] = self.csv_to_text(file)
            else:
                raise Exception(f"Unsupported file type: {file_type}")
            logger.info(f"Done: {file.name}")
        return file_content_dict

    @filecache.filecache(60 * 60)
    def get_all_docs_as_text(self, _dir: Path = None) -> dict:
        if _dir is not None:
            dir_to_use = Path(_dir)
        else:
            dir_to_use = self.dir
        all_files = list(dir_to_use.rglob("*.*"))
        sample_files = [i for i in all_files if not "store" in str(i).lower()]

        # Define some example paths for demonstration
        text_dict = self.get_text_from_docs(sample_files)
        return text_dict


AUDITOR_PROMPT = """You are a helpful compliance analyst tasked with taking a detailed test script and answering the questions step by step. 
For each step you should provide and quote the exact evidence from the attached evidence and explain. 
If you don't have enough evidence to come to a conclusion you should say "I need more evidence in order to answer the question correctly" 
and continue on to the next question Do not hallucinate. """

MODEL = 'o1-preview'

DOCUMENT_AGGREGATOR_PROMPT = "You are a helpful analyst. Your job is to take in a list of documents around a given compliance decision and then Go through each document individually and classify it by document type, Summary, Date, and what happened in document"

import  os
client = OpenAI(api_key=os.environ["OPENAI_KEY"])

CONVERSION_PROMPT_USING_O1 = """
You are a helpful assistant tasked with taking an financial compliance auditing test script and converting it into a internal-facing programmatically executable routine optimized for an LLM. 
The LLM using this routine will be tasked with reading the test script, and answering audit questions, and finding any exceptions or compliance violations.

Please follow these instructions:
1. **Review the financial compliance auditing test script carefully** to ensure every step is accounted for. It is crucial not to skip any steps or policies.
2. **Organize the instructions into a logical, step-by-step order**, using the specified format.
3. **Use the following format**:
   - **Main actions are numbered** (e.g., 1, 2, 3).
   - **Sub-actions are lettered** under their relevant main actions (e.g., 1a, 1b).
      **Sub-actions should start on new lines**
   - **Specify conditions using clear 'if...then...else' statements** (e.g., 'If the , then...').
   - **If you need more data respond with "I need more data to make compliance decision"
4. For every step give evidence from the provided documents and explain the reasoning of why you made that decision,  give a pass or fail, and certainty rate of how certain you are about the decision you made.
   - **End with a final action for case resolution**: calling the `case_resolution` function should always be the final step.
5. Note that the evidence might be redacting names, addresses, and other PII and take that into account when you do the decisioning.

**Important**: If at any point you are uncertain, respond with "I don't know."

Please convert the financial compliance auditing test script into the formatted routine, ensuring it is easy to follow and execute programmatically.

"""

EXPANDED_POLICIES = [{"policy": "Indirect Credit Report Disputes",
                      "content": ["1. Did the CRA notify the Partner (Furnisher) through E-Oscar?",
                                  "2. Did the Partner investigate the dispute and review all relevant information provided by the CRA?",
                                  "3. Did the Partner report back the finding to the CRA?",
                                  "4. If an inaccuracy of information was provided by the Partner, was correction made to each CRA?",
                                  "5. Did the Partner review and provide response to the CRA by the CRA response due date (on ACDV form)?"],
                      "routine": "**Step-by-Step Routine:**\n\n1. **Verify if the CRA notified the Partner (Furnisher) through E-Oscar.**\n\n    a. Check for evidence of notification from the CRA to the Partner via E-Oscar.\n    \n    b. If evidence of E-Oscar notification is found, then proceed to Step 2.\n    \n    c. Else, record that the CRA did not notify the Partner through E-Oscar.\n    \n    **Evidence:** [Insert evidence of E-Oscar notification here]\n    \n    **Reasoning:** Based on the provided evidence, I assessed whether the CRA sent a notification through E-Oscar to the Partner.\n    \n    **Pass/Fail:** [Pass or Fail]\n    \n    **Certainty Rate:** [e.g., 95%]\n\n2. **Verify if the Partner investigated the dispute and reviewed all relevant information provided by the CRA.**\n\n    a. Check for documentation that the Partner conducted an investigation into the dispute.\n    \n    b. Confirm that all relevant information provided by the CRA was reviewed by the Partner.\n    \n    c. If both the investigation was conducted and all relevant information was reviewed, then proceed to Step 3.\n    \n    d. Else, record that the Partner did not adequately investigate the dispute or review all relevant information.\n    \n    **Evidence:** [Insert evidence of investigation and review here]\n    \n    **Reasoning:** Evaluated the Partner's investigation records to determine if they addressed the dispute comprehensively.\n    \n    **Pass/Fail:** [Pass or Fail]\n    \n    **Certainty Rate:** [e.g., 90%]\n\n3. **Verify if the Partner reported back the findings to the CRA.**\n\n    a. Look for evidence that the Partner communicated the investigation findings to the CRA.\n    \n    b. If the findings were reported back to the CRA, then proceed to Step 4.\n    \n    c. Else, record that the Partner did not report back the findings to the CRA.\n    \n    **Evidence:** [Insert evidence of communication to the CRA here]\n    \n    **Reasoning:** Reviewed communication logs to confirm that the Partner reported the findings to the CRA.\n    \n    **Pass/Fail:** [Pass or Fail]\n    \n    **Certainty Rate:** [e.g., 92%]\n\n4. **Verify if, when an inaccuracy of information was provided by the Partner, a correction was made to each CRA.**\n\n    a. Determine if the Partner provided inaccurate information.\n    \n        i. If an inaccuracy was identified:\n        \n            - Check if corrections were made to each CRA.\n            \n            - If corrections were made, proceed to Step 5.\n            \n            - Else, record that corrections were not made to each CRA.\n        \n        ii. Else, if no inaccuracy was provided, proceed to Step 5.\n    \n    **Evidence:** [Insert evidence of inaccuracies and corrections here]\n    \n    **Reasoning:** Investigated whether any inaccuracies existed and if corrective actions were taken accordingly.\n    \n    **Pass/Fail:** [Pass or Fail]\n    \n    **Certainty Rate:** [e.g., 88%]\n\n5. **Verify if the Partner reviewed and provided response to the CRA by the CRA response due date (on ACDV form).**\n\n    a. Identify the CRA response due date as indicated on the ACDV form.\n    \n    b. Check if the Partner's response was provided on or before the due date.\n    \n    c. If the response was timely, proceed to Final Action.\n    \n    d. Else, record that the Partner did not provide a response by the due date.\n    \n    **Evidence:** [Insert ACDV form and response timestamps here]\n    \n    **Reasoning:** Cross-referenced the response date with the due date to assess timeliness.\n    \n    **Pass/Fail:** [Pass or Fail]\n    \n    **Certainty Rate:** [e.g., 97%]\n\n6. **Final Action:**\n\n    a. Call the `case_resolution` function with all findings and decisions.\n\n---\n\n**Note:** If any required data is missing or redacted (e.g., due to redaction of PII such as names or addresses), and it affects the ability to make a compliance decision, state: \"I need more data to make compliance decision.\""},
                     {"policy": "Direct Credit Report Disputes",
                      "content": ["1. Did the Partner resolve the issue accurately, if applicable?",
                                  "2.a. Did the Partner complete the investigation and report the results and action taken to the consumer within 30 days of receipt?",
                                  "2.b. If resolved after 30 days, did partner resolve the claim within 45 days (i.e., if the consumer provided additional information relevant to the dispute during the 30-day period, and the partner required additional time for review)?",
                                  "3. Did the Partner report the corrected information to the CRA, if applicable? (Please request supporting documentation if applicable)"],
                      "routine": "1. Determine if the Partner resolved the issue accurately, if applicable.\n   1a. Review the case details from the provided documents to understand the issue.\n   1b. If the issue is applicable for resolution, then:\n       - Provide evidence from the documents to determine if the Partner resolved the issue accurately.\n       - Explain your reasoning for the decision.\n       - Give a pass or fail for this step.\n       - Provide a certainty rate for your decision.\n   1c. Else, if the issue is not applicable for resolution, then:\n       - State that the issue is not applicable for resolution.\n       - Proceed to step 2.\n\n2. Determine if the Partner completed the investigation and reported the results and action taken to the consumer within 30 days of receipt.\n   2a. Calculate the number of days between the receipt of the issue and the communication of the results to the consumer using dates from the provided documents.\n   2b. If the investigation and reporting were completed within 30 days, then:\n       - Provide evidence from the documents.\n       - Explain your reasoning.\n       - Give a pass or fail for this step.\n       - Provide a certainty rate for your decision.\n   2c. Else, if resolved after 30 days, then:\n       - Determine if the consumer provided additional information relevant to the dispute during the 30-day period, and the Partner required additional time for review.\n           - If both conditions are met, and the Partner resolved the claim within 45 days, then:\n               - Provide evidence from the documents.\n               - Explain your reasoning.\n               - Give a pass or fail for this step.\n               - Provide a certainty rate for your decision.\n           - Else:\n               - Provide evidence from the documents indicating the delay.\n               - Explain your reasoning.\n               - Give a pass or fail for this step.\n               - Provide a certainty rate for your decision.\n\n3. Determine if the Partner reported the corrected information to the CRA, if applicable.\n   3a. If the issue involved incorrect information reported to the CRA (Consumer Reporting Agency), then:\n       - Check if the Partner reported the corrected information to the CRA.\n           - Provide evidence from the documents (request supporting documentation if necessary).\n           - Explain your reasoning.\n           - Give a pass or fail for this step.\n           - Provide a certainty rate for your decision.\n   3b. Else, if not applicable, state that reporting to the CRA is not applicable.\n\n4. If you need more data to make a compliance decision at any point, respond with \"I need more data to make compliance decision\".\n\n5. If at any point you are uncertain, respond with \"I don't know.\"\n\n6. End with a final action for case resolution by calling the `case_resolution` function."},
                     {"policy": "Adverse Action Notices", "content": ["2a. Statement of action taken",
                                                                      "   - Examples may include text such as: unable to approve, declined, or similar",
                                                                      "2b. Name and Address of the creditor",
                                                                      "   - WebBank name",
                                                                      "   - An address as provided by the partner",
                                                                      "2c. A statement of the provisions of Section 701(a) of the ECOA",
                                                                      "   - Anti-discrimination language is included (language is specific and can be provided by the CT team)",
                                                                      "2d. Name and address of the federal agency that administers compliance with respect to the creditor",
                                                                      "   - FDIC Consumer Response Center, 1100 Walnut st. Box #111, Kansas City, MO 64106",
                                                                      "5. Was the FICO score listed on the AAN and does it have information about credit (Credit score disclosures), if the credit score was used to decline the application?",
                                                                      "   - May be listed as FICO score, Vantage score or similar",
                                                                      "6. If the credit score was used to decline the application, does the AAN have the consumer's right to:",
                                                                      "   a. Obtain a free copy of his or her consumer report from the consumer reporting agency providing the information if requested within 60 days,",
                                                                      "   b. To dispute the accuracy or completeness of any information in a consumer report furnished by the consumer reporting agency?",
                                                                      "7. If adverse action taken is based on information contained in the consumer report, does the AAN contain Name, address, and telephone number of the consumer reporting agency, per FCRA?",
                                                                      "   - Transunion, Experian, Equifax, etc.",
                                                                      "8. If adverse action taken is based on information contained in the consumer report, does the AAN contain a statement that the Credit Reporting Agency did not make the decision to take adverse action, per FCRA?",
                                                                      "9. If adverse action taken is based on information contained in the consumer report, does the AAN contain a statement that the Credit Reporting Agency is unable to provide the consumer with specific reasons why the adverse action was taken, per FCRA?"],
                      "routine": "**Formatted Routine:**\n\n1. **Verify that the Adverse Action Notice (AAN) contains a statement of action taken.**\n\n   1a. Check if the AAN includes phrases such as \"unable to approve,\" \"declined,\" or similar.\n\n       - **If** these phrases are present **then**:\n         - **Evidence**: Extracted text from AAN showing the statement of action taken.\n         - **Reasoning**: The presence of such phrases indicates compliance with the requirement to state the action taken.\n         - **Decision**: Pass\n         - **Certainty Rate**: 100%\n       - **Else**:\n         - **Evidence**: No such phrases found in the AAN.\n         - **Reasoning**: Absence of the statement of action taken violates the compliance requirement.\n         - **Decision**: Fail\n         - **Certainty Rate**: 100%\n\n2. **Verify that the AAN contains the name and address of the creditor.**\n\n   2a. Check if the creditor's name is \"WebBank.\"\n\n       - **If** \"WebBank\" is present **then**:\n         - **Evidence**: Extracted creditor name from AAN.\n         - **Reasoning**: Correct creditor name ensures accurate identification.\n         - **Decision**: Pass\n         - **Certainty Rate**: 100%\n       - **Else**:\n         - **Evidence**: Creditor name is missing or incorrect.\n         - **Reasoning**: Incorrect or missing creditor name does not meet compliance standards.\n         - **Decision**: Fail\n         - **Certainty Rate**: 100%\n\n   2b. Check if an address is provided by the partner.\n\n       - **If** an address is present **then**:\n         - **Evidence**: Extracted address from AAN (PII may be redacted).\n         - **Reasoning**: Presence of address satisfies the requirement.\n         - **Decision**: Pass\n         - **Certainty Rate**: 100%\n       - **Else**:\n         - **Evidence**: No address found in the AAN.\n         - **Reasoning**: Missing address fails to meet compliance standards.\n         - **Decision**: Fail\n         - **Certainty Rate**: 100%\n\n3. **Verify that the AAN includes a statement of the provisions of Section 701(a) of the ECOA (Anti-discrimination language).**\n\n   3a. Check if the anti-discrimination language is included in the AAN.\n\n       - **If** the language is included **then**:\n         - **Evidence**: Extracted anti-discrimination statement from AAN.\n         - **Reasoning**: Inclusion demonstrates compliance with ECOA requirements.\n         - **Decision**: Pass\n         - **Certainty Rate**: 100%\n       - **Else**:\n         - **Evidence**: Anti-discrimination language not found.\n         - **Reasoning**: Omission violates ECOA compliance standards.\n         - **Decision**: Fail\n         - **Certainty Rate**: 100%\n\n4. **Verify that the AAN includes the name and address of the federal agency that administers compliance with respect to the creditor.**\n\n   4a. Check if the following is included: \"FDIC Consumer Response Center, 1100 Walnut St. Box #111, Kansas City, MO 64106.\"\n\n       - **If** the address is present **then**:\n         - **Evidence**: Extracted federal agency information from AAN.\n         - **Reasoning**: Presence meets the requirement for federal agency disclosure.\n         - **Decision**: Pass\n         - **Certainty Rate**: 100%\n       - **Else**:\n         - **Evidence**: Federal agency information not found.\n         - **Reasoning**: Missing information does not meet compliance standards.\n         - **Decision**: Fail\n         - **Certainty Rate**: 100%\n\n5. **Determine if the credit score was used to decline the application.**\n\n   5a. **If** the credit score was used **then** proceed to Step 6.\n\n   5b. **Else** proceed to Step 8.\n\n6. **Verify that the AAN includes the credit score and related disclosures if the credit score was used to decline the application.**\n\n   6a. Check if the AAN lists the FICO score, Vantage score, or similar.\n\n       - **If** the credit score is listed **then**:\n         - **Evidence**: Extracted credit score and disclosure information from AAN.\n         - **Reasoning**: Disclosure of credit score is required when used in the decision.\n         - **Decision**: Pass\n         - **Certainty Rate**: 100%\n       - **Else**:\n         - **Evidence**: Credit score information not found.\n         - **Reasoning**: Missing credit score disclosure violates compliance standards.\n         - **Decision**: Fail\n         - **Certainty Rate**: 100%\n\n7. **Verify that the AAN includes the consumer's rights regarding their consumer report.**\n\n   7a. Check if the AAN states the consumer's right to obtain a free copy of their consumer report from the consumer reporting agency if requested within 60 days.\n\n       - **If** the right is stated **then**:\n         - **Evidence**: Extracted statement of consumer's right from AAN.\n         - **Reasoning**: Informing consumers of their rights is a compliance requirement.\n         - **Decision**: Pass\n         - **Certainty Rate**: 100%\n       - **Else**:\n         - **Evidence**: Statement of consumer's right not found.\n         - **Reasoning**: Omission fails to meet compliance standards.\n         - **Decision**: Fail\n         - **Certainty Rate**: 100%\n\n   7b. Check if the AAN states the consumer's right to dispute the accuracy or completeness of any information in a consumer report furnished by the consumer reporting agency.\n\n       - **If** the right is stated **then**:\n         - **Evidence**: Extracted dispute rights information from AAN.\n         - **Reasoning**: Disclosure ensures consumer is informed of their rights under FCRA.\n         - **Decision**: Pass\n         - **Certainty Rate**: 100%\n       - **Else**:\n         - **Evidence**: Dispute rights information not found.\n         - **Reasoning**: Lack of disclosure violates FCRA requirements.\n         - **Decision**: Fail\n         - **Certainty Rate**: 100%\n\n8. **If adverse action is based on information contained in the consumer report, verify that the AAN includes the following per FCRA:**\n\n   8a. Check if the AAN includes the name, address, and telephone number of the consumer reporting agency (e.g., TransUnion, Experian, Equifax).\n\n       - **If** the information is included **then**:\n         - **Evidence**: Extracted consumer reporting agency details from AAN.\n         - **Reasoning**: Inclusion complies with FCRA disclosure requirements.\n         - **Decision**: Pass\n         - **Certainty Rate**: 100%\n       - **Else**:\n         - **Evidence**: Consumer reporting agency details not found.\n         - **Reasoning**: Missing information violates FCRA compliance.\n         - **Decision**: Fail\n         - **Certainty Rate**: 100%\n\n   8b. Check if the AAN includes a statement that the consumer reporting agency did not make the decision to take adverse action.\n\n       - **If** the statement is included **then**:\n         - **Evidence**: Extracted statement from AAN.\n         - **Reasoning**: Statement fulfills FCRA requirement to inform consumer.\n         - **Decision**: Pass\n         - **Certainty Rate**: 100%\n       - **Else**:\n         - **Evidence**: Statement not found.\n         - **Reasoning**: Omission fails FCRA compliance.\n         - **Decision**: Fail\n         - **Certainty Rate**: 100%\n\n   8c. Check if the AAN includes a statement that the consumer reporting agency is unable to provide the consumer with specific reasons why the adverse action was taken.\n\n       - **If** the statement is included **then**:\n         - **Evidence**: Extracted statement from AAN.\n         - **Reasoning**: Inclusion meets FCRA disclosure obligations.\n         - **Decision**: Pass\n         - **Certainty Rate**: 100%\n       - **Else**:\n         - **Evidence**: Statement not found.\n         - **Reasoning**: Missing statement does not comply with FCRA.\n         - **Decision**: Fail\n         - **Certainty Rate**: 100%\n\n9. **Final Action:**\n\n   - Call the `case_resolution` function.\n\n**Note:** If at any point the necessary information is not available or is unclear due to redactions or missing data, respond with \"I need more data to make compliance decision.\""}]


class RawTestPolicies(TypedDict):
    policy: str
    content: list[str]


RAW_TEST_SCRIPTS_FROM_BANK: list[RawTestPolicies] = [
    {
        "policy": "Indirect Credit Report Disputes",
        "content": [
            "1. Did the CRA notify the Partner (Furnisher) through E-Oscar?",
            "2. Did the Partner investigate the dispute and review all relevant information provided by the CRA?",
            "3. Did the Partner report back the finding to the CRA?",
            "4. If an inaccuracy of information was provided by the Partner, was correction made to each CRA?",
            "5. Did the Partner review and provide response to the CRA by the CRA response due date (on ACDV form)?"
        ]
    },
    {
        "policy": "Direct Credit Report Disputes",
        "content": [
            "1. Did the Partner resolve the issue accurately, if applicable?",
            "2.a. Did the Partner complete the investigation and report the results and action taken to the consumer within 30 days of receipt?",
            "2.b. If resolved after 30 days, did partner resolve the claim within 45 days (i.e., if the consumer provided additional information relevant to the dispute during the 30-day period, and the partner required additional time for review)?",
            "3. Did the Partner report the corrected information to the CRA, if applicable? (Please request supporting documentation if applicable)"
        ]
    },
    {
        "policy": "Adverse Action Notices",
        "content": [
            "2a. Statement of action taken",
            "   - Examples may include text such as: unable to approve, declined, or similar",
            "2b. Name and Address of the creditor",
            "   - WebBank name",
            "   - An address as provided by the partner",
            "2c. A statement of the provisions of Section 701(a) of the ECOA",
            "   - Anti-discrimination language is included (language is specific and can be provided by the CT team)",
            "2d. Name and address of the federal agency that administers compliance with respect to the creditor",
            "   - FDIC Consumer Response Center, 1100 Walnut st. Box #111, Kansas City, MO 64106",
            "5. Was the FICO score listed on the AAN and does it have information about credit (Credit score disclosures), if the credit score was used to decline the application?",
            "   - May be listed as FICO score, Vantage score or similar",
            "6. If the credit score was used to decline the application, does the AAN have the consumer's right to:",
            "   a. Obtain a free copy of his or her consumer report from the consumer reporting agency providing the information if requested within 60 days,",
            "   b. To dispute the accuracy or completeness of any information in a consumer report furnished by the consumer reporting agency?",
            "7. If adverse action taken is based on information contained in the consumer report, does the AAN contain Name, address, and telephone number of the consumer reporting agency, per FCRA?",
            "   - Transunion, Experian, Equifax, etc.",
            "8. If adverse action taken is based on information contained in the consumer report, does the AAN contain a statement that the Credit Reporting Agency did not make the decision to take adverse action, per FCRA?",
            "9. If adverse action taken is based on information contained in the consumer report, does the AAN contain a statement that the Credit Reporting Agency is unable to provide the consumer with specific reasons why the adverse action was taken, per FCRA?"
        ]
    }
]

ImprovedQuery = str


class PolicyContentImproved(TypedDict):
    policy: str
    content: list[str]
    routine: str


class O1AuditTestRunner:
    def __init__(self, raw_test_policies: list[RawTestPolicies]):
        self.raw_test_policies = raw_test_policies

    @staticmethod
    def run_audit(tests: dict, document: dict):
        try:
            messages = [
                {
                    "role": "user",
                    "content": f"""
                        {AUDITOR_PROMPT}

                        TESTING INSTRUCTIONS: 
                        {tests}

                        POLICY:
                        {document}
                    """
                }
            ]

            response = client.chat.completions.create(
                model=MODEL,
                messages=messages
            )

            return response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")

    def expand_query_using_o1(self) -> list[PolicyContentImproved]:

        def generate_routine(policy: list[str]) -> str:
            try:
                messages = [
                    {
                        "role": "user",
                        "content": f"""
                            {CONVERSION_PROMPT_USING_O1}

                            POLICY:
                            {policy}
                        """
                    }
                ]

                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages
                )

                return response.choices[0].message.content
            except Exception as e:
                print(f"An error occurred: {e}")

        def process_testing_script(test: RawTestPolicies):
            routine = generate_routine(test['content'])
            print(routine)
            return {"policy": test['policy'], "content": test['content'], "routine": routine}

        with ThreadPoolExecutor() as executor:
            prompts = list(executor.map(process_testing_script, self.raw_test_policies))
            return prompts

    @staticmethod
    def _classify_documents(dispute_content_from_file: dict[str, str]):
        try:
            messages = [
                {
                    "role": "user",
                    "content": f"""
                        {DOCUMENT_AGGREGATOR_PROMPT}

                        DOCUMENTS:
                        {dispute_content_from_file}
                    """
                }
            ]

            response = client.chat.completions.create(
                model=MODEL,
                messages=messages
            )

            return response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")

    def main(self, docs_to_audit: dict[str, str], test_to_run: Optional[int] = None, expanded_policies=None):
        if expanded_policies is None:
            expanded_policies = self.expand_query_using_o1()
        if test_to_run:
            _tests = [expanded_policies[test_to_run]]
        else:
            _tests = expanded_policies

        def run_testing_script(single_test: dict):
            """"""
            execution = self.run_audit(single_test['routine'], docs_to_audit)
            return {"policy": single_test['policy'], "routine": single_test['routine'], "execution": execution}

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(run_testing_script, _tests))
            print(results)
            return results


class SingleTestRuleResultSchema(BaseModel):
    compliance_test_name: str
    reason: str
    confidence: float
    test_rule_name: str
    passed: bool = Field(..., description="Whether the test rule passed or failed")
    exact_text_from_test_file: str
    filename: str = Field(..., description="If filename is present return the filename, else return empty string '' ")


class AllTestOutputs(BaseModel):
    results: list[SingleTestRuleResultSchema]


def get_structured_output() -> list[SingleTestRuleResultSchema]:
    """"""
    content = """**1. Determine if the Partner resolved the issue accurately, if applicable.**

**1a. Review the case details from the provided documents to understand the issue.**

The consumer submitted a dispute regarding negative information on their credit report. As per **'Dispute.pdf'** dated **April 29, 2024**:

*"...over that term I managed to accrue several negatives forwarded to my credit file. In a while, I desire to apply for a loan, and I am truly nervous referencing the possibility that these bad credit bureau tradelines may cause me damage. Please review the relationship as a favor."*

**1b. Provide evidence from the documents to determine if the Partner resolved the issue accurately.**

The Partner responded on **May 17, 2024**, as shown in **'Dispute Response.pdf'**:

*"We have reviewed the payment history on your account and confirmed the information reported to the credit reporting agencies is accurate. Federal law requires the reporting of accurate information to the credit reporting agencies to ensure the integrity of the credit system."*

Reviewing the **'Transaction History.csv'**, the account shows multiple late payments and fee assessments:

- **1/11/2024**: Payment reversal due to insufficient funds (*Descriptor Code: R01x*).
- **2/9/2024**: Payment reversal due to insufficient funds (*Descriptor Code: R01x*).
- **Late Charge Assessments** on:
  - **2/21/2024**
  - **3/21/2024**
  - **4/21/2024**
  - **5/21/2024**
  - **6/21/2024**
  - **7/21/2024**

These entries indicate the consumer had delinquencies and late payments on their account.

**Explain your reasoning for the decision.**

The evidence from the transaction history confirms that the negative information reported to the credit agencies was accurate due to missed payments, payment reversals, and late fees. Therefore, the Partner accurately resolved the issue by confirming the validity of the reported information.

**Give a pass or fail for this step.**

**Pass**

**Provide a certainty rate for your decision.**

**Certainty Rate: 95%**

---

**2. Determine if the Partner completed the investigation and reported the results and action taken to the consumer within 30 days of receipt.**

**2a. Calculate the number of days between the receipt of the issue and the communication of the results to the consumer using dates from the provided documents.**

- **Dispute Received Date**: April 29, 2024 (**'Initial Population.xlsx'** and **'Dispute.pdf'**)
- **Response Date**: May 17, 2024 (**'Dispute Response.pdf'**)

**Number of days between receipt and response**: 18 days

**2b. Provide evidence from the documents.**

The Partner received the dispute on **April 29, 2024**, and responded on **May 17, 2024**, which is 18 days later.

**Explain your reasoning.**

The response was provided within the 30-day timeframe required for dispute investigations, complying with regulatory standards.

**Give a pass or fail for this step.**

**Pass**

**Provide a certainty rate for your decision.**

**Certainty Rate: 100%**

---

**3. Determine if the Partner reported the corrected information to the CRA, if applicable.**

Since the Partner confirmed that the information reported was accurate and no corrections were necessary, reporting corrected information to the Consumer Reporting Agency (CRA) is not applicable.
"""
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        response_format=AllTestOutputs,
        messages=[
            {
                "role": "system",
                "content": f"""
                You are a data engineer in financial sector trying to store the output of compliance testing into a database .
                    """
            },
            {"role": "user", "content": "There are exactly 3 compliance tests"},
            {"role": "user", "content": content}
        ]

    )

    return completion.choices[0].message.parsed


if __name__ == '__main__':
    # need to get documents
    direct_dispute_extractor = ContentExtractor(_dir=None)
    direct_dispute_cases = direct_dispute_extractor.get_all_docs_as_text(
        _dir=Path("/Users/dudeman/Downloads/FCRA_Direct_Disputes"))
    test_runner = O1AuditTestRunner(raw_test_policies=RAW_TEST_SCRIPTS_FROM_BANK)
    # results = test_runner.main(docs_to_audit=direct_dispute_cases, expanded_policies=[EXPANDED_POLICIES[1]])
    parsed_results = get_structured_output()
