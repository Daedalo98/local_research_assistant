import re
import requests
from gemini_manager import GeminiManager

class TrustPipeline:
    def __init__(self, model_name, zotero_client=None):
        self.manager = GeminiManager()
        self.model = model_name
        self.zotero_client = zotero_client

    def format_sources(self, retrieved_chunks):
        formatted = ""
        for i, chunk in enumerate(retrieved_chunks):
            if isinstance(chunk, dict):
                filename = chunk.get('filename', f"Source_{i+1}")
                text = chunk.get('text', str(chunk))
            else:
                filename = f"Source_{i+1}"
                text = str(chunk)
            formatted += f"--- {filename} ---\n{text}\n\n"
        return formatted

    def extract_dois(self, text):
        doi_regex = r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+"
        matches = re.findall(doi_regex, text)
        return list(set(matches))

    def fetch_citation(self, doi):
        try:
            url = f"https://doi.org/{doi}"
            headers = {"Accept": "text/x-bibliography; style=apa"}
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                return response.text.strip()
            return f"DOI found ({doi}) but citation could not be formatted."
        except Exception as e:
            return f"Error fetching citation for {doi}: {e}"

    def step_a_enhance_text(self, user_idea, sources_text, prompt_instruction):
        system = "You are an expert research assistant. Enhance the user's base text using the provided sources. Use inline citations."
        user_prompt = f"User's Base Text:\n{user_idea}\n\nSources:\n{sources_text}\n\nInstructions:\n{prompt_instruction}"
        return self.manager.generate_sync(user_prompt, system, self.model)

    def step_b_retrieve_citations(self, pulled_chunks):
        import asyncio
        citations = []
        
        loop = None
        if self.zotero_client:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
        for chunk in pulled_chunks:
            filename = chunk.get('filename', 'Unknown') if isinstance(chunk, dict) else 'Unknown'
            filepath = chunk.get('filepath', '') if isinstance(chunk, dict) else ''
            
            if filepath.startswith("zotero://"):
                item_key = filepath.replace("zotero://", "")
                if self.zotero_client and loop:
                    try:
                        cit_obj = loop.run_until_complete(self.zotero_client.generate_citation([item_key], "apa"))
                        # zotero_client might return a string directly or a JSON output. 
                        # Handle potential JSON parsing error just in case.
                        cit = cit_obj
                        
                        cit_entry = f"[{filename}] {cit}"
                        if cit_entry not in citations:
                            citations.append(cit_entry)
                        continue
                    except Exception as e:
                        print(f"Failed to generate Zotero citation for {item_key}: {e}")
                        
        return citations

    def step_c_create_academic_paragraph(self, user_idea, enhanced_text, citations_text, prompt_instruction):
        system = "You are an expert academic writer. Synthesize the final academic text seamlessly."
        user_prompt = (
            f"Original User Idea:\n{user_idea}\n\n"
            f"Enhanced Text (with retrieved info):\n{enhanced_text}\n\n"
            f"Extracted APA Citations:\n{citations_text}\n\n"
            f"Task:\n{prompt_instruction}\n\n"
            "Format the output as a coherent academic section. At the bottom, include a distinct 'References' section containing the APA citations provided."
        )
        return self.manager.generate_sync(user_prompt, system, self.model)