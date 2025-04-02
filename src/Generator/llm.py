import json
import jsonschema

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import os
import google.generativeai as genai

from duckduckgo_search import DDGS

class LearningHubAgent:
    def __init__(self):
        """Initialize the Learning Hub Agent with Vertex AI."""
        
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        # Initialize Gemini model through Vertex AI
        self.model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                system_instruction="Always return valid JSON."
                )
        
        # Initialize search engine for RAG
        self.ddgs = DDGS()
    
    def generate_headlines(self, query):
        """Generate course headlines/syllabus based on the query."""
        print(f"Generating headlines for: {query}")

        # Define the response schema
        schema = {
            "type": "object",
            "properties": {
                "main_headline": {"type": "string"},
                "topics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                    "headline": {"type": "string"},
                    "subtopics": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                    },
                    "required": ["headline", "subtopics"]
                }
                }
            },
            "required": ["main_headline", "topics"]
            }
        
        prompt = f"""
        Create a comprehensive course syllabus with 5-7 main headlines 
        for a learning module about: "{query}".
        
        For each headline, provide 2-3 sub-topics that should be covered.
        Create a detailed outline with a main headline, multiple topics, and related subtopics. The number of topics should be determined based on what makes sense for the subject matter.
        Format your response as a JSON object that strictly follows this JSON Schema:
        {json.dumps(schema, indent=2)} 
        """

        response = self.model.generate_content(prompt)
        
        # Parse the response
        try:
            # Try to parse directly first
            json_data = json.loads(response.text)
        except json.JSONDecodeError:
            # If that fails, try to extract from markdown
            text = response.text
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0].strip()
            else:
                json_str = text
            json_data = json.loads(json_str)

        # After parsing the JSON response:
        try:
            jsonschema.validate(instance=json_data, schema=schema)
            print("JSON validation successful!")
        except jsonschema.exceptions.ValidationError as e:
            print(f"JSON validation error: {e}")

        return json_data
    
    def web_search(self, query, headlines):
        """Perform web search (RAG) based on query and headlines."""
        print(f"Retrieving web content for: {query}")
        search_results = []
        
        # Search for the main query
        main_results = list(self.ddgs.text(query, max_results=5))
        search_results.extend(main_results)
        
        # Search for specific topics to get targeted information
        for topic in headlines.get("topics", [])[:3]:
            headline = topic.get("headline", "")
            specific_query = f"{query} {headline}"
            headline_results = list(self.ddgs.text(specific_query, max_results=2))
            search_results.extend(headline_results)
        
        # Deduplicate results
        unique_results = []
        seen_urls = set()
        for result in search_results:
            if result.get("href") not in seen_urls:
                seen_urls.add(result.get("href"))
                unique_results.append(result)
        
        return unique_results[:10]  # Limit total results
    
    def retrieve_images(self, query, headlines):
        """Retrieve relevant images based on query and headlines."""
        print(f"Finding images for: {headlines["main_headline"]}")
        images_data = []
        
        # Search for images related to the main query
        main_query = f"{headlines["main_headline"]} diagram educational"
        main_images = list(self.ddgs.images(main_query, max_results=2))
        
        # Search for topic-specific images
        for topic in headlines.get("topics", [])[:2]:
            headline = topic.get("headline", "")
            topic_query = f"{headline} {query} illustration"
            topic_images = list(self.ddgs.images(topic_query, max_results=1))
            main_images.extend(topic_images)
        
        # Process images
        for i, img in enumerate(main_images[:4]):
            image_data = {
                "id": i + 1,
                "url": img.get("image"),
                "title": img.get("title", f"Image related to {query}"),
                "source": img.get("url", "")
            }
            images_data.append(image_data)
        
        return images_data
    
    def generate_course_content(self, query, headlines, search_results, images):
        """Generate course content using Gemini with retrieved data."""
        print("Generating comprehensive course content...")
        # Format search results
        search_content = "\n\n".join([
            f"--- SOURCE {i+1} ---\n"
            f"Title: {result.get('title', 'No Title')}\n"
            f"Content: {result.get('body', '')}\n"
            f"URL: {result.get('href', '')}"
            for i, result in enumerate(search_results)
        ])
        
        # Format image data
        image_content = "\n".join([
            f"IMAGE {img.get('id')}: {img.get('title')} (from {img.get('source')})"
            for img in images
        ])
        
        # Format headlines into syllabus
        headlines_content = f"Course Title: {headlines.get('main_headline', query)}\n\n"
        for i, topic in enumerate(headlines.get("topics", [])):
            headlines_content += f"Topic {i+1}: {topic.get('headline', '')}\n"
            for j, subtopic in enumerate(topic.get("subtopics", [])):
                headlines_content += f"  - Subtopic {i+1}.{j+1}: {subtopic}\n"
        
        # Create the content generation prompt for Gemini
        prompt = f"""
        Create a comprehensive educational course about "{query}".
        
        Use the provided course structure, web search results, and image descriptions to create 
        a well-formatted, educational course.
        
        COURSE OUTLINE:
        {headlines_content}
        
        WEB SEARCH RESULTS:
        {search_content}
        
        AVAILABLE IMAGES:
        {image_content}
        
        Please create a well-structured course following these guidelines:
        1. Begin with an engaging introduction to the topic
        2. Follow the provided course outline structure
        3. For each main topic:
           - Provide clear explanations using information from the search results
           - Reference relevant images where appropriate using: ![IMAGE X](image_url_X)
           - Include examples, facts, and interesting information
        4. End with a summary and suggestions for further learning
        
        Format the content in Markdown with proper headings and sections.
        """
        
        response = self.model.generate_content(prompt)
        course_content = response.text
        
        # Replace image placeholders with actual image URLs
        for img in images:
            img_id = img.get('id')
            img_url = img.get('url')
            img_placeholder = f"![IMAGE {img_id}](image_url_{img_id})"
            img_markdown = f"![{img.get('title')}]({img_url})"
            course_content = course_content.replace(img_placeholder, img_markdown)
        
        return course_content
    
    def process_query(self, query):
        """Process the user query and generate the complete learning course."""
        print(f"Processing query: {query}")
        
        # Step 1: Generate headlines/syllabus
        headlines = self.generate_headlines(query)
        
        # Step 2: Retrieve web search data (RAG)
        search_results = self.web_search(query, headlines)
        
        # Step 3: Retrieve images
        images = self.retrieve_images(query, headlines)
        
        # Step 4: Generate course content using Gemini
        course_content = self.generate_course_content(query, headlines, search_results, images)
        
        # Prepare final output
        output = {
            "query": query,
            "headlines": headlines,
            "course_content": course_content,
            "images": images
        }
        
        print("Learning hub content generation complete!")
        return output
