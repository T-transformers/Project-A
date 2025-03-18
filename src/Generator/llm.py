import json

# Vertex AI imports
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel

from duckduckgo_search import DDGS

class LearningHubAgent:
    def __init__(self, project_id, location="us-central1"):
        """Initialize the Learning Hub Agent with Vertex AI."""
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)
        
        # Initialize Gemini model through Vertex AI
        self.model = GenerativeModel("gemini-2.0-flash")
        
        # Initialize search engine for RAG
        self.ddgs = DDGS()
    
    def generate_headlines(self, query):
        """Generate course headlines/syllabus based on the query."""
        print(f"Generating headlines for: {query}")
        
        prompt = f"""
        Create a comprehensive course syllabus with 5-7 main headlines 
        for a learning module about: "{query}".
        
        For each headline, provide 2-3 sub-topics that should be covered.
        Format your response as JSON with this structure:
        {{
            "main_headline": "The course title",
            "topics": [
                {{
                    "headline": "Topic 1",
                    "subtopics": ["Subtopic 1.1", "Subtopic 1.2"]
                }},
                ...
            ]
        }}
        """
        
        response = self.model.generate_content(prompt)
        
        # Extract JSON from the response
        json_str = response.text
        # Preprocessing to handle single quotes in JSON
        
        headlines = json.loads(json_str)
        return headlines
    
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
        print(f"Finding images for: {query}")
        images_data = []
        
        # Search for images related to the main query
        main_query = f"{query} diagram educational"
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
