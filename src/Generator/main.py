from llm import LearningHubAgent

def main():
    # Initialize the Learning Hub Agent
    agent = LearningHubAgent()
    
    # Example query
    query = "Artificial Intelligence in Education"
    course_curriculum = agent.process_query(query)
    print(course_curriculum)


if __name__ == "__main__":
    main()