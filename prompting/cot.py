import re
import openai

def create_cot_prompt(course_title, description, learning_objective):
    """Create a Chain of Thought prompt with examples that show step-by-step reasoning."""
    return f"""Rate how well this course covers the learning objective:

Course Title: {course_title}
Description: {description}
Learning Objective: {learning_objective}

SCORING RUBRIC:
5 - Direct Match (core skill): The real-world task is a core learning objective explicitly covered in the course description.
4 - Strongly Related: The task aligns with course content; students should be able to perform it after completion.
3 - Indirectly Related but transferable: Not explicitly covered, but students develop transferable skills.
2 - Unrelated but Within Same Domain: Within the discipline but not relevant to course content.
1 - Completely Unrelated: The task is outside the scope of the course and belongs to a different domain.

Let me work through some examples first to show my reasoning process:

Example 1:
Course: Introduction to Database Management
Description: This course introduces fundamental concepts of database design and management, including data models, normalization, SQL queries, and database administration.
Learning Objective: Write SQL queries to extract and manipulate data
Analysis: The course description explicitly mentions SQL queries as one of the core topics covered. The learning objective of writing SQL queries is directly mentioned as part of the course content. This is a clear case where the learning objective is explicitly covered in the course description.
Rating: 5 - Direct Match

Example 2:
Course: Web Development Fundamentals
Description: This course covers HTML, CSS, JavaScript, and responsive design principles for creating modern websites. Students will learn to build interactive web pages and implement common UI patterns.
Learning Objective: Develop a full-stack web application with authentication
Analysis: The course covers front-end technologies (HTML, CSS, JavaScript) but doesn't mention back-end development or authentication, which are necessary for full-stack applications. However, the front-end skills taught would be transferable to this objective, though incomplete for the full task.
Rating: 3 - Indirectly Related but transferable

Now, let me analyze the current case:
Course Title: {course_title}
Description: {description}
Learning Objective: {learning_objective}

Analysis: """

def extract_score(response):
    """Extract numerical score from model response."""
    if not response:
        return None
        
    # First try to find a standalone digit at the beginning
    first_line = response.strip().split('\n')[0].strip()
    if first_line.isdigit() and 1 <= int(first_line) <= 5:
        return int(first_line)
    
    # Then try to find patterns like "Rating: 4" or "Score: 3"
    rating_pattern = r'(?:rating|score):\s*([1-5])'
    matches = re.search(rating_pattern, response.lower())
    if matches:
        return int(matches.group(1))
    
    # Finally, just find any digit 1-5
    matches = re.findall(r'[1-5]', response)
    if matches:
        return int(matches[0])
        
    return None

def get_cot_prediction(course_title, description, learning_objective, client):
    """Get prediction using Chain of Thought prompting."""
    prompt = create_cot_prompt(course_title, description, learning_objective)
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at evaluating how well course descriptions align with learning objectives. Follow the examples to provide step-by-step reasoning and a clear numerical rating."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        
        return extract_score(response.choices[0].message.content)
        
    except Exception as e:
        print(f"Error getting prediction: {str(e)}")
        return None