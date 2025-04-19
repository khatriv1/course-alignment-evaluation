import re
import openai

def create_step_back_prompt(course_title, description, learning_objective):
    """Create a Take a Step Back prompt."""
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

Let me take a step back and think about this at a more abstract level:

1. What is the general domain or field of knowledge this course belongs to?
2. What broader learning categories does this course aim to develop (e.g., technical skills, theoretical knowledge, analytical abilities)?
3. What general type of capability does the learning objective represent?
4. At an abstract level, what transferable skills might connect the course to the objective?

Now, using this more abstract understanding, I'll analyze the specific alignment between the course and learning objective to provide a rating."""

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

def get_step_back_prediction(course_title, description, learning_objective, client):
    """Get prediction using Take a Step Back prompting."""
    prompt = create_step_back_prompt(course_title, description, learning_objective)
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at evaluating how well course descriptions align with learning objectives. First take a step back to consider the question from a more abstract perspective, then provide a single numerical rating (1-5) followed by justification."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        
        return extract_score(response.choices[0].message.content)
        
    except Exception as e:
        print(f"Error getting prediction: {str(e)}")
        return None