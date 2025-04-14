import re
import openai

def create_cot_prompt(course_title, description, learning_objective):
    """Create a Chain of Thought prompt."""
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

Let's think about this step by step:

1. First, identify and list the key concepts and skills from the course description:
   - What are the main topics covered?
   - What specific skills are taught?
   - What are the core learning outcomes?

2. Next, analyze what the learning objective requires:
   - What specific skills or knowledge does it need?
   - What level of proficiency is implied?
   - In what context would this objective be applied?

3. Compare the course content and the objective:
   - Are there direct overlaps in content?
   - Are the required skills explicitly taught?
   - Could the course skills transfer to this objective?

4. Consider the level of alignment:
   - Is this a core focus of the course?
   - Is it indirectly supported by course content?
   - Are they in the same domain?

5. Based on this analysis, determine the appropriate score according to the rubric.

First walk through each step of the analysis, then provide a single numerical rating (1-5) followed by your final justification."""

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
                {"role": "system", "content": "You are an expert at evaluating how well course descriptions align with learning objectives. Use careful step-by-step reasoning to analyze the alignment and provide a single numerical rating (1-5) followed by justification."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        
        return extract_score(response.choices[0].message.content)
        
    except Exception as e:
        print(f"Error getting prediction: {str(e)}")
        return None
