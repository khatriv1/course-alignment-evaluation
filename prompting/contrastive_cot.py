import re
import openai

def create_contrastive_cot_prompt(course_title, description, learning_objective):
    """Create a contrastive chain of thought prompt."""
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

Let me think about this step by step:

First, here's an incorrect reasoning path to avoid:
1. A quick comparison of keywords might suggest a certain rating.
2. Assigning a rating solely based on overlapping terms.
3. Missing nuances in how skills in the course might transfer to the objective.
4. Overlooking implicit connections between course content and objective requirements.
5. This incorrect approach would likely lead to a wrong rating.

Now, let's analyze this correctly:
1. Let me identify key skills and knowledge areas taught in the course.
2. Next, I'll identify what skills and knowledge the learning objective requires.
3. I'll compare these carefully, looking for both explicit and implicit connections.
4. I'll consider how directly the course prepares students for the learning objective.
5. Based on this thorough analysis, I'll assign a rating according to the rubric.

Now I'll analyze the course and learning objective using the correct reasoning path and provide a rating."""

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

def get_contrastive_cot_prediction(course_title, description, learning_objective, client):
    """Get prediction using contrastive chain of thought prompting."""
    prompt = create_contrastive_cot_prompt(course_title, description, learning_objective)
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at evaluating how well course descriptions align with learning objectives. Use careful step-by-step reasoning, avoid common pitfalls, and provide a single numerical rating (1-5) followed by justification."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        
        return extract_score(response.choices[0].message.content)
        
    except Exception as e:
        print(f"Error getting prediction: {str(e)}")
        return None