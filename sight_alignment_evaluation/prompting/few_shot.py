# sight_alignment_evaluation/prompting/few_shot.py

"""
Few-shot prompting for SIGHT comment classification.
Provides examples before asking for classification.
"""

import time
from typing import List, Optional
from utils.sight_rubric import SIGHTRubric

def get_few_shot_examples(category: str) -> str:
    """Get few-shot examples for each category."""
    examples = {
        'general': [
            ("Best video I have watched so far, I was with him all the way and my concentration never dipped.", "true"),
            ("Amazing lectures!", "true"),
            ("What's the weather like today?", "false")
        ],
        'confusion': [
            ("I don't understand why we multiply here instead of divide", "true"),
            ("Wait, how did you get from step 2 to step 3?", "true"),
            ("Thanks for the video!", "false")
        ],
        'pedagogy': [
            ("You should show more examples before moving to harder problems", "true"),
            ("It would be better if you explained why this formula works", "true"),
            ("I got the same answer", "false")
        ],
        'setup': [
            ("The audio is too quiet, can't hear properly", "true"),
            ("The board is blurry at 5:30", "true"),
            ("Great teaching method", "false")
        ],
        'gratitude': [
            ("Thank you so much for this explanation!", "true"),
            ("You're the best teacher, thanks!", "true"),
            ("I don't understand this", "false")
        ],
        'personal_experience': [
            ("I struggled with this in my calculus class last semester", "true"),
            ("When I was learning this, my teacher used a different method", "true"),
            ("What is the formula?", "false")
        ],
        'clarification': [
            ("@john Actually, you need to factor out the common term first", "true"),
            ("@sarah The answer is 42 because you multiply by 2", "true"),
            ("Great video!", "false")
        ],
        'non_english': [
            ("这个视频很有帮助", "true"),
            ("Très bien expliqué!", "true"),
            ("This is in English", "false")
        ],
        'na': [
            ("First! 😎", "true"),
            ("Anyone watching in 2025?", "true"),
            ("The integral of x^2 is x^3/3 + C", "false")
        ]
    }
    
    # Format examples for prompt
    formatted_examples = []
    for comment, label in examples.get(category, []):
        formatted_examples.append(f"Comment: {comment}\nAnswer: {label}")
    
    return "\n\n".join(formatted_examples)

def get_few_shot_prediction(comment: str,
                          video_name: str,
                          playlist_name: str, 
                          client,
                          category: str) -> Optional[bool]:
    """
    Get few-shot prediction for a single category.
    
    Args:
        comment: The YouTube comment text
        video_name: Name of the video
        playlist_name: Course playlist name
        client: OpenAI client
        category: Category to classify for
    
    Returns:
        Boolean indicating if comment belongs to category, None if failed
    """
    rubric = SIGHTRubric()
    prompt_descriptions = rubric.get_prompt_descriptions()
    
    if category not in prompt_descriptions:
        raise ValueError(f"Unknown category: {category}")
    
    # Get examples for this category
    examples = get_few_shot_examples(category)
    
    # Create few-shot prompt
    prompt = f"""You are classifying YouTube comments from educational math videos.

Category: {category}
Definition: {prompt_descriptions[category]}

Here are some examples:

{examples}

Now classify this comment:

Playlist name: {playlist_name}
Video name: {video_name}
Comment: {comment}

Answer:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing educational video comments. Respond only with 'true' or 'false'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        if result == "true":
            return True
        elif result == "false":
            return False
        else:
            print(f"Unexpected response for {category}: {result}")
            return None
            
    except Exception as e:
        print(f"Error getting few-shot prediction for {category}: {str(e)}")
        return None


def get_few_shot_prediction_all_categories(comment: str,
                                         video_name: str,
                                         playlist_name: str,
                                         client) -> List[str]:
    """
    Get few-shot predictions for all SIGHT categories.
    
    Args:
        comment: The YouTube comment text
        video_name: Name of the video
        playlist_name: Course playlist name
        client: OpenAI client
    
    Returns:
        List of categories assigned to the comment
    """
    categories = [
        'general', 'confusion', 'pedagogy', 'setup', 'gratitude',
        'personal_experience', 'clarification', 'non_english', 'na'
    ]
    
    assigned_categories = []
    
    for category in categories:
        prediction = get_few_shot_prediction(
            comment, video_name, playlist_name, client, category
        )
        
        if prediction is True:
            assigned_categories.append(category)
        
        # Rate limiting
        time.sleep(0.5)
    
    return assigned_categories