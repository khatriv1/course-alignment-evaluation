# sight_alignment_evaluation/prompting/take_a_step_back.py

"""
Take a Step Back prompting for SIGHT comment classification.
First derives high-level principles, then applies them to classification.
"""

import time
from typing import List, Optional, Dict
from utils.sight_rubric import SIGHTRubric

def derive_classification_principles(category: str, client) -> Optional[str]:
    """
    Derive high-level principles for classifying a category.
    
    Args:
        category: The category to derive principles for
        client: OpenAI client
    
    Returns:
        String containing derived principles or None if failed
    """
    rubric = SIGHTRubric()
    prompt_descriptions = rubric.get_prompt_descriptions()
    
    prompt = f"""Take a step back and think about the fundamental principles for identifying comments in the '{category}' category.

Category definition: {prompt_descriptions[category]}

What are the key characteristics, patterns, and principles that would help identify if a YouTube comment on an educational math video belongs to this category? 

List 3-5 high-level principles:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing patterns in educational video comments. Derive clear, high-level principles."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error deriving principles for {category}: {str(e)}")
        return None

def get_take_step_back_prediction(comment: str,
                                video_name: str,
                                playlist_name: str, 
                                client,
                                category: str) -> Optional[bool]:
    """
    Get Take a Step Back prediction for a single category.
    
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
    
    # Step 1: Derive high-level principles
    principles = derive_classification_principles(category, client)
    if principles is None:
        print(f"Failed to derive principles for {category}")
        principles = "Consider the category definition carefully."
    
    # Step 2: Apply principles to classify
    prompt = f"""You are classifying YouTube comments from educational math videos.

Category: {category}
Definition: {prompt_descriptions[category]}

High-level principles for this category:
{principles}

Now apply these principles to classify this specific comment:

Playlist name: {playlist_name}
Video name: {video_name}
Comment: {comment}

Based on the principles above, does this comment belong to the '{category}' category?
Answer 'true' or 'false'."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing educational video comments. Apply the given principles to make accurate classifications. Respond only with 'true' or 'false'."},
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
        print(f"Error getting take-step-back prediction for {category}: {str(e)}")
        return None


def get_take_step_back_prediction_all_categories(comment: str,
                                               video_name: str,
                                               playlist_name: str,
                                               client) -> List[str]:
    """
    Get Take a Step Back predictions for all SIGHT categories.
    
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
    
    # Pre-derive principles for all categories to be efficient
    principles_cache = {}
    print("Deriving classification principles for all categories...")
    for category in categories:
        principles = derive_classification_principles(category, client)
        if principles:
            principles_cache[category] = principles
        else:
            # Fallback principles if derivation fails
            principles_cache[category] = f"Consider if the comment fits the definition of {category}."
        time.sleep(0.3)  # Rate limiting for principle derivation
    
    # Now classify using cached principles
    for category in categories:
        rubric = SIGHTRubric()
        prompt_descriptions = rubric.get_prompt_descriptions()
        
        prompt = f"""You are classifying YouTube comments from educational math videos.

Category: {category}
Definition: {prompt_descriptions[category]}

High-level principles for this category:
{principles_cache[category]}

Now apply these principles to classify this specific comment:

Playlist name: {playlist_name}
Video name: {video_name}
Comment: {comment}

Based on the principles above, does this comment belong to the '{category}' category?
Answer 'true' or 'false'."""

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing educational video comments. Apply the given principles to make accurate classifications. Respond only with 'true' or 'false'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip().lower()
            
            if result == "true":
                assigned_categories.append(category)
                
        except Exception as e:
            print(f"Error getting take-step-back prediction for {category}: {str(e)}")
            continue
        
        # Rate limiting
        time.sleep(0.5)
    
    return assigned_categories