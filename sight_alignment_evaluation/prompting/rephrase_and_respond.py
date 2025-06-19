# sight_alignment_evaluation/prompting/rephrase_and_respond.py

"""
Rephrase and Respond prompting for SIGHT comment classification.
First rephrases the comment to better understand it, then classifies.
"""

import time
from typing import List, Optional, Tuple
from utils.sight_rubric import SIGHTRubric

def rephrase_comment(comment: str, client) -> Optional[str]:
    """
    Rephrase the comment to clarify its meaning.
    
    Args:
        comment: The original comment
        client: OpenAI client
    
    Returns:
        Rephrased comment or None if failed
    """
    prompt = f"""Rephrase the following YouTube comment to make its intent and meaning clearer, while preserving all important information:

Original comment: "{comment}"

Rephrased comment:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at understanding and clarifying the meaning of comments. Rephrase to make intent clear while preserving all information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error rephrasing comment: {str(e)}")
        return None

def get_rephrase_respond_prediction(comment: str,
                                  video_name: str,
                                  playlist_name: str, 
                                  client,
                                  category: str) -> Optional[bool]:
    """
    Get Rephrase and Respond prediction for a single category.
    
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
    
    # Step 1: Rephrase the comment
    rephrased = rephrase_comment(comment, client)
    if rephrased is None:
        print(f"Failed to rephrase comment for {category}, using original")
        rephrased = comment
    
    # Step 2: Create classification prompt with both original and rephrased
    prompt = f"""You are classifying YouTube comments from educational math videos.

Category: {category}
Definition: {prompt_descriptions[category]}

Video Context:
- Playlist: {playlist_name}
- Video: {video_name}

Original comment: "{comment}"
Rephrased for clarity: "{rephrased}"

Based on both the original comment and its clarified meaning, does this comment belong to the '{category}' category?

Answer with 'true' if it belongs to this category, or 'false' if it does not."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing educational video comments. Consider both the original and rephrased versions to make accurate classifications. Respond only with 'true' or 'false'."},
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
        print(f"Error getting rephrase-respond prediction for {category}: {str(e)}")
        return None


def get_rephrase_respond_prediction_all_categories(comment: str,
                                                 video_name: str,
                                                 playlist_name: str,
                                                 client) -> List[str]:
    """
    Get Rephrase and Respond predictions for all SIGHT categories.
    
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
    
    # First, get the rephrased version once (to avoid rephrasing multiple times)
    rephrased = rephrase_comment(comment, client)
    if rephrased is None:
        rephrased = comment
    
    for category in categories:
        # For efficiency, we'll use the already rephrased version
        rubric = SIGHTRubric()
        prompt_descriptions = rubric.get_prompt_descriptions()
        
        prompt = f"""You are classifying YouTube comments from educational math videos.

Category: {category}
Definition: {prompt_descriptions[category]}

Video Context:
- Playlist: {playlist_name}
- Video: {video_name}

Original comment: "{comment}"
Rephrased for clarity: "{rephrased}"

Based on both the original comment and its clarified meaning, does this comment belong to the '{category}' category?

Answer with 'true' if it belongs to this category, or 'false' if it does not."""

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing educational video comments. Consider both the original and rephrased versions to make accurate classifications. Respond only with 'true' or 'false'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip().lower()
            
            if result == "true":
                assigned_categories.append(category)
                
        except Exception as e:
            print(f"Error getting rephrase-respond prediction for {category}: {str(e)}")
            continue
        
        # Rate limiting
        time.sleep(0.5)
    
    return assigned_categories