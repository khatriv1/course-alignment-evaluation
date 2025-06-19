# sight_alignment_evaluation/prompting/contrastive_cot.py

"""
Contrastive Chain of Thought prompting for SIGHT comment classification.
Uses both positive and negative reasoning to improve classification.
"""

import time
from typing import List, Optional, Dict
from utils.sight_rubric import SIGHTRubric

def get_contrastive_examples(category: str) -> str:
    """Generate contrastive reasoning examples for each category."""
    contrastive_examples = {
        'general': {
            "positive": {
                "comment": "Amazing lecture, the professor is brilliant!",
                "positive_reasoning": "This expresses a general positive opinion about both the lecture and professor without focusing on specific content.",
                "negative_reasoning": "This is NOT about confusion (no questions asked), NOT about pedagogy (doesn't mention teaching methods), NOT setup-related (no technical issues).",
                "answer": "true"
            },
            "negative": {
                "comment": "How do I solve for x in this equation?",
                "positive_reasoning": "This could seem like engagement with the video.",
                "negative_reasoning": "However, this is a specific math question (confusion category), NOT a general opinion about the video or teaching.",
                "answer": "false"
            }
        },
        'confusion': {
            "positive": {
                "comment": "Can someone explain why the limit equals infinity here?",
                "positive_reasoning": "This asks a specific mathematical question about limits, showing confusion about the content.",
                "negative_reasoning": "This is NOT general praise, NOT about teaching style, NOT a technical issue - it's specifically about mathematical understanding.",
                "answer": "true"
            },
            "negative": {
                "comment": "The way you break down problems is excellent",
                "positive_reasoning": "This mentions how problems are explained.",
                "negative_reasoning": "However, this is praising the teaching method (pedagogy), NOT expressing confusion or asking for help.",
                "answer": "false"
            }
        },
        'pedagogy': {
            "positive": {
                "comment": "Your step-by-step approach makes complex topics accessible",
                "positive_reasoning": "This specifically praises the instructional method (step-by-step approach) and its effectiveness.",
                "negative_reasoning": "This is NOT just general praise, NOT confusion, NOT about technical setup - it's specifically about teaching methodology.",
                "answer": "true"
            },
            "negative": {
                "comment": "Thank you so much!",
                "positive_reasoning": "This expresses appreciation.",
                "negative_reasoning": "However, this is gratitude, NOT specific feedback about teaching methods or instructional techniques.",
                "answer": "false"
            }
        }
    }
    
    # Get examples for the category
    if category not in contrastive_examples:
        # Provide default structure
        return "No contrastive examples available for this category."
    
    examples = contrastive_examples[category]
    
    # Format contrastive examples
    formatted = []
    for example_type, example in examples.items():
        formatted.append(f"Example ({example_type}):\nComment: {example['comment']}\nPositive reasoning: {example['positive_reasoning']}\nNegative reasoning: {example['negative_reasoning']}\nAnswer: {example['answer']}")
    
    return "\n\n".join(formatted)

def get_contrastive_cot_prediction(comment: str,
                                 video_name: str,
                                 playlist_name: str, 
                                 client,
                                 category: str) -> Optional[bool]:
    """
    Get Contrastive CoT prediction for a single category.
    
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
    
    # Get contrastive examples
    contrastive_examples = get_contrastive_examples(category)
    
    # Create Contrastive CoT prompt
    prompt = f"""You are classifying YouTube comments from educational math videos.

Category: {category}
Definition: {prompt_descriptions[category]}

Here are contrastive examples showing both why comments do and don't belong to this category:

{contrastive_examples}

Now classify this comment using both positive and negative reasoning:

Playlist name: {playlist_name}
Video name: {video_name}
Comment: {comment}

First, provide positive reasoning (why it MIGHT belong to {category}).
Then, provide negative reasoning (why it might NOT belong to {category}).
Finally, based on both reasonings, answer with 'true' or 'false'."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing educational video comments. Use contrastive reasoning to make accurate classifications. Always end with 'true' or 'false'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=250
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        # Extract final answer
        if "true" in result.split()[-5:]:
            return True
        elif "false" in result.split()[-5:]:
            return False
        else:
            print(f"Could not extract answer from Contrastive CoT response for {category}: {result}")
            return None
            
    except Exception as e:
        print(f"Error getting Contrastive CoT prediction for {category}: {str(e)}")
        return None


def get_contrastive_cot_prediction_all_categories(comment: str,
                                                video_name: str,
                                                playlist_name: str,
                                                client) -> List[str]:
    """
    Get Contrastive CoT predictions for all SIGHT categories.
    
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
        prediction = get_contrastive_cot_prediction(
            comment, video_name, playlist_name, client, category
        )
        
        if prediction is True:
            assigned_categories.append(category)
        
        # Rate limiting
        time.sleep(0.5)
    
    return assigned_categories