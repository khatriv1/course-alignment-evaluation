# sight_alignment_evaluation/prompting/active_prompt.py
# FIXED: Added Self-Consistency to final predictions

import time
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple
from collections import Counter
from utils.sight_rubric import SIGHTRubric

class ActivePromptSelector:
    """Implements Active Prompting methodology for SIGHT classification"""
    
    def __init__(self, pool_size: int = 50, k_samples: int = 3, consistency_samples: int = 5):
        self.pool_size = pool_size
        self.k_samples = k_samples
        self.consistency_samples = consistency_samples  # NEW: For self-consistency
        self.uncertainty_scores = {}
        self.selected_examples = {}
        
    def estimate_uncertainty_for_category(self, comments: List[str], video_names: List[str], 
                                        playlist_names: List[str], client, category: str) -> Dict[str, float]:
        """Estimate uncertainty for comments using multiple inference passes."""
        print(f"Estimating uncertainty for {len(comments)} comments in category: {category}")
        
        uncertainty_scores = {}
        failed_count = 0
        
        for i, (comment, video_name, playlist_name) in enumerate(zip(comments, video_names, playlist_names)):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"Processing comment {i + 1}/{len(comments)}")
            
            # Get k predictions for this comment (unchanged)
            predictions = []
            for sample_idx in range(self.k_samples):
                pred = self._get_single_prediction(comment, video_name, playlist_name, client, category)
                if pred is not None:
                    predictions.append(pred)
                else:
                    failed_count += 1
                
                time.sleep(0.05)
            
            if predictions:
                unique_predictions = len(set(predictions))
                disagreement = unique_predictions / len(predictions)
                uncertainty_scores[comment] = disagreement
            else:
                uncertainty_scores[comment] = 0.0
            
            if failed_count > len(comments) * 0.3:
                print(f"⚠ Many API failures ({failed_count}), but continuing...")
        
        print(f"Completed uncertainty estimation for {category}. Failed calls: {failed_count}")
        return uncertainty_scores
    
    def _get_single_prediction(self, comment: str, video_name: str, playlist_name: str, 
                             client, category: str) -> Optional[bool]:
        """Get a single binary prediction for uncertainty estimation (unchanged)"""
        
        try:
            rubric = SIGHTRubric()
            prompt_descriptions = rubric.get_prompt_descriptions()
            
            if category not in prompt_descriptions:
                return None
            
            prompt = f"""Does this comment belong to category "{category}"?

Definition: {prompt_descriptions[category]}

Comment: "{comment[:150]}"

Answer "yes" or "no":"""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "Answer only 'yes' or 'no'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=2,
                timeout=8
            )
            
            result = response.choices[0].message.content.strip().lower()
            
            if "yes" in result:
                return True
            elif "no" in result:
                return False
            
            return None
                
        except Exception as e:
            return None
    
    def select_uncertain_comments(self, uncertainty_scores: Dict[str, float], n_select: int = 4) -> List[str]:
        """Select the most uncertain comments for annotation"""
        
        sorted_comments = sorted(uncertainty_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [comment for comment, score in sorted_comments[:n_select]]
        
        print(f"Selected {len(selected)} most uncertain comments:")
        for i, comment in enumerate(selected):
            score = uncertainty_scores[comment]
            print(f"  {i+1}. (uncertainty: {score:.3f}) {comment[:60]}...")
        
        return selected

def get_active_prompt_prediction(comment: str, video_name: str, playlist_name: str, 
                               client, category: str, 
                               selected_examples: List[Tuple[str, str]] = None,
                               use_self_consistency: bool = True,
                               consistency_samples: int = 5) -> Optional[bool]:
    """Get active prompting prediction for a single category with SELF-CONSISTENCY"""
    
    rubric = SIGHTRubric()
    prompt_descriptions = rubric.get_prompt_descriptions()
    
    if category not in prompt_descriptions:
        raise ValueError(f"Unknown category: {category}")
    
    # Format actively selected examples with CoT reasoning
    examples_text = ""
    if selected_examples and len(selected_examples) > 0:
        examples_text = "Examples from uncertain cases:\n\n"
        for ex_comment, ex_label in selected_examples[:3]:
            # Create detailed CoT reasoning
            cot_reasoning = create_cot_reasoning_sight(ex_comment, category, ex_label)
            examples_text += f'Comment: "{ex_comment[:80]}..."\n'
            examples_text += f'Reasoning: {cot_reasoning}\n'
            examples_text += f'Answer: {ex_label}\n\n'
    
    prompt = f"""Classify this comment for category: {category}

{prompt_descriptions[category]}

{examples_text}Comment to classify: "{comment[:200]}"

Instructions:
- Only answer "yes" if the comment clearly fits the category definition
- When in doubt, answer "no"
- Be conservative in your classification

Does this comment belong to category "{category}"? Answer "yes" or "no":"""

    if not use_self_consistency:
        # Single prediction (original method)
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You classify YouTube comments. Be conservative - only answer 'yes' if you're confident the comment fits the category. Answer only 'yes' or 'no'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=3,
                timeout=15
            )
            
            result = response.choices[0].message.content.strip().lower()
            return "yes" in result
                
        except Exception as e:
            print(f"Error getting active prompt prediction for {category}: {str(e)}")
            return False
    
    else:
        # NEW: SELF-CONSISTENCY - Multiple predictions + most common answer
        predictions = []
        
        for i in range(consistency_samples):
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo-0125",
                    messages=[
                        {"role": "system", "content": "You classify YouTube comments. Be conservative - only answer 'yes' if you're confident the comment fits the category. Answer only 'yes' or 'no'."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,  # Higher temperature for diversity
                    max_tokens=3,
                    timeout=15
                )
                
                result = response.choices[0].message.content.strip().lower()
                if "yes" in result:
                    predictions.append(True)
                elif "no" in result:
                    predictions.append(False)
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"Error in self-consistency sample {i+1}: {str(e)}")
                continue
        
        if not predictions:
            return False
        
        # Take most common answer (SELF-CONSISTENCY)
        counter = Counter(predictions)
        most_common_answer = counter.most_common(1)[0][0]
        
        print(f"Self-consistency for {category}: {predictions} → {most_common_answer}")
        return most_common_answer

def get_active_prompt_prediction_all_categories(comment: str, video_name: str, playlist_name: str,
                                              client, 
                                              uncertainty_examples: Dict[str, List[Tuple[str, str]]] = None,
                                              use_self_consistency: bool = True,
                                              consistency_samples: int = 5) -> List[str]:
    """Get active prompting predictions for all SIGHT categories with SELF-CONSISTENCY"""
    categories = [
        'general', 'confusion', 'pedagogy', 'setup', 'gratitude',
        'personal_experience', 'clarification', 'non_english', 'na'
    ]
    
    category_scores = {}
    
    # Get predictions for each category with self-consistency
    for category in categories:
        category_examples = uncertainty_examples.get(category, []) if uncertainty_examples else []
        
        prediction = get_active_prompt_prediction(
            comment, video_name, playlist_name, client, category, category_examples,
            use_self_consistency=use_self_consistency,
            consistency_samples=consistency_samples
        )
        
        category_scores[category] = prediction
        time.sleep(0.1)
    
    # CONSERVATIVE MULTI-LABEL ASSIGNMENT (unchanged)
    assigned_categories = []
    
    primary_cats = ['confusion', 'non_english', 'clarification']
    for cat in primary_cats:
        if category_scores.get(cat, False):
            assigned_categories.append(cat)
    
    if not assigned_categories:
        secondary_cats = ['general', 'gratitude', 'pedagogy', 'setup', 'personal_experience']
        for cat in secondary_cats:
            if category_scores.get(cat, False):
                assigned_categories.append(cat)
    
    if not assigned_categories:
        if category_scores.get('na', False):
            assigned_categories.append('na')
        else:
            assigned_categories.append('na')
    
    return assigned_categories

def create_cot_reasoning_sight(comment: str, category: str, label: str) -> str:
    """Create detailed Chain-of-Thought reasoning for SIGHT examples"""
    
    # Extract key features
    comment_lower = comment.lower()
    
    if label == "yes":
        if category == "confusion":
            if any(word in comment_lower for word in ['confused', 'confusing', 'understand', 'unclear', 'help']):
                return "Let me analyze step by step: 1) The comment contains explicit confusion indicators like 'confused' or 'don't understand', 2) This shows the student is experiencing uncertainty about the content, 3) This type of language indicates a need for clarification. Therefore, this belongs to the confusion category."
            else:
                return "Let me analyze step by step: 1) While not using explicit confusion words, the tone suggests uncertainty, 2) The context implies the student needs additional explanation, 3) This represents cognitive struggle with the material. Therefore, this belongs to the confusion category."
        
        elif category == "general":
            if any(word in comment_lower for word in ['good', 'great', 'nice', 'excellent', 'awesome', 'love']):
                return "Let me analyze step by step: 1) The comment contains positive language about the content, 2) This is a general appreciation statement, 3) It doesn't ask questions or request specific help, 4) This represents overall feedback about the video. Therefore, this belongs to the general category."
            else:
                return "Let me analyze step by step: 1) The comment provides general feedback about the content, 2) It doesn't fall into specific categories like confusion or technical setup, 3) This represents a broad comment about the educational material. Therefore, this belongs to the general category."
        
        elif category == "pedagogy":
            return "Let me analyze step by step: 1) The comment discusses teaching methods or educational approaches, 2) It refers to how the material is presented or explained, 3) This focuses on the instructional aspect rather than content. Therefore, this belongs to the pedagogy category."
        
        else:
            return f"Let me analyze step by step: 1) The comment shows characteristics specific to {category}, 2) The language and context align with this category's definition, 3) This represents the type of feedback typical for {category}. Therefore, this belongs to the {category} category."
    
    else:  # label == "no"
        return f"Let me analyze step by step: 1) While the comment might seem related, it doesn't show the key characteristics of {category}, 2) The language and tone are more aligned with a different category, 3) The specific indicators for {category} are not present. Therefore, this does NOT belong to the {category} category."

# Rest of the functions remain the same...
def create_active_examples_for_category(selected_comments: List[str], ground_truth_data: pd.DataFrame, 
                                      category: str) -> List[Tuple[str, str]]:
    """Create examples from selected uncertain comments using ground truth annotations."""
    examples = []
    
    h1_col = f'annotator_H1_{category}'
    h2_col = f'annotator_H2_{category}'
    
    for comment in selected_comments:
        matching_rows = ground_truth_data[ground_truth_data['comment'] == comment]
        
        if not matching_rows.empty:
            row = matching_rows.iloc[0]
            
            h1_vote = row.get(h1_col, 0)
            h2_vote = row.get(h2_col, 0)
            
            if pd.isna(h1_vote):
                h1_vote = 0
            if pd.isna(h2_vote):
                h2_vote = 0
                
            label = "yes" if (h1_vote + h2_vote) >= 1 else "no"
            examples.append((comment, label))
    
    return examples

def prepare_active_prompting_data(df: pd.DataFrame, client, n_examples: int = 4) -> Dict[str, List[Tuple[str, str]]]:
    """Prepare active prompting examples (unchanged)"""
    print("Preparing Active Prompting data for SIGHT classification...")
    
    categories = ['general', 'confusion', 'pedagogy']
    all_categories = ['general', 'confusion', 'pedagogy', 'setup', 'gratitude',
                     'personal_experience', 'clarification', 'non_english', 'na']
    
    max_samples = min(len(df), 50)
    if len(df) > max_samples:
        sample_indices = np.random.choice(len(df), max_samples, replace=False)
        comments = [df.iloc[i]['comment'] for i in sample_indices]
        video_names = [df.iloc[i]['video_name'] for i in sample_indices]
        playlist_names = [df.iloc[i]['playlist_name'] for i in sample_indices]
        df_sample = df.iloc[sample_indices]
    else:
        comments = df['comment'].tolist()
        video_names = df['video_name'].tolist()
        playlist_names = df['playlist_name'].tolist()
        df_sample = df
    
    print(f"Using {len(comments)} comments for uncertainty estimation")
    
    active_examples = {}
    selector = ActivePromptSelector(k_samples=3)
    
    for category in categories:
        print(f"\n{'='*50}")
        print(f"Processing category: {category}")
        print(f"{'='*50}")
        
        try:
            uncertainty_scores = selector.estimate_uncertainty_for_category(
                comments, video_names, playlist_names, client, category
            )
            
            if not uncertainty_scores:
                raise Exception("No uncertainty scores")
            
            selected_comments = selector.select_uncertain_comments(uncertainty_scores, n_examples)
            examples = create_active_examples_for_category(selected_comments, df_sample, category)
            active_examples[category] = examples
            
            print(f"✓ Created {len(examples)} examples for {category}")
            
        except Exception as e:
            print(f"⚠ Error processing {category}: {e}")
            active_examples[category] = []
    
    remaining_categories = ['setup', 'gratitude', 'personal_experience', 'clarification', 'non_english', 'na']
    for category in remaining_categories:
        active_examples[category] = []
    
    print(f"\n✓ Active Prompting preparation completed!")
    return active_examples

def prepare_active_prompting_data_fallback(df: pd.DataFrame, client=None, n_examples: int = 4) -> Dict[str, List[Tuple[str, str]]]:
    """FALLBACK: Skip uncertainty estimation and use random sampling"""
    print("⚠ Using FALLBACK mode - random sampling with ground truth...")
    
    categories = ['general', 'confusion', 'pedagogy', 'setup', 'gratitude',
                 'personal_experience', 'clarification', 'non_english', 'na']
    
    active_examples = {}
    
    for category in categories:
        h1_col = f'annotator_H1_{category}'
        examples = []
        
        if h1_col in df.columns:
            positive_df = df[df[h1_col] == 1]
            negative_df = df[df[h1_col] == 0]
            
            if len(positive_df) > 0:
                pos_samples = positive_df.sample(n=min(n_examples//2, len(positive_df)), random_state=42)
                examples.extend([(row['comment'], "yes") for _, row in pos_samples.iterrows()])
            
            if len(negative_df) > 0:
                neg_samples = negative_df.sample(n=min(n_examples//2, len(negative_df)), random_state=42)
                examples.extend([(row['comment'], "no") for _, row in neg_samples.iterrows()])
        
        active_examples[category] = examples
    
    return active_examples