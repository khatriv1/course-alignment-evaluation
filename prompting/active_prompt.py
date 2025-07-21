# course_alignment_evaluation/prompting/active_prompt.py
# FIXED: Added Self-Consistency to final predictions

import re
import time
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple
from collections import Counter

class ActivePromptSelector:
    """Implements Active Prompting methodology with MINIMAL API calls"""
    
    def __init__(self, pool_size: int = 20, k_samples: int = 2, consistency_samples: int = 5):
        self.pool_size = pool_size
        self.k_samples = k_samples
        self.consistency_samples = consistency_samples  # NEW: For self-consistency
        self.uncertainty_scores = {}
        
    def estimate_uncertainty(self, course_data: List[Tuple[str, str, str]], client) -> Dict[Tuple[str, str, str], float]:
        """Estimate uncertainty with MINIMAL API calls (unchanged)"""
        print(f"Estimating uncertainty for {len(course_data)} course-objective pairs")
        
        uncertainty_scores = {}
        
        for i, (course_title, description, learning_objective) in enumerate(course_data):
            if (i + 1) % 5 == 0:
                print(f"Processing pair {i + 1}/{len(course_data)}")
            
            predictions = []
            for sample_idx in range(self.k_samples):
                pred = self._get_single_prediction(course_title, description, learning_objective, client)
                if pred is not None:
                    predictions.append(pred)
                time.sleep(0.1)
            
            if predictions:
                unique_predictions = len(set(predictions))
                disagreement = unique_predictions / len(predictions)
                uncertainty_scores[(course_title, description, learning_objective)] = disagreement
            else:
                uncertainty_scores[(course_title, description, learning_objective)] = 0.0
        
        return uncertainty_scores
    
    def _get_single_prediction(self, course_title: str, description: str, learning_objective: str, client) -> Optional[int]:
        """Get a single score prediction with SIMPLE prompt (unchanged)"""
        
        prompt = f"""Rate course-objective alignment (1-5):

Course: {course_title[:50]}
Objective: {learning_objective[:100]}

1=Unrelated, 5=Perfect match. Answer:"""

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "Answer only with a number 1-5."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=2,
                timeout=8
            )
            
            result = response.choices[0].message.content.strip()
            score_match = re.search(r'[1-5]', result)
            if score_match:
                return int(score_match.group(0))
                
        except Exception as e:
            pass
            
        return None
    
    def select_uncertain_pairs(self, uncertainty_scores: Dict[Tuple[str, str, str], float], n_select: int = 3) -> List[Tuple[str, str, str]]:
        """Select the most uncertain pairs (unchanged)"""
        sorted_pairs = sorted(uncertainty_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [pair for pair, score in sorted_pairs[:n_select]]
        
        print(f"Selected {len(selected)} most uncertain pairs:")
        for i, (title, desc, obj) in enumerate(selected):
            score = uncertainty_scores[(title, desc, obj)]
            print(f"  {i+1}. (uncertainty: {score:.3f}) {title[:30]} - {obj[:40]}...")
        
        return selected

def prepare_active_prompting_data(df: pd.DataFrame, client, n_examples: int = 3) -> List[Dict]:
    """Prepare active prompting examples with MINIMAL uncertainty estimation (unchanged)"""
    print("Preparing Active Prompting data (MINIMAL VERSION)...")
    
    max_samples = min(len(df), 10)
    sample_df = df.sample(n=max_samples, random_state=42)
    
    course_data = []
    for _, row in sample_df.iterrows():
        course_data.append((
            row['course_title'],
            row['description'], 
            row['learning_objective']
        ))
    
    try:
        selector = ActivePromptSelector(k_samples=2)
        
        uncertainty_scores = selector.estimate_uncertainty(course_data, client)
        
        if not uncertainty_scores:
            raise Exception("No uncertainty scores")
        
        selected_pairs = selector.select_uncertain_pairs(uncertainty_scores, n_examples)
        
        examples = []
        for course_title, description, learning_objective in selected_pairs:
            matching_rows = sample_df[
                (sample_df['course_title'] == course_title) & 
                (sample_df['learning_objective'] == learning_objective)
            ]
            
            if not matching_rows.empty:
                score = matching_rows.iloc[0]['human_score']
                
                if score >= 4:
                    reasoning = "Strong alignment - course content directly supports the objective."
                elif score == 3:
                    reasoning = "Moderate alignment - transferable skills but not direct coverage."
                elif score == 2:
                    reasoning = "Weak alignment - same domain but limited relevance."
                else:
                    reasoning = "Poor alignment - minimal connection between course and objective."
                
                examples.append({
                    'course': course_title,
                    'description': description,
                    'objective': learning_objective,
                    'reasoning': reasoning,
                    'score': score
                })
        
        print(f"✓ Created {len(examples)} active prompting examples")
        return examples
        
    except Exception as e:
        print(f"Error in uncertainty estimation: {e}")
        print("Using fallback examples...")
        
        examples = []
        sample_pairs = sample_df.head(n_examples)
        
        for _, row in sample_pairs.iterrows():
            score = row['human_score']
            
            if score >= 4:
                reasoning = "Strong alignment - course content directly supports the objective."
            elif score == 3:
                reasoning = "Moderate alignment - transferable skills but not direct coverage."
            elif score == 2:
                reasoning = "Weak alignment - same domain but limited relevance."
            else:
                reasoning = "Poor alignment - minimal connection between course and objective."
            
            examples.append({
                'course': row['course_title'],
                'description': row['description'],
                'objective': row['learning_objective'],
                'reasoning': reasoning,
                'score': score
            })
        
        print(f"✓ Created {len(examples)} fallback examples")
        return examples

def calculate_uncertainty(responses):
    """Calculate uncertainty metrics for Active Prompting selection (unchanged)"""
    if not responses or len(responses) < 2:
        return {
            "disagreement": 0,
            "entropy": 0,
            "confidence": 1.0
        }
    
    scores = np.array([r for r in responses if r is not None])
    
    if len(scores) == 0:
        return {
            "disagreement": 0,
            "entropy": 0,
            "confidence": 1.0
        }
    
    # Disagreement metric (variance)
    disagreement = np.var(scores)
    
    # Calculate probability distribution
    unique, counts = np.unique(scores, return_counts=True)
    probs = counts / len(scores)
    
    # Entropy metric
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    
    # Self-confidence metric
    most_common_idx = np.argmax(counts)
    most_common = unique[most_common_idx]
    confidence = 1.0 - np.mean(np.abs(scores - most_common)) / 4.0
    
    return {
        "disagreement": disagreement,
        "entropy": entropy,
        "confidence": confidence
    }

def get_examples_for_active_prompt(example_pool, client, num_examples=2):
    """Select the most informative examples with MINIMAL API calls (unchanged)"""
    if len(example_pool) <= num_examples:
        return example_pool
    
    return example_pool[:num_examples]

def create_active_prompt(course_title, description, learning_objective, cot_examples=None):
    """Create an Active Prompt with SIMPLIFIED structure (unchanged)"""
    
    base_prompt = f"""Rate course-objective alignment (1-5):

Course: {course_title}
Description: {description[:200]}
Objective: {learning_objective}

SCALE:
5=Perfect match, 4=Strong, 3=Moderate, 2=Weak, 1=None"""

    if cot_examples:
        example_text = "\n\nExamples:\n"
        
        for i, ex in enumerate(cot_examples[:2]):
            # Create detailed CoT reasoning
            detailed_reasoning = create_cot_reasoning_alignment(ex['course'], ex['objective'], ex['reasoning'], ex['score'])
            example_text += f"Course: {ex['course'][:30]}...\n"
            example_text += f"Objective: {ex['objective'][:50]}...\n"
            example_text += f"Reasoning: {detailed_reasoning}\n"
            example_text += f"Score: {ex['score']}\n\n"
        
        prompt = f"{base_prompt}{example_text}Your score:"
    else:
        prompt = f"{base_prompt}\n\nYour score:"
    
    return prompt

def create_cot_reasoning_alignment(course: str, objective: str, basic_reasoning: str, score: int) -> str:
    """Create detailed Chain-of-Thought reasoning for course alignment examples"""
    
    course_lower = course.lower()
    objective_lower = objective.lower()
    
    # Find overlapping concepts
    course_words = set(course_lower.split())
    objective_words = set(objective_lower.split())
    overlap = course_words.intersection(objective_words)
    overlap = [word for word in overlap if len(word) > 3]  # Filter out short words
    
    if score >= 4:  # Strong alignment
        if overlap:
            detailed_reasoning = f"Let me analyze this step by step: 1) I identify shared concepts between course and objective: '{', '.join(overlap[:2])}', 2) The course content directly addresses the skills needed for this objective, 3) There's clear conceptual overlap and the course provides relevant knowledge, 4) Students completing this course would gain the specific competencies required. Therefore, this shows strong alignment (Score: {score})."
        else:
            detailed_reasoning = f"Let me analyze this step by step: 1) While specific words don't overlap, the course content is highly relevant to the objective, 2) The skills taught in the course directly transfer to achieving this objective, 3) The course provides the foundational knowledge and practical abilities needed, 4) There's excellent conceptual alignment between course outcomes and this objective. Therefore, this shows strong alignment (Score: {score})."
    
    elif score == 3:  # Moderate alignment
        detailed_reasoning = f"Let me analyze this step by step: 1) The course covers some relevant content for this objective, 2) There are transferable skills but not direct coverage of all required competencies, 3) Students would gain some relevant knowledge but might need additional preparation, 4) The course provides a foundation but doesn't fully address all aspects of the objective. Therefore, this shows moderate alignment (Score: {score})."
    
    elif score == 2:  # Weak alignment
        detailed_reasoning = f"Let me analyze this step by step: 1) The course and objective are in related domains but with limited overlap, 2) Some general skills might transfer but specific competencies are not addressed, 3) The course provides background knowledge but doesn't directly prepare students for this objective, 4) Significant additional learning would be needed to achieve the objective. Therefore, this shows weak alignment (Score: {score})."
    
    else:  # Poor alignment (score 1)
        detailed_reasoning = f"Let me analyze this step by step: 1) The course content is largely unrelated to the objective requirements, 2) There's minimal skill transfer or conceptual overlap, 3) The course doesn't provide relevant preparation for achieving this objective, 4) Students would need entirely different coursework to develop these competencies. Therefore, this shows poor alignment (Score: {score})."
    
    return detailed_reasoning

def extract_score(response):
    """Extract numerical score from model response (unchanged)"""
    if not response:
        return None
        
    matches = re.findall(r'[1-5]', response)
    if matches:
        return int(matches[0])
        
    return None

def get_active_prompt_prediction(course_title, description, learning_objective, client, 
                                uncertainty_examples=None,
                                use_self_consistency: bool = True,
                                consistency_samples: int = 5):
    """Get prediction using Active Prompting with SELF-CONSISTENCY"""
    
    # Use provided examples or create simple defaults
    if not uncertainty_examples:
        uncertainty_examples = [
            {
                "course": "Data Science",
                "description": "Introduction to data analysis and machine learning",
                "objective": "Build predictive models",
                "reasoning": "Course covers machine learning, directly relevant.",
                "score": 4
            },
            {
                "course": "History",
                "description": "Medieval European history",
                "objective": "Build predictive models",
                "reasoning": "No connection between history and data modeling.",
                "score": 1
            }
        ]
    
    # Create prompt with examples
    prompt = create_active_prompt(course_title, description, learning_objective, uncertainty_examples)
    
    if not use_self_consistency:
        # Single prediction (original method)
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "Rate course-objective alignment. Answer only with a number 1-5."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=10,
                timeout=10
            )
            
            return extract_score(response.choices[0].message.content)
            
        except Exception as e:
            print(f"Error getting prediction: {str(e)}")
            return None
    
    else:
        # NEW: SELF-CONSISTENCY - Multiple predictions + most common answer
        predictions = []
        
        for i in range(consistency_samples):
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo-0125",
                    messages=[
                        {"role": "system", "content": "Rate course-objective alignment. Answer only with a number 1-5."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,  # Higher temperature for diversity
                    max_tokens=10,
                    timeout=10
                )
                
                score = extract_score(response.choices[0].message.content)
                if score is not None:
                    predictions.append(score)
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error in self-consistency sample {i+1}: {str(e)}")
                continue
        
        if not predictions:
            return None
        
        # Take most common score (SELF-CONSISTENCY)
        counter = Counter(predictions)
        most_common_score = counter.most_common(1)[0][0]
        
        print(f"Self-consistency for alignment: {predictions} → {most_common_score}")
        return most_common_score