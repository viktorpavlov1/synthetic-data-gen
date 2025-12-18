from typing import Dict, List

def distribute_counts(total_count: int, distribution: Dict[str, float]) -> Dict[str, int]:
    """
    Distribute a total count among categories based on percentages using the Largest Remainder Method.
    This ensures that the sum of counts equals the total_count and handles rounding errors gracefully.
    
    Args:
        total_count: The total number of items to distribute.
        distribution: A dictionary mapping category keys to their float probability/percentage (0.0 to 1.0).
        
    Returns:
        A dictionary mapping category keys to their integer counts.
    """
    counts = {}
    remainders = []
    
    current_sum = 0
    # Calculate integer parts and remainders
    for key, prob in distribution.items():
        val = total_count * prob
        integer_part = int(val)
        fractional_part = val - integer_part
        
        counts[key] = integer_part
        current_sum += integer_part
        remainders.append((fractional_part, key))
    
    # Calculate the difference that needs to be distributed
    diff = total_count - current_sum
    
    # Sort by remainder in descending order to give priority to largest fractions
    remainders.sort(key=lambda x: x[0], reverse=True)
    
    # Distribute the remaining count to the categories with the largest remainders
    for i in range(diff):
        if i < len(remainders):
            key = remainders[i][1]
            counts[key] += 1
            
    return counts
