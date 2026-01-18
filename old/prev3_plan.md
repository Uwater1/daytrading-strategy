# Prev3 Weighted Index Implementation Plan

## Summary
This plan outlines the changes needed to convert the `prev3` check in the Big Bar Trading Strategy from a simple binary condition to a weighted index that dynamically adjusts the entry threshold based on the direction consistency of previous bars.

## Current Implementation Analysis
In the current [`bigbar.py`](bigbar.py) strategy:
- `prev3` is calculated as the sum of the previous 3 bars' body sizes (close - open)
- It's compared against a fixed threshold: `prev3_sum >= prev3_min_ratio * body` (for long) or `prev3_sum <= -prev3_min_ratio * body` (for short)
- This is a binary check that only determines if the condition is met or not

## Key Changes Required

### 1. Weighted Index Calculation
- Assign weights to previous 3 bars:
  - Farthest bar (t-4): weight = 1
  - Middle bar (t-3): weight = 2  
  - Closest bar (t-2): weight = 3
- Calculate weighted sum: `weighted_sum = 1*(t-4) + 2*(t-3) + 3*(t-2)`

### 2. Direction Consistency Measurement
- Determine if each previous bar follows the direction of the current big bar
- For long entry (green big bar): positive body size indicates direction match
- For short entry (red big bar): negative body size indicates direction match
- Calculate a consistency score based on how many bars follow the direction

### 3. Dynamic Threshold Adjustment
- Multiply the weighted sum by a new `previous_weight` parameter (to be optimized)
- Add the result (positive or negative) to the entry threshold
- This creates a dynamic threshold that adapts based on previous bars' behavior

### 4. Parameter Optimization
- Replace the `prev3_min_ratio` parameter with `previous_weight`
- Add `previous_weight` to the optimization grid
- Update the strategy parameters and optimization function

## Implementation Steps

### Step 1: Update Strategy Parameters
```python
# In BigBarAllIn class
class BigBarAllIn(Strategy):
    # Strategy parameters (optimizable)
    atr_period = 20
    k_atr = 2.0
    uptail_max_ratio = 0.7
    previous_weight = 0.1  # New parameter to be optimized
    buffer_ratio = 0.01
```

### Step 2: Calculate Weighted Index
```python
# In next() method
def next(self):
    # ... existing code ...
    
    # Calculate weighted prev3 index
    try:
        # Weights: 1 (farthest), 2 (middle), 3 (closest)
        bar1 = (self.data.Close[-4] - self.data.Open[-4])  # t-4 (farthest)
        bar2 = (self.data.Close[-3] - self.data.Open[-3])  # t-3 (middle)  
        bar3 = (self.data.Close[-2] - self.data.Open[-2])  # t-2 (closest)
        
        weighted_sum = (1 * bar1) + (2 * bar2) + (3 * bar3)
    except Exception:
        return
```

### Step 3: Determine Direction Consistency
```python
# For long entry conditions
cond_green = close_p > open_p
# Check if previous bars follow upward direction
consistent_bars = 0
if bar1 > 0: consistent_bars += 1
if bar2 > 0: consistent_bars += 1
if bar3 > 0: consistent_bars += 1

# For short entry conditions
cond_red = close_p < open_p
# Check if previous bars follow downward direction
consistent_bars = 0
if bar1 < 0: consistent_bars += 1
if bar2 < 0: consistent_bars += 1
if bar3 < 0: consistent_bars += 1
```

### Step 4: Calculate Dynamic Threshold Adjustment
```python
# Normalize weighted sum by current bar body
normalized_weighted_sum = weighted_sum / body if body != 0 else 0

# Calculate threshold adjustment
threshold_adjustment = normalized_weighted_sum * self.previous_weight

# Dynamic entry threshold
dynamic_threshold = threshold_adjustment
```

### Step 5: Update Entry Conditions
```python
# For long entry
# Replace fixed threshold with dynamic threshold
cond_prev3_long = (normalized_weighted_sum >= self.previous_weight)  # Adjust logic based on testing

# For short entry
cond_prev3_short = (normalized_weighted_sum <= -self.previous_weight)  # Adjust logic based on testing
```

### Step 6: Update Optimization Function
```python
def optimize_strategy(filepath, return_heatmap=True):
    # ... existing code ...
    
    # Updated optimization grid
    previous_weights = [0.1, 0.2, 0.3, 0.4, 0.5]  # New parameter range
    
    # ... existing code ...
    
    # Define optimization parameters
    if return_heatmap:
        optimize_result, heatmap = bt.optimize(
            atr_period=atr_periods,
            k_atr=k_atr_values,
            uptail_max_ratio=uptail_ratios,
            previous_weight=previous_weights,  # Replace prev3_min_ratio
            maximize='Return [%]',
            constraint=lambda param: param.uptail_max_ratio > 0.5 and param.previous_weight > 0.0,
            return_heatmap=True
        )
    # ... rest of code ...
```

## Testing and Validation
1. Run backtests with different values of `previous_weight`
2. Compare performance with original strategy
3. Optimize `previous_weight` parameter
4. Validate results on different timeframes and instruments
5. Test sensitivity to parameter changes

## Expected Benefits
- More nuanced entry decision based on previous bars' behavior
- Dynamic threshold adjustment improves adaptability
- Weighted calculation gives more importance to recent bars
- Better alignment with trend continuation patterns

## Files to Modify
1. `bigbar.py` - Main strategy implementation
2. `requirements.txt` - No changes expected
3. `README.md` - Update strategy documentation

## Risks and Mitigations
- **Overfitting**: Use walk-forward optimization to validate
- **Parameter sensitivity**: Test across different market conditions
- **Performance impact**: Monitor backtest speed with new calculations
