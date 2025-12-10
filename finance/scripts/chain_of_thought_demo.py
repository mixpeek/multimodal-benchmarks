#!/usr/bin/env python3
"""
Chain-of-Thought Reasoning Demo
===============================

Demonstrates how explicit step-by-step reasoning improves calculation accuracy.

This script shows:
1. Why direct prompting fails on financial calculations
2. How Chain-of-Thought (CoT) forces explicit reasoning
3. The prompt structure that achieves 76.9% calculation accuracy

Usage:
    python scripts/chain_of_thought_demo.py
"""

print("""
================================================================================
            CHAIN-OF-THOUGHT REASONING DEMO
================================================================================

PROBLEM: Direct LLM prompting on financial calculations produces:
  - Wrong formulas applied
  - Hallucinated numbers (not from sources)
  - Missing units or scale
  - Inconsistent reasoning

SOLUTION: Force explicit step-by-step reasoning with structured prompts.
""")

# ============================================================================
# DEMO: Direct Prompting vs Chain-of-Thought
# ============================================================================

print("""
================================================================================
                    DIRECT PROMPTING (BAD)
================================================================================
""")

direct_prompt = '''
PROMPT:
-------
Based on the following financial data, answer the question.

Data:
- Cash from operations (2020): $8,738 million
- Current liabilities (2020): $6,567 million
- Net income (2020): $2,345 million
- Total revenue (2020): $45,678 million

Question: What is the operating cash flow ratio for 2020?

Answer:
'''

direct_response_bad = '''
TYPICAL BAD RESPONSE:
---------------------
The operating cash flow ratio is approximately 5.2.
'''

print(direct_prompt)
print(direct_response_bad)

print("""
PROBLEMS WITH THIS RESPONSE:
1. No formula shown - we don't know what calculation was done
2. Wrong answer (5.2 doesn't match any reasonable calculation)
3. No source citation - which numbers were used?
4. No units - is this 5.2x? 5.2%?
5. May have hallucinated or used wrong formula
""")

# ============================================================================
# DEMO: Chain-of-Thought Prompting
# ============================================================================

print("""
================================================================================
                    CHAIN-OF-THOUGHT PROMPTING (GOOD)
================================================================================
""")

cot_prompt = '''
PROMPT:
-------
You are an expert financial analyst. Answer this question using ONLY the
provided financial data. Use explicit step-by-step reasoning.

Financial Data from 10-K filings:

Source 1 (Cash Flow Statement, page 45):
  "Cash from operating activities: $8,738 million"

Source 2 (Balance Sheet, page 43):
  "Total current liabilities: $6,567 million"

Source 3 (Income Statement, page 41):
  "Net income: $2,345 million"
  "Total revenue: $45,678 million"

Question: What is the operating cash flow ratio for 2020?

Answer using this EXACT structure:

## Step 1: IDENTIFY the relevant data points
[List the specific numbers, tables, and statements you need]

## Step 2: EXTRACT the exact numbers needed
[Quote the exact values from the sources above, with units and source reference]

## Step 3: PERFORM calculations (if needed)
[Show your work: formula → substitution → result]

## Step 4: VERIFY your answer makes sense
[Sanity check: Is this reasonable? Check units, scale, sign]

## Step 5: FORMAT final answer
[Provide the final answer in the requested format]

CRITICAL RULES:
- If data is missing, state "Cannot be determined from provided data"
- Always include units (millions, billions, percent)
- Show calculation steps for transparency
- ONLY use numbers from the provided sources

ANSWER:
'''

cot_response_good = '''
CHAIN-OF-THOUGHT RESPONSE:
--------------------------
## Step 1: IDENTIFY the relevant data points
To calculate the operating cash flow ratio, I need:
- Cash from operating activities (numerator)
- Total current liabilities (denominator)

Formula: Operating Cash Flow Ratio = Cash from Operations / Current Liabilities

## Step 2: EXTRACT the exact numbers needed
From Source 1 (Cash Flow Statement, page 45):
  Cash from operating activities: $8,738 million

From Source 2 (Balance Sheet, page 43):
  Total current liabilities: $6,567 million

## Step 3: PERFORM calculations
Operating Cash Flow Ratio = Cash from Operations / Current Liabilities
                         = $8,738 million / $6,567 million
                         = 1.3307...
                         = 1.33 (rounded to 2 decimal places)

## Step 4: VERIFY your answer makes sense
Checks:
- Both values are from 2020 ✓
- Both values in same units (millions) - units cancel out ✓
- Ratio of 1.33 indicates company generates $1.33 in operating cash
  for every $1.00 of current liabilities - this is healthy (>1.0 is good) ✓
- Value is reasonable for a profitable company ✓

## Step 5: FORMAT final answer
ANSWER: 1.33
'''

print(cot_prompt)
print(cot_response_good)

print("""
WHY THIS IS BETTER:

1. EXPLICIT FORMULA: Shows exactly what calculation is being done
2. SOURCE CITATIONS: Every number traced to a specific source
3. STEP-BY-STEP: Each step is verifiable
4. VERIFICATION: Sanity checks catch errors
5. TRACEABLE: If wrong, we can identify where the error occurred
""")

# ============================================================================
# DEMO: The Prompt Template
# ============================================================================

print("""
================================================================================
                    THE PROMPT TEMPLATE
================================================================================

Here's the exact prompt structure that achieves 76.9% calculation accuracy:
""")

prompt_template = '''
PROMPT_TEMPLATE = """
You are an expert financial analyst. Answer this question using ONLY the
provided financial data. Use explicit step-by-step reasoning.

Financial Data from 10-K filings:
{sources_text}

Question: {question}

Answer using this EXACT structure:

## Step 1: IDENTIFY the relevant data points
[List the specific numbers, tables, and statements you need]

## Step 2: EXTRACT the exact numbers needed
[Quote the exact values from the sources above, with units and source reference]

## Step 3: PERFORM calculations (if needed)
[Show your work: formula → substitution → result]

## Step 4: VERIFY your answer makes sense
[Sanity check: Is this reasonable? Check units, scale, sign]

## Step 5: FORMAT final answer
[Provide the final answer in the requested format]

CRITICAL RULES:
- If data is missing, state "Cannot be determined from provided data"
- Always include units (millions, billions, percent)
- Round to requested precision
- Show calculation steps for transparency
- ONLY use numbers from the provided sources

ANSWER:
"""
'''

print(prompt_template)

# ============================================================================
# DEMO: Implementation Code
# ============================================================================

print("""
================================================================================
                    IMPLEMENTATION CODE
================================================================================
""")

implementation = '''
async def _llm_cot_reasoning(
    self,
    question: str,
    sources: List[SearchResult]
) -> str:
    """Chain-of-Thought reasoning with Claude."""

    # Format sources with structure
    sources_text = self._format_sources_with_structure(sources)

    prompt = PROMPT_TEMPLATE.format(
        sources_text=sources_text,
        question=question
    )

    response = await self.claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=0,  # Deterministic for reproducibility
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text


def _format_sources_with_structure(
    self,
    sources: List[SearchResult]
) -> str:
    """Format sources with clear structure and references."""

    formatted = []

    for i, source in enumerate(sources, 1):
        # Include source reference
        formatted.append(f"Source {i} ({source.statement_type}, page {source.page}):")

        # Include the actual text
        formatted.append(f"  \"{source.text}\"")
        formatted.append("")

    return "\\n".join(formatted)
'''

print(implementation)

# ============================================================================
# DEMO: More Examples
# ============================================================================

print("""
================================================================================
                    MORE EXAMPLES
================================================================================
""")

examples = [
    {
        "question": "What is the FY2020 fixed asset turnover ratio for Nike?",
        "formula": "Revenue / Average PP&E",
        "cot": '''
## Step 1: IDENTIFY
Need: FY2020 Revenue, FY2020 PP&E, FY2019 PP&E (for average)

## Step 2: EXTRACT
From Income Statement: FY2020 Revenue = $37,403 million
From Balance Sheet: FY2020 PP&E = $4,866 million
From Balance Sheet: FY2019 PP&E = $4,744 million

## Step 3: CALCULATE
Average PP&E = ($4,866M + $4,744M) / 2 = $4,805M
Fixed Asset Turnover = $37,403M / $4,805M = 7.78

## Step 4: VERIFY
- All values from correct years ✓
- Ratio ~7.8 is reasonable for asset-light retailer ✓

ANSWER: 7.78
''',
        "actual": "7.78",
    },
    {
        "question": "What is the gross margin for 2021?",
        "formula": "(Revenue - COGS) / Revenue × 100",
        "cot": '''
## Step 1: IDENTIFY
Need: Revenue and Cost of Goods Sold (COGS) for 2021

## Step 2: EXTRACT
From Income Statement: Revenue = $33,067 million
From Income Statement: Cost of sales = $21,445 million

## Step 3: CALCULATE
Gross Profit = $33,067M - $21,445M = $11,622M
Gross Margin = $11,622M / $33,067M × 100 = 35.15%

## Step 4: VERIFY
- Both values from 2021 ✓
- Margin of ~35% is reasonable for manufacturing ✓

ANSWER: 35.15%
''',
        "actual": "35.15%",
    },
]

for ex in examples:
    print(f"Question: {ex['question']}")
    print(f"Formula: {ex['formula']}")
    print(f"\nChain-of-Thought:")
    print(ex['cot'])
    print(f"Correct Answer: {ex['actual']}")
    print("-" * 60)

# ============================================================================
# DEMO: Answer Parsing
# ============================================================================

print("""
================================================================================
                    ANSWER PARSING
================================================================================

After CoT reasoning, we need to extract the final answer:
""")

parsing_code = '''
def _parse_llm_answer(self, answer_text: str) -> str:
    """Extract final answer from CoT response."""

    # Strategy 1: Look for "ANSWER:" marker
    patterns = [
        r'ANSWER:\\s*([^\\n]+)',
        r'Final Answer:\\s*([^\\n]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, answer_text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            # Clean formatting
            answer = answer.replace('$', '').replace(',', '')
            return answer

    # Strategy 2: Extract last number (skip years)
    numbers = re.findall(r'-?[\\d,]+\\.?\\d*', answer_text)

    for num in reversed(numbers):
        value = float(num.replace(',', ''))
        # Skip year values (2015-2025)
        if not (2015 <= value <= 2025):
            return num

    # Fallback: return last line
    return answer_text.strip().split('\\n')[-1]
'''

print(parsing_code)

# ============================================================================
# DEMO: Results Comparison
# ============================================================================

print("""
================================================================================
                    RESULTS COMPARISON
================================================================================
""")

comparison_table = """
| Metric                | Without CoT | With CoT  | Improvement |
|-----------------------|-------------|-----------|-------------|
| Calculation Accuracy  | 60.0%       | 76.9%     | +16.9 pts   |
| Hallucination Rate    | ~15%        | ~5%       | -10 pts     |
| Traceable Reasoning   | No          | Yes       | -           |
| Source Citations      | Rare        | Always    | -           |
| Error Diagnosability  | Poor        | Excellent | -           |
"""

print(comparison_table)

print("""
KEY INSIGHT:

The LLM (Claude Sonnet 4) is excellent at arithmetic. The 76.9% calculation
accuracy shows that when given the right data and forced to show work,
it performs correctly most of the time.

The remaining 23.1% failures are almost always because:
1. Required data wasn't retrieved (retrieval failure, not reasoning failure)
2. Question was ambiguous
3. Formula wasn't well-known

This confirms: THE BOTTLENECK IS RETRIEVAL, NOT REASONING.
""")

# ============================================================================
# Summary
# ============================================================================

print("""
================================================================================
                    SUMMARY
================================================================================

Chain-of-Thought Reasoning provides:

1. EXPLICIT REASONING: Every step is shown and verifiable
2. SOURCE TRACING: Numbers are tied to specific retrieved chunks
3. ERROR REDUCTION: Hallucination drops from ~15% to ~5%
4. CALCULATION ACCURACY: 60% → 76.9% (+16.9 percentage points)
5. DIAGNOSABILITY: When wrong, we can identify exactly where

The structured prompt template forces:
- Step 1: Identify what's needed
- Step 2: Extract exact numbers with sources
- Step 3: Show calculation work
- Step 4: Verify reasonableness
- Step 5: Format final answer

This is why calculation accuracy (76.9%) exceeds overall accuracy (44%).
The LLM is good at math - retrieval is the bottleneck.

================================================================================
""")

if __name__ == "__main__":
    print("\nDemo complete! See TECHNICAL_GUIDE.md for full details.")
