# C-RES Prompt Templates

This appendix gives the prompt templates used at each step of C-RES, referenced
from the Method. Highlighted phrases in the paper mark what each prompt type
adds; here they appear as plain text. Placeholders in braces (`{country}`,
`{story}`, `{rot}`, `{value}`) are filled per item from NormAd-ETI.

--------------------------------------------------------------------------------
## Step 1 — Baseline prompts

### Baseline Prompt (SC)
```
Task: You will be given a short story involving 2 or more characters about
everyday situations. You will also be given a country of relevance. Read the
story and reason if the action by the character in the story is socially
acceptable in the given country. The answer options are Yes, No or Neither.

Country: {country}
Story: {story}

Answer:
Explanation:
```

### Baseline Prompt (SCV)
```
Task: You will be given a short story involving 2 or more characters about
everyday situations. You will also be given a country of relevance, along with
a rule-of-thumb. Read the story and reason if the action by the character in
the story is socially acceptable in the given country with the associated
rule-of-thumb. The answer options are Yes, No or Neither.

Country: {country}
RoT: {value}
Story: {story}

Answer:
Explanation:
```

### Baseline Prompt (SRoT)
The country is deliberately withheld from the model. Retrieval, however, is
still conditioned on the NormAd country label, so the cultural evidence supplied
in Step 2 is country-matched even though the model is country-blind. This is
what makes SRoT the cleanest probe for over-culturalisation.
```
Task: You will be given a short story involving 2 or more characters about
everyday situations. You will also be given a rule-of-thumb. Read the story and
reason if the action by the character in the story is socially acceptable under
the given rule-of-thumb alone. The answer options are Yes, No or Neither.

Rule-of-thumb: {rot}
Story: {story}

Answer:
Explanation:
```

--------------------------------------------------------------------------------
## Step 2 — Evidence retrieval

### Source Selection (One-shot tool selection)
```
You are about to analyse a cultural scenario.
Before answering, decide which research tools (if any) would help.

Available tools (choose by exact name):
  -- hofstede_tool : Quantitative cultural dimension scores (0-100) --- power
     distance, individualism, masculinity, uncertainty avoidance, long-term
     orientation, indulgence. Best for questions about cultural values and
     behavioural tendencies.
  -- cultural_atlas_tool : Detailed cultural practices, etiquette, communication
     styles, and social norms by country. Best for specific behavioural rules
     and social expectations.
  -- wikipedia_rag : General cultural background, history, religion, and social
     context retrieved from Wikipedia. Best for broad cultural context.

Select between 0 and 3 tools. Only select tools that will genuinely help. If no
tool is needed, write 'none'.

Format your response as:
REASONING: <brief explanation of why you chose these tools>
SELECTED_TOOLS: <comma-separated tool names, or 'none'>
```

### Thought and Action (ReAct)
```
You previously analysed a cultural scenario and gave an initial answer with
reasoning. You can now use research tools iteratively to gather evidence before
finalising your answer.

Do not make any extra inferences about actions outside of the given context and
the tool observations you collect.

{tool descriptions, as above}

FORMAT --- you must follow this exact format on each turn:

Thought: <what information do I still need? which tool will give it to me?>
Action: <tool_name> OR finish

If you choose a tool, you will receive an Observation with the tool's output.
Then you continue with another Thought/Action.

When you have enough information (or no remaining tool will help), use:
Action: finish

Each Thought should plan the NEXT step --- do not state a final yes/no answer
during iterations. Save that for after Action: finish.

RULES:
  -- Each tool can only be called ONCE --- no repeated calls
  -- You do not have to use all tools --- stop when you have enough information
  -- A maximum of 3 iterations is allowed
```

--------------------------------------------------------------------------------
## Step 3 — Reading and synthesis

### Reading Turn
```
[System]
You will be given a situation and cultural evidence. Your task is to extract
from the evidence the cultural information that is relevant to the situation.

Quote relevant passages directly from the evidence, then briefly explain how
each relates to the situation. Focus on cultural norms, attitudes, or practices.

[User]
{scenario: Country/Story or Rule-of-thumb/Story}

{evidence from one tool}

What in this evidence is relevant to the situation above?
```

### Synthesis
```
You previously analysed a cultural scenario and gave an initial answer with
reasoning. You now have new cultural evidence from research tools.

Do not make any extra inferences about actions outside of the given context and
the provided cultural evidence. Only align to the scenario and the evidence
given.

INSTRUCTIONS:
1. Review your initial reasoning and answer.
2. Read the evidence. Identify which specific pieces are relevant to the
   scenario, quoting them in your reasoning. If no evidence is relevant, state
   exactly: "No relevant evidence found for this situation."
3. For the relevant evidence, analyse whether it supports or contradicts your
   initial answer.
4. State your final answer (Yes, No, or Neither). If it differs from your
   initial answer, explain what specific evidence justifies the change. If it
   matches your initial answer, explain what supports keeping it.

Format your response EXACTLY as:
Answer: <Yes|No|Neither>
Explanation: <your analysis>
```

--------------------------------------------------------------------------------
## Judge rubric 

The LLM judge scores each explanation on three independent questions, returned
as JSON with a one-sentence justification each. (In the code the third question
is named `general_sufficient`; the paper numbers the three retained questions
Q1-Q3.)

- **Q1 Cultural attribution** (`cultural_reasoning`, yes/no): does the
  explanation reach its answer by appealing to what a specific group does,
  values, or believes?
- **Q2 Applied given rule** (`used_given_clue`, yes/no; asked only for SCV and
  SRoT): does the explanation apply the given rule/value to reach the answer?
- **Q3 Plain explanation suffices** (`general_sufficient`,
  yes/no/no_cultural_claim): could a general, non-cultural explanation have
  reached the same verdict?

Derived measures:
- `over_cult_general = Q1 AND (Q3 == yes)` --- cultural reasoning used where a
  plain explanation would have sufficed.
- `over_cult_clue = Q1 AND (NOT Q2)` --- cultural reasoning while abandoning the
  given rule (SRoT/SCV only).
