# ... (existing imports, pydantic models, and FastAPI setup) ...

# --- 3. CrewAI Components ---

# Initialize the Search Tool
search_tool = SerperDevTool() 

# --- NEW: LLM Setup ---
# We define a single LLM configuration using the key we have on Render.
# We are assuming you are using the 'gemini-2.5-flash' model.
from crewai.llm import LLM 

# Check for GEMINI_API_KEY first
if os.environ.get("GEMINI_API_KEY"):
    GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
    llm_config = {
        "model": "gemini-2.5-flash", # Use a fast model for the web service
        "api_key": GEMINI_KEY 
    }
else:
    # If GEMINI is not set, we fall back to check for the OpenAI key (as the agent is doing)
    # The log shows it is looking for OPENAI_API_KEY, so we handle it here:
    llm_config = {
        "model": "gpt-4o-mini",
        "api_key": os.environ.get("OPENAI_API_KEY") # This will crash if not found
    }

# The actual LLM instance
agent_llm = LLM(config=llm_config)


def run_fashion_scout_crew(query: str) -> dict:
    """Initializes and runs the CrewAI process."""
    
    # 3.1. Define Agents
    # We now pass the configured 'agent_llm' to each agent explicitly.
    fashion_researcher = Agent(
        role='Search Commander',
        goal=f"Accurately find the top 10 relevant online shopping results for: {query}.",
        backstory="An expert at filtering noise and finding retail links across major e-commerce platforms.",
        tools=[search_tool],
        llm=agent_llm,  # <-- PASS THE LLM HERE
        verbose=True,
        allow_delegation=False
    )

    fashion_analyst = Agent(
        role='Fashion Analyst',
        goal='Analyze search results to find the top 3 items that strictly meet all user-specified criteria.',
        backstory="A meticulous product analyst who filters search data based on constraints and prepares structured recommendations.",
        llm=agent_llm, # <-- PASS THE LLM HERE
        verbose=True,
        allow_delegation=False,
        output_json=CrewOutput,
        cache=True 
    )

    # ... (rest of the tasks and crew definition is the same) ...

    # 3.2. Define Tasks
    research_task = Task(
        description=f"1. Perform a deep web search for the user request: '{query}'. 2. Compile all relevant product titles, prices, URLs, and snippets into a clean markdown list for the Analyst.",
        agent=fashion_researcher,
        expected_output="A clean, structured markdown list of at least 10 relevant product listings (Title, Price, URL, Source Snippet)."
    )

    analysis_task = Task(
        description=(
            f"1. Review the research data provided by the Search Commander. 2. Strictly filter the results to match the user's constraints from the query: '{query}'. 3. Synthesize the final, best 3 matches into the required JSON format."
        ),
        agent=fashion_analyst,
        context=[research_task],
        expected_output="A JSON object conforming to the CrewOutput Pydantic schema, containing a friendly summary and the top 3 best-fit product recommendations."
    )

    # 3.3. Define the Crew
    fashion_crew = Crew(
        agents=[fashion_researcher, fashion_analyst],
        tasks=[research_task, analysis_task],
        process=Process.sequential,
        verbose=2
    )

    # 3.4. Kickoff the Crew
    print(f"--- Kicking off crew for query: {query} ---")
    
    result = fashion_crew.kickoff(inputs={'query': query})
    
    return CrewOutput.model_validate_json(result).model_dump()
