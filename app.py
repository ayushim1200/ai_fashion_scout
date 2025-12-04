import os
# NEW: Import SerperDevTool from crewai_tools
from crewai_tools import SerperDevTool
from crewai import Agent, Task, Crew, Process
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
from starlette.responses import FileResponse, HTMLResponse
from contextlib import asynccontextmanager

# Load environment variables (for local testing only)
load_dotenv()

# --- 1. Pydantic Output Schema for structured data ---
class Recommendation(BaseModel):
    """A single product recommendation item."""
    title: str = Field(description="The product title or a brief description.")
    price: str = Field(description="The price of the item, including currency.")
    url: str = Field(description="The direct link to the product page.")
    source: str = Field(description="The retailer or website the product was found on.")

class CrewOutput(BaseModel):
    """The final structured output of the Crew."""
    summary: str = Field(description="A friendly, concise summary explaining the top recommendations.")
    recommendations: List[Recommendation] = Field(description="A list of the top 3 recommended items.")

# --- 2. FastAPI Setup ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup/Shutdown logic
    print("FastAPI application starting up.")
    yield
    print("FastAPI application shutting down.")

app = FastAPI(
    title="AI Fashion Scout API", 
    description="CrewAI powered fashion recommendation service.",
    lifespan=lifespan
)

class QueryInput(BaseModel):
    # This query contains the full, combined request from the chatbot's frontend state.
    query: str = Field(..., description="The user's full fashion request (style, budget, details).")

# --- Frontend Serving Endpoint ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    """Serves the index.html file at the root path."""
    return FileResponse('index.html')


# --- 3. CrewAI Components ---

# Initialize the Search Tool (Now correctly defined)
search_tool = SerperDevTool() 

# --- LLM Setup ---
# We use a provider-specific import for the Gemini model to ensure the correct class is used.
from crewai_tools.llms import Gemini as GeminiLLM # Aliasing the import to avoid name conflict with crewai.llm.LLM
from crewai.llm import LLM # Keep this import for the crew

# Define the model to use
MODEL_NAME = "gemini-2.5-flash"

# Check for GEMINI_API_KEY first
if os.environ.get("GEMINI_API_KEY"):
    GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
    # Initialize the LLM using the model name and config.
    # We use a standard provider setup which should work with the environment variable.
    
    # NOTE: To use the environment key, the cleanest way is often to use the 
    # specific class (like GeminiLLM) or pass the parameters directly to the base LLM wrapper.
    # Let's use the explicit Gemini class for the best stability.
    
    # Fallback to the generic LLM wrapper if the specific provider class fails or is not preferred.
    try:
        agent_llm = GeminiLLM(
            model=MODEL_NAME, 
            api_key=GEMINI_KEY
        )
    except ImportError:
        # If the direct provider import fails, fall back to the generic LLM constructor
        # and rely on the model name and key being passed explicitly.
        agent_llm = LLM(
            model=MODEL_NAME, 
            config={"api_key": GEMINI_KEY}
        )

else:
    # If GEMINI is not set, we fall back to check for the OpenAI key (as the agent was doing)
    # NOTE: This fallback might still be prone to import errors if the corresponding
    # OpenAI model library (like `crewai-tools.llms.OpenAI`) isn't available/configured.
    
    # Since the system is failing due to LLM initialization, we must ensure the fallback
    # is also correctly defined or completely removed if only Gemini is intended.
    # For now, let's prioritize the Gemini path and raise a clean error if keys are missing.
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("FATAL ERROR: Neither GEMINI_API_KEY nor OPENAI_API_KEY is set in environment variables.")
    
    # If OpenAI key exists (as a safety net)
    agent_llm = LLM(
        model="gpt-4o-mini",
        config={"api_key": os.environ.get("OPENAI_API_KEY")}
    )


def run_fashion_scout_crew(query: str) -> dict:
# ... (rest of the run_fashion_scout_crew function remains the same)
    
    # 3.1. Define Agents
    fashion_researcher = Agent(
        role='Search Commander',
        goal=f"Accurately find the top 10 relevant online shopping results for: {query}.",
        backstory="An expert at filtering noise and finding retail links across major e-commerce platforms.",
        tools=[search_tool],
        llm=agent_llm, 
        verbose=True,
        allow_delegation=False
    )

    fashion_analyst = Agent(
        role='Fashion Analyst',
        goal='Analyze search results to find the top 3 items that strictly meet all user-specified criteria.',
        backstory="A meticulous product analyst who filters search data based on constraints and prepares structured recommendations.",
        llm=agent_llm, 
        verbose=True,
        allow_delegation=False,
        output_json=CrewOutput,
        cache=True 
    )

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


# --- 4. FastAPI Endpoint ---

@app.post("/recommend_outfit")
async def recommend_outfit(input: QueryInput):
    """
    Triggers the CrewAI Fashion Scout process with the user query.
    """
    
    try:
        recommendations = run_fashion_scout_crew(input.query)
        return recommendations
    except Exception as e:
        print(f"Error processing query '{input.query}': {e}")
        raise HTTPException(status_code=500, detail="The AI Agents failed to complete the search.")
