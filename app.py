import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
from starlette.responses import FileResponse, HTMLResponse
from contextlib import asynccontextmanager

# Load environment variables (API keys) from .env file
# NOTE: This only works locally. On Render, keys are provided via environment variables.
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

# Define the lifespan for startup/shutdown events (optional but good practice)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize any resources here if needed
    print("FastAPI application starting up.")
    yield
    # Shutdown: Clean up resources here if needed
    print("FastAPI application shutting down.")

app = FastAPI(
    title="AI Fashion Scout API", 
    description="CrewAI powered fashion recommendation service.",
    lifespan=lifespan
)

class QueryInput(BaseModel):
    query: str = Field(..., description="The user's fashion request, e.g., 'black floral dress, midi length, under $100'")

# --- Frontend Serving Endpoint ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    """Serves the index.html file at the root path."""
    # Ensure index.html is copied to the root directory in the Dockerfile
    return FileResponse('index.html')


# --- 3. CrewAI Components ---

# Initialize the Search Tool
search_tool = SerperDevTool() 

def run_fashion_scout_crew(query: str) -> dict:
    """Initializes and runs the CrewAI process."""
    
    # 3.1. Define Agents
    fashion_researcher = Agent(
        role='Search Commander',
        goal=f"Accurately find the top 10 relevant online shopping results for: {query}.",
        backstory="An expert at filtering noise and finding retail links across major e-commerce platforms.",
        tools=[search_tool],
        verbose=True,
        allow_delegation=False
    )

    fashion_analyst = Agent(
        role='Fashion Analyst',
        goal='Analyze search results to find the top 3 items that strictly meet all user-specified criteria (price, style, color, length).',
        backstory="A meticulous product analyst who filters search data based on constraints and prepares structured recommendations.",
        verbose=True,
        allow_delegation=False,
        output_json=CrewOutput, # Enforce structured JSON output
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
            f"1. Review the research data provided by the Search Commander. 2. Strictly filter the results to match the user's constraints, which include: '{query}'. 3. Synthesize the final, best 3 matches into the required JSON format."
        ),
        agent=fashion_analyst,
        context=[research_task], # Analyst relies on Researcher's output
        expected_output="A JSON object conforming to the CrewOutput Pydantic schema, containing a friendly summary and the top 3 best-fit product recommendations."
    )

    # 3.3. Define the Crew
    fashion_crew = Crew(
        agents=[fashion_researcher, fashion_analyst],
        tasks=[research_task, analysis_task],
        process=Process.sequential,
        verbose=2 # Show detailed agent thought process in logs
    )

    # 3.4. Kickoff the Crew
    print(f"--- Kicking off crew for query: {query} ---")
    
    # The kickoff returns the final JSON string from the analysis_task
    result = fashion_crew.kickoff(inputs={'query': query})
    
    # Since the final task enforces a JSON output, we parse and return the data
    return CrewOutput.model_validate_json(result).model_dump()


# --- 4. FastAPI Endpoint ---

@app.post("/recommend_outfit")
async def recommend_outfit(input: QueryInput):
    """
    Triggers the CrewAI Fashion Scout process with the user query.
    """
    
    try:
        # Note: This is synchronous, which can lead to timeouts. For a real production app, 
        # you would use a job queue (like Celery) or a background task manager.
        # For a lab setting, this demonstrates the core function.
        recommendations = run_fashion_scout_crew(input.query)
        return recommendations
    except Exception as e:
        # Log the error and return a 500
        print(f"Error processing query '{input.query}': {e}")
        # Return a generic error to the frontend
        raise HTTPException(status_code=500, detail="The AI Agents failed to complete the search.")
