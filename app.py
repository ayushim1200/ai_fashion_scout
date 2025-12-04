import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool # Tool for general web search
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

# Load environment variables (API keys) from .env file
load_dotenv()

# --- 1. Pydantic Output Schema for structured data ---
# This ensures the output is predictable and easy for the frontend to parse.
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
app = FastAPI(title="AI Fashion Scout API", description="CrewAI powered fashion recommendation service.")

class QueryInput(BaseModel):
    query: str = Field(..., description="The user's fashion request, e.g., 'black floral dress, midi length, under $100'")

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
    
    # Since the final task enforces a JSON output, we return the parsed result
    return CrewOutput.model_validate_json(result).model_dump()


# --- 4. FastAPI Endpoint ---

@app.post("/recommend_outfit")
async def recommend_outfit(input: QueryInput, background_tasks: BackgroundTasks):
    """
    Triggers the CrewAI Fashion Scout process with the user query.
    Note: For long-running tasks, you'd typically use a proper queue like Celery 
    or return a Job ID for polling, but this synchronous call demonstrates the core concept.
    """
    
    # Run the crew and return the result
    try:
        # Note: CrewAI is synchronous, so this call will block the thread
        # In production, use background task processing or async LLM clients.
        recommendations = run_fashion_scout_crew(input.query)
        return recommendations
    except Exception as e:
        # Log the error and return a 500
        print(f"Error processing query '{input.query}': {e}")
        raise HTTPException(status_code=500, detail=f"AI Agent failed to complete the task: {e}")

# Example command to run locally: uvicorn app:app --reload --port 8000
