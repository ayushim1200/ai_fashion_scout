import os
# --- Core Imports ---
from crewai_tools import SerperDevTool
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM 
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
    query: str = Field(..., description="The user's full fashion request (style, budget, details).")

# --- Frontend Serving Endpoint ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    """Serves the index.html file at the root path."""
    return FileResponse('index.html')


# --- 3. CrewAI Components ---

# Initialize the Search Tool
search_tool = SerperDevTool() 

# --- LLM Setup (Stable) ---

# Check for required environment variables early
if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
    # If no key is set, the crew cannot run the analysis, but we can still search.
    agent_llm = None
    LLM_AVAILABLE = False
else:
    LLM_AVAILABLE = True
    # Stable initialization logic:
    if os.environ.get("GEMINI_API_KEY"):
        agent_llm = LLM(
            model="gemini-2.5-flash", 
            config={"api_key": os.environ.get("GEMINI_API_KEY")} 
        )
    elif os.environ.get("OPENAI_API_KEY"):
        agent_llm = LLM(
            model="gpt-4o-mini",
            config={"api_key": os.environ.get("OPENAI_API_KEY")}
        )
    else:
        # Should be caught by the check above, but keeps the flow clean
        agent_llm = None


def run_fashion_scout_crew(query: str) -> dict:
    """Initializes and runs the CrewAI process."""
    
    # --- SOLUTION 1: Skip LLM Analysis if Quota is Hit or Key is Missing ---
    if not LLM_AVAILABLE:
        print("--- LLM UNAVAILABLE: Performing Direct Search ---")
        try:
            # Execute the search tool directly to get raw results
            raw_results = search_tool.run(f"online shopping for {query}")
            
            # Simple manual parsing (less precise than AI, but fast and avoids LLM quota)
            import json
            results = json.loads(raw_results)
            
            recommendations = []
            for item in results.get('organic', [])[:3]:
                recommendations.append(Recommendation(
                    title=item.get('title', 'N/A'),
                    price=item.get('snippet', 'Price not found'),
                    url=item.get('link', '#'),
                    source=item.get('source', 'Web Search')
                ))

            return CrewOutput(
                summary="Quota exceeded or LLM key missing. Displaying top 3 raw search results (less precise filtering).",
                recommendations=recommendations
            ).model_dump()
            
        except Exception as e:
            # If Serper search fails (e.g., SERPER_API_KEY is missing/wrong), raise generic error
            raise ValueError(f"Search failed. Check SERPER_API_KEY. Error: {e}")

    
    # --- SOLUTION 2: Full CrewAI (If LLM is Available) ---

    # 3.1. Define Agents
    fashion_researcher = Agent(
        role='Search Commander',
        goal=f"Accurately find the top 5 relevant online shopping results for: {query}.",
        backstory="An expert at filtering noise and finding retail links across major e-commerce platforms.",
        tools=[search_tool],
        llm=agent_llm, 
        verbose=True,
        allow_delegation=False
    )

    fashion_analyst = Agent(
        role='Fashion Analyst',
        goal='Select the 3 best products that match the user request based on the provided data, and format the output perfectly.',
        backstory="A meticulous product analyst focused solely on formatting the final structured recommendations from the raw search data.",
        llm=agent_llm, 
        verbose=True,
        allow_delegation=False,
        output_json=CrewOutput,
        cache=True 
    )

    # 3.2. Define Tasks
    research_task = Task(
        description=(
            f"1. Perform a deep web search for the user request: '{query}'. 2. Compile the top 5 most relevant product titles, prices, URLs, and snippets from the organic search results into a clean markdown list for the Analyst."
        ),
        agent=fashion_researcher,
        expected_output="A clean, structured markdown list of the top 5 relevant product listings (Title, Price, URL, Source Snippet)."
    )

    analysis_task = Task(
        description=(
            f"Review the list of search results. Select the 3 items that most closely match the user's request: '{query}'. For each selected item, extract the Title, Price, URL, and Source (website name) and package the final result into the required JSON schema. The summary must be friendly and confirm the search criteria."
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
        verbose=True 
    )

    # 3.4. Kickoff the Crew
    print(f"--- Kicking off crew for query: {query} ---")
    
    result = fashion_crew.kickoff(inputs={'query': query})
    
    # *** FINAL FIX: Extract the raw string from the result before parsing ***
    if hasattr(result, 'raw'):
        # If result is a CrewAI object, extract the raw string
        final_json_string = str(result.raw).strip()
    elif isinstance(result, str):
        # If result is already a string (which it should be), clean it up
        final_json_string = result.strip()
    else:
        raise TypeError(f"Crew output format unexpected: {type(result)}. Expected string or Crew object.")

    # Remove markdown code fences if present (e.g., ```json\n...\n```)
    if final_json_string.startswith("```"):
        # This handles cases where the LLM wraps the JSON in markdown code blocks
        import re
        match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", final_json_string)
        if match:
            final_json_string = match.group(1)
        
    # Final validation and return
    return CrewOutput.model_validate_json(final_json_string).model_dump()


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
        # Check if the failure is related to the LLM quota
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            # This allows the frontend to display a cleaner error
            raise HTTPException(status_code=503, detail="LLM Quota Exceeded. Please wait 5 minutes and try again.")
        
        print(f"Error processing query '{input.query}': {e}")
        # Return a generic 500 error for all other issues
        raise HTTPException(status_code=500, detail="The AI Agents failed to complete the search.")
