"""
MarketForge AI — LangGraph compiled graphs.

All nine department graphs and the master pipeline are exposed here.
Import the graph objects directly for use in Airflow DAGs, tests, or the API.

Usage:
    from marketforge.agents.graphs import master_graph, user_insights_graph
    from marketforge.agents.graphs.master import run_full_pipeline
    from marketforge.agents.graphs.user_insights import run_career_analysis

LangSmith Studio setup:
    Set the following environment variables to enable full graph tracing
    and the interactive graph view in LangSmith Studio:

        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_API_KEY=<your-langsmith-api-key>
        LANGCHAIN_PROJECT=marketforge-ai

    Once set, every graph.ainvoke() call automatically traces to LangSmith.
    Open studio.langsmith.com → Projects → marketforge-ai → any run →
    click "View Trace" to see the full interactive node graph.
"""

from marketforge.agents.graphs.data_collection import data_collection_graph
from marketforge.agents.graphs.market_analysis  import market_analysis_graph
from marketforge.agents.graphs.research         import research_graph
from marketforge.agents.graphs.content_studio   import content_studio_graph
from marketforge.agents.graphs.user_insights    import user_insights_graph
from marketforge.agents.graphs.security         import security_graph
from marketforge.agents.graphs.ml_engineering   import ml_engineering_graph
from marketforge.agents.graphs.qa_testing       import qa_graph
from marketforge.agents.graphs.ops_monitor      import ops_graph
from marketforge.agents.graphs.master           import master_graph

__all__ = [
    "data_collection_graph",
    "market_analysis_graph",
    "research_graph",
    "content_studio_graph",
    "user_insights_graph",
    "security_graph",
    "ml_engineering_graph",
    "qa_graph",
    "ops_graph",
    "master_graph",
]
