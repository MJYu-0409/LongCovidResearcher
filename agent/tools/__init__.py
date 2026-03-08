from agent.tools.search    import search_literature
from agent.tools.paper     import get_paper_detail
from agent.tools.sentiment import analyze_sentiment
from agent.tools.qa        import answer_question
from agent.tools.synthesis import synthesize_review

ALL_TOOLS = [
    search_literature,
    get_paper_detail,
    analyze_sentiment,
    answer_question,
    synthesize_review,
]
