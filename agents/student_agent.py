# Student agent: Add your own agent here
from agents.minmax_agent import MinMaxAgent
from store import register_agent

@register_agent("student_agent")
# use minmax agent
class StudentAgent(MinMaxAgent):
    def __init__(self):
        super(StudentAgent, self).__init__()