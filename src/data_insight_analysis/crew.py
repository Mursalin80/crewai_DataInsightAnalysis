import yaml
from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FileReadTool

from .llm.llms import llm_google_gemini, llm_groq_llama_3, llm_together

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators
csv_tool = FileReadTool(file_path="./support_tickets_data.csv")


@CrewBase
class DataInsightAnalysis:
    """DataInsightAnalysis crew"""

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools

    # suggestion_generation_agent
    @agent
    def suggestion_generation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["suggestion_generation_agent"],
            verbose=True,
            tools=[csv_tool],
            llm=llm_together,
        )

    # reporting_agent
    @agent
    def reporting_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["reporting_agent"],
            verbose=True,
            tools=[csv_tool],
            llm=llm_groq_llama_3,
        )

    # chart_generation_agent
    @agent
    def chart_generation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["chart_generation_agent"],
            verbose=True,
            allow_code_execution=True,
            llm=llm_google_gemini,
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def suggestion_generation(self) -> Task:
        return Task(
            config=self.tasks_config["suggestion_generation"],
            agent=self.suggestion_generation_agent(),
        )

    @task
    def table_generation(self) -> Task:
        return Task(
            config=self.tasks_config["table_generation"], agent=self.reporting_agent()
        )

    @task
    def chart_generation(self) -> Task:
        return Task(
            config=self.tasks_config["chart_generation"],
            agent=self.chart_generation_agent(),
        )

    @task
    def final_report_assembly(self) -> Task:
        return Task(
            config=self.tasks_config["final_report_assembly"],
            agent=self.reporting_agent(),
            context=[
                self.suggestion_generation(),
                self.table_generation(),
                self.chart_generation(),
            ],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the DataInsightAnalysis crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical,  # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
